"""Uncertainty calibration script for multi-scale neural network.

This script implements the uncertainty calibration pipeline:
- Train ensemble or MC dropout model
- Calibrate temperature on validation set
- Save calibrated model with uncertainty estimates

Requirements:
    - Validates: Requirements 10.1, 10.2, 10.3, 10.4, 10.5
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader

from src.config.config_loader import load_config, create_model_from_config, setup_experiment
from src.training.uncertainty import (
    DeepEnsemble, MCDropoutModel, AleatoricModel,
    UncertaintyConfig, calibrate_temperature, TemperatureScaledModel
)
from src.data.loader import load_md_trajectories
from src.data.structures import GraphDataset
from src.models.multi_scale_model import ModelConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Calibrate uncertainty for multi-scale neural network'
    )
    
    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        default='config',
        help='Name of config file (without .yaml extension)'
    )
    parser.add_argument(
        '--config-path',
        type=str,
        default=None,
        help='Path to config directory (default: configs/)'
    )
    parser.add_argument(
        '--overrides',
        nargs='*',
        default=[],
        help='Config overrides in format key=value'
    )
    
    # Data
    parser.add_argument(
        '--train-data',
        type=str,
        default='data/finetune_train',
        help='Path to training data directory'
    )
    parser.add_argument(
        '--val-data',
        type=str,
        default='data/finetune_val',
        help='Path to validation data directory'
    )
    
    # Pretrained model
    parser.add_argument(
        '--pretrained',
        type=str,
        required=True,
        help='Path to pretrained/fine-tuned model checkpoint'
    )
    
    # Uncertainty method
    parser.add_argument(
        '--method',
        type=str,
        choices=['ensemble', 'mc_dropout', 'aleatoric'],
        default='ensemble',
        help='Uncertainty estimation method'
    )
    
    # Ensemble parameters
    parser.add_argument(
        '--num-ensemble',
        type=int,
        default=None,
        help='Number of models in ensemble (overrides config)'
    )
    
    # MC Dropout parameters
    parser.add_argument(
        '--dropout-rate',
        type=float,
        default=None,
        help='Dropout rate for MC dropout (overrides config)'
    )
    parser.add_argument(
        '--num-mc-samples',
        type=int,
        default=None,
        help='Number of MC samples (overrides config)'
    )
    
    # Training
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs (overrides config)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size (overrides config)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Learning rate (overrides config)'
    )
    
    # Temperature scaling
    parser.add_argument(
        '--no-temperature-scaling',
        action='store_true',
        help='Disable temperature scaling'
    )
    
    # Checkpointing
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='checkpoints/uncertainty',
        help='Directory to save checkpoints'
    )
    
    # Device
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to train on (cuda or cpu)'
    )
    
    # Logging
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        help='Log metrics every N batches'
    )
    
    return parser.parse_args()


def load_data(args, cfg):
    """Load and preprocess training data.
    
    Args:
        args: Command line arguments
        cfg: Configuration object
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    print("Loading uncertainty calibration data...")
    
    # Get data paths
    train_path = args.train_data if args.train_data else cfg.training.data.train_path
    val_path = args.val_data if args.val_data else cfg.training.data.get('val_path', None)
    
    # Load training trajectories
    if os.path.exists(train_path):
        train_trajectories = load_md_trajectories(train_path)
        print(f"Loaded {len(train_trajectories)} training trajectories")
    else:
        print(f"Warning: Training data path {train_path} does not exist")
        print("Creating dummy dataset for demonstration")
        train_trajectories = []
    
    # Load validation trajectories
    val_trajectories = None
    if val_path and os.path.exists(val_path):
        val_trajectories = load_md_trajectories(val_path)
        print(f"Loaded {len(val_trajectories)} validation trajectories")
    
    # Create datasets
    train_dataset = GraphDataset(train_trajectories) if train_trajectories else None
    val_dataset = GraphDataset(val_trajectories) if val_trajectories else None
    
    # Get batch size
    batch_size = args.batch_size if args.batch_size else cfg.training.data.batch_size
    num_workers = cfg.training.data.get('num_workers', 0)
    
    # Create data loaders
    train_loader = None
    if train_dataset:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=cfg.training.data.get('shuffle', True),
            num_workers=num_workers,
            collate_fn=train_dataset.collate_fn
        )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=val_dataset.collate_fn
        )
    
    return train_loader, val_loader


def create_uncertainty_config(args, cfg) -> UncertaintyConfig:
    """Create uncertainty calibration configuration.
    
    Args:
        args: Command line arguments
        cfg: Configuration object
    
    Returns:
        UncertaintyConfig instance
    """
    # Get method-specific parameters
    if args.method == 'ensemble':
        num_ensemble = args.num_ensemble if args.num_ensemble else cfg.training.ensemble.num_models
        dropout_rate = 0.0
        num_mc_samples = 1
    elif args.method == 'mc_dropout':
        num_ensemble = 1
        dropout_rate = args.dropout_rate if args.dropout_rate else cfg.training.mc_dropout.dropout_rate
        num_mc_samples = args.num_mc_samples if args.num_mc_samples else cfg.training.mc_dropout.num_samples
    else:  # aleatoric
        num_ensemble = 1
        dropout_rate = 0.0
        num_mc_samples = 1
    
    # Create config
    config = UncertaintyConfig(
        # Ensemble parameters
        num_ensemble_models=num_ensemble,
        
        # MC Dropout parameters
        dropout_rate=dropout_rate,
        num_mc_samples=num_mc_samples,
        
        # Optimization
        learning_rate=args.lr if args.lr else cfg.training.optimizer.lr,
        weight_decay=cfg.training.optimizer.weight_decay,
        max_grad_norm=cfg.training.training.gradient_clip_norm,
        
        # Training
        num_epochs=args.epochs if args.epochs else cfg.training.training.epochs,
        batch_size=args.batch_size if args.batch_size else cfg.training.data.batch_size,
        
        # Temperature scaling
        use_temperature_scaling=not args.no_temperature_scaling and cfg.training.temperature_scaling.enabled,
        
        # Checkpointing
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=cfg.training.training.checkpoint_interval,
        pretrained_checkpoint=args.pretrained,
        
        # Logging
        log_interval=args.log_interval,
        
        # Device
        device=args.device if args.device else cfg.model.device
    )
    
    return config


def create_loss_function():
    """Create loss function for training.
    
    Returns:
        Loss function
    """
    # Simple MSE loss for demonstration
    # In practice, use appropriate loss from finetuning_losses
    return torch.nn.MSELoss()


def train_ensemble(model_config: ModelConfig,
                  train_loader: DataLoader,
                  val_loader: DataLoader,
                  uncertainty_config: UncertaintyConfig) -> DeepEnsemble:
    """Train deep ensemble for epistemic uncertainty.
    
    Args:
        model_config: Model configuration
        train_loader: Training data loader
        val_loader: Validation data loader
        uncertainty_config: Uncertainty configuration
    
    Returns:
        Trained DeepEnsemble model
    
    Requirements:
        - Validates: Requirement 10.1
    """
    print("\n" + "="*60)
    print("Training Deep Ensemble")
    print("="*60)
    print(f"Number of models: {uncertainty_config.num_ensemble_models}")
    
    # Create ensemble
    ensemble = DeepEnsemble(
        model_config=model_config,
        num_models=uncertainty_config.num_ensemble_models
    )
    
    # Create loss function
    loss_fn = create_loss_function()
    
    # Train ensemble
    histories = ensemble.train_ensemble(
        train_loader=train_loader,
        val_loader=val_loader,
        config=uncertainty_config,
        loss_fn=loss_fn
    )
    
    # Save ensemble
    ensemble.save_ensemble(uncertainty_config.checkpoint_dir)
    
    return ensemble


def train_mc_dropout(model_config: ModelConfig,
                    train_loader: DataLoader,
                    val_loader: DataLoader,
                    uncertainty_config: UncertaintyConfig) -> MCDropoutModel:
    """Train MC Dropout model for epistemic uncertainty.
    
    Args:
        model_config: Model configuration
        train_loader: Training data loader
        val_loader: Validation data loader
        uncertainty_config: Uncertainty configuration
    
    Returns:
        Trained MCDropoutModel
    
    Requirements:
        - Validates: Requirement 10.2
    """
    print("\n" + "="*60)
    print("Training MC Dropout Model")
    print("="*60)
    print(f"Dropout rate: {uncertainty_config.dropout_rate}")
    print(f"Number of MC samples: {uncertainty_config.num_mc_samples}")
    
    # Create MC Dropout model
    mc_model = MCDropoutModel(
        model_config=model_config,
        dropout_rate=uncertainty_config.dropout_rate
    )
    
    # Create loss function
    loss_fn = create_loss_function()
    
    # Train model
    history = mc_model.train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        config=uncertainty_config,
        loss_fn=loss_fn
    )
    
    return mc_model


def train_aleatoric(model_config: ModelConfig,
                   train_loader: DataLoader,
                   val_loader: DataLoader,
                   uncertainty_config: UncertaintyConfig) -> AleatoricModel:
    """Train aleatoric uncertainty model.
    
    Args:
        model_config: Model configuration
        train_loader: Training data loader
        val_loader: Validation data loader
        uncertainty_config: Uncertainty configuration
    
    Returns:
        Trained AleatoricModel
    
    Requirements:
        - Validates: Requirement 10.3
    """
    print("\n" + "="*60)
    print("Training Aleatoric Uncertainty Model")
    print("="*60)
    
    # Create aleatoric model
    aleatoric_model = AleatoricModel(model_config=model_config)
    
    # Train model (uses NLL loss internally)
    history = aleatoric_model.train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        config=uncertainty_config
    )
    
    return aleatoric_model


def apply_temperature_scaling(model,
                             val_loader: DataLoader,
                             device: torch.device) -> TemperatureScaledModel:
    """Apply temperature scaling for calibration.
    
    Args:
        model: Trained model
        val_loader: Validation data loader
        device: Device
    
    Returns:
        Temperature-scaled model
    
    Requirements:
        - Validates: Requirement 10.4
    """
    print("\n" + "="*60)
    print("Applying Temperature Scaling")
    print("="*60)
    
    # Calibrate temperature
    optimal_temp = calibrate_temperature(
        model=model,
        val_loader=val_loader,
        device=device,
        max_iter=50
    )
    
    # Create temperature-scaled model
    scaled_model = TemperatureScaledModel(
        base_model=model,
        temperature=optimal_temp
    )
    
    return scaled_model


def evaluate_uncertainty(model,
                        val_loader: DataLoader,
                        device: torch.device,
                        method: str):
    """Evaluate uncertainty estimates on validation set.
    
    Args:
        model: Trained uncertainty model
        val_loader: Validation data loader
        device: Device
        method: Uncertainty method
    
    Requirements:
        - Validates: Requirement 10.5
    """
    print("\n" + "="*60)
    print("Evaluating Uncertainty Estimates")
    print("="*60)
    
    model.eval()
    model.to(device)
    
    total_epistemic = 0.0
    total_aleatoric = 0.0
    count = 0
    
    with torch.no_grad():
        for batch in val_loader:
            # Move batch to device
            if 'data' not in batch:
                continue
            
            data = batch['data'].to(device)
            t_span = batch.get('t_span', torch.tensor([0.0, 1.0], device=device))
            
            # Get predictions with uncertainty
            if method == 'ensemble':
                mean_pred, epistemic_unc = model.predict_with_uncertainty(data, t_span)
                aleatoric_unc = None
            elif method == 'mc_dropout':
                mean_pred, epistemic_unc = model.predict_with_uncertainty(
                    data, t_span, num_samples=50
                )
                aleatoric_unc = None
            elif method == 'aleatoric':
                mean_pred, aleatoric_unc = model(data, t_span)
                epistemic_unc = None
            else:
                continue
            
            # Accumulate uncertainties
            if epistemic_unc is not None and 'observables' in epistemic_unc:
                total_epistemic += epistemic_unc['observables'].mean().item()
            
            if aleatoric_unc is not None and 'observables' in aleatoric_unc:
                total_aleatoric += aleatoric_unc['observables'].mean().item()
            
            count += 1
    
    # Print summary
    if count > 0:
        if total_epistemic > 0:
            print(f"Average epistemic uncertainty: {total_epistemic / count:.6f}")
        if total_aleatoric > 0:
            print(f"Average aleatoric uncertainty: {total_aleatoric / count:.6f}")
        if total_epistemic > 0 and total_aleatoric > 0:
            total_unc = (total_epistemic + total_aleatoric) / count
            print(f"Average total uncertainty: {total_unc:.6f}")


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    print("="*60)
    print("Multi-Scale Neural Network - Uncertainty Calibration")
    print("="*60)
    
    # Load configuration
    print("\nLoading configuration...")
    
    # Override config to use uncertainty config
    overrides = ['training=uncertainty'] + args.overrides
    
    cfg = load_config(
        config_name=args.config,
        config_path=args.config_path,
        overrides=overrides
    )
    
    # Setup experiment
    print("Setting up experiment...")
    setup_experiment(cfg)
    
    # Print configuration
    print("\nConfiguration:")
    print(f"  Model: {cfg.model.get('name', 'MultiScaleModel')}")
    print(f"  Pretrained checkpoint: {args.pretrained}")
    print(f"  Uncertainty method: {args.method}")
    print(f"  Device: {args.device if args.device else cfg.model.device}")
    print(f"  Checkpoint dir: {args.checkpoint_dir}")
    
    if args.method == 'ensemble':
        num_models = args.num_ensemble if args.num_ensemble else cfg.training.ensemble.num_models
        print(f"  Ensemble size: {num_models}")
    elif args.method == 'mc_dropout':
        dropout = args.dropout_rate if args.dropout_rate else cfg.training.mc_dropout.dropout_rate
        samples = args.num_mc_samples if args.num_mc_samples else cfg.training.mc_dropout.num_samples
        print(f"  Dropout rate: {dropout}")
        print(f"  MC samples: {samples}")
    
    # Load data
    train_loader, val_loader = load_data(args, cfg)
    
    if train_loader is None:
        print("\nError: No training data available")
        print("Please provide training data or create synthetic data")
        return
    
    if val_loader is None:
        print("\nWarning: No validation data available")
        print("Temperature scaling will be skipped")
    
    # Create model config
    print("\nCreating model configuration...")
    model_config = ModelConfig(
        # GNN
        gnn_input_dim=cfg.model.gnn.input_dim,
        gnn_hidden_dim=cfg.model.gnn.hidden_dim,
        gnn_num_layers=cfg.model.gnn.num_layers,
        gnn_cutoff_radius=cfg.model.gnn.cutoff_radius,
        
        # Transformer
        transformer_hidden_dim=cfg.model.transformer.hidden_dim,
        transformer_num_heads=cfg.model.transformer.num_heads,
        transformer_dropout=cfg.model.transformer.dropout,
        
        # Encoder
        encoder_hidden_dims=cfg.model.encoder.hidden_dims,
        latent_dim=cfg.model.encoder.latent_dim,
        
        # Neural ODE
        ode_hidden_dims=cfg.model.neural_ode.hidden_dims,
        ode_solver=cfg.model.neural_ode.solver,
        ode_rtol=cfg.model.neural_ode.rtol,
        ode_atol=cfg.model.neural_ode.atol,
        
        # Decoder
        decoder_hidden_dims=cfg.model.decoder.hidden_dims,
        num_observables=cfg.model.decoder.num_observables,
        
        # Conditioning
        conditioning_dim=cfg.model.conditioning.dim,
        
        # Device
        device=args.device if args.device else cfg.model.device
    )
    
    # Create uncertainty config
    uncertainty_config = create_uncertainty_config(args, cfg)
    
    # Train model based on method
    device = torch.device(uncertainty_config.device)
    
    try:
        if args.method == 'ensemble':
            # Train deep ensemble
            model = train_ensemble(
                model_config=model_config,
                train_loader=train_loader,
                val_loader=val_loader,
                uncertainty_config=uncertainty_config
            )
        
        elif args.method == 'mc_dropout':
            # Train MC dropout model
            model = train_mc_dropout(
                model_config=model_config,
                train_loader=train_loader,
                val_loader=val_loader,
                uncertainty_config=uncertainty_config
            )
        
        elif args.method == 'aleatoric':
            # Train aleatoric model
            model = train_aleatoric(
                model_config=model_config,
                train_loader=train_loader,
                val_loader=val_loader,
                uncertainty_config=uncertainty_config
            )
        
        else:
            raise ValueError(f"Unknown uncertainty method: {args.method}")
        
        # Apply temperature scaling if enabled
        if uncertainty_config.use_temperature_scaling and val_loader is not None:
            model = apply_temperature_scaling(model, val_loader, device)
        
        # Evaluate uncertainty
        if val_loader is not None:
            evaluate_uncertainty(model, val_loader, device, args.method)
        
        print("\n" + "="*60)
        print("Uncertainty calibration completed successfully!")
        print("="*60)
        print(f"\nCheckpoints saved to: {args.checkpoint_dir}")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Checkpoints may have been saved during training")
    
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
