"""Pretraining script for multi-scale neural network.

This script implements the pretraining pipeline:
- Load configuration
- Load and preprocess data
- Initialize model and optimizer
- Run pretraining loop with logging and checkpointing

Requirements:
    - Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5
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
from src.training.pretraining import PretrainingTrainer, PretrainingConfig
from src.training.pretraining_losses import PretrainingLossWeights
from src.data.loader import load_md_trajectories
from src.data.structures import GraphDataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Pretrain multi-scale neural network on diverse MD simulations'
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
        default='data/train',
        help='Path to training data directory'
    )
    parser.add_argument(
        '--val-data',
        type=str,
        default=None,
        help='Path to validation data directory'
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
    
    # Checkpointing
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='checkpoints/pretraining',
        help='Directory to save checkpoints'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
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
    print("Loading training data...")
    
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


def create_trainer(model, args, cfg):
    """Create pretraining trainer.
    
    Args:
        model: MultiScaleModel instance
        args: Command line arguments
        cfg: Configuration object
    
    Returns:
        PretrainingTrainer instance
    """
    # Get loss weights from config
    loss_weights = PretrainingLossWeights(
        autoregressive=cfg.training.loss_weights.autoregressive,
        contrastive=cfg.training.loss_weights.contrastive,
        masked_particle=cfg.training.loss_weights.masked_particle
    )
    
    # Create pretraining config
    config = PretrainingConfig(
        # Optimization
        learning_rate=args.lr if args.lr else cfg.training.optimizer.lr,
        weight_decay=cfg.training.optimizer.weight_decay,
        max_grad_norm=cfg.training.training.gradient_clip_norm,
        
        # Training
        num_epochs=args.epochs if args.epochs else cfg.training.training.epochs,
        batch_size=args.batch_size if args.batch_size else cfg.training.data.batch_size,
        
        # Loss weights
        loss_weights=loss_weights,
        
        # Data
        dt=cfg.training.data.get('dt', 1.0),
        
        # Checkpointing
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=cfg.training.training.checkpoint_interval,
        
        # Logging
        log_interval=args.log_interval,
        
        # Device
        device=args.device if args.device else cfg.model.device
    )
    
    # Create trainer
    trainer = PretrainingTrainer(model, config)
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    return trainer


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    print("="*60)
    print("Multi-Scale Neural Network - Pretraining")
    print("="*60)
    
    # Load configuration
    print("\nLoading configuration...")
    cfg = load_config(
        config_name=args.config,
        config_path=args.config_path,
        overrides=args.overrides
    )
    
    # Setup experiment
    print("Setting up experiment...")
    setup_experiment(cfg)
    
    # Print configuration
    print("\nConfiguration:")
    print(f"  Model: {cfg.model.get('name', 'MultiScaleModel')}")
    print(f"  Device: {args.device if args.device else cfg.model.device}")
    print(f"  Epochs: {args.epochs if args.epochs else cfg.training.training.epochs}")
    print(f"  Batch size: {args.batch_size if args.batch_size else cfg.training.data.batch_size}")
    print(f"  Learning rate: {args.lr if args.lr else cfg.training.optimizer.lr}")
    print(f"  Checkpoint dir: {args.checkpoint_dir}")
    
    # Load data
    train_loader, val_loader = load_data(args, cfg)
    
    if train_loader is None:
        print("\nError: No training data available")
        print("Please provide training data or create synthetic data")
        return
    
    # Create model
    print("\nCreating model...")
    model = create_model_from_config(cfg)
    
    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Create trainer
    print("\nCreating trainer...")
    trainer = create_trainer(model, args, cfg)
    
    # Train
    print("\n" + "="*60)
    print("Starting pretraining...")
    print("="*60 + "\n")
    
    try:
        history = trainer.train(train_loader, val_loader)
        
        # Save training history
        trainer.save_history()
        
        print("\n" + "="*60)
        print("Pretraining completed successfully!")
        print("="*60)
        print(f"\nCheckpoints saved to: {args.checkpoint_dir}")
        print(f"Final training loss: {history['train_loss'][-1]:.4f}")
        if val_loader is not None:
            print(f"Best validation loss: {trainer.best_loss:.4f}")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Saving checkpoint...")
        trainer.save_checkpoint('interrupted.pt')
        print(f"Checkpoint saved to: {args.checkpoint_dir}/interrupted.pt")
    
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()
        print("\nSaving checkpoint...")
        trainer.save_checkpoint('error.pt')
        print(f"Checkpoint saved to: {args.checkpoint_dir}/error.pt")
        raise


if __name__ == '__main__':
    main()
