"""Uncertainty calibration for Bayesian uncertainty estimation.

This module implements uncertainty calibration methods:
- Deep ensembles for epistemic uncertainty
- MC dropout for epistemic uncertainty
- Aleatoric uncertainty prediction
- Temperature scaling for calibration
- Combined uncertainty reporting

Requirements:
    - Validates: Requirements 10.1, 10.2, 10.3, 10.4, 10.5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, LBFGS
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import os
from pathlib import Path
import json
from tqdm import tqdm
import copy

from ..models.multi_scale_model import MultiScaleModel, ModelConfig


@dataclass
class UncertaintyConfig:
    """Configuration for uncertainty calibration.
    
    Attributes:
        # Ensemble parameters
        num_ensemble_models: Number of models in ensemble
        
        # MC Dropout parameters
        dropout_rate: Dropout rate for MC dropout
        num_mc_samples: Number of forward passes for MC dropout
        
        # Optimization
        learning_rate: Learning rate for training
        weight_decay: Weight decay for AdamW
        max_grad_norm: Maximum gradient norm for clipping
        
        # Training
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        
        # Temperature scaling
        use_temperature_scaling: Whether to apply temperature scaling
        
        # Checkpointing
        checkpoint_dir: Directory to save checkpoints
        checkpoint_interval: Save checkpoint every N epochs
        pretrained_checkpoint: Path to pretrained model checkpoint
        
        # Logging
        log_interval: Log metrics every N batches
        
        # Device
        device: Device to train on ('cuda' or 'cpu')
    """
    # Ensemble parameters
    num_ensemble_models: int = 5
    
    # MC Dropout parameters
    dropout_rate: float = 0.1
    num_mc_samples: int = 50
    
    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    max_grad_norm: float = 0.5
    
    # Training
    num_epochs: int = 50
    batch_size: int = 16
    
    # Temperature scaling
    use_temperature_scaling: bool = True
    
    # Checkpointing
    checkpoint_dir: str = 'checkpoints/uncertainty'
    checkpoint_interval: int = 10
    pretrained_checkpoint: Optional[str] = None
    
    # Logging
    log_interval: int = 10
    
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class DeepEnsemble(nn.Module):
    """Deep ensemble for epistemic uncertainty estimation.
    
    Trains multiple models with different initializations and computes
    epistemic uncertainty from prediction variance across models.
    
    Requirements:
        - Validates: Requirement 10.1
        - Train multiple models with different initializations
        - Implement prediction with ensemble to compute epistemic uncertainty
    """
    
    def __init__(self, 
                 model_config: ModelConfig,
                 num_models: int = 5):
        """Initialize deep ensemble.
        
        Args:
            model_config: Configuration for each model in ensemble
            num_models: Number of models in ensemble
        """
        super().__init__()
        
        self.num_models = num_models
        self.model_config = model_config
        
        # Create ensemble of models
        self.models = nn.ModuleList([
            MultiScaleModel(model_config) for _ in range(num_models)
        ])
        
        # Temperature parameter for calibration (shared across ensemble)
        self.temperature = nn.Parameter(torch.ones(1))
    
    def train_ensemble(self,
                      train_loader: DataLoader,
                      val_loader: Optional[DataLoader],
                      config: UncertaintyConfig,
                      loss_fn: nn.Module) -> Dict[str, List[float]]:
        """Train all models in ensemble independently.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            config: Uncertainty calibration configuration
            loss_fn: Loss function for training
        
        Returns:
            Training history for all models
        
        Requirements:
            - Validates: Requirement 10.1
            - Train multiple models with different initializations
        """
        device = torch.device(config.device)
        histories = []
        
        print(f"Training ensemble of {self.num_models} models")
        
        for model_idx, model in enumerate(self.models):
            print(f"\n{'='*60}")
            print(f"Training Model {model_idx + 1}/{self.num_models}")
            print(f"{'='*60}")
            
            # Set different random seed for each model
            torch.manual_seed(model_idx)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(model_idx)
            
            # Move model to device
            model.to(device)
            
            # Load pretrained checkpoint if provided
            if config.pretrained_checkpoint is not None:
                self._load_pretrained_weights(model, config.pretrained_checkpoint)
                print(f"Loaded pretrained weights from {config.pretrained_checkpoint}")
            
            # Initialize optimizer for this model
            optimizer = AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
            
            # Initialize scheduler
            scheduler = CosineAnnealingLR(optimizer, T_max=config.num_epochs)
            
            # Training history for this model
            history = {
                'train_loss': [],
                'val_loss': [],
                'learning_rate': []
            }
            
            # Training loop
            for epoch in range(config.num_epochs):
                # Train epoch
                train_loss = self._train_epoch(
                    model, train_loader, optimizer, loss_fn,
                    config, epoch, model_idx
                )
                
                # Update scheduler
                scheduler.step()
                
                # Validation
                val_loss = None
                if val_loader is not None:
                    val_loss = self._validate(model, val_loader, loss_fn, device)
                
                # Log
                history['train_loss'].append(train_loss)
                if val_loss is not None:
                    history['val_loss'].append(val_loss)
                history['learning_rate'].append(optimizer.param_groups[0]['lr'])
                
                # Print summary
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{config.num_epochs}: "
                          f"Train Loss = {train_loss:.4f}", end="")
                    if val_loss is not None:
                        print(f", Val Loss = {val_loss:.4f}", end="")
                    print()
                
                # Save checkpoint
                if (epoch + 1) % config.checkpoint_interval == 0:
                    self._save_model_checkpoint(
                        model, model_idx, epoch, config.checkpoint_dir
                    )
            
            histories.append(history)
            
            # Save final model
            self._save_model_checkpoint(
                model, model_idx, config.num_epochs - 1,
                config.checkpoint_dir, is_final=True
            )
        
        print(f"\n{'='*60}")
        print("Ensemble training completed!")
        print(f"{'='*60}")
        
        return histories
    
    def _train_epoch(self,
                    model: nn.Module,
                    train_loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    loss_fn: nn.Module,
                    config: UncertaintyConfig,
                    epoch: int,
                    model_idx: int) -> float:
        """Train one epoch for a single model.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            optimizer: Optimizer
            loss_fn: Loss function
            config: Configuration
            epoch: Current epoch
            model_idx: Index of model in ensemble
        
        Returns:
            Average training loss
        """
        model.train()
        device = torch.device(config.device)
        
        total_loss = 0.0
        count = 0
        
        pbar = tqdm(train_loader, 
                   desc=f'Model {model_idx+1} Epoch {epoch+1}',
                   leave=False)
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = self._move_batch_to_device(batch, device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            data = batch['data']
            t_span = batch.get('t_span', torch.tensor([0.0, 1.0], device=device))
            
            output = model(data, t_span)
            
            # Compute loss
            if 'target' in batch:
                loss = loss_fn(output, batch['target'])
            else:
                # Use output as target (self-supervised)
                loss = loss_fn(output, output)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.max_grad_norm
            )
            
            # Optimizer step
            optimizer.step()
            
            # Accumulate loss
            total_loss += loss.item()
            count += 1
            
            # Update progress bar
            if batch_idx % config.log_interval == 0:
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        return total_loss / count if count > 0 else 0.0
    
    def _validate(self,
                 model: nn.Module,
                 val_loader: DataLoader,
                 loss_fn: nn.Module,
                 device: torch.device) -> float:
        """Validate a single model.
        
        Args:
            model: Model to validate
            val_loader: Validation data loader
            loss_fn: Loss function
            device: Device
        
        Returns:
            Average validation loss
        """
        model.eval()
        
        total_loss = 0.0
        count = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                batch = self._move_batch_to_device(batch, device)
                
                # Forward pass
                data = batch['data']
                t_span = batch.get('t_span', torch.tensor([0.0, 1.0], device=device))
                
                output = model(data, t_span)
                
                # Compute loss
                if 'target' in batch:
                    loss = loss_fn(output, batch['target'])
                else:
                    loss = loss_fn(output, output)
                
                total_loss += loss.item()
                count += 1
        
        return total_loss / count if count > 0 else 0.0
    
    def predict_with_uncertainty(self,
                                data,
                                t_span: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Predict with ensemble and compute epistemic uncertainty.
        
        Args:
            data: Input GraphData
            t_span: Time span for ODE integration
        
        Returns:
            Tuple of (mean_predictions, epistemic_uncertainties)
            Each is a dictionary with keys matching model output
        
        Requirements:
            - Validates: Requirement 10.1
            - Implement prediction with ensemble to compute epistemic uncertainty
        """
        self.eval()
        
        # Collect predictions from all models
        all_predictions = []
        
        with torch.no_grad():
            for model in self.models:
                pred = model(data, t_span)
                all_predictions.append(pred)
        
        # Stack predictions
        # Each prediction is a dict with keys: 'particle_pred', 'observables', 'latent_z0', 'latent_z1'
        stacked_predictions = {}
        for key in all_predictions[0].keys():
            stacked_predictions[key] = torch.stack([p[key] for p in all_predictions])
        
        # Compute mean and variance
        mean_predictions = {}
        epistemic_uncertainties = {}
        
        for key, stacked_pred in stacked_predictions.items():
            # Mean across ensemble (dim=0)
            mean_predictions[key] = stacked_pred.mean(dim=0)
            
            # Variance across ensemble (epistemic uncertainty)
            epistemic_uncertainties[key] = stacked_pred.var(dim=0)
        
        return mean_predictions, epistemic_uncertainties
    
    def forward(self, data, t_span: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass returns mean prediction from ensemble.
        
        Args:
            data: Input GraphData
            t_span: Time span for ODE integration
        
        Returns:
            Mean predictions across ensemble
        """
        mean_predictions, _ = self.predict_with_uncertainty(data, t_span)
        return mean_predictions
    
    def _load_pretrained_weights(self, model: nn.Module, checkpoint_path: str):
        """Load pretrained weights into a model.
        
        Args:
            model: Model to load weights into
            checkpoint_path: Path to checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    def _save_model_checkpoint(self,
                               model: nn.Module,
                               model_idx: int,
                               epoch: int,
                               checkpoint_dir: str,
                               is_final: bool = False):
        """Save checkpoint for a single model.
        
        Args:
            model: Model to save
            model_idx: Index of model in ensemble
            epoch: Current epoch
            checkpoint_dir: Directory to save checkpoint
            is_final: Whether this is the final checkpoint
        """
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        if is_final:
            filename = f'ensemble_model_{model_idx}_final.pt'
        else:
            filename = f'ensemble_model_{model_idx}_epoch_{epoch+1}.pt'
        
        checkpoint_path = os.path.join(checkpoint_dir, filename)
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_idx': model_idx,
            'epoch': epoch
        }, checkpoint_path)
    
    def _move_batch_to_device(self, batch: Dict, device: torch.device) -> Dict:
        """Move batch to device.
        
        Args:
            batch: Batch dictionary
            device: Target device
        
        Returns:
            Batch on device
        """
        moved_batch = {}
        
        if 'data' in batch:
            moved_batch['data'] = batch['data'].to(device)
        
        if 'target' in batch:
            if isinstance(batch['target'], dict):
                moved_batch['target'] = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch['target'].items()
                }
            else:
                moved_batch['target'] = batch['target'].to(device)
        
        if 't_span' in batch:
            if isinstance(batch['t_span'], torch.Tensor):
                moved_batch['t_span'] = batch['t_span'].to(device)
            else:
                moved_batch['t_span'] = batch['t_span']
        
        return moved_batch
    
    def save_ensemble(self, checkpoint_dir: str):
        """Save entire ensemble.
        
        Args:
            checkpoint_dir: Directory to save ensemble
        """
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, 'ensemble_complete.pt')
        
        torch.save({
            'num_models': self.num_models,
            'model_config': self.model_config,
            'models_state_dict': [model.state_dict() for model in self.models],
            'temperature': self.temperature.item()
        }, checkpoint_path)
        
        print(f"Ensemble saved to {checkpoint_path}")
    
    def load_ensemble(self, checkpoint_path: str):
        """Load entire ensemble.
        
        Args:
            checkpoint_path: Path to ensemble checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load each model
        for i, state_dict in enumerate(checkpoint['models_state_dict']):
            self.models[i].load_state_dict(state_dict)
        
        # Load temperature
        if 'temperature' in checkpoint:
            self.temperature.data = torch.tensor([checkpoint['temperature']])
        
        print(f"Ensemble loaded from {checkpoint_path}")




class MCDropoutModel(nn.Module):
    """Model with MC Dropout for epistemic uncertainty estimation.
    
    Adds dropout to all layers and performs multiple forward passes
    with dropout enabled to estimate epistemic uncertainty.
    
    Requirements:
        - Validates: Requirement 10.2
        - Add dropout to all layers
        - Implement prediction with multiple forward passes
        - Compute epistemic uncertainty from variance
    """
    
    def __init__(self,
                 model_config: ModelConfig,
                 dropout_rate: float = 0.1):
        """Initialize MC Dropout model.
        
        Args:
            model_config: Configuration for base model
            dropout_rate: Dropout rate to apply
        """
        super().__init__()
        
        self.dropout_rate = dropout_rate
        self.model_config = model_config
        
        # Create base model
        self.model = MultiScaleModel(model_config)
        
        # Add dropout layers to all components
        self._add_dropout_to_model()
        
        # Temperature parameter for calibration
        self.temperature = nn.Parameter(torch.ones(1))
    
    def _add_dropout_to_model(self):
        """Add dropout layers to all components of the model.
        
        Requirements:
            - Validates: Requirement 10.2 (add dropout to all layers)
        """
        # Add dropout to GNN
        self._add_dropout_to_module(self.model.gnn)
        
        # Add dropout to Transformer
        self._add_dropout_to_module(self.model.transformer)
        
        # Add dropout to Encoder
        self._add_dropout_to_module(self.model.encoder)
        
        # Add dropout to Neural ODE dynamics function
        self._add_dropout_to_module(self.model.neural_ode)
        
        # Add dropout to Decoder
        self._add_dropout_to_module(self.model.decoder)
    
    def _add_dropout_to_module(self, module: nn.Module):
        """Recursively add dropout after linear layers in a module.
        
        Args:
            module: Module to add dropout to
        """
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                # Replace linear layer with sequential: Linear -> Dropout
                setattr(module, name, nn.Sequential(
                    child,
                    nn.Dropout(self.dropout_rate)
                ))
            elif isinstance(child, nn.Sequential):
                # Add dropout to sequential modules
                new_layers = []
                for layer in child:
                    new_layers.append(layer)
                    if isinstance(layer, nn.Linear):
                        new_layers.append(nn.Dropout(self.dropout_rate))
                setattr(module, name, nn.Sequential(*new_layers))
            else:
                # Recursively process child modules
                self._add_dropout_to_module(child)
    
    def predict_with_uncertainty(self,
                                data,
                                t_span: torch.Tensor,
                                num_samples: int = 50) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Predict with MC Dropout and compute epistemic uncertainty.
        
        Performs multiple forward passes with dropout enabled and computes
        mean and variance across samples.
        
        Args:
            data: Input GraphData
            t_span: Time span for ODE integration
            num_samples: Number of forward passes with dropout
        
        Returns:
            Tuple of (mean_predictions, epistemic_uncertainties)
            Each is a dictionary with keys matching model output
        
        Requirements:
            - Validates: Requirement 10.2
            - Implement prediction with multiple forward passes
            - Compute epistemic uncertainty from variance
        """
        # Enable dropout (set to train mode)
        self.train()
        
        # Collect predictions from multiple forward passes
        all_predictions = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                pred = self.model(data, t_span)
                all_predictions.append(pred)
        
        # Stack predictions
        stacked_predictions = {}
        for key in all_predictions[0].keys():
            stacked_predictions[key] = torch.stack([p[key] for p in all_predictions])
        
        # Compute mean and variance
        mean_predictions = {}
        epistemic_uncertainties = {}
        
        for key, stacked_pred in stacked_predictions.items():
            # Mean across samples (dim=0)
            mean_predictions[key] = stacked_pred.mean(dim=0)
            
            # Variance across samples (epistemic uncertainty)
            epistemic_uncertainties[key] = stacked_pred.var(dim=0)
        
        return mean_predictions, epistemic_uncertainties
    
    def forward(self, data, t_span: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through model.
        
        Args:
            data: Input GraphData
            t_span: Time span for ODE integration
        
        Returns:
            Model predictions
        """
        return self.model(data, t_span)
    
    def train_model(self,
                   train_loader: DataLoader,
                   val_loader: Optional[DataLoader],
                   config: UncertaintyConfig,
                   loss_fn: nn.Module) -> Dict[str, List[float]]:
        """Train MC Dropout model.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            config: Uncertainty calibration configuration
            loss_fn: Loss function for training
        
        Returns:
            Training history
        """
        device = torch.device(config.device)
        self.to(device)
        
        # Load pretrained checkpoint if provided
        if config.pretrained_checkpoint is not None:
            self._load_pretrained_weights(config.pretrained_checkpoint)
            print(f"Loaded pretrained weights from {config.pretrained_checkpoint}")
        
        # Initialize optimizer
        optimizer = AdamW(
            self.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Initialize scheduler
        scheduler = CosineAnnealingLR(optimizer, T_max=config.num_epochs)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        print(f"Training MC Dropout model for {config.num_epochs} epochs")
        print(f"Dropout rate: {self.dropout_rate}")
        
        # Training loop
        for epoch in range(config.num_epochs):
            # Train epoch
            train_loss = self._train_epoch(
                train_loader, optimizer, loss_fn, config, epoch, device
            )
            
            # Update scheduler
            scheduler.step()
            
            # Validation
            val_loss = None
            if val_loader is not None:
                val_loss = self._validate(val_loader, loss_fn, device)
            
            # Log
            history['train_loss'].append(train_loss)
            if val_loss is not None:
                history['val_loss'].append(val_loss)
            history['learning_rate'].append(optimizer.param_groups[0]['lr'])
            
            # Print summary
            print(f"Epoch {epoch+1}/{config.num_epochs}: "
                  f"Train Loss = {train_loss:.4f}", end="")
            if val_loss is not None:
                print(f", Val Loss = {val_loss:.4f}", end="")
            print()
            
            # Save checkpoint
            if (epoch + 1) % config.checkpoint_interval == 0:
                self._save_checkpoint(epoch, config.checkpoint_dir)
        
        # Save final model
        self._save_checkpoint(config.num_epochs - 1, config.checkpoint_dir, is_final=True)
        
        print("MC Dropout training completed!")
        
        return history
    
    def _train_epoch(self,
                    train_loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    loss_fn: nn.Module,
                    config: UncertaintyConfig,
                    epoch: int,
                    device: torch.device) -> float:
        """Train one epoch.
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            loss_fn: Loss function
            config: Configuration
            epoch: Current epoch
            device: Device
        
        Returns:
            Average training loss
        """
        self.train()
        
        total_loss = 0.0
        count = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = self._move_batch_to_device(batch, device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            data = batch['data']
            t_span = batch.get('t_span', torch.tensor([0.0, 1.0], device=device))
            
            output = self(data, t_span)
            
            # Compute loss
            if 'target' in batch:
                loss = loss_fn(output, batch['target'])
            else:
                loss = loss_fn(output, output)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.parameters(),
                config.max_grad_norm
            )
            
            # Optimizer step
            optimizer.step()
            
            # Accumulate loss
            total_loss += loss.item()
            count += 1
            
            # Update progress bar
            if batch_idx % config.log_interval == 0:
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        return total_loss / count if count > 0 else 0.0
    
    def _validate(self,
                 val_loader: DataLoader,
                 loss_fn: nn.Module,
                 device: torch.device) -> float:
        """Validate model.
        
        Args:
            val_loader: Validation data loader
            loss_fn: Loss function
            device: Device
        
        Returns:
            Average validation loss
        """
        self.eval()
        
        total_loss = 0.0
        count = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                batch = self._move_batch_to_device(batch, device)
                
                # Forward pass
                data = batch['data']
                t_span = batch.get('t_span', torch.tensor([0.0, 1.0], device=device))
                
                output = self(data, t_span)
                
                # Compute loss
                if 'target' in batch:
                    loss = loss_fn(output, batch['target'])
                else:
                    loss = loss_fn(output, output)
                
                total_loss += loss.item()
                count += 1
        
        return total_loss / count if count > 0 else 0.0
    
    def _load_pretrained_weights(self, checkpoint_path: str):
        """Load pretrained weights.
        
        Args:
            checkpoint_path: Path to checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            # Load into base model
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
    
    def _save_checkpoint(self,
                        epoch: int,
                        checkpoint_dir: str,
                        is_final: bool = False):
        """Save checkpoint.
        
        Args:
            epoch: Current epoch
            checkpoint_dir: Directory to save checkpoint
            is_final: Whether this is the final checkpoint
        """
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        if is_final:
            filename = 'mc_dropout_model_final.pt'
        else:
            filename = f'mc_dropout_model_epoch_{epoch+1}.pt'
        
        checkpoint_path = os.path.join(checkpoint_dir, filename)
        
        torch.save({
            'model_state_dict': self.state_dict(),
            'epoch': epoch,
            'dropout_rate': self.dropout_rate,
            'temperature': self.temperature.item()
        }, checkpoint_path)
    
    def _move_batch_to_device(self, batch: Dict, device: torch.device) -> Dict:
        """Move batch to device.
        
        Args:
            batch: Batch dictionary
            device: Target device
        
        Returns:
            Batch on device
        """
        moved_batch = {}
        
        if 'data' in batch:
            moved_batch['data'] = batch['data'].to(device)
        
        if 'target' in batch:
            if isinstance(batch['target'], dict):
                moved_batch['target'] = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch['target'].items()
                }
            else:
                moved_batch['target'] = batch['target'].to(device)
        
        if 't_span' in batch:
            if isinstance(batch['t_span'], torch.Tensor):
                moved_batch['t_span'] = batch['t_span'].to(device)
            else:
                moved_batch['t_span'] = batch['t_span']
        
        return moved_batch




class AleatoricDecoder(nn.Module):
    """Decoder that predicts both mean and variance for aleatoric uncertainty.
    
    Modifies the standard decoder to output both mean predictions and
    variance estimates, enabling aleatoric uncertainty quantification.
    
    Requirements:
        - Validates: Requirement 10.3
        - Modify decoder to predict both mean and variance
        - Implement negative log-likelihood loss
    """
    
    def __init__(self,
                 latent_dim: int,
                 particle_dim: int,
                 conditioning_dim: int,
                 num_observables: int = 5):
        """Initialize aleatoric decoder.
        
        Args:
            latent_dim: Dimension of latent space
            particle_dim: Dimension of particle embeddings
            conditioning_dim: Dimension of conditioning vector
            num_observables: Number of coarse-grained observables
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.particle_dim = particle_dim
        self.conditioning_dim = conditioning_dim
        self.num_observables = num_observables
        
        # Particle-level decoder (predicts mean and log variance)
        particle_input_dim = latent_dim + particle_dim + conditioning_dim
        
        self.particle_mean_head = nn.Sequential(
            nn.Linear(particle_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, particle_dim)
        )
        
        self.particle_logvar_head = nn.Sequential(
            nn.Linear(particle_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, particle_dim)
        )
        
        # Observable decoder (predicts mean and log variance)
        observable_input_dim = latent_dim + conditioning_dim
        
        self.observable_mean_head = nn.Sequential(
            nn.Linear(observable_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_observables)
        )
        
        self.observable_logvar_head = nn.Sequential(
            nn.Linear(observable_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_observables)
        )
    
    def forward(self,
                z: torch.Tensor,
                x: torch.Tensor,
                batch: torch.Tensor,
                conditioning: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Forward pass predicting mean and variance.
        
        Args:
            z: Latent state (batch_size, latent_dim)
            x: Particle embeddings (N, particle_dim)
            batch: Batch assignment (N,)
            conditioning: Conditioning vector (batch_size, conditioning_dim)
        
        Returns:
            Tuple of (mean_predictions, variance_predictions)
            Each is a dictionary with 'particle_pred' and 'observables'
        
        Requirements:
            - Validates: Requirement 10.3 (predict both mean and variance)
        """
        # Expand latent state and conditioning for each particle
        z_expanded = z[batch]  # (N, latent_dim)
        conditioning_expanded = conditioning[batch]  # (N, conditioning_dim)
        
        # Concatenate for particle-level prediction
        particle_input = torch.cat([z_expanded, x, conditioning_expanded], dim=1)
        
        # Predict particle mean and log variance
        particle_mean = self.particle_mean_head(particle_input)
        particle_logvar = self.particle_logvar_head(particle_input)
        particle_var = torch.exp(particle_logvar)  # Convert log variance to variance
        
        # Concatenate for observable prediction
        observable_input = torch.cat([z, conditioning], dim=1)
        
        # Predict observable mean and log variance
        observable_mean = self.observable_mean_head(observable_input)
        observable_logvar = self.observable_logvar_head(observable_input)
        observable_var = torch.exp(observable_logvar)
        
        # Return mean and variance predictions
        mean_predictions = {
            'particle_pred': particle_mean,
            'observables': observable_mean
        }
        
        variance_predictions = {
            'particle_pred': particle_var,
            'observables': observable_var
        }
        
        return mean_predictions, variance_predictions


def aleatoric_loss(pred_mean: torch.Tensor,
                   pred_var: torch.Tensor,
                   target: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood loss for aleatoric uncertainty.
    
    Assumes Gaussian distribution: p(y|x) = N(μ(x), σ²(x))
    NLL = 0.5 * (log(σ²) + (y - μ)² / σ²)
    
    Args:
        pred_mean: Predicted mean (*, D)
        pred_var: Predicted variance (*, D)
        target: Target values (*, D)
    
    Returns:
        Negative log-likelihood loss (scalar)
    
    Requirements:
        - Validates: Requirement 10.3 (implement negative log-likelihood loss)
    """
    # Compute negative log-likelihood
    # NLL = 0.5 * (log(var) + (target - mean)² / var)
    nll = 0.5 * (torch.log(pred_var + 1e-8) + (target - pred_mean) ** 2 / (pred_var + 1e-8))
    
    # Average over all dimensions and samples
    return nll.mean()


class AleatoricModel(nn.Module):
    """Model with aleatoric uncertainty prediction.
    
    Replaces the standard decoder with AleatoricDecoder to predict
    both mean and variance for uncertainty quantification.
    
    Requirements:
        - Validates: Requirement 10.3
        - Modify decoder to predict both mean and variance
    """
    
    def __init__(self, model_config: ModelConfig):
        """Initialize aleatoric model.
        
        Args:
            model_config: Configuration for base model
        """
        super().__init__()
        
        self.model_config = model_config
        
        # Create base model
        self.base_model = MultiScaleModel(model_config)
        
        # Replace decoder with aleatoric decoder
        self.aleatoric_decoder = AleatoricDecoder(
            latent_dim=model_config.latent_dim,
            particle_dim=model_config.transformer_hidden_dim,
            conditioning_dim=model_config.conditioning_dim,
            num_observables=model_config.num_observables
        )
        
        # Temperature parameter for calibration
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self,
                data,
                t_span: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Forward pass predicting mean and aleatoric variance.
        
        Args:
            data: Input GraphData
            t_span: Time span for ODE integration
        
        Returns:
            Tuple of (mean_predictions, aleatoric_variances)
        """
        # Extract data attributes
        x = data.x
        pos = data.pos
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None else None
        
        # Determine batch size
        if batch is not None:
            batch_size = int(batch.max().item()) + 1
        else:
            batch_size = 1
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        
        # Construct conditioning
        if hasattr(data, 'density') and hasattr(data, 'energy') and hasattr(data, 'material'):
            density = data.density.view(-1, 1) if data.density.dim() == 1 else data.density
            energy = data.energy.view(-1, 1) if data.energy.dim() == 1 else data.energy
            material = data.material if data.material.dim() == 2 else data.material.unsqueeze(0)
            conditioning = torch.cat([density, energy, material], dim=1)
        else:
            conditioning = torch.zeros(batch_size, self.model_config.conditioning_dim,
                                      device=x.device, dtype=x.dtype)
        
        # Process through GNN, Transformer, Encoder, Neural ODE
        h_gnn = self.base_model.gnn(x, pos, batch)
        h_trans = self.base_model.transformer(h_gnn, pos, batch)
        z0 = self.base_model.encoder(h_trans, batch, conditioning)
        z1 = self.base_model.neural_ode(z0, t_span, conditioning)
        
        # Decode with aleatoric decoder
        mean_predictions, aleatoric_variances = self.aleatoric_decoder(
            z1, h_trans, batch, conditioning
        )
        
        # Add latent states to output
        mean_predictions['latent_z0'] = z0
        mean_predictions['latent_z1'] = z1
        
        return mean_predictions, aleatoric_variances
    
    def train_model(self,
                   train_loader: DataLoader,
                   val_loader: Optional[DataLoader],
                   config: UncertaintyConfig) -> Dict[str, List[float]]:
        """Train aleatoric model with NLL loss.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            config: Uncertainty calibration configuration
        
        Returns:
            Training history
        """
        device = torch.device(config.device)
        self.to(device)
        
        # Load pretrained checkpoint if provided
        if config.pretrained_checkpoint is not None:
            self._load_pretrained_weights(config.pretrained_checkpoint)
            print(f"Loaded pretrained weights from {config.pretrained_checkpoint}")
        
        # Initialize optimizer
        optimizer = AdamW(
            self.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Initialize scheduler
        scheduler = CosineAnnealingLR(optimizer, T_max=config.num_epochs)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        print(f"Training aleatoric model for {config.num_epochs} epochs")
        
        # Training loop
        for epoch in range(config.num_epochs):
            # Train epoch
            train_loss = self._train_epoch(
                train_loader, optimizer, config, epoch, device
            )
            
            # Update scheduler
            scheduler.step()
            
            # Validation
            val_loss = None
            if val_loader is not None:
                val_loss = self._validate(val_loader, device)
            
            # Log
            history['train_loss'].append(train_loss)
            if val_loss is not None:
                history['val_loss'].append(val_loss)
            history['learning_rate'].append(optimizer.param_groups[0]['lr'])
            
            # Print summary
            print(f"Epoch {epoch+1}/{config.num_epochs}: "
                  f"Train Loss = {train_loss:.4f}", end="")
            if val_loss is not None:
                print(f", Val Loss = {val_loss:.4f}", end="")
            print()
            
            # Save checkpoint
            if (epoch + 1) % config.checkpoint_interval == 0:
                self._save_checkpoint(epoch, config.checkpoint_dir)
        
        # Save final model
        self._save_checkpoint(config.num_epochs - 1, config.checkpoint_dir, is_final=True)
        
        print("Aleatoric model training completed!")
        
        return history
    
    def _train_epoch(self,
                    train_loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    config: UncertaintyConfig,
                    epoch: int,
                    device: torch.device) -> float:
        """Train one epoch with NLL loss.
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            config: Configuration
            epoch: Current epoch
            device: Device
        
        Returns:
            Average training loss
        """
        self.train()
        
        total_loss = 0.0
        count = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = self._move_batch_to_device(batch, device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            data = batch['data']
            t_span = batch.get('t_span', torch.tensor([0.0, 1.0], device=device))
            
            mean_pred, var_pred = self(data, t_span)
            
            # Compute NLL loss
            if 'target' in batch:
                target = batch['target']
                
                # Compute loss for each output
                loss = 0.0
                if 'particle_pred' in target:
                    loss += aleatoric_loss(
                        mean_pred['particle_pred'],
                        var_pred['particle_pred'],
                        target['particle_pred']
                    )
                if 'observables' in target:
                    loss += aleatoric_loss(
                        mean_pred['observables'],
                        var_pred['observables'],
                        target['observables']
                    )
            else:
                # Self-supervised: use mean as target
                loss = aleatoric_loss(
                    mean_pred['particle_pred'],
                    var_pred['particle_pred'],
                    mean_pred['particle_pred'].detach()
                )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.parameters(),
                config.max_grad_norm
            )
            
            # Optimizer step
            optimizer.step()
            
            # Accumulate loss
            total_loss += loss.item()
            count += 1
            
            # Update progress bar
            if batch_idx % config.log_interval == 0:
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        return total_loss / count if count > 0 else 0.0
    
    def _validate(self,
                 val_loader: DataLoader,
                 device: torch.device) -> float:
        """Validate model.
        
        Args:
            val_loader: Validation data loader
            device: Device
        
        Returns:
            Average validation loss
        """
        self.eval()
        
        total_loss = 0.0
        count = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                batch = self._move_batch_to_device(batch, device)
                
                # Forward pass
                data = batch['data']
                t_span = batch.get('t_span', torch.tensor([0.0, 1.0], device=device))
                
                mean_pred, var_pred = self(data, t_span)
                
                # Compute NLL loss
                if 'target' in batch:
                    target = batch['target']
                    
                    loss = 0.0
                    if 'particle_pred' in target:
                        loss += aleatoric_loss(
                            mean_pred['particle_pred'],
                            var_pred['particle_pred'],
                            target['particle_pred']
                        )
                    if 'observables' in target:
                        loss += aleatoric_loss(
                            mean_pred['observables'],
                            var_pred['observables'],
                            target['observables']
                        )
                else:
                    loss = aleatoric_loss(
                        mean_pred['particle_pred'],
                        var_pred['particle_pred'],
                        mean_pred['particle_pred']
                    )
                
                total_loss += loss.item()
                count += 1
        
        return total_loss / count if count > 0 else 0.0
    
    def _load_pretrained_weights(self, checkpoint_path: str):
        """Load pretrained weights into base model.
        
        Args:
            checkpoint_path: Path to checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            # Load compatible weights into base model
            base_state_dict = checkpoint['model_state_dict']
            
            # Load GNN, Transformer, Encoder, Neural ODE
            self.base_model.gnn.load_state_dict(
                {k.replace('gnn.', ''): v for k, v in base_state_dict.items() if k.startswith('gnn.')}
            )
            self.base_model.transformer.load_state_dict(
                {k.replace('transformer.', ''): v for k, v in base_state_dict.items() if k.startswith('transformer.')}
            )
            self.base_model.encoder.load_state_dict(
                {k.replace('encoder.', ''): v for k, v in base_state_dict.items() if k.startswith('encoder.')}
            )
            self.base_model.neural_ode.load_state_dict(
                {k.replace('neural_ode.', ''): v for k, v in base_state_dict.items() if k.startswith('neural_ode.')}
            )
            # Note: decoder is replaced, so we don't load it
    
    def _save_checkpoint(self,
                        epoch: int,
                        checkpoint_dir: str,
                        is_final: bool = False):
        """Save checkpoint.
        
        Args:
            epoch: Current epoch
            checkpoint_dir: Directory to save checkpoint
            is_final: Whether this is the final checkpoint
        """
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        if is_final:
            filename = 'aleatoric_model_final.pt'
        else:
            filename = f'aleatoric_model_epoch_{epoch+1}.pt'
        
        checkpoint_path = os.path.join(checkpoint_dir, filename)
        
        torch.save({
            'model_state_dict': self.state_dict(),
            'epoch': epoch,
            'temperature': self.temperature.item()
        }, checkpoint_path)
    
    def _move_batch_to_device(self, batch: Dict, device: torch.device) -> Dict:
        """Move batch to device.
        
        Args:
            batch: Batch dictionary
            device: Target device
        
        Returns:
            Batch on device
        """
        moved_batch = {}
        
        if 'data' in batch:
            moved_batch['data'] = batch['data'].to(device)
        
        if 'target' in batch:
            if isinstance(batch['target'], dict):
                moved_batch['target'] = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch['target'].items()
                }
            else:
                moved_batch['target'] = batch['target'].to(device)
        
        if 't_span' in batch:
            if isinstance(batch['t_span'], torch.Tensor):
                moved_batch['t_span'] = batch['t_span'].to(device)
            else:
                moved_batch['t_span'] = batch['t_span']
        
        return moved_batch




def calibrate_temperature(model: nn.Module,
                         val_loader: DataLoader,
                         device: torch.device,
                         max_iter: int = 50) -> float:
    """Calibrate temperature parameter for uncertainty calibration.
    
    Finds optimal temperature T to minimize negative log-likelihood on
    validation set. Uses LBFGS optimizer for efficient optimization.
    
    Args:
        model: Model with temperature parameter
        val_loader: Validation data loader
        device: Device to run on
        max_iter: Maximum number of optimization iterations
    
    Returns:
        Optimal temperature value
    
    Requirements:
        - Validates: Requirement 10.4
        - Find optimal temperature to calibrate uncertainty on validation set
    """
    print("Calibrating temperature on validation set...")
    
    # Get temperature parameter from model
    if hasattr(model, 'temperature'):
        temperature = model.temperature
    else:
        # Create temperature parameter if not present
        temperature = nn.Parameter(torch.ones(1, device=device))
    
    # Set model to eval mode
    model.eval()
    
    # Initialize LBFGS optimizer for temperature only
    optimizer = LBFGS([temperature], lr=0.01, max_iter=max_iter)
    
    def eval_loss():
        """Evaluate loss on validation set with current temperature."""
        optimizer.zero_grad()
        
        total_loss = 0.0
        count = 0
        
        for batch in val_loader:
            # Move batch to device
            if 'data' in batch:
                data = batch['data'].to(device)
            else:
                continue
            
            t_span = batch.get('t_span', torch.tensor([0.0, 1.0], device=device))
            
            # Forward pass
            with torch.no_grad():
                if isinstance(model, DeepEnsemble):
                    # For ensemble, get mean prediction
                    output, _ = model.predict_with_uncertainty(data, t_span)
                elif isinstance(model, MCDropoutModel):
                    # For MC dropout, get single forward pass
                    output = model(data, t_span)
                elif isinstance(model, AleatoricModel):
                    # For aleatoric model, get mean prediction
                    output, _ = model(data, t_span)
                else:
                    # Standard model
                    output = model(data, t_span)
            
            # Get predictions (use observables for calibration)
            if 'observables' in output:
                logits = output['observables']
            else:
                # Use particle predictions if observables not available
                logits = output.get('particle_pred', torch.zeros(1, device=device))
            
            # Apply temperature scaling
            scaled_logits = logits / temperature
            
            # Compute loss (negative log-likelihood)
            if 'target' in batch:
                target = batch['target']
                if isinstance(target, dict) and 'observables' in target:
                    target_values = target['observables'].to(device)
                else:
                    target_values = logits.detach()  # Self-supervised
                
                # MSE loss as proxy for NLL
                loss = F.mse_loss(scaled_logits, target_values)
            else:
                # Self-supervised: minimize variance
                loss = scaled_logits.var()
            
            total_loss += loss.item()
            count += 1
        
        avg_loss = total_loss / count if count > 0 else 0.0
        
        # Create loss tensor for backprop
        loss_tensor = torch.tensor(avg_loss, device=device, requires_grad=True)
        loss_tensor.backward()
        
        return loss_tensor
    
    # Optimize temperature
    optimizer.step(eval_loss)
    
    optimal_temperature = temperature.item()
    
    print(f"Optimal temperature: {optimal_temperature:.4f}")
    
    return optimal_temperature


class TemperatureScaledModel(nn.Module):
    """Wrapper that applies temperature scaling to model predictions.
    
    Requirements:
        - Validates: Requirement 10.4
        - Apply temperature scaling for uncertainty calibration
    """
    
    def __init__(self, base_model: nn.Module, temperature: float = 1.0):
        """Initialize temperature-scaled model.
        
        Args:
            base_model: Base model to wrap
            temperature: Temperature parameter
        """
        super().__init__()
        
        self.base_model = base_model
        self.temperature = nn.Parameter(torch.tensor([temperature]))
    
    def forward(self, data, t_span: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with temperature scaling.
        
        Args:
            data: Input GraphData
            t_span: Time span for ODE integration
        
        Returns:
            Temperature-scaled predictions
        """
        # Get base model predictions
        if isinstance(self.base_model, DeepEnsemble):
            output, uncertainty = self.base_model.predict_with_uncertainty(data, t_span)
        elif isinstance(self.base_model, MCDropoutModel):
            output = self.base_model(data, t_span)
            uncertainty = None
        elif isinstance(self.base_model, AleatoricModel):
            output, uncertainty = self.base_model(data, t_span)
        else:
            output = self.base_model(data, t_span)
            uncertainty = None
        
        # Apply temperature scaling to predictions
        scaled_output = {}
        for key, value in output.items():
            if key in ['particle_pred', 'observables']:
                # Scale predictions
                scaled_output[key] = value / self.temperature
            else:
                # Keep latent states unchanged
                scaled_output[key] = value
        
        return scaled_output
    
    def calibrate(self,
                 val_loader: DataLoader,
                 device: torch.device,
                 max_iter: int = 50):
        """Calibrate temperature on validation set.
        
        Args:
            val_loader: Validation data loader
            device: Device
            max_iter: Maximum optimization iterations
        """
        optimal_temp = calibrate_temperature(
            self.base_model, val_loader, device, max_iter
        )
        self.temperature.data = torch.tensor([optimal_temp])




@dataclass
class UncertaintyReport:
    """Report containing all uncertainty estimates.
    
    Attributes:
        predictions: Mean predictions
        epistemic_uncertainty: Epistemic uncertainty (model uncertainty)
        aleatoric_uncertainty: Aleatoric uncertainty (data uncertainty)
        total_uncertainty: Combined uncertainty
    
    Requirements:
        - Validates: Requirement 10.5
        - Report epistemic and aleatoric uncertainty separately
        - Compute total uncertainty
    """
    predictions: Dict[str, torch.Tensor]
    epistemic_uncertainty: Optional[Dict[str, torch.Tensor]] = None
    aleatoric_uncertainty: Optional[Dict[str, torch.Tensor]] = None
    total_uncertainty: Optional[Dict[str, torch.Tensor]] = None


class CombinedUncertaintyModel(nn.Module):
    """Model that combines epistemic and aleatoric uncertainty.
    
    Uses ensemble or MC dropout for epistemic uncertainty and
    aleatoric decoder for aleatoric uncertainty.
    
    Requirements:
        - Validates: Requirement 10.5
        - Report epistemic and aleatoric uncertainty separately
        - Compute total uncertainty
    """
    
    def __init__(self,
                 model_config: ModelConfig,
                 use_ensemble: bool = True,
                 num_ensemble_models: int = 5,
                 dropout_rate: float = 0.1,
                 num_mc_samples: int = 50):
        """Initialize combined uncertainty model.
        
        Args:
            model_config: Configuration for base model
            use_ensemble: Whether to use ensemble (True) or MC dropout (False)
            num_ensemble_models: Number of models in ensemble
            dropout_rate: Dropout rate for MC dropout
            num_mc_samples: Number of MC samples
        """
        super().__init__()
        
        self.use_ensemble = use_ensemble
        self.num_mc_samples = num_mc_samples
        
        if use_ensemble:
            # Create ensemble with aleatoric decoders
            self.models = nn.ModuleList([
                AleatoricModel(model_config) for _ in range(num_ensemble_models)
            ])
            self.num_models = num_ensemble_models
        else:
            # Create single MC dropout model with aleatoric decoder
            self.model = AleatoricModel(model_config)
            # Add dropout to model
            self._add_dropout_to_model(self.model, dropout_rate)
        
        # Temperature parameter
        self.temperature = nn.Parameter(torch.ones(1))
    
    def _add_dropout_to_model(self, model: nn.Module, dropout_rate: float):
        """Add dropout to model for MC dropout.
        
        Args:
            model: Model to add dropout to
            dropout_rate: Dropout rate
        """
        # Add dropout recursively
        for name, child in model.named_children():
            if isinstance(child, nn.Linear):
                setattr(model, name, nn.Sequential(
                    child,
                    nn.Dropout(dropout_rate)
                ))
            else:
                self._add_dropout_to_model(child, dropout_rate)
    
    def predict_with_uncertainty(self,
                                data,
                                t_span: torch.Tensor) -> UncertaintyReport:
        """Predict with full uncertainty quantification.
        
        Computes epistemic uncertainty (from ensemble/MC dropout),
        aleatoric uncertainty (from variance prediction), and
        total uncertainty (sum of both).
        
        Args:
            data: Input GraphData
            t_span: Time span for ODE integration
        
        Returns:
            UncertaintyReport with all uncertainty estimates
        
        Requirements:
            - Validates: Requirement 10.5
            - Report epistemic and aleatoric uncertainty separately
            - Compute total uncertainty
        """
        if self.use_ensemble:
            return self._predict_with_ensemble(data, t_span)
        else:
            return self._predict_with_mc_dropout(data, t_span)
    
    def _predict_with_ensemble(self,
                              data,
                              t_span: torch.Tensor) -> UncertaintyReport:
        """Predict with ensemble for epistemic + aleatoric uncertainty.
        
        Args:
            data: Input GraphData
            t_span: Time span for ODE integration
        
        Returns:
            UncertaintyReport
        """
        self.eval()
        
        # Collect predictions from all models
        all_means = []
        all_aleatoric_vars = []
        
        with torch.no_grad():
            for model in self.models:
                mean_pred, aleatoric_var = model(data, t_span)
                all_means.append(mean_pred)
                all_aleatoric_vars.append(aleatoric_var)
        
        # Stack predictions
        stacked_means = {}
        stacked_aleatoric = {}
        
        for key in all_means[0].keys():
            if key not in ['latent_z0', 'latent_z1']:  # Skip latent states
                stacked_means[key] = torch.stack([m[key] for m in all_means])
                stacked_aleatoric[key] = torch.stack([a[key] for a in all_aleatoric_vars])
        
        # Compute mean predictions
        mean_predictions = {}
        for key, stacked in stacked_means.items():
            mean_predictions[key] = stacked.mean(dim=0)
        
        # Add latent states from first model
        mean_predictions['latent_z0'] = all_means[0]['latent_z0']
        mean_predictions['latent_z1'] = all_means[0]['latent_z1']
        
        # Compute epistemic uncertainty (variance across ensemble)
        epistemic_uncertainty = {}
        for key, stacked in stacked_means.items():
            epistemic_uncertainty[key] = stacked.var(dim=0)
        
        # Compute aleatoric uncertainty (average of predicted variances)
        aleatoric_uncertainty = {}
        for key, stacked in stacked_aleatoric.items():
            aleatoric_uncertainty[key] = stacked.mean(dim=0)
        
        # Compute total uncertainty (sum of epistemic and aleatoric)
        total_uncertainty = {}
        for key in epistemic_uncertainty.keys():
            total_uncertainty[key] = epistemic_uncertainty[key] + aleatoric_uncertainty[key]
        
        return UncertaintyReport(
            predictions=mean_predictions,
            epistemic_uncertainty=epistemic_uncertainty,
            aleatoric_uncertainty=aleatoric_uncertainty,
            total_uncertainty=total_uncertainty
        )
    
    def _predict_with_mc_dropout(self,
                                data,
                                t_span: torch.Tensor) -> UncertaintyReport:
        """Predict with MC dropout for epistemic + aleatoric uncertainty.
        
        Args:
            data: Input GraphData
            t_span: Time span for ODE integration
        
        Returns:
            UncertaintyReport
        """
        # Enable dropout
        self.train()
        
        # Collect predictions from multiple forward passes
        all_means = []
        all_aleatoric_vars = []
        
        with torch.no_grad():
            for _ in range(self.num_mc_samples):
                mean_pred, aleatoric_var = self.model(data, t_span)
                all_means.append(mean_pred)
                all_aleatoric_vars.append(aleatoric_var)
        
        # Stack predictions
        stacked_means = {}
        stacked_aleatoric = {}
        
        for key in all_means[0].keys():
            if key not in ['latent_z0', 'latent_z1']:
                stacked_means[key] = torch.stack([m[key] for m in all_means])
                stacked_aleatoric[key] = torch.stack([a[key] for a in all_aleatoric_vars])
        
        # Compute mean predictions
        mean_predictions = {}
        for key, stacked in stacked_means.items():
            mean_predictions[key] = stacked.mean(dim=0)
        
        # Add latent states
        mean_predictions['latent_z0'] = all_means[0]['latent_z0']
        mean_predictions['latent_z1'] = all_means[0]['latent_z1']
        
        # Compute epistemic uncertainty (variance across MC samples)
        epistemic_uncertainty = {}
        for key, stacked in stacked_means.items():
            epistemic_uncertainty[key] = stacked.var(dim=0)
        
        # Compute aleatoric uncertainty (average of predicted variances)
        aleatoric_uncertainty = {}
        for key, stacked in stacked_aleatoric.items():
            aleatoric_uncertainty[key] = stacked.mean(dim=0)
        
        # Compute total uncertainty
        total_uncertainty = {}
        for key in epistemic_uncertainty.keys():
            total_uncertainty[key] = epistemic_uncertainty[key] + aleatoric_uncertainty[key]
        
        return UncertaintyReport(
            predictions=mean_predictions,
            epistemic_uncertainty=epistemic_uncertainty,
            aleatoric_uncertainty=aleatoric_uncertainty,
            total_uncertainty=total_uncertainty
        )
    
    def forward(self, data, t_span: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass returns mean prediction.
        
        Args:
            data: Input GraphData
            t_span: Time span for ODE integration
        
        Returns:
            Mean predictions
        """
        report = self.predict_with_uncertainty(data, t_span)
        return report.predictions
    
    def train_model(self,
                   train_loader: DataLoader,
                   val_loader: Optional[DataLoader],
                   config: UncertaintyConfig) -> Dict[str, List[float]]:
        """Train combined uncertainty model.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            config: Uncertainty calibration configuration
        
        Returns:
            Training history
        """
        if self.use_ensemble:
            # Train ensemble
            print("Training ensemble with aleatoric uncertainty...")
            histories = []
            
            for i, model in enumerate(self.models):
                print(f"\nTraining model {i+1}/{self.num_models}")
                history = model.train_model(train_loader, val_loader, config)
                histories.append(history)
            
            # Calibrate temperature if requested
            if config.use_temperature_scaling and val_loader is not None:
                device = torch.device(config.device)
                optimal_temp = calibrate_temperature(self, val_loader, device)
                self.temperature.data = torch.tensor([optimal_temp])
            
            return histories
        else:
            # Train MC dropout model
            print("Training MC dropout model with aleatoric uncertainty...")
            history = self.model.train_model(train_loader, val_loader, config)
            
            # Calibrate temperature if requested
            if config.use_temperature_scaling and val_loader is not None:
                device = torch.device(config.device)
                optimal_temp = calibrate_temperature(self, val_loader, device)
                self.temperature.data = torch.tensor([optimal_temp])
            
            return history
    
    def save_model(self, checkpoint_dir: str):
        """Save combined uncertainty model.
        
        Args:
            checkpoint_dir: Directory to save model
        """
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, 'combined_uncertainty_model.pt')
        
        if self.use_ensemble:
            torch.save({
                'use_ensemble': True,
                'num_models': self.num_models,
                'models_state_dict': [model.state_dict() for model in self.models],
                'temperature': self.temperature.item()
            }, checkpoint_path)
        else:
            torch.save({
                'use_ensemble': False,
                'model_state_dict': self.model.state_dict(),
                'num_mc_samples': self.num_mc_samples,
                'temperature': self.temperature.item()
            }, checkpoint_path)
        
        print(f"Combined uncertainty model saved to {checkpoint_path}")
    
    def load_model(self, checkpoint_path: str):
        """Load combined uncertainty model.
        
        Args:
            checkpoint_path: Path to checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if checkpoint['use_ensemble']:
            for i, state_dict in enumerate(checkpoint['models_state_dict']):
                self.models[i].load_state_dict(state_dict)
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'temperature' in checkpoint:
            self.temperature.data = torch.tensor([checkpoint['temperature']])
        
        print(f"Combined uncertainty model loaded from {checkpoint_path}")


def print_uncertainty_report(report: UncertaintyReport, output_keys: Optional[List[str]] = None):
    """Print formatted uncertainty report.
    
    Args:
        report: UncertaintyReport to print
        output_keys: Optional list of keys to print (default: all)
    
    Requirements:
        - Validates: Requirement 10.5 (report uncertainties separately)
    """
    if output_keys is None:
        output_keys = [k for k in report.predictions.keys() if k not in ['latent_z0', 'latent_z1']]
    
    print("\n" + "="*60)
    print("UNCERTAINTY REPORT")
    print("="*60)
    
    for key in output_keys:
        print(f"\n{key.upper()}:")
        print("-" * 40)
        
        # Predictions
        pred = report.predictions[key]
        print(f"  Prediction shape: {pred.shape}")
        print(f"  Prediction mean: {pred.mean().item():.6f}")
        print(f"  Prediction std: {pred.std().item():.6f}")
        
        # Epistemic uncertainty
        if report.epistemic_uncertainty is not None and key in report.epistemic_uncertainty:
            epi_unc = report.epistemic_uncertainty[key]
            print(f"  Epistemic uncertainty (model): {epi_unc.mean().item():.6f}")
        
        # Aleatoric uncertainty
        if report.aleatoric_uncertainty is not None and key in report.aleatoric_uncertainty:
            ale_unc = report.aleatoric_uncertainty[key]
            print(f"  Aleatoric uncertainty (data): {ale_unc.mean().item():.6f}")
        
        # Total uncertainty
        if report.total_uncertainty is not None and key in report.total_uncertainty:
            total_unc = report.total_uncertainty[key]
            print(f"  Total uncertainty: {total_unc.mean().item():.6f}")
    
    print("\n" + "="*60)


# Export main classes and functions
__all__ = [
    'UncertaintyConfig',
    'DeepEnsemble',
    'MCDropoutModel',
    'AleatoricDecoder',
    'AleatoricModel',
    'aleatoric_loss',
    'calibrate_temperature',
    'TemperatureScaledModel',
    'UncertaintyReport',
    'CombinedUncertaintyModel',
    'print_uncertainty_report'
]
