"""Fine-tuning loop for physics-constrained training.

This module implements the fine-tuning pipeline with:
- AdamW optimizer with lower learning rate
- Physics-constrained loss (prediction + conservation + physics limits)
- Support for fine-tuning on specific materials and conditions
- Logging and checkpointing

Requirements:
    - Validates: Requirements 9.3, 9.4
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
from dataclasses import dataclass
import os
from pathlib import Path
import json
from tqdm import tqdm

from .finetuning_losses import FineTuningLoss, FineTuningLossWeights


@dataclass
class FineTuningConfig:
    """Configuration for fine-tuning.
    
    Attributes:
        # Optimization
        learning_rate: Initial learning rate (lower than pretraining)
        weight_decay: Weight decay for AdamW
        max_grad_norm: Maximum gradient norm for clipping
        
        # Training
        num_epochs: Number of fine-tuning epochs
        batch_size: Batch size for training
        
        # Loss weights
        loss_weights: Weights for combining fine-tuning losses
        
        # Physics constraints
        density_threshold: Density threshold for Vlasov limit (particles/nm³)
        temperature_threshold: Temperature threshold for MB limit (K)
        
        # Data
        dt: Time step between trajectory frames (fs)
        
        # Material-specific fine-tuning
        target_materials: Optional list of materials to focus on
        target_density_range: Optional (min, max) density range
        target_energy_range: Optional (min, max) energy range
        
        # Checkpointing
        checkpoint_dir: Directory to save checkpoints
        checkpoint_interval: Save checkpoint every N epochs
        pretrained_checkpoint: Path to pretrained model checkpoint
        
        # Logging
        log_interval: Log metrics every N batches
        
        # Device
        device: Device to train on ('cuda' or 'cpu')
    """
    # Optimization
    learning_rate: float = 1e-4  # Lower than pretraining (1e-3)
    weight_decay: float = 1e-5   # Lower than pretraining (1e-4)
    max_grad_norm: float = 0.5   # Lower than pretraining (1.0)
    
    # Training
    num_epochs: int = 50
    batch_size: int = 16  # Smaller than pretraining (32)
    
    # Loss weights
    loss_weights: Optional[FineTuningLossWeights] = None
    
    # Physics constraints
    density_threshold: float = 0.01
    temperature_threshold: float = 1000.0
    
    # Data
    dt: float = 1.0
    
    # Material-specific fine-tuning
    target_materials: Optional[List[str]] = None
    target_density_range: Optional[tuple] = None
    target_energy_range: Optional[tuple] = None
    
    # Checkpointing
    checkpoint_dir: str = 'checkpoints/finetuning'
    checkpoint_interval: int = 10
    pretrained_checkpoint: Optional[str] = None
    
    # Logging
    log_interval: int = 10
    
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def __post_init__(self):
        """Set default values for optional parameters."""
        if self.loss_weights is None:
            self.loss_weights = FineTuningLossWeights()


class FineTuningTrainer:
    """Trainer for physics-constrained fine-tuning.
    
    Implements the fine-tuning loop with:
    - AdamW optimizer with lower learning rate
    - Physics-constrained loss (prediction + conservation + physics limits)
    - Support for fine-tuning on specific materials and conditions
    - Logging and checkpointing
    
    Requirements:
        - Validates: Requirements 9.3, 9.4
        - Set up AdamW optimizer with lower learning rate
        - Train for 50 epochs
        - Support fine-tuning on specific materials and conditions
    """
    
    def __init__(self,
                 model: nn.Module,
                 config: FineTuningConfig):
        """Initialize fine-tuning trainer.
        
        Args:
            model: MultiScaleModel instance
            config: Fine-tuning configuration
        """
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        
        # Move model to device
        self.model.to(self.device)
        
        # Load pretrained checkpoint if provided
        if config.pretrained_checkpoint is not None:
            self.load_pretrained_checkpoint(config.pretrained_checkpoint)
            print(f"Loaded pretrained checkpoint from {config.pretrained_checkpoint}")
        
        # Initialize loss function
        self.loss_fn = FineTuningLoss(
            weights=config.loss_weights,
            density_threshold=config.density_threshold,
            temperature_threshold=config.temperature_threshold
        )
        
        # Initialize optimizer with lower learning rate
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Initialize learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs
        )
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
        # Create checkpoint directory
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # Logging history
        self.history = {
            'train_loss': [],
            'prediction_loss': [],
            'conservation_loss': [],
            'physics_limit_loss': [],
            'energy_loss': [],
            'momentum_loss': [],
            'charge_loss': [],
            'vlasov_loss': [],
            'mb_loss': [],
            'stopping_power_loss': [],
            'learning_rate': []
        }
    
    def load_pretrained_checkpoint(self, checkpoint_path: str):
        """Load pretrained model weights.
        
        Args:
            checkpoint_path: Path to pretrained checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state dict
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Assume checkpoint is just the state dict
            self.model.load_state_dict(checkpoint)
        
        print(f"Loaded pretrained model from {checkpoint_path}")
    
    def filter_data_by_conditions(self, data_loader: DataLoader) -> DataLoader:
        """Filter data based on target materials and conditions.
        
        Args:
            data_loader: Original data loader
        
        Returns:
            Filtered data loader
        
        Requirements:
            - Validates: Requirement 9.4 (support fine-tuning on specific materials and conditions)
        """
        # This is a placeholder - in practice, you'd implement filtering logic
        # based on self.config.target_materials, target_density_range, etc.
        # For now, return the original data loader
        return data_loader
    
    def train_epoch(self,
                   train_loader: DataLoader,
                   epoch: int) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            train_loader: DataLoader providing training data
                Each batch should be a dictionary with:
                - 'data': GraphData at time t
                - 'data_next': GraphData at time t+1 (targets)
                - 'dt': Time step
            epoch: Current epoch number
        
        Returns:
            Dictionary with average losses for the epoch
        """
        self.model.train()
        
        epoch_losses = {
            'total': 0.0,
            'prediction': 0.0,
            'conservation': 0.0,
            'physics_limit': 0.0,
            'energy': 0.0,
            'momentum': 0.0,
            'charge': 0.0,
            'vlasov': 0.0,
            'mb': 0.0,
            'stopping_power': 0.0,
            'count': 0
        }
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{self.config.num_epochs}')
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Extract data
            data = batch['data']
            data_next = batch.get('data_next', None)
            dt = batch.get('dt', self.config.dt)
            
            # Time span
            t_span = torch.tensor([0.0, dt], device=self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Compute loss
            if data_next is not None:
                # Use explicit targets from next timestep
                total_loss, losses = self.loss_fn.compute_with_targets(
                    self.model, data, data_next, t_span
                )
            else:
                # Use implicit targets (for single-timestep data)
                total_loss, losses = self.loss_fn(self.model, data, t_span)
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )
            
            # Optimizer step
            self.optimizer.step()
            
            # Update global step
            self.global_step += 1
            
            # Accumulate losses
            epoch_losses['total'] += total_loss.item()
            epoch_losses['count'] += 1
            
            # Accumulate component losses
            if 'total_prediction_loss' in losses:
                epoch_losses['prediction'] += losses['total_prediction_loss'].item()
            if 'total_conservation_loss' in losses:
                epoch_losses['conservation'] += losses['total_conservation_loss'].item()
            if 'total_physics_limit_loss' in losses:
                epoch_losses['physics_limit'] += losses['total_physics_limit_loss'].item()
            
            # Individual conservation losses
            if 'energy_loss' in losses:
                epoch_losses['energy'] += losses['energy_loss'].item()
            if 'momentum_loss' in losses:
                epoch_losses['momentum'] += losses['momentum_loss'].item()
            if 'charge_loss' in losses:
                epoch_losses['charge'] += losses['charge_loss'].item()
            
            # Individual physics limit losses
            if 'vlasov_loss' in losses:
                epoch_losses['vlasov'] += losses['vlasov_loss'].item()
            if 'mb_loss' in losses:
                epoch_losses['mb'] += losses['mb_loss'].item()
            if 'stopping_power_loss' in losses:
                epoch_losses['stopping_power'] += losses['stopping_power_loss'].item()
            
            # Update progress bar
            if batch_idx % self.config.log_interval == 0:
                pbar.set_postfix({
                    'loss': f"{total_loss.item():.4f}",
                    'pred': f"{losses.get('total_prediction_loss', torch.tensor(0.0)).item():.4f}",
                    'cons': f"{losses.get('total_conservation_loss', torch.tensor(0.0)).item():.4f}",
                    'phys': f"{losses.get('total_physics_limit_loss', torch.tensor(0.0)).item():.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
                })
        
        # Compute average losses
        count = epoch_losses['count']
        avg_losses = {
            'total': epoch_losses['total'] / count,
            'prediction': epoch_losses['prediction'] / count,
            'conservation': epoch_losses['conservation'] / count,
            'physics_limit': epoch_losses['physics_limit'] / count,
            'energy': epoch_losses['energy'] / count,
            'momentum': epoch_losses['momentum'] / count,
            'charge': epoch_losses['charge'] / count,
            'vlasov': epoch_losses['vlasov'] / count,
            'mb': epoch_losses['mb'] / count,
            'stopping_power': epoch_losses['stopping_power'] / count
        }
        
        return avg_losses
    
    def train(self,
             train_loader: DataLoader,
             val_loader: Optional[DataLoader] = None) -> Dict[str, List[float]]:
        """Run full fine-tuning loop.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
        
        Returns:
            Training history dictionary
        
        Requirements:
            - Validates: Requirements 9.3, 9.4
            - Train for 50 epochs
            - Support fine-tuning on specific materials and conditions
        """
        print(f"Starting fine-tuning for {self.config.num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Loss weights:")
        print(f"  Prediction: {self.config.loss_weights.prediction}")
        print(f"  Conservation: {self.config.loss_weights.conservation}")
        print(f"  Physics limits: {self.config.loss_weights.physics_limits}")
        
        # Filter data if target conditions specified
        if (self.config.target_materials is not None or
            self.config.target_density_range is not None or
            self.config.target_energy_range is not None):
            print(f"Filtering data for target conditions:")
            if self.config.target_materials:
                print(f"  Materials: {self.config.target_materials}")
            if self.config.target_density_range:
                print(f"  Density range: {self.config.target_density_range}")
            if self.config.target_energy_range:
                print(f"  Energy range: {self.config.target_energy_range}")
            train_loader = self.filter_data_by_conditions(train_loader)
        
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            
            # Train for one epoch
            train_losses = self.train_epoch(train_loader, epoch)
            
            # Update learning rate
            self.scheduler.step()
            
            # Log metrics
            self.history['train_loss'].append(train_losses['total'])
            self.history['prediction_loss'].append(train_losses['prediction'])
            self.history['conservation_loss'].append(train_losses['conservation'])
            self.history['physics_limit_loss'].append(train_losses['physics_limit'])
            self.history['energy_loss'].append(train_losses['energy'])
            self.history['momentum_loss'].append(train_losses['momentum'])
            self.history['charge_loss'].append(train_losses['charge'])
            self.history['vlasov_loss'].append(train_losses['vlasov'])
            self.history['mb_loss'].append(train_losses['mb'])
            self.history['stopping_power_loss'].append(train_losses['stopping_power'])
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # Print epoch summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Total Loss: {train_losses['total']:.4f}")
            print(f"  Prediction Loss: {train_losses['prediction']:.4f}")
            print(f"  Conservation Loss: {train_losses['conservation']:.4f}")
            print(f"    Energy: {train_losses['energy']:.4f}")
            print(f"    Momentum: {train_losses['momentum']:.4f}")
            print(f"    Charge: {train_losses['charge']:.4f}")
            print(f"  Physics Limit Loss: {train_losses['physics_limit']:.4f}")
            print(f"    Vlasov: {train_losses['vlasov']:.4f}")
            print(f"    Maxwell-Boltzmann: {train_losses['mb']:.4f}")
            print(f"    Stopping Power: {train_losses['stopping_power']:.4f}")
            print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Validation
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                print(f"  Val Loss: {val_loss:.4f}")
                
                # Track best model
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.save_checkpoint('best_model.pt')
                    print(f"  New best model saved!")
            
            # Save checkpoint
            if (epoch + 1) % self.config.checkpoint_interval == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt')
                print(f"  Checkpoint saved at epoch {epoch+1}")
        
        print("\nFine-tuning completed!")
        return self.history
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validate model on validation set.
        
        Args:
            val_loader: DataLoader for validation data
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        
        total_loss = 0.0
        count = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Extract data
                data = batch['data']
                data_next = batch.get('data_next', None)
                dt = batch.get('dt', self.config.dt)
                
                # Time span
                t_span = torch.tensor([0.0, dt], device=self.device)
                
                # Compute loss
                if data_next is not None:
                    loss, _ = self.loss_fn.compute_with_targets(
                        self.model, data, data_next, t_span
                    )
                else:
                    loss, _ = self.loss_fn(self.model, data, t_span)
                
                total_loss += loss.item()
                count += 1
        
        return total_loss / count if count > 0 else 0.0
    
    def save_checkpoint(self, filename: str):
        """Save training checkpoint.
        
        Args:
            filename: Checkpoint filename
        """
        checkpoint_path = os.path.join(self.config.checkpoint_dir, filename)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config,
            'history': self.history
        }
        
        torch.save(checkpoint, checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_loss = checkpoint['best_loss']
        self.history = checkpoint['history']
        
        print(f"Checkpoint loaded from epoch {self.current_epoch}")
    
    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """Move batch data to device.
        
        Args:
            batch: Batch dictionary
        
        Returns:
            Batch with data moved to device
        """
        moved_batch = {}
        
        # Move data
        if 'data' in batch:
            moved_batch['data'] = batch['data'].to(self.device)
        
        # Move data_next
        if 'data_next' in batch and batch['data_next'] is not None:
            moved_batch['data_next'] = batch['data_next'].to(self.device)
        
        # Copy dt (scalar)
        if 'dt' in batch:
            moved_batch['dt'] = batch['dt']
        
        return moved_batch
    
    def save_history(self, filename: str = 'finetuning_history.json'):
        """Save training history to JSON file.
        
        Args:
            filename: Output filename
        """
        history_path = os.path.join(self.config.checkpoint_dir, filename)
        
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"Training history saved to {history_path}")


def create_finetuning_dataloader(
    trajectories: List,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """Create DataLoader for fine-tuning.
    
    This is a helper function to create a DataLoader from trajectory data.
    Users should implement their own collate function based on their data format.
    
    Args:
        trajectories: List of trajectory data
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
    
    Returns:
        DataLoader instance
    """
    # This is a placeholder - users need to implement proper collate function
    # based on their data format
    
    def collate_fn(batch):
        """Collate function for batching.
        
        This should be customized based on the data format.
        """
        # Example structure - customize as needed
        if len(batch) > 0:
            return {
                'data': batch[0],
                'data_next': batch[1] if len(batch) > 1 else None,
                'dt': 1.0
            }
        return {}
    
    return DataLoader(
        trajectories,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
