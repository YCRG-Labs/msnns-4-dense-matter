"""Pretraining loop for self-supervised learning.

This module implements the pretraining pipeline with:
- AdamW optimizer with cosine annealing
- Gradient clipping
- Support for diverse materials, densities, energies, temperatures
- Logging and checkpointing

Requirements:
    - Validates: Requirement 8.5
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
import os
from pathlib import Path
import json
from tqdm import tqdm

from .pretraining_losses import CombinedPretrainingLoss, PretrainingLossWeights


@dataclass
class PretrainingConfig:
    """Configuration for pretraining.
    
    Attributes:
        # Optimization
        learning_rate: Initial learning rate
        weight_decay: Weight decay for AdamW
        max_grad_norm: Maximum gradient norm for clipping
        
        # Training
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        
        # Loss weights
        loss_weights: Weights for combining pretraining losses
        
        # Data
        dt: Time step between trajectory frames (fs)
        
        # Checkpointing
        checkpoint_dir: Directory to save checkpoints
        checkpoint_interval: Save checkpoint every N epochs
        
        # Logging
        log_interval: Log metrics every N batches
        
        # Device
        device: Device to train on ('cuda' or 'cpu')
    """
    # Optimization
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    
    # Training
    num_epochs: int = 100
    batch_size: int = 32
    
    # Loss weights
    loss_weights: Optional[PretrainingLossWeights] = None
    
    # Data
    dt: float = 1.0
    
    # Checkpointing
    checkpoint_dir: str = 'checkpoints/pretraining'
    checkpoint_interval: int = 10
    
    # Logging
    log_interval: int = 10
    
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def __post_init__(self):
        """Set default values for optional parameters."""
        if self.loss_weights is None:
            self.loss_weights = PretrainingLossWeights()


class PretrainingTrainer:
    """Trainer for self-supervised pretraining.
    
    Implements the pretraining loop with:
    - AdamW optimizer with cosine annealing
    - Gradient clipping
    - Logging and checkpointing
    
    Requirements:
        - Validates: Requirement 8.5
        - Set up AdamW optimizer with cosine annealing
        - Train for 100 epochs with gradient clipping
        - Support diverse materials, densities, energies, temperatures
    """
    
    def __init__(self,
                 model: nn.Module,
                 config: PretrainingConfig):
        """Initialize pretraining trainer.
        
        Args:
            model: MultiScaleModel instance
            config: Pretraining configuration
        """
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize loss function
        self.loss_fn = CombinedPretrainingLoss(config.loss_weights)
        
        # Initialize optimizer
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
            'autoregressive_loss': [],
            'contrastive_loss': [],
            'masked_particle_loss': [],
            'learning_rate': []
        }
    
    def train_epoch(self,
                   train_loader: DataLoader,
                   epoch: int) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            train_loader: DataLoader providing training data
                Each batch should be a dictionary with:
                - 'trajectory': List of GraphData for autoregressive loss (optional)
                - 'batch_data': List of GraphData for contrastive loss (optional)
                - 'single_data': GraphData for masked particle loss (optional)
            epoch: Current epoch number
        
        Returns:
            Dictionary with average losses for the epoch
        """
        self.model.train()
        
        epoch_losses = {
            'total': 0.0,
            'autoregressive': 0.0,
            'contrastive': 0.0,
            'masked_particle': 0.0,
            'count': 0
        }
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{self.config.num_epochs}')
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Compute loss
            losses = self.loss_fn(
                model=self.model,
                trajectory=batch.get('trajectory'),
                batch_data=batch.get('batch_data'),
                single_data=batch.get('single_data'),
                dt=self.config.dt
            )
            
            # Backward pass
            total_loss = losses['total']
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
            
            if 'autoregressive' in losses:
                epoch_losses['autoregressive'] += losses['autoregressive'].item()
            if 'contrastive' in losses:
                epoch_losses['contrastive'] += losses['contrastive'].item()
            if 'masked_particle' in losses:
                epoch_losses['masked_particle'] += losses['masked_particle'].item()
            
            # Update progress bar
            if batch_idx % self.config.log_interval == 0:
                pbar.set_postfix({
                    'loss': f"{total_loss.item():.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
                })
        
        # Compute average losses
        avg_losses = {
            'total': epoch_losses['total'] / epoch_losses['count'],
            'autoregressive': epoch_losses['autoregressive'] / epoch_losses['count'],
            'contrastive': epoch_losses['contrastive'] / epoch_losses['count'],
            'masked_particle': epoch_losses['masked_particle'] / epoch_losses['count']
        }
        
        return avg_losses
    
    def train(self,
             train_loader: DataLoader,
             val_loader: Optional[DataLoader] = None) -> Dict[str, List[float]]:
        """Run full pretraining loop.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
        
        Returns:
            Training history dictionary
        """
        print(f"Starting pretraining for {self.config.num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Batch size: {self.config.batch_size}")
        
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            
            # Train for one epoch
            train_losses = self.train_epoch(train_loader, epoch)
            
            # Update learning rate
            self.scheduler.step()
            
            # Log metrics
            self.history['train_loss'].append(train_losses['total'])
            self.history['autoregressive_loss'].append(train_losses['autoregressive'])
            self.history['contrastive_loss'].append(train_losses['contrastive'])
            self.history['masked_particle_loss'].append(train_losses['masked_particle'])
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # Print epoch summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_losses['total']:.4f}")
            print(f"  Autoregressive: {train_losses['autoregressive']:.4f}")
            print(f"  Contrastive: {train_losses['contrastive']:.4f}")
            print(f"  Masked Particle: {train_losses['masked_particle']:.4f}")
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
        
        print("\nPretraining completed!")
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
                
                # Compute loss
                losses = self.loss_fn(
                    model=self.model,
                    trajectory=batch.get('trajectory'),
                    batch_data=batch.get('batch_data'),
                    single_data=batch.get('single_data'),
                    dt=self.config.dt
                )
                
                total_loss += losses['total'].item()
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
        
        # Move trajectory
        if 'trajectory' in batch and batch['trajectory'] is not None:
            moved_batch['trajectory'] = [
                self._move_data_to_device(data) for data in batch['trajectory']
            ]
        
        # Move batch_data
        if 'batch_data' in batch and batch['batch_data'] is not None:
            moved_batch['batch_data'] = [
                self._move_data_to_device(data) for data in batch['batch_data']
            ]
        
        # Move single_data
        if 'single_data' in batch and batch['single_data'] is not None:
            moved_batch['single_data'] = self._move_data_to_device(batch['single_data'])
        
        return moved_batch
    
    def _move_data_to_device(self, data):
        """Move GraphData to device.
        
        Args:
            data: GraphData object
        
        Returns:
            GraphData on device
        """
        return data.to(self.device)
    
    def save_history(self, filename: str = 'training_history.json'):
        """Save training history to JSON file.
        
        Args:
            filename: Output filename
        """
        history_path = os.path.join(self.config.checkpoint_dir, filename)
        
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"Training history saved to {history_path}")


def create_pretraining_dataloader(
    trajectories: List,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """Create DataLoader for pretraining.
    
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
        return {
            'trajectory': batch[0] if len(batch) > 0 else None,
            'batch_data': batch if len(batch) > 1 else None,
            'single_data': batch[0] if len(batch) > 0 else None
        }
    
    return DataLoader(
        trajectories,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
