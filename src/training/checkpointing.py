"""Checkpointing utilities for saving and loading training state.

This module provides centralized checkpoint management for:
- Saving model state, optimizer state, epoch, hyperparameters
- Loading checkpoints to resume training
- Configurable save intervals
- Best model tracking

Requirements:
    - Validates: Requirements 17.1, 17.3, 17.4
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from pathlib import Path
import os
import json
from datetime import datetime


class CheckpointManager:
    """Manager for saving and loading training checkpoints.
    
    Handles checkpoint creation, saving, loading, and management with
    support for:
    - Regular interval checkpoints
    - Best model checkpoints
    - Hyperparameter and configuration saving
    - Resume training from saved state
    
    Requirements:
        - Validates: Requirement 17.1 (save checkpoints at configurable intervals)
        - Validates: Requirement 17.3 (resume training from saved checkpoints)
        - Validates: Requirement 17.4 (save hyperparameters with checkpoints)
    """
    
    def __init__(self,
                 checkpoint_dir: str,
                 save_interval: int = 10,
                 keep_last_n: Optional[int] = None):
        """Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            save_interval: Save checkpoint every N epochs
            keep_last_n: Keep only last N checkpoints (None = keep all)
        
        Requirements:
            - Validates: Requirement 17.1 (configurable intervals)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_interval = save_interval
        self.keep_last_n = keep_last_n
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Track saved checkpoints
        self.saved_checkpoints = []
        
        # Best model tracking
        self.best_metric = float('inf')
        self.best_checkpoint_path = None
    
    def save_checkpoint(self,
                       model: nn.Module,
                       optimizer: torch.optim.Optimizer,
                       epoch: int,
                       config: Any,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                       metrics: Optional[Dict[str, float]] = None,
                       history: Optional[Dict[str, list]] = None,
                       extra_state: Optional[Dict[str, Any]] = None,
                       filename: Optional[str] = None) -> str:
        """Save training checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer to save
            epoch: Current epoch number
            config: Training configuration object
            scheduler: Optional learning rate scheduler
            metrics: Optional dictionary of current metrics
            history: Optional training history
            extra_state: Optional extra state to save
            filename: Optional custom filename (default: checkpoint_epoch_{epoch}.pt)
        
        Returns:
            Path to saved checkpoint
        
        Requirements:
            - Validates: Requirement 17.1 (save model state, optimizer state, epoch, hyperparameters)
            - Validates: Requirement 17.4 (save hyperparameters and configuration)
        """
        # Create checkpoint dictionary
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': self._serialize_config(config),
            'timestamp': datetime.now().isoformat(),
        }
        
        # Add optional components
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        if metrics is not None:
            checkpoint['metrics'] = metrics
        
        if history is not None:
            checkpoint['history'] = history
        
        if extra_state is not None:
            checkpoint['extra_state'] = extra_state
        
        # Determine filename
        if filename is None:
            filename = f'checkpoint_epoch_{epoch}.pt'
        
        checkpoint_path = self.checkpoint_dir / filename
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        
        # Track saved checkpoint
        self.saved_checkpoints.append(checkpoint_path)
        
        # Clean up old checkpoints if needed
        if self.keep_last_n is not None:
            self._cleanup_old_checkpoints()
        
        return str(checkpoint_path)
    
    def save_best_checkpoint(self,
                            model: nn.Module,
                            optimizer: torch.optim.Optimizer,
                            epoch: int,
                            config: Any,
                            metric_value: float,
                            scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                            metrics: Optional[Dict[str, float]] = None,
                            history: Optional[Dict[str, list]] = None,
                            extra_state: Optional[Dict[str, Any]] = None,
                            lower_is_better: bool = True) -> Optional[str]:
        """Save checkpoint if it's the best so far.
        
        Args:
            model: Model to save
            optimizer: Optimizer to save
            epoch: Current epoch number
            config: Training configuration object
            metric_value: Metric value to compare
            scheduler: Optional learning rate scheduler
            metrics: Optional dictionary of current metrics
            history: Optional training history
            extra_state: Optional extra state to save
            lower_is_better: Whether lower metric values are better
        
        Returns:
            Path to saved checkpoint if it's the best, None otherwise
        
        Requirements:
            - Validates: Requirement 17.1 (save model state, optimizer state, epoch, hyperparameters)
        """
        is_best = False
        
        if lower_is_better:
            if metric_value < self.best_metric:
                is_best = True
                self.best_metric = metric_value
        else:
            if metric_value > self.best_metric:
                is_best = True
                self.best_metric = metric_value
        
        if is_best:
            checkpoint_path = self.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                config=config,
                scheduler=scheduler,
                metrics=metrics,
                history=history,
                extra_state=extra_state,
                filename='best_model.pt'
            )
            
            self.best_checkpoint_path = checkpoint_path
            return checkpoint_path
        
        return None
    
    def load_checkpoint(self,
                       checkpoint_path: str,
                       model: nn.Module,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                       device: Optional[torch.device] = None) -> Dict[str, Any]:
        """Load checkpoint and restore training state.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optional optimizer to load state into
            scheduler: Optional scheduler to load state into
            device: Optional device to map checkpoint to
        
        Returns:
            Dictionary containing checkpoint metadata (epoch, config, metrics, etc.)
        
        Requirements:
            - Validates: Requirement 17.3 (load model and optimizer state, resume from saved epoch)
        """
        # Load checkpoint
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if provided
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Extract metadata
        metadata = {
            'epoch': checkpoint.get('epoch', 0),
            'config': checkpoint.get('config', None),
            'metrics': checkpoint.get('metrics', None),
            'history': checkpoint.get('history', None),
            'extra_state': checkpoint.get('extra_state', None),
            'timestamp': checkpoint.get('timestamp', None)
        }
        
        return metadata
    
    def should_save_checkpoint(self, epoch: int) -> bool:
        """Check if checkpoint should be saved at this epoch.
        
        Args:
            epoch: Current epoch number
        
        Returns:
            True if checkpoint should be saved
        
        Requirements:
            - Validates: Requirement 17.1 (save at configurable intervals)
        """
        return (epoch + 1) % self.save_interval == 0
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to the most recent checkpoint.
        
        Returns:
            Path to latest checkpoint or None if no checkpoints exist
        """
        checkpoints = list(self.checkpoint_dir.glob('checkpoint_epoch_*.pt'))
        
        if not checkpoints:
            return None
        
        # Sort by epoch number
        def extract_epoch(path):
            try:
                return int(path.stem.split('_')[-1])
            except:
                return -1
        
        checkpoints.sort(key=extract_epoch, reverse=True)
        return str(checkpoints[0])
    
    def get_best_checkpoint(self) -> Optional[str]:
        """Get path to the best model checkpoint.
        
        Returns:
            Path to best checkpoint or None if not saved yet
        """
        best_path = self.checkpoint_dir / 'best_model.pt'
        
        if best_path.exists():
            return str(best_path)
        
        return self.best_checkpoint_path
    
    def list_checkpoints(self) -> list:
        """List all saved checkpoints.
        
        Returns:
            List of checkpoint paths sorted by epoch
        """
        checkpoints = list(self.checkpoint_dir.glob('checkpoint_epoch_*.pt'))
        
        # Sort by epoch number
        def extract_epoch(path):
            try:
                return int(path.stem.split('_')[-1])
            except:
                return -1
        
        checkpoints.sort(key=extract_epoch)
        return [str(p) for p in checkpoints]
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints keeping only the last N.
        
        Requirements:
            - Validates: Requirement 17.1 (checkpoint management)
        """
        if self.keep_last_n is None:
            return
        
        # Get all epoch checkpoints (exclude best_model.pt)
        checkpoints = [p for p in self.saved_checkpoints 
                      if 'best_model' not in str(p)]
        
        # Remove oldest checkpoints
        if len(checkpoints) > self.keep_last_n:
            to_remove = checkpoints[:-self.keep_last_n]
            
            for checkpoint_path in to_remove:
                try:
                    if os.path.exists(checkpoint_path):
                        os.remove(checkpoint_path)
                    self.saved_checkpoints.remove(checkpoint_path)
                except Exception as e:
                    print(f"Warning: Failed to remove checkpoint {checkpoint_path}: {e}")
    
    def _serialize_config(self, config: Any) -> Dict[str, Any]:
        """Serialize configuration object to dictionary.
        
        Args:
            config: Configuration object (dataclass or dict)
        
        Returns:
            Dictionary representation of config
        
        Requirements:
            - Validates: Requirement 17.4 (save hyperparameters and configuration)
        """
        if isinstance(config, dict):
            return config
        
        # Try to convert dataclass to dict
        if hasattr(config, '__dataclass_fields__'):
            import dataclasses
            return dataclasses.asdict(config)
        
        # Try to get __dict__
        if hasattr(config, '__dict__'):
            return {k: v for k, v in config.__dict__.items() 
                   if not k.startswith('_')}
        
        # Fallback: convert to string
        return {'config_str': str(config)}
    
    def save_config_json(self, config: Any, filename: str = 'config.json'):
        """Save configuration to JSON file.
        
        Args:
            config: Configuration object
            filename: Output filename
        
        Requirements:
            - Validates: Requirement 17.4 (save hyperparameters and configuration)
        """
        config_dict = self._serialize_config(config)
        config_path = self.checkpoint_dir / filename
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        
        print(f"Configuration saved to {config_path}")
    
    def load_config_json(self, filename: str = 'config.json') -> Dict[str, Any]:
        """Load configuration from JSON file.
        
        Args:
            filename: Config filename
        
        Returns:
            Configuration dictionary
        """
        config_path = self.checkpoint_dir / filename
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        return config


def save_checkpoint_simple(
    filepath: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    **kwargs
):
    """Simple checkpoint saving function.
    
    Convenience function for quick checkpoint saving without CheckpointManager.
    
    Args:
        filepath: Path to save checkpoint
        model: Model to save
        optimizer: Optimizer to save
        epoch: Current epoch
        **kwargs: Additional items to save in checkpoint
    
    Requirements:
        - Validates: Requirement 17.1 (save model state, optimizer state, epoch)
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        **kwargs
    }
    
    # Create directory if needed
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(checkpoint, filepath)


def load_checkpoint_simple(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """Simple checkpoint loading function.
    
    Convenience function for quick checkpoint loading without CheckpointManager.
    
    Args:
        filepath: Path to checkpoint file
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        device: Optional device to map checkpoint to
    
    Returns:
        Checkpoint dictionary
    
    Requirements:
        - Validates: Requirement 17.3 (load model and optimizer state)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint
