"""Training logging utilities with TensorBoard support.

This module provides centralized logging for:
- Loss values and metrics to TensorBoard
- Conservation errors
- Training progress
- Hyperparameters

Requirements:
    - Validates: Requirement 17.2
"""

import torch
from typing import Dict, Any, Optional, Union
from pathlib import Path
import json
from datetime import datetime

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: TensorBoard not available. Install with: pip install tensorboard")


class TrainingLogger:
    """Logger for training metrics and progress.
    
    Provides unified interface for logging to:
    - TensorBoard (if available)
    - JSON files
    - Console output
    
    Requirements:
        - Validates: Requirement 17.2 (log loss values, conservation errors, and metrics)
    """
    
    def __init__(self,
                 log_dir: str,
                 experiment_name: Optional[str] = None,
                 use_tensorboard: bool = True,
                 console_log: bool = True):
        """Initialize training logger."""
        self.log_dir = Path(log_dir)
        self.console_log = console_log
        
        # Create experiment name
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.experiment_name = experiment_name
        self.experiment_dir = self.log_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize TensorBoard writer
        self.tensorboard_writer = None
        if use_tensorboard and TENSORBOARD_AVAILABLE:
            tensorboard_dir = self.experiment_dir / 'tensorboard'
            tensorboard_dir.mkdir(exist_ok=True)
            self.tensorboard_writer = SummaryWriter(log_dir=str(tensorboard_dir))
        
        # Initialize JSON log file
        self.json_log_path = self.experiment_dir / 'metrics.json'
        self.metrics_history = []
        
        # Console formatting
        self.console_width = 80

    def log_hyperparameters(self, hparams: Dict[str, Any]):
        """Log hyperparameters for the experiment."""
        # Save to JSON
        hparams_path = self.experiment_dir / 'hyperparameters.json'
        with open(hparams_path, 'w') as f:
            json.dump(hparams, f, indent=2)
        
        # Log to TensorBoard
        if self.tensorboard_writer is not None:
            hparams_str = {k: str(v) for k, v in hparams.items()}
            self.tensorboard_writer.add_hparams(hparams_str, {})
        
        # Console output
        if self.console_log:
            print("\n" + "=" * self.console_width)
            print("HYPERPARAMETERS".center(self.console_width))
            print("=" * self.console_width)
            for key, value in hparams.items():
                print(f"  {key}: {value}")
            print("=" * self.console_width + "\n")
    
    def log_metrics(self,
                   metrics: Dict[str, float],
                   step: int,
                   prefix: str = ""):
        """Log metrics at a given step."""
        # Add prefix to metric names
        if prefix and not prefix.endswith('/'):
            prefix = prefix + '/'
        
        prefixed_metrics = {f"{prefix}{k}": v for k, v in metrics.items()}
        
        # Log to TensorBoard
        if self.tensorboard_writer is not None:
            for name, value in prefixed_metrics.items():
                self.tensorboard_writer.add_scalar(name, value, step)
        
        # Save to JSON history
        log_entry = {
            'step': step,
            'metrics': prefixed_metrics,
            'timestamp': datetime.now().isoformat()
        }
        self.metrics_history.append(log_entry)
        
        # Periodically save JSON (every 10 steps)
        if len(self.metrics_history) % 10 == 0:
            self._save_json_log()
        
        # Console output
        if self.console_log:
            metrics_str = ", ".join([f"{k}: {v:.6f}" for k, v in metrics.items()])
            print(f"Step {step:5d} | {prefix[:-1] if prefix else 'metrics'} | {metrics_str}")
    
    def log_conservation_errors(self,
                                errors: Dict[str, float],
                                step: int,
                                prefix: str = "conservation"):
        """Log conservation errors (energy, momentum, charge)."""
        self.log_metrics(errors, step, prefix=prefix)
    
    def log_loss_components(self,
                           losses: Dict[str, float],
                           step: int,
                           prefix: str = "loss"):
        """Log individual loss components."""
        self.log_metrics(losses, step, prefix=prefix)
    
    def log_epoch_summary(self,
                         epoch: int,
                         train_metrics: Dict[str, float],
                         val_metrics: Optional[Dict[str, float]] = None):
        """Log summary for an epoch."""
        if self.console_log:
            print("\n" + "-" * self.console_width)
            print(f"EPOCH {epoch} SUMMARY".center(self.console_width))
            print("-" * self.console_width)
            
            print("Training:")
            for key, value in train_metrics.items():
                print(f"  {key}: {value:.6f}")
            
            if val_metrics:
                print("\nValidation:")
                for key, value in val_metrics.items():
                    print(f"  {key}: {value:.6f}")
            
            print("-" * self.console_width + "\n")
        
        # Log to TensorBoard
        self.log_metrics(train_metrics, epoch, prefix="epoch/train")
        if val_metrics:
            self.log_metrics(val_metrics, epoch, prefix="epoch/val")

    def log_model_graph(self, model: torch.nn.Module, input_data: Any):
        """Log model computational graph to TensorBoard."""
        if self.tensorboard_writer is not None:
            try:
                self.tensorboard_writer.add_graph(model, input_data)
            except Exception as e:
                print(f"Warning: Could not log model graph: {e}")
    
    def log_histogram(self,
                     name: str,
                     values: torch.Tensor,
                     step: int):
        """Log histogram of tensor values."""
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_histogram(name, values, step)
    
    def log_gradients(self,
                     model: torch.nn.Module,
                     step: int):
        """Log gradient histograms for model parameters."""
        if self.tensorboard_writer is not None:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    self.tensorboard_writer.add_histogram(
                        f"gradients/{name}",
                        param.grad,
                        step
                    )
                    self.tensorboard_writer.add_scalar(
                        f"gradient_norm/{name}",
                        param.grad.norm().item(),
                        step
                    )
    
    def log_weights(self,
                   model: torch.nn.Module,
                   step: int):
        """Log weight histograms for model parameters."""
        if self.tensorboard_writer is not None:
            for name, param in model.named_parameters():
                self.tensorboard_writer.add_histogram(
                    f"weights/{name}",
                    param,
                    step
                )
    
    def _save_json_log(self):
        """Save metrics history to JSON file."""
        with open(self.json_log_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
    
    def close(self):
        """Close logger and save final state."""
        # Save final JSON log
        self._save_json_log()
        
        # Close TensorBoard writer
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.close()
        
        if self.console_log:
            print(f"\nLogs saved to: {self.experiment_dir}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def create_logger(log_dir: str = "logs",
                 experiment_name: Optional[str] = None,
                 use_tensorboard: bool = True,
                 console_log: bool = True) -> TrainingLogger:
    """Factory function to create a training logger."""
    return TrainingLogger(
        log_dir=log_dir,
        experiment_name=experiment_name,
        use_tensorboard=use_tensorboard,
        console_log=console_log
    )
