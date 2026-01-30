"""Weights & Biases experiment tracking integration.

This module provides optional W&B integration for cloud-based experiment tracking.

Requirements:
    - Validates: Requirement 17.5
"""

import torch
from typing import Dict, Any, Optional, Union
from pathlib import Path

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: Weights & Biases not available. Install with: pip install wandb")


class WandbLogger:
    """Weights & Biases logger for experiment tracking.
    
    Provides cloud-based experiment tracking with:
    - Automatic hyperparameter logging
    - Real-time metric visualization
    - Model artifact versioning
    - Collaborative experiment comparison
    
    Requirements:
        - Validates: Requirement 17.5 (experiment tracking with W&B)
    """
    
    def __init__(self,
                 project: str,
                 name: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None,
                 tags: Optional[list] = None,
                 notes: Optional[str] = None,
                 entity: Optional[str] = None,
                 mode: str = "online"):
        """Initialize W&B logger."""
        if not WANDB_AVAILABLE:
            raise ImportError(
                "Weights & Biases is not installed. "
                "Install with: pip install wandb"
            )
        
        self.project = project
        self.run = None
        
        # Initialize W&B run
        self.run = wandb.init(
            project=project,
            name=name,
            config=config,
            tags=tags,
            notes=notes,
            entity=entity,
            mode=mode
        )
        
        # Store config
        self.config = wandb.config if self.run else {}
    
    def log_metrics(self,
                   metrics: Dict[str, float],
                   step: Optional[int] = None,
                   commit: bool = True):
        """Log metrics to W&B."""
        if self.run is not None:
            wandb.log(metrics, step=step, commit=commit)
    
    def log_conservation_errors(self,
                                errors: Dict[str, float],
                                step: Optional[int] = None):
        """Log conservation errors to W&B."""
        prefixed_errors = {f"conservation/{k}": v for k, v in errors.items()}
        self.log_metrics(prefixed_errors, step=step)
    
    def log_loss_components(self,
                           losses: Dict[str, float],
                           step: Optional[int] = None):
        """Log individual loss components to W&B."""
        prefixed_losses = {f"loss/{k}": v for k, v in losses.items()}
        self.log_metrics(prefixed_losses, step=step)
    
    def log_model_artifact(self,
                          model: torch.nn.Module,
                          name: str,
                          metadata: Optional[Dict[str, Any]] = None):
        """Log model as W&B artifact for versioning."""
        if self.run is not None:
            # Create artifact
            artifact = wandb.Artifact(
                name=name,
                type="model",
                metadata=metadata or {}
            )
            
            # Save model to temporary file
            model_path = Path(wandb.run.dir) / f"{name}.pt"
            torch.save(model.state_dict(), model_path)
            
            # Add file to artifact
            artifact.add_file(str(model_path))
            
            # Log artifact
            wandb.log_artifact(artifact)
    
    def log_checkpoint_artifact(self,
                               checkpoint_path: Union[str, Path],
                               name: str,
                               metadata: Optional[Dict[str, Any]] = None):
        """Log checkpoint as W&B artifact."""
        if self.run is not None:
            artifact = wandb.Artifact(
                name=name,
                type="checkpoint",
                metadata=metadata or {}
            )
            artifact.add_file(str(checkpoint_path))
            wandb.log_artifact(artifact)
    
    def log_histogram(self,
                     name: str,
                     values: torch.Tensor,
                     step: Optional[int] = None):
        """Log histogram to W&B."""
        if self.run is not None:
            wandb.log({name: wandb.Histogram(values.cpu().numpy())}, step=step)
    
    def log_gradients(self,
                     model: torch.nn.Module,
                     step: Optional[int] = None):
        """Log gradient histograms to W&B."""
        if self.run is not None:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    self.log_histogram(
                        f"gradients/{name}",
                        param.grad,
                        step=step
                    )
    
    def log_weights(self,
                   model: torch.nn.Module,
                   step: Optional[int] = None):
        """Log weight histograms to W&B."""
        if self.run is not None:
            for name, param in model.named_parameters():
                self.log_histogram(
                    f"weights/{name}",
                    param,
                    step=step
                )
    
    def watch_model(self,
                   model: torch.nn.Module,
                   log: str = "gradients",
                   log_freq: int = 100):
        """Watch model for automatic gradient and parameter logging."""
        if self.run is not None:
            wandb.watch(model, log=log, log_freq=log_freq)
    
    def log_config(self, config: Dict[str, Any]):
        """Update run configuration."""
        if self.run is not None:
            wandb.config.update(config)
    
    def log_summary(self, summary: Dict[str, Any]):
        """Log summary metrics (final values)."""
        if self.run is not None:
            for key, value in summary.items():
                wandb.run.summary[key] = value
    
    def finish(self):
        """Finish the W&B run."""
        if self.run is not None:
            wandb.finish()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finish()


def create_wandb_logger(project: str,
                       name: Optional[str] = None,
                       config: Optional[Dict[str, Any]] = None,
                       tags: Optional[list] = None,
                       notes: Optional[str] = None,
                       entity: Optional[str] = None,
                       mode: str = "online") -> Optional[WandbLogger]:
    """Factory function to create a W&B logger."""
    if not WANDB_AVAILABLE:
        print("Warning: Weights & Biases not available. Skipping W&B logging.")
        return None
    
    try:
        return WandbLogger(
            project=project,
            name=name,
            config=config,
            tags=tags,
            notes=notes,
            entity=entity,
            mode=mode
        )
    except Exception as e:
        print(f"Warning: Could not initialize W&B logger: {e}")
        return None
