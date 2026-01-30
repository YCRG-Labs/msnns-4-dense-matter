"""Numerical stability utilities for training."""

import torch
import torch.nn as nn
import logging
from typing import Optional, Union, Iterable


logger = logging.getLogger(__name__)


def clip_gradients(
    model: nn.Module,
    max_norm: float = 1.0,
    norm_type: Union[float, str] = 2.0,
    error_if_nonfinite: bool = True
) -> Optional[torch.Tensor]:
    """Clip gradients by norm to prevent exploding gradients.
    
    This function clips the gradients of all parameters in the model
    to have a maximum norm. This is crucial for numerical stability
    during training, especially with Neural ODEs.
    
    Args:
        model: Neural network model
        max_norm: Maximum norm of gradients (default: 1.0)
        norm_type: Type of norm to use (default: 2.0 for L2 norm)
        error_if_nonfinite: If True, raises error if gradients contain NaN/Inf
    
    Returns:
        Total norm of gradients before clipping, or None if no gradients
    
    Raises:
        RuntimeError: If error_if_nonfinite=True and gradients contain NaN/Inf
    
    Example:
        >>> optimizer.zero_grad()
        >>> loss.backward()
        >>> total_norm = clip_gradients(model, max_norm=1.0)
        >>> optimizer.step()
    """
    # Get all parameters with gradients
    parameters = [p for p in model.parameters() if p.grad is not None]
    
    if not parameters:
        logger.warning("No gradients found to clip")
        return None
    
    # Check for NaN/Inf in gradients
    if error_if_nonfinite:
        for i, p in enumerate(parameters):
            if torch.isnan(p.grad).any():
                raise RuntimeError(
                    f"NaN detected in gradients of parameter {i}. "
                    "This indicates numerical instability."
                )
            if torch.isinf(p.grad).any():
                raise RuntimeError(
                    f"Inf detected in gradients of parameter {i}. "
                    "This indicates numerical instability or exploding gradients."
                )
    
    # Clip gradients
    total_norm = torch.nn.utils.clip_grad_norm_(
        parameters,
        max_norm=max_norm,
        norm_type=norm_type,
        error_if_nonfinite=error_if_nonfinite
    )
    
    # Log if gradients were clipped significantly
    if total_norm > max_norm * 2:
        logger.warning(
            f"Large gradient norm detected: {total_norm:.4f} "
            f"(clipped to {max_norm})"
        )
    
    return total_norm


def check_model_outputs(
    outputs: dict,
    stage: str = "forward"
) -> bool:
    """Check model outputs for NaN or Inf values.
    
    Args:
        outputs: Dictionary of model outputs
        stage: Name of the stage for error messages
    
    Returns:
        True if all outputs are finite
    
    Raises:
        ValueError: If NaN or Inf detected in any output
    """
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            if torch.isnan(value).any():
                raise ValueError(
                    f"NaN detected in {key} during {stage}. "
                    "This indicates numerical instability."
                )
            if torch.isinf(value).any():
                raise ValueError(
                    f"Inf detected in {key} during {stage}. "
                    "This indicates numerical overflow or instability."
                )
    
    return True


def check_loss_value(
    loss: torch.Tensor,
    max_loss: float = 1e6
) -> bool:
    """Check if loss value is valid and reasonable.
    
    Args:
        loss: Loss tensor
        max_loss: Maximum reasonable loss value
    
    Returns:
        True if loss is valid
    
    Raises:
        ValueError: If loss is NaN, Inf, or unreasonably large
    """
    if torch.isnan(loss):
        raise ValueError(
            "NaN loss detected. This indicates numerical instability. "
            "Try reducing learning rate or checking input data."
        )
    
    if torch.isinf(loss):
        raise ValueError(
            "Inf loss detected. This indicates numerical overflow. "
            "Try reducing learning rate or using gradient clipping."
        )
    
    if loss.item() > max_loss:
        logger.warning(
            f"Very large loss detected: {loss.item():.4e}. "
            "This may indicate training instability."
        )
    
    return True


def safe_divide(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    epsilon: float = 1e-8
) -> torch.Tensor:
    """Safely divide two tensors, avoiding division by zero.
    
    Args:
        numerator: Numerator tensor
        denominator: Denominator tensor
        epsilon: Small value to add to denominator
    
    Returns:
        Result of safe division
    """
    return numerator / (denominator + epsilon)


def safe_log(
    x: torch.Tensor,
    epsilon: float = 1e-8
) -> torch.Tensor:
    """Safely compute logarithm, avoiding log(0).
    
    Args:
        x: Input tensor
        epsilon: Small value to add before taking log
    
    Returns:
        Result of safe logarithm
    """
    return torch.log(x + epsilon)


def safe_sqrt(
    x: torch.Tensor,
    epsilon: float = 1e-8
) -> torch.Tensor:
    """Safely compute square root, avoiding sqrt of negative numbers.
    
    Args:
        x: Input tensor
        epsilon: Small value to ensure positivity
    
    Returns:
        Result of safe square root
    """
    return torch.sqrt(torch.maximum(x, torch.tensor(epsilon, device=x.device)))


class GradientMonitor:
    """Monitor gradient statistics during training.
    
    Tracks gradient norms, detects vanishing/exploding gradients,
    and provides warnings for numerical issues.
    
    Example:
        >>> monitor = GradientMonitor()
        >>> for epoch in range(num_epochs):
        ...     for batch in dataloader:
        ...         optimizer.zero_grad()
        ...         loss.backward()
        ...         stats = monitor.update(model)
        ...         optimizer.step()
    """
    
    def __init__(self, window_size: int = 100):
        """Initialize gradient monitor.
        
        Args:
            window_size: Number of steps to track for moving statistics
        """
        self.window_size = window_size
        self.gradient_norms = []
        self.step = 0
    
    def update(self, model: nn.Module) -> dict:
        """Update gradient statistics.
        
        Args:
            model: Neural network model
        
        Returns:
            Dictionary with gradient statistics
        """
        # Compute total gradient norm
        total_norm = 0.0
        num_params = 0
        
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                num_params += 1
        
        total_norm = total_norm ** 0.5
        
        # Track gradient norms
        self.gradient_norms.append(total_norm)
        if len(self.gradient_norms) > self.window_size:
            self.gradient_norms.pop(0)
        
        self.step += 1
        
        # Compute statistics
        avg_norm = sum(self.gradient_norms) / len(self.gradient_norms)
        max_norm = max(self.gradient_norms)
        min_norm = min(self.gradient_norms)
        
        # Check for vanishing gradients
        if avg_norm < 1e-6:
            logger.warning(
                f"Vanishing gradients detected (avg norm: {avg_norm:.2e}). "
                "Consider increasing learning rate or checking model architecture."
            )
        
        # Check for exploding gradients
        if avg_norm > 100:
            logger.warning(
                f"Exploding gradients detected (avg norm: {avg_norm:.2e}). "
                "Consider using gradient clipping or reducing learning rate."
            )
        
        return {
            'step': self.step,
            'total_norm': total_norm,
            'avg_norm': avg_norm,
            'max_norm': max_norm,
            'min_norm': min_norm,
            'num_params': num_params
        }
    
    def reset(self):
        """Reset gradient statistics."""
        self.gradient_norms = []
        self.step = 0
