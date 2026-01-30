"""Physics-informed conservation law constraints.

This module implements functions to compute and enforce conservation laws:
- Energy conservation
- Momentum conservation
- Charge conservation

These can be used as auxiliary loss terms during training to ensure
physically consistent predictions.

Requirements:
    - Validates: Requirements 6.1, 6.2, 6.3, 6.4
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


def compute_kinetic_energy(
    momentum: torch.Tensor,
    mass: torch.Tensor
) -> torch.Tensor:
    """Compute kinetic energy for particles.
    
    E_kinetic = Σ ||p_i||² / (2 * m_i)
    
    Args:
        momentum: Particle momenta (N, 3) in atomic units
        mass: Particle masses (N,) in atomic mass units
    
    Returns:
        Total kinetic energy (scalar)
    """
    # Compute ||p_i||² for each particle
    p_squared = torch.sum(momentum ** 2, dim=-1)  # (N,)
    
    # Compute E_i = ||p_i||² / (2 * m_i)
    kinetic_energy_per_particle = p_squared / (2.0 * mass)  # (N,)
    
    # Sum over all particles
    total_energy = torch.sum(kinetic_energy_per_particle)
    
    return total_energy


def compute_total_energy(
    momentum: torch.Tensor,
    mass: torch.Tensor,
    potential_energy: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Compute total energy (kinetic + potential) for particles.
    
    E_total = Σ (||p_i||² / (2 * m_i) + U_i)
    
    Args:
        momentum: Particle momenta (N, 3) in atomic units
        mass: Particle masses (N,) in atomic mass units
        potential_energy: Optional potential energy per particle (N,)
                         If None, only kinetic energy is computed
    
    Returns:
        Total energy (scalar)
    """
    kinetic = compute_kinetic_energy(momentum, mass)
    
    if potential_energy is not None:
        potential = torch.sum(potential_energy)
        return kinetic + potential
    
    return kinetic


def energy_conservation_error(
    pred_momentum: torch.Tensor,
    true_momentum: torch.Tensor,
    mass: torch.Tensor,
    pred_potential: Optional[torch.Tensor] = None,
    true_potential: Optional[torch.Tensor] = None,
    relative: bool = True
) -> torch.Tensor:
    """Compute energy conservation error between predicted and true states.
    
    Computes the relative error:
        error = |E_pred - E_true| / E_true
    
    Or absolute error if relative=False:
        error = |E_pred - E_true|
    
    Args:
        pred_momentum: Predicted particle momenta (N, 3)
        true_momentum: True particle momenta (N, 3)
        mass: Particle masses (N,)
        pred_potential: Optional predicted potential energy per particle (N,)
        true_potential: Optional true potential energy per particle (N,)
        relative: If True, return relative error; if False, return absolute error
    
    Returns:
        Energy conservation error (scalar)
    
    Requirements:
        - Validates: Requirement 6.1 (energy conservation)
    """
    # Compute total energies
    E_pred = compute_total_energy(pred_momentum, mass, pred_potential)
    E_true = compute_total_energy(true_momentum, mass, true_potential)
    
    # Compute absolute error
    abs_error = torch.abs(E_pred - E_true)
    
    if relative:
        # Compute relative error with small epsilon to avoid division by zero
        rel_error = abs_error / (torch.abs(E_true) + 1e-10)
        return rel_error
    
    return abs_error


def energy_conservation_loss(
    pred_momentum: torch.Tensor,
    true_momentum: torch.Tensor,
    mass: torch.Tensor,
    pred_potential: Optional[torch.Tensor] = None,
    true_potential: Optional[torch.Tensor] = None,
    reduction: str = 'mean'
) -> torch.Tensor:
    """Compute energy conservation loss for training.
    
    This can be used as an auxiliary loss term:
        L_total = L_prediction + λ_energy * L_energy_conservation
    
    Args:
        pred_momentum: Predicted particle momenta (batch_size, N, 3) or (N, 3)
        true_momentum: True particle momenta (batch_size, N, 3) or (N, 3)
        mass: Particle masses (batch_size, N) or (N,)
        pred_potential: Optional predicted potential energy (batch_size, N) or (N,)
        true_potential: Optional true potential energy (batch_size, N) or (N,)
        reduction: How to reduce the loss ('mean', 'sum', or 'none')
    
    Returns:
        Energy conservation loss
    
    Requirements:
        - Validates: Requirement 6.1
    """
    # Handle batched inputs
    if pred_momentum.dim() == 3:
        # Batched: (batch_size, N, 3)
        batch_size = pred_momentum.shape[0]
        errors = []
        
        for i in range(batch_size):
            pred_mom_i = pred_momentum[i]
            true_mom_i = true_momentum[i]
            mass_i = mass[i] if mass.dim() == 2 else mass
            
            pred_pot_i = pred_potential[i] if pred_potential is not None else None
            true_pot_i = true_potential[i] if true_potential is not None else None
            
            error = energy_conservation_error(
                pred_mom_i, true_mom_i, mass_i,
                pred_pot_i, true_pot_i,
                relative=True
            )
            errors.append(error)
        
        errors = torch.stack(errors)
        
        if reduction == 'mean':
            return errors.mean()
        elif reduction == 'sum':
            return errors.sum()
        else:
            return errors
    
    else:
        # Single sample: (N, 3)
        error = energy_conservation_error(
            pred_momentum, true_momentum, mass,
            pred_potential, true_potential,
            relative=True
        )
        
        if reduction == 'none':
            return error.unsqueeze(0)
        
        return error


class EnergyConservationLoss(nn.Module):
    """Energy conservation loss module for training.
    
    This can be added to the training pipeline as a physics constraint:
        L_total = L_prediction + λ_E * L_energy_conservation
    
    Requirements:
        - Validates: Requirement 6.1
    
    Example:
        >>> energy_loss = EnergyConservationLoss(weight=0.1)
        >>> loss = prediction_loss + energy_loss(pred_mom, true_mom, mass)
    """
    
    def __init__(self, weight: float = 1.0, reduction: str = 'mean'):
        """Initialize energy conservation loss.
        
        Args:
            weight: Weight for the loss term (default: 1.0)
            reduction: How to reduce the loss ('mean', 'sum', or 'none')
        """
        super().__init__()
        self.weight = weight
        self.reduction = reduction
    
    def forward(
        self,
        pred_momentum: torch.Tensor,
        true_momentum: torch.Tensor,
        mass: torch.Tensor,
        pred_potential: Optional[torch.Tensor] = None,
        true_potential: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute weighted energy conservation loss.
        
        Args:
            pred_momentum: Predicted particle momenta
            true_momentum: True particle momenta
            mass: Particle masses
            pred_potential: Optional predicted potential energy
            true_potential: Optional true potential energy
        
        Returns:
            Weighted energy conservation loss
        """
        loss = energy_conservation_loss(
            pred_momentum, true_momentum, mass,
            pred_potential, true_potential,
            reduction=self.reduction
        )
        return self.weight * loss


def compute_total_momentum(momentum: torch.Tensor) -> torch.Tensor:
    """Compute total momentum for particles.
    
    P_total = Σ p_i
    
    Args:
        momentum: Particle momenta (N, 3) in atomic units
    
    Returns:
        Total momentum vector (3,)
    """
    # Sum over all particles
    total_momentum = torch.sum(momentum, dim=0)  # (3,)
    
    return total_momentum


def momentum_conservation_error(
    pred_momentum: torch.Tensor,
    true_momentum: torch.Tensor,
    relative: bool = True
) -> torch.Tensor:
    """Compute momentum conservation error between predicted and true states.
    
    Computes the relative error:
        error = ||P_pred - P_true|| / ||P_true||
    
    Or absolute error if relative=False:
        error = ||P_pred - P_true||
    
    Args:
        pred_momentum: Predicted particle momenta (N, 3)
        true_momentum: True particle momenta (N, 3)
        relative: If True, return relative error; if False, return absolute error
    
    Returns:
        Momentum conservation error (scalar)
    
    Requirements:
        - Validates: Requirement 6.2 (momentum conservation)
    """
    # Compute total momenta
    P_pred = compute_total_momentum(pred_momentum)
    P_true = compute_total_momentum(true_momentum)
    
    # Compute absolute error (L2 norm of difference)
    abs_error = torch.norm(P_pred - P_true)
    
    if relative:
        # Compute relative error with small epsilon to avoid division by zero
        P_true_norm = torch.norm(P_true)
        rel_error = abs_error / (P_true_norm + 1e-10)
        return rel_error
    
    return abs_error


def momentum_conservation_loss(
    pred_momentum: torch.Tensor,
    true_momentum: torch.Tensor,
    reduction: str = 'mean'
) -> torch.Tensor:
    """Compute momentum conservation loss for training.
    
    This can be used as an auxiliary loss term:
        L_total = L_prediction + λ_momentum * L_momentum_conservation
    
    Args:
        pred_momentum: Predicted particle momenta (batch_size, N, 3) or (N, 3)
        true_momentum: True particle momenta (batch_size, N, 3) or (N, 3)
        reduction: How to reduce the loss ('mean', 'sum', or 'none')
    
    Returns:
        Momentum conservation loss
    
    Requirements:
        - Validates: Requirement 6.2
    """
    # Handle batched inputs
    if pred_momentum.dim() == 3:
        # Batched: (batch_size, N, 3)
        batch_size = pred_momentum.shape[0]
        errors = []
        
        for i in range(batch_size):
            pred_mom_i = pred_momentum[i]
            true_mom_i = true_momentum[i]
            
            error = momentum_conservation_error(
                pred_mom_i, true_mom_i,
                relative=True
            )
            errors.append(error)
        
        errors = torch.stack(errors)
        
        if reduction == 'mean':
            return errors.mean()
        elif reduction == 'sum':
            return errors.sum()
        else:
            return errors
    
    else:
        # Single sample: (N, 3)
        error = momentum_conservation_error(
            pred_momentum, true_momentum,
            relative=True
        )
        
        if reduction == 'none':
            return error.unsqueeze(0)
        
        return error


class MomentumConservationLoss(nn.Module):
    """Momentum conservation loss module for training.
    
    This can be added to the training pipeline as a physics constraint:
        L_total = L_prediction + λ_P * L_momentum_conservation
    
    Requirements:
        - Validates: Requirement 6.2
    
    Example:
        >>> momentum_loss = MomentumConservationLoss(weight=0.1)
        >>> loss = prediction_loss + momentum_loss(pred_mom, true_mom)
    """
    
    def __init__(self, weight: float = 1.0, reduction: str = 'mean'):
        """Initialize momentum conservation loss.
        
        Args:
            weight: Weight for the loss term (default: 1.0)
            reduction: How to reduce the loss ('mean', 'sum', or 'none')
        """
        super().__init__()
        self.weight = weight
        self.reduction = reduction
    
    def forward(
        self,
        pred_momentum: torch.Tensor,
        true_momentum: torch.Tensor
    ) -> torch.Tensor:
        """Compute weighted momentum conservation loss.
        
        Args:
            pred_momentum: Predicted particle momenta
            true_momentum: True particle momenta
        
        Returns:
            Weighted momentum conservation loss
        """
        loss = momentum_conservation_loss(
            pred_momentum, true_momentum,
            reduction=self.reduction
        )
        return self.weight * loss


def compute_total_charge(charge: torch.Tensor) -> torch.Tensor:
    """Compute total charge for particles.
    
    Q_total = Σ q_i
    
    Args:
        charge: Particle charges (N,) in elementary charge units
    
    Returns:
        Total charge (scalar)
    """
    # Sum over all particles
    total_charge = torch.sum(charge)
    
    return total_charge


def charge_conservation_error(
    pred_charge: torch.Tensor,
    true_charge: torch.Tensor,
    relative: bool = True
) -> torch.Tensor:
    """Compute charge conservation error between predicted and true states.
    
    Computes the relative error:
        error = |Q_pred - Q_true| / |Q_true|
    
    Or absolute error if relative=False:
        error = |Q_pred - Q_true|
    
    Args:
        pred_charge: Predicted particle charges (N,)
        true_charge: True particle charges (N,)
        relative: If True, return relative error; if False, return absolute error
    
    Returns:
        Charge conservation error (scalar)
    
    Requirements:
        - Validates: Requirement 6.3 (charge conservation)
    """
    # Compute total charges
    Q_pred = compute_total_charge(pred_charge)
    Q_true = compute_total_charge(true_charge)
    
    # Compute absolute error
    abs_error = torch.abs(Q_pred - Q_true)
    
    if relative:
        # Compute relative error with small epsilon to avoid division by zero
        rel_error = abs_error / (torch.abs(Q_true) + 1e-10)
        return rel_error
    
    return abs_error


def charge_conservation_loss(
    pred_charge: torch.Tensor,
    true_charge: torch.Tensor,
    reduction: str = 'mean'
) -> torch.Tensor:
    """Compute charge conservation loss for training.
    
    This can be used as an auxiliary loss term:
        L_total = L_prediction + λ_charge * L_charge_conservation
    
    Args:
        pred_charge: Predicted particle charges (batch_size, N) or (N,)
        true_charge: True particle charges (batch_size, N) or (N,)
        reduction: How to reduce the loss ('mean', 'sum', or 'none')
    
    Returns:
        Charge conservation loss
    
    Requirements:
        - Validates: Requirement 6.3
    """
    # Handle batched inputs
    if pred_charge.dim() == 2:
        # Batched: (batch_size, N)
        batch_size = pred_charge.shape[0]
        errors = []
        
        for i in range(batch_size):
            pred_charge_i = pred_charge[i]
            true_charge_i = true_charge[i]
            
            error = charge_conservation_error(
                pred_charge_i, true_charge_i,
                relative=True
            )
            errors.append(error)
        
        errors = torch.stack(errors)
        
        if reduction == 'mean':
            return errors.mean()
        elif reduction == 'sum':
            return errors.sum()
        else:
            return errors
    
    else:
        # Single sample: (N,)
        error = charge_conservation_error(
            pred_charge, true_charge,
            relative=True
        )
        
        if reduction == 'none':
            return error.unsqueeze(0)
        
        return error


class ChargeConservationLoss(nn.Module):
    """Charge conservation loss module for training.
    
    This can be added to the training pipeline as a physics constraint:
        L_total = L_prediction + λ_Q * L_charge_conservation
    
    Requirements:
        - Validates: Requirement 6.3
    
    Example:
        >>> charge_loss = ChargeConservationLoss(weight=0.1)
        >>> loss = prediction_loss + charge_loss(pred_charge, true_charge)
    """
    
    def __init__(self, weight: float = 1.0, reduction: str = 'mean'):
        """Initialize charge conservation loss.
        
        Args:
            weight: Weight for the loss term (default: 1.0)
            reduction: How to reduce the loss ('mean', 'sum', or 'none')
        """
        super().__init__()
        self.weight = weight
        self.reduction = reduction
    
    def forward(
        self,
        pred_charge: torch.Tensor,
        true_charge: torch.Tensor
    ) -> torch.Tensor:
        """Compute weighted charge conservation loss.
        
        Args:
            pred_charge: Predicted particle charges
            true_charge: True particle charges
        
        Returns:
            Weighted charge conservation loss
        """
        loss = charge_conservation_loss(
            pred_charge, true_charge,
            reduction=self.reduction
        )
        return self.weight * loss


def combined_conservation_loss(
    pred_momentum: torch.Tensor,
    true_momentum: torch.Tensor,
    mass: torch.Tensor,
    pred_charge: torch.Tensor,
    true_charge: torch.Tensor,
    pred_potential: Optional[torch.Tensor] = None,
    true_potential: Optional[torch.Tensor] = None,
    lambda_energy: float = 1.0,
    lambda_momentum: float = 1.0,
    lambda_charge: float = 1.0,
    reduction: str = 'mean'
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute combined conservation loss with configurable weights.
    
    L_conservation = λ_E * L_energy + λ_P * L_momentum + λ_Q * L_charge
    
    Args:
        pred_momentum: Predicted particle momenta (batch_size, N, 3) or (N, 3)
        true_momentum: True particle momenta (batch_size, N, 3) or (N, 3)
        mass: Particle masses (batch_size, N) or (N,)
        pred_charge: Predicted particle charges (batch_size, N) or (N,)
        true_charge: True particle charges (batch_size, N) or (N,)
        pred_potential: Optional predicted potential energy (batch_size, N) or (N,)
        true_potential: Optional true potential energy (batch_size, N) or (N,)
        lambda_energy: Weight for energy conservation loss (default: 1.0)
        lambda_momentum: Weight for momentum conservation loss (default: 1.0)
        lambda_charge: Weight for charge conservation loss (default: 1.0)
        reduction: How to reduce the loss ('mean', 'sum', or 'none')
    
    Returns:
        total_loss: Combined conservation loss
        loss_dict: Dictionary with individual loss components
    
    Requirements:
        - Validates: Requirement 6.4 (combined conservation loss)
    """
    # Compute individual conservation losses
    L_energy = energy_conservation_loss(
        pred_momentum, true_momentum, mass,
        pred_potential, true_potential,
        reduction=reduction
    )
    
    L_momentum = momentum_conservation_loss(
        pred_momentum, true_momentum,
        reduction=reduction
    )
    
    L_charge = charge_conservation_loss(
        pred_charge, true_charge,
        reduction=reduction
    )
    
    # Combine with weights
    total_loss = (
        lambda_energy * L_energy +
        lambda_momentum * L_momentum +
        lambda_charge * L_charge
    )
    
    # Return total loss and individual components
    loss_dict = {
        'energy_loss': L_energy,
        'momentum_loss': L_momentum,
        'charge_loss': L_charge,
        'total_conservation_loss': total_loss
    }
    
    return total_loss, loss_dict


class CombinedConservationLoss(nn.Module):
    """Combined conservation loss module for training.
    
    Combines energy, momentum, and charge conservation losses with
    configurable weights:
        L_conservation = λ_E * L_energy + λ_P * L_momentum + λ_Q * L_charge
    
    This can be added to the training pipeline as a physics constraint:
        L_total = L_prediction + L_conservation
    
    Requirements:
        - Validates: Requirement 6.4
    
    Example:
        >>> conservation_loss = CombinedConservationLoss(
        ...     lambda_energy=1.0,
        ...     lambda_momentum=1.0,
        ...     lambda_charge=1.0
        ... )
        >>> loss, loss_dict = conservation_loss(
        ...     pred_mom, true_mom, mass, pred_charge, true_charge
        ... )
        >>> total_loss = prediction_loss + loss
    """
    
    def __init__(
        self,
        lambda_energy: float = 1.0,
        lambda_momentum: float = 1.0,
        lambda_charge: float = 1.0,
        reduction: str = 'mean'
    ):
        """Initialize combined conservation loss.
        
        Args:
            lambda_energy: Weight for energy conservation loss (default: 1.0)
            lambda_momentum: Weight for momentum conservation loss (default: 1.0)
            lambda_charge: Weight for charge conservation loss (default: 1.0)
            reduction: How to reduce the loss ('mean', 'sum', or 'none')
        """
        super().__init__()
        self.lambda_energy = lambda_energy
        self.lambda_momentum = lambda_momentum
        self.lambda_charge = lambda_charge
        self.reduction = reduction
    
    def forward(
        self,
        pred_momentum: torch.Tensor,
        true_momentum: torch.Tensor,
        mass: torch.Tensor,
        pred_charge: torch.Tensor,
        true_charge: torch.Tensor,
        pred_potential: Optional[torch.Tensor] = None,
        true_potential: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute combined conservation loss.
        
        Args:
            pred_momentum: Predicted particle momenta
            true_momentum: True particle momenta
            mass: Particle masses
            pred_charge: Predicted particle charges
            true_charge: True particle charges
            pred_potential: Optional predicted potential energy
            true_potential: Optional true potential energy
        
        Returns:
            total_loss: Combined conservation loss
            loss_dict: Dictionary with individual loss components
        """
        return combined_conservation_loss(
            pred_momentum, true_momentum, mass,
            pred_charge, true_charge,
            pred_potential, true_potential,
            lambda_energy=self.lambda_energy,
            lambda_momentum=self.lambda_momentum,
            lambda_charge=self.lambda_charge,
            reduction=self.reduction
        )
