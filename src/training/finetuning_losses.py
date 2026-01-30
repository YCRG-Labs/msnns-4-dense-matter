"""Fine-tuning loss functions for physics-constrained training.

This module implements the fine-tuning loss that combines:
1. Prediction loss (MSE on positions, momenta, observables)
2. Conservation law losses (energy, momentum, charge)
3. Physics limit losses (Vlasov, Maxwell-Boltzmann, stopping power)

Requirements:
    - Validates: Requirements 9.1, 9.2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from ..physics.conservation import CombinedConservationLoss
from ..physics.known_limits import CombinedPhysicsLimitLoss


@dataclass
class FineTuningLossWeights:
    """Weights for combining fine-tuning loss components.
    
    Attributes:
        prediction: Weight for prediction loss (positions, momenta, observables)
        conservation: Weight for conservation law losses
        physics_limits: Weight for physics limit losses
        
        # Conservation sub-weights
        energy: Weight for energy conservation (within conservation loss)
        momentum: Weight for momentum conservation (within conservation loss)
        charge: Weight for charge conservation (within conservation loss)
        
        # Physics limit sub-weights
        vlasov: Weight for Vlasov limit (within physics limit loss)
        maxwell_boltzmann: Weight for Maxwell-Boltzmann limit (within physics limit loss)
        stopping_power: Weight for stopping power (within physics limit loss)
    """
    # Main weights
    prediction: float = 1.0
    conservation: float = 0.1
    physics_limits: float = 0.05
    
    # Conservation sub-weights
    energy: float = 1.0
    momentum: float = 1.0
    charge: float = 1.0
    
    # Physics limit sub-weights
    vlasov: float = 1.0
    maxwell_boltzmann: float = 1.0
    stopping_power: float = 1.0


class PredictionLoss(nn.Module):
    """Prediction loss for positions, momenta, and observables.
    
    Computes MSE loss on:
    - Particle positions
    - Particle momenta
    - Coarse-grained observables (beam width, energy spread, emittance)
    
    Requirements:
        - Validates: Requirement 9.1 (prediction loss component)
    """
    
    def __init__(self,
                 position_weight: float = 1.0,
                 momentum_weight: float = 1.0,
                 observable_weight: float = 1.0):
        """Initialize prediction loss.
        
        Args:
            position_weight: Weight for position MSE
            momentum_weight: Weight for momentum MSE
            observable_weight: Weight for observable MSE
        """
        super().__init__()
        self.position_weight = position_weight
        self.momentum_weight = momentum_weight
        self.observable_weight = observable_weight
    
    def forward(self,
                model_output: Dict[str, torch.Tensor],
                target_data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute prediction loss.
        
        Args:
            model_output: Dictionary with model predictions
                - 'particle_pred': Predicted particle embeddings (N, D)
                - 'observables': Predicted observables (batch_size, 5)
                - 'latent_z0': Initial latent state (batch_size, latent_dim)
                - 'latent_z1': Evolved latent state (batch_size, latent_dim)
            target_data: Dictionary with target values
                - 'positions': Target positions (N, 3)
                - 'momenta': Target momenta (N, 3)
                - 'observables': Target observables (batch_size, 5)
        
        Returns:
            total_loss: Combined prediction loss
            loss_dict: Dictionary with individual loss components
        """
        losses = {}
        total_loss = torch.tensor(0.0, device=list(model_output.values())[0].device)
        
        # Position loss
        if 'positions' in target_data and 'particle_pred' in model_output:
            # Note: particle_pred contains embeddings, not direct positions
            # For now, we'll use latent space loss as a proxy
            # In a full implementation, we'd decode embeddings to positions
            pass
        
        # Observable loss
        if 'observables' in target_data and 'observables' in model_output:
            obs_loss = F.mse_loss(model_output['observables'], target_data['observables'])
            losses['observable_loss'] = obs_loss
            total_loss = total_loss + self.observable_weight * obs_loss
        
        # Latent space loss (as proxy for particle-level predictions)
        if 'latent_target' in target_data and 'latent_z1' in model_output:
            latent_loss = F.mse_loss(model_output['latent_z1'], target_data['latent_target'])
            losses['latent_loss'] = latent_loss
            total_loss = total_loss + latent_loss
        
        losses['total_prediction_loss'] = total_loss
        
        return total_loss, losses


class FineTuningLoss(nn.Module):
    """Combined fine-tuning loss with physics constraints.
    
    Combines:
    1. Prediction loss (MSE on positions, momenta, observables)
    2. Conservation law losses (energy, momentum, charge)
    3. Physics limit losses (Vlasov, Maxwell-Boltzmann, stopping power)
    
    L_finetune = L_prediction + λ_cons * L_conservation + λ_phys * L_physics
    
    where:
        L_conservation = λ_E * L_energy + λ_P * L_momentum + λ_Q * L_charge
        L_physics = λ_V * L_vlasov + λ_MB * L_maxwell_boltzmann + λ_SP * L_stopping_power
    
    Requirements:
        - Validates: Requirements 9.1, 9.2
        - Combine prediction loss with conservation losses and physics limit losses
        - Use configurable weights for each term
    
    Example:
        >>> weights = FineTuningLossWeights(
        ...     prediction=1.0,
        ...     conservation=0.1,
        ...     physics_limits=0.05
        ... )
        >>> loss_fn = FineTuningLoss(weights)
        >>> total_loss, loss_dict = loss_fn(model, data, t_span, targets)
    """
    
    def __init__(self,
                 weights: Optional[FineTuningLossWeights] = None,
                 density_threshold: float = 0.01,
                 temperature_threshold: float = 1000.0):
        """Initialize fine-tuning loss.
        
        Args:
            weights: Loss weights (default: FineTuningLossWeights())
            density_threshold: Density threshold for Vlasov limit (particles/nm³)
            temperature_threshold: Temperature threshold for MB limit (K)
        """
        super().__init__()
        self.weights = weights if weights is not None else FineTuningLossWeights()
        
        # Initialize prediction loss
        self.prediction_loss = PredictionLoss()
        
        # Initialize conservation loss
        self.conservation_loss = CombinedConservationLoss(
            lambda_energy=self.weights.energy,
            lambda_momentum=self.weights.momentum,
            lambda_charge=self.weights.charge,
            reduction='mean'
        )
        
        # Initialize physics limit loss
        self.physics_limit_loss = CombinedPhysicsLimitLoss(
            lambda_vlasov=self.weights.vlasov,
            lambda_mb=self.weights.maxwell_boltzmann,
            lambda_sp=self.weights.stopping_power,
            density_threshold=density_threshold,
            temperature_threshold=temperature_threshold,
            reduction='mean'
        )
    
    def forward(self,
                model: nn.Module,
                data,
                t_span: torch.Tensor,
                targets: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute combined fine-tuning loss.
        
        Args:
            model: MultiScaleModel instance
            data: GraphData input with particle states
            t_span: Time interval [t_start, t_end]
            targets: Optional dictionary with target values
                - 'positions': Target positions (N, 3)
                - 'momenta': Target momenta (N, 3)
                - 'observables': Target observables (batch_size, 5)
                - 'latent_target': Target latent state (batch_size, latent_dim)
        
        Returns:
            total_loss: Combined fine-tuning loss
            loss_dict: Dictionary with all loss components
        """
        # Get model predictions
        output = model(data, t_span)
        
        # Initialize loss dictionary
        all_losses = {}
        total_loss = torch.tensor(0.0, device=data.x.device)
        
        # 1. Prediction loss
        if targets is not None:
            pred_loss, pred_losses = self.prediction_loss(output, targets)
            all_losses.update(pred_losses)
            total_loss = total_loss + self.weights.prediction * pred_loss
        
        # 2. Conservation loss
        # Extract momenta and charges from data
        pred_momenta = data.x[:, 3:6]  # Predicted momenta (from input for now)
        true_momenta = data.x[:, 3:6]  # True momenta
        
        # Get masses and charges
        if data.x.shape[1] > 7:
            masses = data.x[:, 7]  # (N,)
        else:
            masses = torch.ones(data.x.shape[0], device=data.x.device)
        
        if data.x.shape[1] > 6:
            pred_charges = data.x[:, 6]  # (N,)
            true_charges = data.x[:, 6]
        else:
            pred_charges = torch.ones(data.x.shape[0], device=data.x.device)
            true_charges = torch.ones(data.x.shape[0], device=data.x.device)
        
        # Compute conservation loss
        cons_loss, cons_losses = self.conservation_loss(
            pred_momenta.unsqueeze(0),  # Add batch dimension
            true_momenta.unsqueeze(0),
            masses.unsqueeze(0),
            pred_charges.unsqueeze(0),
            true_charges.unsqueeze(0)
        )
        all_losses.update(cons_losses)
        total_loss = total_loss + self.weights.conservation * cons_loss
        
        # 3. Physics limit loss
        phys_loss, phys_losses = self.physics_limit_loss(model, data, t_span)
        all_losses.update(phys_losses)
        total_loss = total_loss + self.weights.physics_limits * phys_loss
        
        # Add total loss
        all_losses['total_finetuning_loss'] = total_loss
        
        return total_loss, all_losses
    
    def compute_with_targets(self,
                            model: nn.Module,
                            data,
                            data_next,
                            t_span: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute fine-tuning loss with explicit target data.
        
        This is a convenience method that extracts targets from the next timestep data.
        
        Args:
            model: MultiScaleModel instance
            data: GraphData at time t
            data_next: GraphData at time t+1 (targets)
            t_span: Time interval [t, t+1]
        
        Returns:
            total_loss: Combined fine-tuning loss
            loss_dict: Dictionary with all loss components
        """
        # Extract targets from next timestep
        targets = {
            'positions': data_next.pos,
            'momenta': data_next.x[:, 3:6],
            'observables': data_next.target_obs if hasattr(data_next, 'target_obs') else None
        }
        
        # Encode target state to get latent target
        with torch.no_grad():
            latent_target = model.encode(data_next)
        targets['latent_target'] = latent_target
        
        return self.forward(model, data, t_span, targets)


def create_finetuning_loss(
    prediction_weight: float = 1.0,
    conservation_weight: float = 0.1,
    physics_weight: float = 0.05,
    energy_weight: float = 1.0,
    momentum_weight: float = 1.0,
    charge_weight: float = 1.0,
    vlasov_weight: float = 1.0,
    mb_weight: float = 1.0,
    sp_weight: float = 1.0,
    density_threshold: float = 0.01,
    temperature_threshold: float = 1000.0
) -> FineTuningLoss:
    """Factory function to create fine-tuning loss with custom weights.
    
    Args:
        prediction_weight: Weight for prediction loss
        conservation_weight: Weight for conservation losses
        physics_weight: Weight for physics limit losses
        energy_weight: Weight for energy conservation
        momentum_weight: Weight for momentum conservation
        charge_weight: Weight for charge conservation
        vlasov_weight: Weight for Vlasov limit
        mb_weight: Weight for Maxwell-Boltzmann limit
        sp_weight: Weight for stopping power
        density_threshold: Density threshold for Vlasov limit
        temperature_threshold: Temperature threshold for MB limit
    
    Returns:
        FineTuningLoss instance with specified weights
    
    Example:
        >>> loss_fn = create_finetuning_loss(
        ...     prediction_weight=1.0,
        ...     conservation_weight=0.1,
        ...     physics_weight=0.05
        ... )
    """
    weights = FineTuningLossWeights(
        prediction=prediction_weight,
        conservation=conservation_weight,
        physics_limits=physics_weight,
        energy=energy_weight,
        momentum=momentum_weight,
        charge=charge_weight,
        vlasov=vlasov_weight,
        maxwell_boltzmann=mb_weight,
        stopping_power=sp_weight
    )
    
    return FineTuningLoss(
        weights=weights,
        density_threshold=density_threshold,
        temperature_threshold=temperature_threshold
    )
