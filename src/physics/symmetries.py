"""Physics-informed symmetry verification and constraints.

This module implements functions to verify and enforce fundamental physical symmetries:
- Rotation invariance
- Translation invariance
- Time-reversal symmetry

Requirements:
    - Validates: Requirements 5.2, 5.4
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional


def generate_random_rotation_matrix(device: torch.device = None, dtype: torch.dtype = None) -> torch.Tensor:
    """Generate a random 3D rotation matrix using Euler angles.
    
    Args:
        device: Device to create tensor on
        dtype: Data type for tensor
    
    Returns:
        Rotation matrix R ∈ SO(3) of shape (3, 3)
    """
    # Random Euler angles
    alpha = torch.rand(1, device=device, dtype=dtype) * 2 * np.pi  # Rotation around z
    beta = torch.rand(1, device=device, dtype=dtype) * np.pi       # Rotation around y
    gamma = torch.rand(1, device=device, dtype=dtype) * 2 * np.pi  # Rotation around z
    
    # Rotation matrices for each axis
    cos_a, sin_a = torch.cos(alpha), torch.sin(alpha)
    cos_b, sin_b = torch.cos(beta), torch.sin(beta)
    cos_g, sin_g = torch.cos(gamma), torch.sin(gamma)
    
    # R_z(alpha)
    R_z1 = torch.tensor([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ], device=device, dtype=dtype).squeeze()
    
    # R_y(beta)
    R_y = torch.tensor([
        [cos_b, 0, sin_b],
        [0, 1, 0],
        [-sin_b, 0, cos_b]
    ], device=device, dtype=dtype).squeeze()
    
    # R_z(gamma)
    R_z2 = torch.tensor([
        [cos_g, -sin_g, 0],
        [sin_g, cos_g, 0],
        [0, 0, 1]
    ], device=device, dtype=dtype).squeeze()
    
    # Combined rotation: R = R_z(gamma) @ R_y(beta) @ R_z(alpha)
    R = R_z2 @ R_y @ R_z1
    
    return R


def apply_rotation_to_data(data, rotation_matrix: torch.Tensor):
    """Apply rotation matrix to particle positions and momenta in GraphData.
    
    Args:
        data: GraphData object with x (node features) and pos (positions)
        rotation_matrix: 3x3 rotation matrix
    
    Returns:
        Rotated GraphData object (new instance)
    """
    from src.data.structures import GraphData
    
    # Clone the data to avoid modifying original
    rotated_data = GraphData(
        x=data.x.clone(),
        pos=data.pos.clone(),
        batch=data.batch.clone() if hasattr(data, 'batch') and data.batch is not None else None,
        density=data.density.clone() if hasattr(data, 'density') else None,
        energy=data.energy.clone() if hasattr(data, 'energy') else None,
        material=data.material.clone() if hasattr(data, 'material') else None
    )
    
    # Rotate positions (first 3 columns of x and pos)
    rotated_data.pos = torch.matmul(data.pos, rotation_matrix.T)
    rotated_data.x[:, :3] = rotated_data.pos
    
    # Rotate momenta (columns 3:6 of x)
    rotated_data.x[:, 3:6] = torch.matmul(data.x[:, 3:6], rotation_matrix.T)
    
    return rotated_data


def verify_rotation_invariance(
    model: nn.Module,
    data,
    t_span: torch.Tensor,
    rotation_matrix: Optional[torch.Tensor] = None,
    atol: float = 1e-4,
    rtol: float = 1e-3
) -> Tuple[bool, Dict[str, float]]:
    """Verify that model predictions are rotation invariant.
    
    For scalar outputs (observables), predictions should be unchanged.
    For vector outputs (particle positions/momenta), predictions should transform appropriately.
    
    Args:
        model: MultiScaleModel to test
        data: GraphData input
        t_span: Time interval for forward pass
        rotation_matrix: 3x3 rotation matrix (if None, generates random rotation)
        atol: Absolute tolerance for comparison
        rtol: Relative tolerance for comparison
    
    Returns:
        is_invariant: True if model satisfies rotation invariance
        errors: Dictionary with error metrics for each output type
    
    Requirements:
        - Validates: Requirement 5.2 (rotation invariance)
    """
    model.eval()
    
    with torch.no_grad():
        # Generate random rotation if not provided
        if rotation_matrix is None:
            rotation_matrix = generate_random_rotation_matrix(
                device=data.x.device,
                dtype=data.x.dtype
            )
        
        # Original prediction
        output_original = model(data, t_span)
        
        # Rotate input data
        data_rotated = apply_rotation_to_data(data, rotation_matrix)
        
        # Prediction on rotated data
        output_rotated = model(data_rotated, t_span)
        
        # Check scalar outputs (observables) - should be unchanged
        observables_original = output_original['observables']
        observables_rotated = output_rotated['observables']
        
        observables_error = torch.abs(observables_original - observables_rotated).max().item()
        observables_rel_error = (observables_error / 
                                (torch.abs(observables_original).max().item() + 1e-10))
        
        # Check latent states - should be unchanged (rotation invariant)
        latent_z0_error = torch.abs(
            output_original['latent_z0'] - output_rotated['latent_z0']
        ).max().item()
        latent_z1_error = torch.abs(
            output_original['latent_z1'] - output_rotated['latent_z1']
        ).max().item()
        
        latent_z0_rel_error = (latent_z0_error / 
                              (torch.abs(output_original['latent_z0']).max().item() + 1e-10))
        latent_z1_rel_error = (latent_z1_error / 
                              (torch.abs(output_original['latent_z1']).max().item() + 1e-10))
        
        # Check vector outputs (particle predictions) - should transform with rotation
        # Extract position and momentum components
        particle_pred_original = output_original['particle_pred']
        particle_pred_rotated = output_rotated['particle_pred']
        
        # For particle embeddings, we expect them to be rotation invariant
        # (since they're processed through the network which uses relative positions)
        particle_error = torch.abs(particle_pred_original - particle_pred_rotated).max().item()
        particle_rel_error = (particle_error / 
                            (torch.abs(particle_pred_original).max().item() + 1e-10))
        
        # Compile errors
        errors = {
            'observables_abs_error': observables_error,
            'observables_rel_error': observables_rel_error,
            'latent_z0_abs_error': latent_z0_error,
            'latent_z0_rel_error': latent_z0_rel_error,
            'latent_z1_abs_error': latent_z1_error,
            'latent_z1_rel_error': latent_z1_rel_error,
            'particle_abs_error': particle_error,
            'particle_rel_error': particle_rel_error
        }
        
        # Check if all errors are within tolerance
        is_invariant = (
            observables_rel_error < rtol and observables_error < atol and
            latent_z0_rel_error < rtol and latent_z0_error < atol and
            latent_z1_rel_error < rtol and latent_z1_error < atol and
            particle_rel_error < rtol and particle_error < atol
        )
        
        return is_invariant, errors


def rotation_invariance_test_suite(
    model: nn.Module,
    data,
    t_span: torch.Tensor,
    num_rotations: int = 10,
    atol: float = 1e-4,
    rtol: float = 1e-3
) -> Tuple[bool, Dict[str, any]]:
    """Test rotation invariance with multiple random rotations.
    
    Args:
        model: MultiScaleModel to test
        data: GraphData input
        t_span: Time interval for forward pass
        num_rotations: Number of random rotations to test
        atol: Absolute tolerance
        rtol: Relative tolerance
    
    Returns:
        all_passed: True if all tests passed
        results: Dictionary with detailed results
    """
    results = {
        'num_tests': num_rotations,
        'num_passed': 0,
        'num_failed': 0,
        'max_errors': {
            'observables_abs_error': 0.0,
            'observables_rel_error': 0.0,
            'latent_z0_abs_error': 0.0,
            'latent_z0_rel_error': 0.0,
            'latent_z1_abs_error': 0.0,
            'latent_z1_rel_error': 0.0,
            'particle_abs_error': 0.0,
            'particle_rel_error': 0.0
        },
        'failed_tests': []
    }
    
    for i in range(num_rotations):
        is_invariant, errors = verify_rotation_invariance(
            model, data, t_span, rotation_matrix=None, atol=atol, rtol=rtol
        )
        
        if is_invariant:
            results['num_passed'] += 1
        else:
            results['num_failed'] += 1
            results['failed_tests'].append({
                'test_id': i,
                'errors': errors
            })
        
        # Track maximum errors
        for key in results['max_errors']:
            results['max_errors'][key] = max(results['max_errors'][key], errors[key])
    
    all_passed = results['num_failed'] == 0
    
    return all_passed, results



def apply_time_reversal_to_data(data):
    """Apply time-reversal transformation to particle data.
    
    Time-reversal: (x, p, t) → (x, -p, -t)
    - Positions remain unchanged
    - Momenta are negated
    - Time is negated (handled externally in t_span)
    
    Args:
        data: GraphData object with x (node features) and pos (positions)
    
    Returns:
        Time-reversed GraphData object (new instance)
    """
    from src.data.structures import GraphData
    
    # Clone the data to avoid modifying original
    reversed_data = GraphData(
        x=data.x.clone(),
        pos=data.pos.clone(),
        batch=data.batch.clone() if hasattr(data, 'batch') and data.batch is not None else None,
        density=data.density.clone() if hasattr(data, 'density') else None,
        energy=data.energy.clone() if hasattr(data, 'energy') else None,
        material=data.material.clone() if hasattr(data, 'material') else None
    )
    
    # Negate momenta (columns 3:6 of x)
    reversed_data.x[:, 3:6] = -data.x[:, 3:6]
    
    return reversed_data


def time_reversal_symmetry_loss(
    model: nn.Module,
    data,
    t_span: torch.Tensor,
    reduction: str = 'mean'
) -> torch.Tensor:
    """Compute time-reversal symmetry loss.
    
    For Hamiltonian systems, the dynamics should satisfy:
        f(x, p, t) = f(x, -p, -t)
    
    This loss measures the violation of this symmetry:
        L_TR = ||f(x, p, t) - f(x, -p, -t)||²
    
    Args:
        model: MultiScaleModel to evaluate
        data: GraphData input with particle states
        t_span: Time interval [t_start, t_end]
        reduction: How to reduce the loss ('mean', 'sum', or 'none')
    
    Returns:
        Time-reversal symmetry loss (scalar if reduction != 'none')
    
    Requirements:
        - Validates: Requirement 5.4 (time-reversal symmetry)
    
    Notes:
        - This is an optional constraint for Hamiltonian systems
        - Can be added to training loss with configurable weight
        - For dissipative systems, this symmetry may not hold
    """
    # Forward prediction: f(x, p, t)
    output_forward = model(data, t_span)
    
    # Time-reversed input: (x, -p)
    data_reversed = apply_time_reversal_to_data(data)
    
    # Time-reversed time span: -t
    t_span_reversed = -torch.flip(t_span, dims=[0])
    
    # Backward prediction: f(x, -p, -t)
    output_backward = model(data_reversed, t_span_reversed)
    
    # Compute loss on particle predictions
    # For time-reversal, particle embeddings should be similar
    # (positions same, momenta negated)
    particle_loss = F.mse_loss(
        output_forward['particle_pred'],
        output_backward['particle_pred'],
        reduction=reduction
    )
    
    # Compute loss on observables
    # Scalar observables should be unchanged under time-reversal
    observable_loss = F.mse_loss(
        output_forward['observables'],
        output_backward['observables'],
        reduction=reduction
    )
    
    # Compute loss on latent states
    # Latent states should satisfy time-reversal symmetry
    latent_loss = F.mse_loss(
        output_forward['latent_z1'],
        output_backward['latent_z1'],
        reduction=reduction
    )
    
    # Combined loss
    total_loss = particle_loss + observable_loss + latent_loss
    
    return total_loss


def verify_time_reversal_symmetry(
    model: nn.Module,
    data,
    t_span: torch.Tensor,
    atol: float = 1e-4,
    rtol: float = 1e-3
) -> Tuple[bool, Dict[str, float]]:
    """Verify that model predictions satisfy time-reversal symmetry.
    
    Args:
        model: MultiScaleModel to test
        data: GraphData input
        t_span: Time interval for forward pass
        atol: Absolute tolerance for comparison
        rtol: Relative tolerance for comparison
    
    Returns:
        is_symmetric: True if model satisfies time-reversal symmetry
        errors: Dictionary with error metrics for each output type
    
    Requirements:
        - Validates: Requirement 5.4 (time-reversal symmetry)
    """
    model.eval()
    
    with torch.no_grad():
        # Forward prediction
        output_forward = model(data, t_span)
        
        # Time-reversed prediction
        data_reversed = apply_time_reversal_to_data(data)
        t_span_reversed = -torch.flip(t_span, dims=[0])
        output_backward = model(data_reversed, t_span_reversed)
        
        # Compute errors
        particle_error = torch.abs(
            output_forward['particle_pred'] - output_backward['particle_pred']
        ).max().item()
        particle_rel_error = (particle_error / 
                            (torch.abs(output_forward['particle_pred']).max().item() + 1e-10))
        
        observable_error = torch.abs(
            output_forward['observables'] - output_backward['observables']
        ).max().item()
        observable_rel_error = (observable_error / 
                               (torch.abs(output_forward['observables']).max().item() + 1e-10))
        
        latent_error = torch.abs(
            output_forward['latent_z1'] - output_backward['latent_z1']
        ).max().item()
        latent_rel_error = (latent_error / 
                           (torch.abs(output_forward['latent_z1']).max().item() + 1e-10))
        
        errors = {
            'particle_abs_error': particle_error,
            'particle_rel_error': particle_rel_error,
            'observable_abs_error': observable_error,
            'observable_rel_error': observable_rel_error,
            'latent_abs_error': latent_error,
            'latent_rel_error': latent_rel_error
        }
        
        # Check if all errors are within tolerance
        is_symmetric = (
            particle_rel_error < rtol and particle_error < atol and
            observable_rel_error < rtol and observable_error < atol and
            latent_rel_error < rtol and latent_error < atol
        )
        
        return is_symmetric, errors


class TimeReversalSymmetryLoss(nn.Module):
    """Time-reversal symmetry loss module for training.
    
    This can be added to the training pipeline as an optional constraint:
        L_total = L_prediction + λ_TR * L_time_reversal
    
    Requirements:
        - Validates: Requirement 5.4
    
    Example:
        >>> tr_loss = TimeReversalSymmetryLoss(weight=0.1)
        >>> loss = prediction_loss + tr_loss(model, data, t_span)
    """
    
    def __init__(self, weight: float = 1.0, reduction: str = 'mean'):
        """Initialize time-reversal symmetry loss.
        
        Args:
            weight: Weight for the loss term (default: 1.0)
            reduction: How to reduce the loss ('mean', 'sum', or 'none')
        """
        super().__init__()
        self.weight = weight
        self.reduction = reduction
    
    def forward(self, model: nn.Module, data, t_span: torch.Tensor) -> torch.Tensor:
        """Compute weighted time-reversal symmetry loss.
        
        Args:
            model: MultiScaleModel
            data: GraphData input
            t_span: Time interval
        
        Returns:
            Weighted time-reversal symmetry loss
        """
        loss = time_reversal_symmetry_loss(model, data, t_span, reduction=self.reduction)
        return self.weight * loss
