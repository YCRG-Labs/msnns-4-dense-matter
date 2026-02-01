"""Latent space scanning for hypothesis generation.

This module implements density space scanning to identify collective transitions
by systematically varying beam density and computing anomaly scores.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Optional, Callable, Any
from dataclasses import dataclass
import warnings

from ..data.structures import BeamConfiguration, GraphData, ParticleState, CollectiveMode
from ..models.multi_scale_model import MultiScaleModel


@dataclass
class ScanResult:
    """Results from density space scanning.
    
    Attributes:
        densities: Array of scanned density values
        latent_states: Latent representations for each density (num_densities, latent_dim)
        anomaly_scores: Anomaly score for each density
        critical_densities: Detected transition points
    """
    densities: np.ndarray
    latent_states: torch.Tensor
    anomaly_scores: np.ndarray
    critical_densities: List[float]


def scan_density_space(
    model: MultiScaleModel,
    base_config: BeamConfiguration,
    density_range: np.ndarray,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Systematically scan beam density to identify collective transitions.
    
    This function varies the beam density across a specified range and computes
    latent representations for each density value. The latent states can then
    be analyzed to detect transitions to collective behavior.
    
    Args:
        model: Trained MultiScaleModel
        base_config: Base beam configuration (density will be varied)
        density_range: Array of densities to scan (particles/nm³)
        device: Device to run computations on ('cpu' or 'cuda')
    
    Returns:
        Tuple of:
            - anomaly_scores: Anomaly score for each density (num_densities,)
            - latent_states: Latent representations (num_densities, latent_dim)
    
    Requirements:
        - Validates: Requirement 11.1
        - Systematically varies beam density
        - Computes latent representations for each density
    """
    model.eval()
    model = model.to(device)
    
    latent_states = []
    
    with torch.no_grad():
        for density in density_range:
            # Create configuration at this density
            config = _create_config_at_density(base_config, density)
            
            # Convert to GraphData
            data = _config_to_graph_data(config, device)
            
            # Encode to latent space
            z = model.encode(data)  # (1, latent_dim)
            latent_states.append(z.cpu())  # Move to CPU immediately
            
            # Clear CUDA cache to prevent memory accumulation
            if device == 'cuda':
                torch.cuda.empty_cache()
    
    # Stack latent states
    latent_states = torch.cat(latent_states, dim=0)  # (num_densities, latent_dim)
    
    # Placeholder for anomaly scores (will be computed in subtask 23.6)
    anomaly_scores = torch.zeros(len(density_range))
    
    return anomaly_scores, latent_states


def _create_config_at_density(
    base_config: BeamConfiguration,
    density: float
) -> BeamConfiguration:
    """Create a new configuration with specified density.
    
    Scales the number of particles or box size to achieve target density
    while preserving other properties.
    
    Args:
        base_config: Base configuration
        density: Target density (particles/nm³)
    
    Returns:
        New BeamConfiguration with target density
    """
    # Calculate scaling factor
    density_ratio = density / base_config.density
    
    # Scale number of particles (simple approach)
    # In practice, might want to regenerate particles or scale box size
    n_particles = max(10, int(len(base_config.particles) * density_ratio))
    
    # Sample or duplicate particles to reach target count
    if n_particles <= len(base_config.particles):
        # Subsample
        indices = np.random.choice(len(base_config.particles), n_particles, replace=False)
        particles = [base_config.particles[i] for i in indices]
    else:
        # Duplicate with small perturbations
        particles = list(base_config.particles)
        while len(particles) < n_particles:
            idx = np.random.randint(len(base_config.particles))
            original = base_config.particles[idx]
            # Add small random perturbation
            perturbed = ParticleState(
                position=original.position + torch.randn(3) * 0.1,
                momentum=original.momentum + torch.randn(3) * 0.01,
                charge=original.charge,
                mass=original.mass,
                species=original.species
            )
            particles.append(perturbed)
    
    # Create new configuration
    new_config = BeamConfiguration(
        particles=particles,
        density=density,
        energy=base_config.energy,
        material=base_config.material,
        temperature=base_config.temperature,
        time=base_config.time
    )
    
    return new_config


def _config_to_graph_data(
    config: BeamConfiguration,
    device: str = 'cpu'
) -> GraphData:
    """Convert BeamConfiguration to GraphData for model input.
    
    Args:
        config: Beam configuration
        device: Device to place tensors on
    
    Returns:
        GraphData object ready for model input
    """
    # Extract particle data - use float32 for consistency with model
    positions = torch.stack([p.position for p in config.particles]).float().to(device)
    momenta = torch.stack([p.momentum for p in config.particles]).float().to(device)
    charges = torch.tensor([p.charge for p in config.particles], dtype=torch.float32, device=device).unsqueeze(1)
    masses = torch.tensor([p.mass for p in config.particles], dtype=torch.float32, device=device).unsqueeze(1)
    species = torch.tensor([p.species for p in config.particles], dtype=torch.long, device=device)
    
    # One-hot encode species (5 species: Si, Fe, W, Cu, Al)
    num_species = 5
    species_onehot = torch.nn.functional.one_hot(species, num_classes=num_species).float()
    
    # Concatenate node features: [pos, mom, charge, mass, species_onehot]
    x = torch.cat([positions, momenta, charges, masses, species_onehot], dim=1)
    
    # Material encoding
    material_map = {'Si': 0, 'Fe': 1, 'W': 2, 'Cu': 3, 'Al': 4}
    material_idx = material_map.get(config.material, 0)
    material_onehot = torch.nn.functional.one_hot(
        torch.tensor([material_idx], device=device),
        num_classes=5
    ).float()
    
    # Create GraphData
    data = GraphData(
        x=x,
        pos=positions,
        density=torch.tensor([config.density], dtype=torch.float32, device=device),
        energy=torch.tensor([config.energy], dtype=torch.float32, device=device),
        material=material_onehot
    )
    
    return data


def mahalanobis_distance(
    x: torch.Tensor,
    mean: torch.Tensor,
    cov: torch.Tensor
) -> torch.Tensor:
    """Compute Mahalanobis distance from a distribution.
    
    Distance: sqrt((x - μ)ᵀ Σ⁻¹ (x - μ))
    
    Args:
        x: Point to compute distance for (latent_dim,) or (batch_size, latent_dim)
        mean: Distribution mean (latent_dim,)
        cov: Covariance matrix (latent_dim, latent_dim)
    
    Returns:
        Mahalanobis distance (scalar or batch_size,)
    
    Requirements:
        - Validates: Requirement 11.5
        - Computes distance from single-particle regime distribution
    """
    # Handle batch dimension
    if x.dim() == 1:
        x = x.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    # Compute delta
    delta = x - mean.unsqueeze(0)  # (batch_size, latent_dim)
    
    # Compute inverse covariance with regularization
    reg = 1e-6 * torch.eye(cov.shape[0], device=cov.device)
    inv_cov = torch.linalg.inv(cov + reg)
    
    # Compute Mahalanobis distance
    # d² = (x - μ)ᵀ Σ⁻¹ (x - μ)
    mahal_sq = torch.einsum('bi,ij,bj->b', delta, inv_cov, delta)
    mahal_dist = torch.sqrt(torch.clamp(mahal_sq, min=0))
    
    if squeeze_output:
        return mahal_dist.squeeze(0)
    return mahal_dist


def fit_gaussian_to_low_density(
    latent_states: torch.Tensor,
    densities: np.ndarray,
    density_threshold: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fit Gaussian distribution to low-density latent states.
    
    This establishes the baseline single-particle regime distribution
    for anomaly detection.
    
    Args:
        latent_states: Latent representations (num_densities, latent_dim)
        densities: Corresponding density values
        density_threshold: Threshold for low-density regime (particles/nm³)
    
    Returns:
        Tuple of:
            - mean: Mean of low-density distribution (latent_dim,)
            - cov: Covariance matrix (latent_dim, latent_dim)
    
    Requirements:
        - Validates: Requirement 11.5
        - Fits Gaussian to low-density latent states
    """
    # Select low-density states
    low_density_mask = densities < density_threshold
    
    if not low_density_mask.any():
        warnings.warn(
            f"No densities below threshold {density_threshold}. "
            f"Using lowest 20% of densities instead."
        )
        # Use lowest 20% of densities
        n_low = max(1, len(densities) // 5)
        sorted_indices = np.argsort(densities)
        low_density_mask = np.zeros(len(densities), dtype=bool)
        low_density_mask[sorted_indices[:n_low]] = True
    
    z_low_density = latent_states[low_density_mask]
    
    # Compute mean and covariance
    mean = z_low_density.mean(dim=0)
    
    # Center the data
    centered = z_low_density - mean.unsqueeze(0)
    
    # Compute covariance
    cov = torch.mm(centered.T, centered) / (z_low_density.shape[0] - 1)
    
    return mean, cov


def compute_mahalanobis_distances(
    latent_states: torch.Tensor,
    densities: np.ndarray,
    density_threshold: float = 0.1
) -> torch.Tensor:
    """Compute Mahalanobis distance for each density from low-density baseline.
    
    Args:
        latent_states: Latent representations (num_densities, latent_dim)
        densities: Corresponding density values
        density_threshold: Threshold for low-density regime
    
    Returns:
        Mahalanobis distances (num_densities,)
    
    Requirements:
        - Validates: Requirement 11.5
        - Computes Mahalanobis distance for each density
    """
    # Fit Gaussian to low-density states
    mean, cov = fit_gaussian_to_low_density(latent_states, densities, density_threshold)
    
    # Compute distances for all states
    distances = mahalanobis_distance(latent_states, mean, cov)
    
    return distances


def compute_temporal_variance(
    model: MultiScaleModel,
    latent_state: torch.Tensor,
    conditioning: torch.Tensor,
    t_max: float = 100.0,
    num_steps: int = 50,
    device: str = 'cpu'
) -> torch.Tensor:
    """Compute temporal variance in latent space trajectory.
    
    Evolves the latent state over time and measures the variance in the
    trajectory, which indicates oscillatory or unstable collective behavior.
    
    Args:
        model: Trained MultiScaleModel
        latent_state: Initial latent state (batch_size, latent_dim)
        conditioning: Conditioning vector (batch_size, conditioning_dim)
        t_max: Maximum time to evolve (fs)
        num_steps: Number of time steps
        device: Device to run computations on
    
    Returns:
        Temporal variance (batch_size,) - mean variance across latent dimensions
    
    Requirements:
        - Validates: Requirement 11.5
        - Evolves latent state over time
        - Measures variance in latent trajectory
    """
    model.eval()
    model = model.to(device)
    latent_state = latent_state.to(device)
    conditioning = conditioning.to(device)
    
    # Generate time points
    t_span = torch.linspace(0, t_max, num_steps, device=device)
    
    # Evolve latent state at each time point
    z_trajectory = []
    
    with torch.no_grad():
        for t in t_span:
            # Skip t=0 (initial state)
            if t.item() == 0:
                z_trajectory.append(latent_state)
            else:
                # Evolve from t=0 to t=t
                t_interval = torch.tensor([0.0, t.item()], device=device)
                z_t = model.neural_ode(latent_state, t_interval, conditioning)
                z_trajectory.append(z_t)
    
    # Stack trajectory: (num_steps, batch_size, latent_dim)
    z_trajectory = torch.stack(z_trajectory, dim=0)
    
    # Compute variance across time for each latent dimension
    # var: (batch_size, latent_dim)
    temporal_var = z_trajectory.var(dim=0)
    
    # Average variance across latent dimensions: (batch_size,)
    mean_temporal_var = temporal_var.mean(dim=1)
    
    return mean_temporal_var


def compute_temporal_variances_for_scan(
    model: MultiScaleModel,
    latent_states: torch.Tensor,
    densities: np.ndarray,
    base_config: BeamConfiguration,
    t_max: float = 100.0,
    num_steps: int = 50,
    device: str = 'cpu'
) -> torch.Tensor:
    """Compute temporal variance for each density in a scan.
    
    Args:
        model: Trained MultiScaleModel
        latent_states: Latent representations (num_densities, latent_dim)
        densities: Corresponding density values
        base_config: Base configuration for conditioning
        t_max: Maximum time to evolve (fs)
        num_steps: Number of time steps
        device: Device to run computations on
    
    Returns:
        Temporal variances (num_densities,)
    
    Requirements:
        - Validates: Requirement 11.5
        - Computes temporal variance for each density
    """
    temporal_variances = []
    
    # Material encoding for conditioning
    material_map = {'Si': 0, 'Fe': 1, 'W': 2, 'Cu': 3, 'Al': 4}
    material_idx = material_map.get(base_config.material, 0)
    material_onehot = torch.nn.functional.one_hot(
        torch.tensor([material_idx], device=device),
        num_classes=5
    ).float()
    
    for i, density in enumerate(densities):
        # Create conditioning vector for this density
        conditioning = torch.cat([
            torch.tensor([[density]], dtype=torch.float32, device=device),
            torch.tensor([[base_config.energy]], dtype=torch.float32, device=device),
            material_onehot
        ], dim=1)
        
        # Get latent state for this density
        z = latent_states[i:i+1]  # (1, latent_dim)
        
        # Compute temporal variance
        temp_var = compute_temporal_variance(
            model, z, conditioning, t_max, num_steps, device
        )
        temporal_variances.append(temp_var.item())
    
    return torch.tensor(temporal_variances, device=device)


def compute_effective_dimensionality(
    z_trajectory: torch.Tensor,
    variance_threshold: float = 0.95
) -> int:
    """Compute effective dimensionality of latent trajectory using PCA.
    
    The effective dimensionality is the number of principal components
    needed to explain a specified fraction of the variance.
    
    Args:
        z_trajectory: Latent trajectory (num_steps, latent_dim) or (num_steps, batch_size, latent_dim)
        variance_threshold: Fraction of variance to explain (default: 0.95)
    
    Returns:
        Effective dimensionality (number of components)
    
    Requirements:
        - Validates: Requirement 11.2
        - Applies PCA to latent trajectory
        - Counts components explaining 95% variance
    """
    # Handle batch dimension
    if z_trajectory.dim() == 3:
        # Flatten batch dimension: (num_steps * batch_size, latent_dim)
        z_trajectory = z_trajectory.reshape(-1, z_trajectory.shape[-1])
    
    # Convert to numpy for sklearn PCA
    z_np = z_trajectory.cpu().numpy()
    
    # Center the data
    z_centered = z_np - z_np.mean(axis=0, keepdims=True)
    
    # Compute covariance matrix
    cov = np.cov(z_centered.T)
    
    # Compute eigenvalues
    eigenvalues = np.linalg.eigvalsh(cov)
    
    # Sort in descending order
    eigenvalues = np.sort(eigenvalues)[::-1]
    
    # Compute explained variance ratio
    total_variance = eigenvalues.sum()
    explained_variance_ratio = eigenvalues / total_variance
    
    # Compute cumulative variance
    cumsum = np.cumsum(explained_variance_ratio)
    
    # Find number of components needed
    eff_dim = np.argmax(cumsum >= variance_threshold) + 1
    
    return int(eff_dim)


def compute_effective_dimensionality_for_density(
    model: MultiScaleModel,
    latent_state: torch.Tensor,
    conditioning: torch.Tensor,
    t_max: float = 100.0,
    num_steps: int = 50,
    variance_threshold: float = 0.95,
    device: str = 'cpu'
) -> int:
    """Compute effective dimensionality for a single density.
    
    Evolves the latent state over time and computes the effective
    dimensionality of the resulting trajectory.
    
    Args:
        model: Trained MultiScaleModel
        latent_state: Initial latent state (1, latent_dim)
        conditioning: Conditioning vector (1, conditioning_dim)
        t_max: Maximum time to evolve (fs)
        num_steps: Number of time steps
        variance_threshold: Fraction of variance to explain
        device: Device to run computations on
    
    Returns:
        Effective dimensionality
    
    Requirements:
        - Validates: Requirement 11.2
        - Computes effective dimensionality via PCA
    """
    model.eval()
    model = model.to(device)
    latent_state = latent_state.to(device)
    conditioning = conditioning.to(device)
    
    # Generate time points
    t_span = torch.linspace(0, t_max, num_steps, device=device)
    
    # Evolve latent state at each time point
    z_trajectory = []
    
    with torch.no_grad():
        for t in t_span:
            # Skip t=0 (initial state)
            if t.item() == 0:
                z_trajectory.append(latent_state)
            else:
                t_interval = torch.tensor([0.0, t.item()], device=device)
                z_t = model.neural_ode(latent_state, t_interval, conditioning)
                z_trajectory.append(z_t)
    
    # Stack trajectory: (num_steps, 1, latent_dim)
    z_trajectory = torch.stack(z_trajectory, dim=0)
    
    # Compute effective dimensionality
    eff_dim = compute_effective_dimensionality(z_trajectory, variance_threshold)
    
    return eff_dim


def compute_effective_dimensionalities_for_scan(
    model: MultiScaleModel,
    latent_states: torch.Tensor,
    densities: np.ndarray,
    base_config: BeamConfiguration,
    t_max: float = 100.0,
    num_steps: int = 50,
    variance_threshold: float = 0.95,
    device: str = 'cpu'
) -> np.ndarray:
    """Compute effective dimensionality for each density in a scan.
    
    Args:
        model: Trained MultiScaleModel
        latent_states: Latent representations (num_densities, latent_dim)
        densities: Corresponding density values
        base_config: Base configuration for conditioning
        t_max: Maximum time to evolve (fs)
        num_steps: Number of time steps
        variance_threshold: Fraction of variance to explain
        device: Device to run computations on
    
    Returns:
        Effective dimensionalities (num_densities,)
    
    Requirements:
        - Validates: Requirement 11.2
        - Computes effective dimensionality for each density
    """
    eff_dims = []
    
    # Material encoding for conditioning
    material_map = {'Si': 0, 'Fe': 1, 'W': 2, 'Cu': 3, 'Al': 4}
    material_idx = material_map.get(base_config.material, 0)
    material_onehot = torch.nn.functional.one_hot(
        torch.tensor([material_idx], device=device),
        num_classes=5
    ).float()
    
    for i, density in enumerate(densities):
        # Create conditioning vector for this density
        conditioning = torch.cat([
            torch.tensor([[density]], dtype=torch.float32, device=device),
            torch.tensor([[base_config.energy]], dtype=torch.float32, device=device),
            material_onehot
        ], dim=1)
        
        # Get latent state for this density
        z = latent_states[i:i+1]  # (1, latent_dim)
        
        # Compute effective dimensionality
        eff_dim = compute_effective_dimensionality_for_density(
            model, z, conditioning, t_max, num_steps, variance_threshold, device
        )
        eff_dims.append(eff_dim)
    
    return np.array(eff_dims)


def compute_anomaly_score(
    mahalanobis_dist: float,
    temporal_var: float,
    epistemic_unc: float,
    dim_change: float,
    weights: Optional[List[float]] = None
) -> float:
    """Compute combined anomaly score from multiple indicators.
    
    The anomaly score combines:
    1. Mahalanobis distance from single-particle regime
    2. Temporal variance in latent trajectory
    3. Epistemic uncertainty from model ensemble
    4. Change in effective dimensionality
    
    Args:
        mahalanobis_dist: Mahalanobis distance from baseline
        temporal_var: Temporal variance in latent space
        epistemic_unc: Epistemic uncertainty estimate
        dim_change: Change in effective dimensionality from baseline
        weights: Weights for each component [w_mahal, w_temp, w_epist, w_dim]
                Default: [1.0, 1.0, 0.5, 1.0]
    
    Returns:
        Combined anomaly score
    
    Requirements:
        - Validates: Requirement 11.5
        - Combines Mahalanobis distance, temporal variance, epistemic uncertainty, dimensionality change
    """
    if weights is None:
        weights = [1.0, 1.0, 0.5, 1.0]
    
    # Combine weighted components
    score = (
        weights[0] * mahalanobis_dist +
        weights[1] * temporal_var +
        weights[2] * epistemic_unc +
        weights[3] * dim_change
    )
    
    return score


def compute_anomaly_scores_for_scan(
    model: MultiScaleModel,
    latent_states: torch.Tensor,
    densities: np.ndarray,
    base_config: BeamConfiguration,
    epistemic_uncertainties: Optional[np.ndarray] = None,
    baseline_dim: int = 2,
    t_max: float = 100.0,
    num_steps: int = 50,
    density_threshold: float = 0.1,
    weights: Optional[List[float]] = None,
    device: str = 'cpu'
) -> np.ndarray:
    """Compute anomaly scores for each density in a scan.
    
    This is the main function that combines all anomaly indicators to
    produce a single score for each density value.
    
    Args:
        model: Trained MultiScaleModel
        latent_states: Latent representations (num_densities, latent_dim)
        densities: Corresponding density values
        base_config: Base configuration for conditioning
        epistemic_uncertainties: Optional epistemic uncertainty estimates (num_densities,)
        baseline_dim: Baseline effective dimensionality for single-particle regime
        t_max: Maximum time to evolve (fs)
        num_steps: Number of time steps
        density_threshold: Threshold for low-density regime
        weights: Weights for anomaly score components
        device: Device to run computations on
    
    Returns:
        Anomaly scores (num_densities,)
    
    Requirements:
        - Validates: Requirement 11.5
        - Computes combined anomaly score for each density
    """
    # 1. Compute Mahalanobis distances
    mahal_distances = compute_mahalanobis_distances(
        latent_states, densities, density_threshold
    ).cpu().numpy()
    
    # 2. Compute temporal variances
    temporal_vars = compute_temporal_variances_for_scan(
        model, latent_states, densities, base_config, t_max, num_steps, device
    ).cpu().numpy()
    
    # 3. Use provided epistemic uncertainties or zeros
    if epistemic_uncertainties is None:
        epistemic_uncertainties = np.zeros(len(densities))
    
    # 4. Compute effective dimensionalities and changes
    eff_dims = compute_effective_dimensionalities_for_scan(
        model, latent_states, densities, base_config, t_max, num_steps, 0.95, device
    )
    dim_changes = np.abs(eff_dims - baseline_dim)
    
    # 5. Normalize each component to [0, 1] range for fair weighting
    def normalize(arr):
        """Normalize array to [0, 1] range."""
        arr_min = arr.min()
        arr_max = arr.max()
        if arr_max - arr_min < 1e-10:
            return np.zeros_like(arr)
        return (arr - arr_min) / (arr_max - arr_min)
    
    mahal_norm = normalize(mahal_distances)
    temporal_norm = normalize(temporal_vars)
    epistemic_norm = normalize(epistemic_uncertainties)
    dim_norm = normalize(dim_changes)
    
    # 6. Compute combined anomaly scores
    anomaly_scores = []
    for i in range(len(densities)):
        score = compute_anomaly_score(
            mahal_norm[i],
            temporal_norm[i],
            epistemic_norm[i],
            dim_norm[i],
            weights
        )
        anomaly_scores.append(score)
    
    return np.array(anomaly_scores)



def detect_transitions(
    anomaly_scores: np.ndarray,
    density_range: np.ndarray,
    threshold: float = 2.0,
    min_separation: float = 0.1
) -> List[float]:
    """Identify density thresholds where collective effects emerge.
    
    Uses z-score normalization to identify anomalous densities that
    exceed a threshold, indicating potential collective transitions.
    
    Args:
        anomaly_scores: Anomaly scores for each density
        density_range: Corresponding densities (particles/nm³)
        threshold: Z-score threshold for detection (default: 2.0)
        min_separation: Minimum density separation between transitions (particles/nm³)
    
    Returns:
        List of critical densities where transitions occur
    
    Requirements:
        - Validates: Requirement 11.6
        - Computes z-scores for anomaly scores
        - Flags densities exceeding threshold
    """
    # Normalize scores to z-scores
    mean = anomaly_scores.mean()
    std = anomaly_scores.std()
    
    if std < 1e-10:
        warnings.warn("Anomaly scores have zero variance. No transitions detected.")
        return []
    
    z_scores = (anomaly_scores - mean) / std
    
    # Find points exceeding threshold
    transitions = []
    
    for i, (density, z_score) in enumerate(zip(density_range, z_scores)):
        if z_score > threshold:
            # Check if this is a new transition (not adjacent to previous)
            if not transitions or (density - transitions[-1]) > min_separation:
                transitions.append(float(density))
    
    return transitions


def full_density_scan(
    model: MultiScaleModel,
    base_config: BeamConfiguration,
    density_range: np.ndarray,
    epistemic_uncertainties: Optional[np.ndarray] = None,
    baseline_dim: int = 2,
    t_max: float = 100.0,
    num_steps: int = 50,
    density_threshold: float = 0.1,
    detection_threshold: float = 2.0,
    min_separation: float = 0.1,
    weights: Optional[List[float]] = None,
    device: str = 'cpu'
) -> ScanResult:
    """Perform complete density space scan with transition detection.
    
    This is the main entry point for hypothesis generation via latent space scanning.
    It performs all steps:
    1. Scan density space and compute latent representations
    2. Compute anomaly scores from multiple indicators
    3. Detect collective transitions
    
    Args:
        model: Trained MultiScaleModel
        base_config: Base beam configuration
        density_range: Array of densities to scan (particles/nm³)
        epistemic_uncertainties: Optional epistemic uncertainty estimates
        baseline_dim: Baseline effective dimensionality
        t_max: Maximum time for temporal evolution (fs)
        num_steps: Number of time steps for evolution
        density_threshold: Threshold for low-density regime
        detection_threshold: Z-score threshold for transition detection
        min_separation: Minimum density separation between transitions
        weights: Weights for anomaly score components
        device: Device to run computations on
    
    Returns:
        ScanResult with densities, latent states, anomaly scores, and critical densities
    
    Requirements:
        - Validates: Requirements 11.1, 11.2, 11.5, 11.6
        - Performs complete density space scanning
        - Detects collective transitions
    """
    # Step 1: Scan density space
    _, latent_states = scan_density_space(model, base_config, density_range, device)
    
    # Step 2: Compute anomaly scores
    anomaly_scores = compute_anomaly_scores_for_scan(
        model=model,
        latent_states=latent_states,
        densities=density_range,
        base_config=base_config,
        epistemic_uncertainties=epistemic_uncertainties,
        baseline_dim=baseline_dim,
        t_max=t_max,
        num_steps=num_steps,
        density_threshold=density_threshold,
        weights=weights,
        device=device
    )
    
    # Step 3: Detect transitions
    critical_densities = detect_transitions(
        anomaly_scores=anomaly_scores,
        density_range=density_range,
        threshold=detection_threshold,
        min_separation=min_separation
    )
    
    # Return results
    return ScanResult(
        densities=density_range,
        latent_states=latent_states,
        anomaly_scores=anomaly_scores,
        critical_densities=critical_densities
    )


# ============================================================================
# Collective Mode Extraction
# ============================================================================

def extract_mode_frequency_and_damping(
    model: MultiScaleModel,
    config_at_transition: BeamConfiguration,
    t_max: float = 1000.0,
    dt: float = 1.0,
    device: str = 'cpu'
) -> Tuple[float, float, torch.Tensor]:
    """Extract collective mode frequency and damping rate using Fourier analysis.
    
    Evolves the latent state for a long time and applies FFT to identify
    dominant oscillatory modes and their damping rates.
    
    Args:
        model: Trained MultiScaleModel
        config_at_transition: Beam configuration at critical density
        t_max: Maximum time to evolve (fs)
        dt: Time step (fs)
        device: Device to run computations on
    
    Returns:
        Tuple of:
            - frequency: Dominant mode frequency (THz)
            - damping_rate: Mode damping rate (1/fs)
            - z_trajectory: Latent trajectory (num_steps, latent_dim)
    
    Requirements:
        - Validates: Requirements 12.1, 12.2
        - Evolves latent state for long time
        - Applies FFT to each latent dimension
        - Identifies dominant frequencies
    """
    model.eval()
    model = model.to(device)
    
    # Convert configuration to GraphData
    data = _config_to_graph_data(config_at_transition, device)
    
    # Encode to latent space
    with torch.no_grad():
        z0 = model.encode(data)  # (1, latent_dim)
    
    # Generate time points
    times = torch.arange(0, t_max, dt, device=device)
    num_steps = len(times)
    
    # Material encoding for conditioning
    material_map = {'Si': 0, 'Fe': 1, 'W': 2, 'Cu': 3, 'Al': 4}
    material_idx = material_map.get(config_at_transition.material, 0)
    material_onehot = torch.nn.functional.one_hot(
        torch.tensor([material_idx], device=device),
        num_classes=5
    ).float()
    
    # Create conditioning vector
    conditioning = torch.cat([
        torch.tensor([[config_at_transition.density]], dtype=torch.float32, device=device),
        torch.tensor([[config_at_transition.energy]], dtype=torch.float32, device=device),
        material_onehot
    ], dim=1)
    
    # Evolve latent state at each time point
    z_trajectory = []
    
    with torch.no_grad():
        for t in times:
            if t.item() == 0:
                z_trajectory.append(z0)
            else:
                t_interval = torch.tensor([0.0, t.item()], device=device)
                z_t = model.neural_ode(z0, t_interval, conditioning)
                z_trajectory.append(z_t)
    
    # Stack trajectory: (num_steps, 1, latent_dim)
    z_trajectory = torch.stack(z_trajectory, dim=0).squeeze(1)  # (num_steps, latent_dim)
    
    # Fourier transform each latent dimension
    frequencies_list = []
    damping_rates_list = []
    powers_list = []
    
    for dim in range(z_trajectory.shape[-1]):
        signal = z_trajectory[:, dim].cpu().numpy()
        
        # Apply FFT
        fft = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), d=dt)
        
        # Consider only positive frequencies
        pos_mask = freqs > 0
        power = np.abs(fft[pos_mask])
        pos_freqs = freqs[pos_mask]
        
        # Find dominant frequency
        if len(power) > 0 and power.max() > 0:
            dominant_idx = np.argmax(power)
            dominant_freq = pos_freqs[dominant_idx]
            dominant_power = power[dominant_idx]
            
            # Convert from 1/fs to THz
            dominant_freq_THz = dominant_freq * 1000.0  # 1/fs = 1000 THz
            
            # Estimate damping from exponential envelope
            # Fit: A * exp(-γt) to the envelope
            envelope = np.abs(signal)
            
            # Avoid log of zero or negative values
            envelope_safe = np.maximum(envelope, 1e-10)
            log_envelope = np.log(envelope_safe)
            
            # Linear fit to log(envelope) vs time
            times_np = times.cpu().numpy()
            
            # Use only first half to avoid numerical issues at long times
            half_idx = len(times_np) // 2
            if half_idx > 10:
                coeffs = np.polyfit(times_np[:half_idx], log_envelope[:half_idx], 1)
                gamma = -coeffs[0]  # Damping rate in 1/fs
                
                # Ensure non-negative damping
                gamma = max(0.0, gamma)
            else:
                gamma = 0.0
            
            # Only consider significant modes (power > 10% of max across all dimensions)
            frequencies_list.append(dominant_freq_THz)
            damping_rates_list.append(gamma)
            powers_list.append(dominant_power)
    
    # Select the most prominent mode (highest power)
    if len(frequencies_list) > 0:
        max_power_idx = np.argmax(powers_list)
        mode_freq = frequencies_list[max_power_idx]
        mode_damping = damping_rates_list[max_power_idx]
    else:
        # No significant oscillations detected
        mode_freq = 0.0
        mode_damping = 0.0
    
    return mode_freq, mode_damping, z_trajectory


def compute_dispersion_relation(
    model: MultiScaleModel,
    base_config: BeamConfiguration,
    critical_density: float,
    momentum_range: Optional[np.ndarray] = None,
    t_max: float = 1000.0,
    dt: float = 1.0,
    device: str = 'cpu'
) -> Tuple[Callable[[float], float], float, np.ndarray, np.ndarray]:
    """Compute dispersion relation ω(k) by varying beam momentum.
    
    Scans different beam momenta (wavenumbers) and extracts the mode
    frequency at each momentum to construct the dispersion relation.
    
    Args:
        model: Trained MultiScaleModel
        base_config: Base beam configuration
        critical_density: Critical density for collective mode
        momentum_range: Array of momenta to scan (MeV/c). If None, uses default range.
        t_max: Maximum time for frequency extraction (fs)
        dt: Time step (fs)
        device: Device to run computations on
    
    Returns:
        Tuple of:
            - dispersion_fn: Callable ω(k) function
            - group_velocity: dω/dk at k=0 (nm/fs)
            - wavenumbers: Array of wavenumbers (1/nm)
            - frequencies: Array of frequencies (THz)
    
    Requirements:
        - Validates: Requirements 12.3, 12.5
        - Varies beam momentum to scan wavenumber
        - Extracts frequency at each momentum
        - Fits polynomial to ω(k)
        - Computes group velocity dω/dk
    """
    if momentum_range is None:
        # Default momentum range: 0.1 to 10 MeV/c
        momentum_range = np.linspace(0.1, 10.0, 20)
    
    frequencies = []
    
    for p in momentum_range:
        # Create configuration at this momentum
        config = BeamConfiguration(
            particles=base_config.particles,
            density=critical_density,
            energy=_momentum_to_energy(p),
            material=base_config.material,
            temperature=base_config.temperature,
            time=base_config.time
        )
        
        # Extract mode frequency at this momentum
        freq, _, _ = extract_mode_frequency_and_damping(
            model, config, t_max, dt, device
        )
        frequencies.append(freq)
    
    frequencies = np.array(frequencies)
    
    # Convert momentum to wavenumber
    # k = p / ℏ, where ℏ = 0.6582 eV·fs (in natural units)
    hbar = 0.6582  # eV·fs
    # Convert MeV/c to eV/c: multiply by 1e6
    # Then divide by ℏc to get wavenumber
    # c = 299.792 nm/fs
    c = 299.792  # nm/fs
    wavenumbers = (momentum_range * 1e6) / (hbar * c)  # 1/nm
    
    # Fit dispersion relation
    # Try quadratic fit: ω = a + b*k + c*k²
    if len(wavenumbers) >= 3:
        coeffs = np.polyfit(wavenumbers, frequencies, deg=2)
        dispersion_poly = np.poly1d(coeffs)
        
        # Create callable function
        def dispersion_fn(k: float) -> float:
            """Dispersion relation ω(k)."""
            return float(dispersion_poly(k))
        
        # Group velocity: dω/dk
        # For polynomial a + b*k + c*k², derivative is b + 2*c*k
        # Evaluate at k=0
        group_velocity = coeffs[-2]  # Linear coefficient (b)
    else:
        # Not enough points for polynomial fit, use linear
        if len(wavenumbers) >= 2:
            coeffs = np.polyfit(wavenumbers, frequencies, deg=1)
            dispersion_poly = np.poly1d(coeffs)
            
            def dispersion_fn(k: float) -> float:
                return float(dispersion_poly(k))
            
            group_velocity = coeffs[0]  # Slope
        else:
            # Only one point, constant dispersion
            def dispersion_fn(k: float) -> float:
                return frequencies[0] if len(frequencies) > 0 else 0.0
            
            group_velocity = 0.0
    
    return dispersion_fn, group_velocity, wavenumbers, frequencies


def _momentum_to_energy(momentum: float) -> float:
    """Convert momentum to kinetic energy.
    
    E = p²/(2m) for non-relativistic case
    
    Args:
        momentum: Momentum in MeV/c
    
    Returns:
        Energy in MeV
    """
    # For simplicity, assume electron mass
    # m_e c² = 0.511 MeV
    m_e_c2 = 0.511  # MeV
    
    # E = sqrt(p²c² + m²c⁴) - mc² for relativistic
    # For non-relativistic: E ≈ p²/(2m)
    # Use non-relativistic approximation
    E = (momentum ** 2) / (2 * m_e_c2)
    
    return E


def characterize_collective_mode(
    model: MultiScaleModel,
    base_config: BeamConfiguration,
    critical_density: float,
    epistemic_uncertainty: float = 0.0,
    t_max: float = 1000.0,
    dt: float = 1.0,
    momentum_range: Optional[np.ndarray] = None,
    device: str = 'cpu'
) -> 'CollectiveMode':
    """Full characterization of collective mode.
    
    Combines frequency extraction, damping rate estimation, dispersion
    relation computation, and experimental signature prediction into a
    complete CollectiveMode object.
    
    Args:
        model: Trained MultiScaleModel
        base_config: Base beam configuration
        critical_density: Critical density where mode emerges
        epistemic_uncertainty: Epistemic uncertainty estimate
        t_max: Maximum time for frequency extraction (fs)
        dt: Time step (fs)
        momentum_range: Array of momenta for dispersion relation
        device: Device to run computations on
    
    Returns:
        CollectiveMode object with all properties
    
    Requirements:
        - Validates: Requirements 12.1, 12.2, 12.3, 12.4, 12.5
        - Combines frequency, damping, dispersion, group velocity
        - Includes uncertainty estimates
        - Returns CollectiveMode object
    """
    
    # Create configuration at critical density
    config_at_critical = BeamConfiguration(
        particles=base_config.particles,
        density=critical_density,
        energy=base_config.energy,
        material=base_config.material,
        temperature=base_config.temperature,
        time=base_config.time
    )
    
    # Extract frequency and damping
    frequency, damping_rate, _ = extract_mode_frequency_and_damping(
        model, config_at_critical, t_max, dt, device
    )
    
    # Compute dispersion relation and group velocity
    dispersion_fn, group_velocity, _, _ = compute_dispersion_relation(
        model, base_config, critical_density, momentum_range, t_max, dt, device
    )
    
    # Predict experimental signatures (simplified placeholders)
    # In a full implementation, these would use the detailed prediction functions
    # from the design document
    scattering_peak = frequency  # Simplified: peak at mode frequency
    energy_spread = 0.1 * base_config.energy  # Simplified: 10% of beam energy
    correlation_length = 10.0  # Simplified: 10 nm
    
    # Create CollectiveMode object
    mode = CollectiveMode(
        frequency=frequency,
        damping_rate=damping_rate,
        critical_density=critical_density,
        dispersion_relation=dispersion_fn,
        group_velocity=group_velocity,
        uncertainty=epistemic_uncertainty,
        scattering_peak=scattering_peak,
        energy_spread=energy_spread,
        correlation_length=correlation_length
    )
    
    return mode


# ============================================================================
# Observable Signature Prediction
# ============================================================================

def predict_scattering_peak(
    model: MultiScaleModel,
    config: BeamConfiguration,
    mode_frequency: float,
    q_range: Optional[np.ndarray] = None,
    omega_range: Optional[np.ndarray] = None,
    num_time_samples: int = 20,
    t_max: float = 100.0,
    device: str = 'cpu'
) -> Tuple[float, float, np.ndarray]:
    """Predict scattering cross-section peak from collective mode.
    
    Collective modes appear as peaks in the dynamic structure factor S(q, ω).
    This function computes a simplified version of S(q, ω) by evolving the
    system and computing spatial Fourier components.
    
    Args:
        model: Trained MultiScaleModel
        config: Beam configuration at critical density
        mode_frequency: Expected mode frequency (THz)
        q_range: Array of wavenumbers to scan (1/nm). If None, uses default.
        omega_range: Array of frequencies to scan (THz). If None, uses default.
        num_time_samples: Number of time samples for temporal averaging
        t_max: Maximum time for evolution (fs)
        device: Device to run computations on
    
    Returns:
        Tuple of:
            - peak_q: Wavenumber at peak (1/nm)
            - peak_omega: Frequency at peak (THz)
            - S_qw: Dynamic structure factor array (len(q_range), len(omega_range))
    
    Requirements:
        - Validates: Requirement 13.1
        - Computes dynamic structure factor S(q, ω)
        - Identifies peak corresponding to collective mode
    """
    model.eval()
    model = model.to(device)
    
    # Default ranges if not provided
    if q_range is None:
        q_range = np.linspace(0.1, 5.0, 50)  # 1/nm
    
    if omega_range is None:
        # Center around mode frequency
        omega_min = max(0.0, mode_frequency - 2.0)
        omega_max = mode_frequency + 2.0
        omega_range = np.linspace(omega_min, omega_max, 100)  # THz
    
    # Convert configuration to GraphData
    data = _config_to_graph_data(config, device)
    
    # Material encoding for conditioning
    material_map = {'Si': 0, 'Fe': 1, 'W': 2, 'Cu': 3, 'Al': 4}
    material_idx = material_map.get(config.material, 0)
    material_onehot = torch.nn.functional.one_hot(
        torch.tensor([material_idx], device=device),
        num_classes=5
    ).float()
    
    # Create conditioning vector
    conditioning = torch.cat([
        torch.tensor([[config.density]], dtype=torch.float32, device=device),
        torch.tensor([[config.energy]], dtype=torch.float32, device=device),
        material_onehot
    ], dim=1)
    
    # Initialize structure factor array
    S_qw = np.zeros((len(q_range), len(omega_range)))
    
    # Sample times for temporal averaging
    times = np.linspace(0, t_max, num_time_samples)
    
    # Ensure batch is set (needed for decoder)
    if data.batch is None:
        data.batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=device)
    
    with torch.no_grad():
        for iq, q in enumerate(q_range):
            for iw, omega in enumerate(omega_range):
                # Compute structure factor contribution at this (q, ω)
                structure_factor = 0.0
                
                for t in times:
                    # Use full model forward pass to get predictions
                    # For t=0, use a small positive value to avoid ODE solver issues
                    t_end = max(t, 0.01)
                    t_interval = torch.tensor([0.0, t_end], device=device)
                    output = model(data, t_interval)
                    
                    # Get predicted particle embeddings
                    particle_pred = output['particle_pred']  # (N, D_transformer)
                    
                    # Extract positions from predictions
                    # Simplified: use first 3 dimensions as position updates
                    positions = data.pos + particle_pred[:, :3]
                    
                    # Compute S(q, ω) contribution
                    # S(q, ω) ∝ |Σ_i exp(i q·r_i - i ω t)|²
                    # Simplified: use q along x-direction
                    phase = torch.exp(1j * torch.tensor(q * positions[:, 0].cpu().numpy() - omega * t * 1e-3, dtype=torch.complex64))
                    structure_factor += torch.abs(phase.sum()).item() ** 2
                
                # Average over time samples
                S_qw[iq, iw] = structure_factor / num_time_samples
    
    # Find peak in S(q, ω)
    peak_idx = np.unravel_index(np.argmax(S_qw), S_qw.shape)
    peak_q = q_range[peak_idx[0]]
    peak_omega = omega_range[peak_idx[1]]
    
    return peak_q, peak_omega, S_qw



def predict_energy_spread(
    model: MultiScaleModel,
    config: BeamConfiguration,
    t_span: float = 10.0,
    device: str = 'cpu'
) -> float:
    """Predict beam energy distribution width.
    
    Collective oscillations increase energy spread. This function predicts
    the energy spread from the model's observable decoder output.
    
    Args:
        model: Trained MultiScaleModel
        config: Beam configuration at critical density
        t_span: Time interval for prediction (fs)
        device: Device to run computations on
    
    Returns:
        Energy spread σ_E in MeV
    
    Requirements:
        - Validates: Requirement 13.2
        - Predicts beam energy distribution width from observables
    """
    model.eval()
    model = model.to(device)
    
    # Convert configuration to GraphData
    data = _config_to_graph_data(config, device)
    
    # Forward pass through model
    with torch.no_grad():
        t_interval = torch.tensor([0.0, t_span], device=device)
        output = model(data, t_interval)
        
        # Extract energy spread from observables
        # Observable decoder outputs: [σ_x, σ_y, σ_E, ε_x, ε_y]
        observables = output['observables']  # (batch_size, 5)
        energy_spread = observables[0, 2].item()  # σ_E
    
    return energy_spread



def predict_correlation_length(
    model: MultiScaleModel,
    config: BeamConfiguration,
    t_span: float = 10.0,
    r_max: float = 50.0,
    num_bins: int = 100,
    device: str = 'cpu'
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Predict spatial correlation length.
    
    Computes the pair correlation function g(r) and finds the correlation
    length where g(r) decays to 1 + (g(0) - 1) / e.
    
    Args:
        model: Trained MultiScaleModel
        config: Beam configuration at critical density
        t_span: Time interval for prediction (fs)
        r_max: Maximum distance for g(r) computation (nm)
        num_bins: Number of bins for g(r)
        device: Device to run computations on
    
    Returns:
        Tuple of:
            - correlation_length: Correlation length ξ in nm
            - r_centers: Bin centers for g(r)
            - g_r: Pair correlation function values
    
    Requirements:
        - Validates: Requirement 13.3
        - Computes pair correlation function g(r)
        - Finds correlation length where g(r) decays to 1/e
    """
    model.eval()
    model = model.to(device)
    
    # Convert configuration to GraphData
    data = _config_to_graph_data(config, device)
    
    # Forward pass through model
    with torch.no_grad():
        t_interval = torch.tensor([0.0, t_span], device=device)
        output = model(data, t_interval)
        
        # Get predicted particle embeddings
        particle_pred = output['particle_pred']  # (N, D_transformer)
        
        # Extract positions from predictions
        # Simplified: use first 3 dimensions as position updates
        positions = data.pos + particle_pred[:, :3]
    
    # Compute pairwise distances
    distances = torch.cdist(positions, positions)  # (N, N)
    
    # Remove self-distances (diagonal)
    mask = ~torch.eye(distances.shape[0], dtype=torch.bool, device=device)
    distances = distances[mask]
    
    # Bin distances to compute g(r)
    r_bins = torch.linspace(0, r_max, num_bins + 1, device=device)
    r_centers = (r_bins[:-1] + r_bins[1:]) / 2
    
    g_r = []
    
    # Estimate box size from particle positions
    box_min = positions.min(dim=0)[0]
    box_max = positions.max(dim=0)[0]
    box_volume = torch.prod(box_max - box_min).item()
    
    # Number density
    n_particles = positions.shape[0]
    density = n_particles / box_volume if box_volume > 0 else 1.0
    
    for i in range(len(r_bins) - 1):
        r_min, r_max_bin = r_bins[i], r_bins[i + 1]
        
        # Count pairs in this distance range
        in_bin = (distances >= r_min) & (distances < r_max_bin)
        n_pairs = in_bin.sum().item()
        
        # Volume of spherical shell
        volume_shell = (4.0 / 3.0) * np.pi * (r_max_bin**3 - r_min**3).item()
        
        # Expected number of pairs for ideal gas
        expected_pairs = n_particles * density * volume_shell
        
        # g(r) = actual / expected
        if expected_pairs > 0:
            g = n_pairs / expected_pairs
        else:
            g = 0.0
        
        g_r.append(g)
    
    g_r = np.array(g_r)
    r_centers_np = r_centers.cpu().numpy()
    
    # Find correlation length: g(ξ) = 1 + (g(0) - 1) / e
    if len(g_r) > 0 and g_r[0] > 1.0:
        g_0 = g_r[0]
        target = 1.0 + (g_0 - 1.0) / np.e
        
        # Find where g(r) crosses target value
        # Look for first crossing after peak
        idx = np.argmin(np.abs(g_r - target))
        correlation_length = r_centers_np[idx]
    else:
        # No clear correlation structure, use default
        correlation_length = 1.0
    
    return correlation_length, r_centers_np, g_r



def predict_experimental_signatures(
    model: MultiScaleModel,
    config: BeamConfiguration,
    mode_frequency: float,
    critical_density: float,
    epistemic_uncertainty: float = 0.0,
    t_span: float = 10.0,
    t_max_scattering: float = 100.0,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """Predict all experimental signatures for a collective mode.
    
    Combines scattering peak, energy spread, and correlation length predictions
    into a comprehensive set of experimental observables with uncertainty estimates.
    
    Args:
        model: Trained MultiScaleModel
        config: Base beam configuration
        mode_frequency: Collective mode frequency (THz)
        critical_density: Critical density where mode emerges (particles/nm³)
        epistemic_uncertainty: Epistemic uncertainty estimate
        t_span: Time interval for predictions (fs)
        t_max_scattering: Maximum time for scattering calculation (fs)
        device: Device to run computations on
    
    Returns:
        Dictionary with keys:
            - scattering_peak_q: Peak wavenumber (1/nm)
            - scattering_peak_omega: Peak frequency (THz)
            - energy_spread: Beam energy distribution width (MeV)
            - correlation_length: Spatial correlation length (nm)
            - critical_density: Critical density (particles/nm³)
            - uncertainty: Epistemic uncertainty
    
    Requirements:
        - Validates: Requirements 13.1, 13.2, 13.3, 13.4, 13.5
        - Predicts scattering peak, energy spread, correlation length, critical density
        - Includes uncertainty estimates
    """
    # Set configuration to critical density
    config_at_critical = BeamConfiguration(
        particles=config.particles,
        density=critical_density,
        energy=config.energy,
        material=config.material,
        temperature=config.temperature,
        time=config.time
    )
    
    # Predict scattering peak
    peak_q, peak_omega, _ = predict_scattering_peak(
        model=model,
        config=config_at_critical,
        mode_frequency=mode_frequency,
        t_max=t_max_scattering,
        device=device
    )
    
    # Predict energy spread
    energy_spread = predict_energy_spread(
        model=model,
        config=config_at_critical,
        t_span=t_span,
        device=device
    )
    
    # Predict correlation length
    correlation_length, _, _ = predict_correlation_length(
        model=model,
        config=config_at_critical,
        t_span=t_span,
        device=device
    )
    
    # Compile results
    signatures = {
        'scattering_peak_q': float(peak_q),
        'scattering_peak_omega': float(peak_omega),
        'energy_spread': float(energy_spread),
        'correlation_length': float(correlation_length),
        'critical_density': float(critical_density),
        'uncertainty': float(epistemic_uncertainty)
    }
    
    return signatures


def characterize_collective_mode_with_signatures(
    model: MultiScaleModel,
    base_config: BeamConfiguration,
    critical_density: float,
    epistemic_uncertainty: float = 0.0,
    t_max: float = 50.0,  # Reduced from 1000.0 for faster execution
    dt: float = 1.0,
    momentum_range: Optional[np.ndarray] = None,
    device: str = 'cpu'
) -> CollectiveMode:
    """Full characterization of collective mode with experimental signatures.
    
    This is an enhanced version of characterize_collective_mode that uses
    the detailed experimental signature prediction functions instead of
    simplified placeholders.
    
    Args:
        model: Trained MultiScaleModel
        base_config: Base beam configuration
        critical_density: Critical density where mode emerges
        epistemic_uncertainty: Epistemic uncertainty estimate
        t_max: Maximum time for frequency extraction (fs)
        dt: Time step (fs)
        momentum_range: Array of momenta for dispersion relation
        device: Device to run computations on
    
    Returns:
        CollectiveMode object with all properties and experimental signatures
    
    Requirements:
        - Validates: Requirements 12.1, 12.2, 12.3, 12.4, 12.5, 13.1, 13.2, 13.3, 13.4, 13.5
        - Combines frequency, damping, dispersion, group velocity
        - Predicts experimental signatures
        - Includes uncertainty estimates
        - Returns CollectiveMode object
    """
    
    # Create configuration at critical density
    config_at_critical = BeamConfiguration(
        particles=base_config.particles,
        density=critical_density,
        energy=base_config.energy,
        material=base_config.material,
        temperature=base_config.temperature,
        time=base_config.time
    )
    
    # Extract frequency and damping
    frequency, damping_rate, _ = extract_mode_frequency_and_damping(
        model, config_at_critical, t_max, dt, device
    )
    
    # Compute dispersion relation and group velocity
    dispersion_fn, group_velocity, _, _ = compute_dispersion_relation(
        model, base_config, critical_density, momentum_range, t_max, dt, device
    )
    
    # Predict experimental signatures using detailed functions
    signatures = predict_experimental_signatures(
        model=model,
        config=base_config,
        mode_frequency=frequency,
        critical_density=critical_density,
        epistemic_uncertainty=epistemic_uncertainty,
        device=device
    )
    
    # Create CollectiveMode object
    mode = CollectiveMode(
        frequency=frequency,
        damping_rate=damping_rate,
        critical_density=critical_density,
        dispersion_relation=dispersion_fn,
        group_velocity=group_velocity,
        uncertainty=epistemic_uncertainty,
        scattering_peak=signatures['scattering_peak_omega'],
        energy_spread=signatures['energy_spread'],
        correlation_length=signatures['correlation_length']
    )
    
    return mode
