"""Core data structures for particle states and beam configurations."""

from dataclasses import dataclass
from typing import List, Dict, Any, Callable, Optional
import torch
from torch_geometric.data import Data


@dataclass
class ParticleState:
    """Represents the state of a single particle at a given time.
    
    Attributes:
        position: Particle position (x, y, z) in nm
        momentum: Particle momentum (px, py, pz) in atomic units
        charge: Particle charge in elementary charge units
        mass: Particle mass in atomic mass units
        species: Species identifier (0=Si, 1=Fe, etc.)
    """
    position: torch.Tensor  # Shape: (3,)
    momentum: torch.Tensor  # Shape: (3,)
    charge: float
    mass: float
    species: int
    
    def __post_init__(self):
        """Validate particle state after initialization."""
        if self.position.shape != (3,):
            raise ValueError(f"Position must have shape (3,), got {self.position.shape}")
        if self.momentum.shape != (3,):
            raise ValueError(f"Momentum must have shape (3,), got {self.momentum.shape}")
        if self.mass <= 0:
            raise ValueError(f"Mass must be positive, got {self.mass}")
        if self.species < 0:
            raise ValueError(f"Species must be non-negative, got {self.species}")
        if torch.isnan(self.position).any() or torch.isinf(self.position).any():
            raise ValueError("Position contains NaN or Inf values")
        if torch.isnan(self.momentum).any() or torch.isinf(self.momentum).any():
            raise ValueError("Momentum contains NaN or Inf values")


@dataclass
class BeamConfiguration:
    """Represents the configuration and properties of a particle beam.
    
    Attributes:
        particles: List of N particles
        density: Beam density in particles/nm³
        energy: Beam energy in MeV
        material: Material name ("Si", "Fe", "W", "Cu", "Al")
        temperature: Temperature in Kelvin
        time: Current time in femtoseconds
    """
    particles: List[ParticleState]
    density: float
    energy: float
    material: str
    temperature: float
    time: float
    
    def __post_init__(self):
        """Validate beam configuration after initialization."""
        if len(self.particles) == 0:
            raise ValueError("Beam must contain at least one particle")
        if self.density <= 0:
            raise ValueError(f"Density must be positive, got {self.density}")
        if self.energy < 0:
            raise ValueError(f"Energy must be non-negative, got {self.energy}")
        if self.temperature < 0:
            raise ValueError(f"Temperature must be non-negative, got {self.temperature}")
        valid_materials = {"Si", "Fe", "W", "Cu", "Al"}
        if self.material not in valid_materials:
            raise ValueError(f"Material must be one of {valid_materials}, got {self.material}")


class GraphData(Data):
    """PyTorch Geometric Data object for batch processing.
    
    Extends torch_geometric.data.Data with additional attributes for
    particle beam simulations.
    
    Attributes:
        x: Node features (N, D_particle)
        edge_index: Edge connectivity (2, E)
        edge_attr: Edge features (E, D_edge)
        pos: Particle positions (N, 3)
        batch: Batch assignment (N,)
        density: Beam density (batch_size,)
        energy: Beam energy (batch_size,)
        material: Material encoding (batch_size, D_material)
        target_pos: Target positions (N, 3)
        target_mom: Target momenta (N, 3)
        target_obs: Target observables (batch_size, 5)
    """
    
    def __init__(self, 
                 x: Optional[torch.Tensor] = None,
                 edge_index: Optional[torch.Tensor] = None,
                 edge_attr: Optional[torch.Tensor] = None,
                 pos: Optional[torch.Tensor] = None,
                 **kwargs):
        super().__init__(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos, **kwargs)
    
    def validate(self):
        """Validate graph data structure."""
        if self.x is not None:
            if torch.isnan(self.x).any() or torch.isinf(self.x).any():
                raise ValueError("Node features contain NaN or Inf values")
        
        if self.pos is not None:
            if self.pos.shape[1] != 3:
                raise ValueError(f"Positions must have shape (N, 3), got {self.pos.shape}")
            if torch.isnan(self.pos).any() or torch.isinf(self.pos).any():
                raise ValueError("Positions contain NaN or Inf values")
        
        if self.edge_index is not None:
            if self.edge_index.shape[0] != 2:
                raise ValueError(f"Edge index must have shape (2, E), got {self.edge_index.shape}")
            if self.x is not None:
                max_idx = self.edge_index.max().item()
                if max_idx >= self.x.shape[0]:
                    raise ValueError(f"Edge index contains invalid node index {max_idx}")
        
        return True


@dataclass
class LatentState:
    """Represents the latent collective state.
    
    Attributes:
        z: Latent vector (batch_size, D_latent)
        time: Time point in femtoseconds
        conditioning: Conditioning vector (batch_size, D_cond)
    """
    z: torch.Tensor
    time: float
    conditioning: torch.Tensor
    
    def __post_init__(self):
        """Validate latent state after initialization."""
        if len(self.z.shape) != 2:
            raise ValueError(f"Latent vector must be 2D (batch_size, D_latent), got shape {self.z.shape}")
        if torch.isnan(self.z).any() or torch.isinf(self.z).any():
            raise ValueError("Latent vector contains NaN or Inf values")
        if len(self.conditioning.shape) != 2:
            raise ValueError(f"Conditioning must be 2D (batch_size, D_cond), got shape {self.conditioning.shape}")
        if self.z.shape[0] != self.conditioning.shape[0]:
            raise ValueError(f"Batch size mismatch: z has {self.z.shape[0]}, conditioning has {self.conditioning.shape[0]}")


@dataclass
class TrajectoryData:
    """Represents a time series of particle states.
    
    Attributes:
        states: Time series of beam configurations
        times: Corresponding times in femtoseconds
        metadata: Simulation metadata (e.g., ground truth for synthetic data)
    """
    states: List[BeamConfiguration]
    times: List[float]
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        """Validate trajectory data after initialization."""
        if len(self.states) == 0:
            raise ValueError("Trajectory must contain at least one state")
        if len(self.states) != len(self.times):
            raise ValueError(f"Number of states ({len(self.states)}) must match number of times ({len(self.times)})")
        if len(self.times) > 1:
            for i in range(len(self.times) - 1):
                if self.times[i] >= self.times[i + 1]:
                    raise ValueError(f"Times must be strictly increasing, got {self.times[i]} >= {self.times[i + 1]}")


@dataclass
class CollectiveMode:
    """Represents an identified collective mode.
    
    Attributes:
        frequency: Mode frequency in THz
        damping_rate: Damping rate in 1/fs
        critical_density: Density threshold in particles/nm³
        dispersion_relation: ω(k) function
        group_velocity: dω/dk in nm/fs
        uncertainty: Epistemic uncertainty
        scattering_peak: Predicted scattering cross-section peak
        energy_spread: Beam energy distribution width in MeV
        correlation_length: Spatial correlation length in nm
    """
    frequency: float
    damping_rate: float
    critical_density: float
    dispersion_relation: Callable[[float], float]
    group_velocity: float
    uncertainty: float
    scattering_peak: float
    energy_spread: float
    correlation_length: float
    
    def __post_init__(self):
        """Validate collective mode after initialization."""
        if self.frequency < 0:
            raise ValueError(f"Frequency must be non-negative, got {self.frequency}")
        if self.critical_density <= 0:
            raise ValueError(f"Critical density must be positive, got {self.critical_density}")
        if self.uncertainty < 0:
            raise ValueError(f"Uncertainty must be non-negative, got {self.uncertainty}")
        if self.correlation_length <= 0:
            raise ValueError(f"Correlation length must be positive, got {self.correlation_length}")


def validate_particle_state(state: ParticleState) -> bool:
    """Validate a particle state.
    
    Args:
        state: ParticleState to validate
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If validation fails
    """
    # Validation is done in __post_init__, so just return True if no exception
    return True


def validate_beam_configuration(config: BeamConfiguration) -> bool:
    """Validate a beam configuration.
    
    Args:
        config: BeamConfiguration to validate
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If validation fails
    """
    # Validate all particles
    for i, particle in enumerate(config.particles):
        try:
            validate_particle_state(particle)
        except ValueError as e:
            raise ValueError(f"Particle {i} validation failed: {e}")
    
    return True


def validate_graph_data(data: GraphData) -> bool:
    """Validate graph data structure.
    
    Args:
        data: GraphData to validate
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If validation fails
    """
    return data.validate()


def validate_trajectory_data(trajectory: TrajectoryData) -> bool:
    """Validate trajectory data.
    
    Args:
        trajectory: TrajectoryData to validate
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If validation fails
    """
    # Validate all beam configurations
    for i, state in enumerate(trajectory.states):
        try:
            validate_beam_configuration(state)
        except ValueError as e:
            raise ValueError(f"State {i} at time {trajectory.times[i]} validation failed: {e}")
    
    return True



class GraphDataset(torch.utils.data.Dataset):
    """Dataset wrapper for graph data used in training.
    
    This wrapper converts TrajectoryData to GraphData for training.
    """
    
    def __init__(self, trajectories: List, preprocessor=None):
        """Initialize dataset.
        
        Args:
            trajectories: List of TrajectoryData objects
            preprocessor: Optional DataPreprocessor (will create one if not provided)
        """
        self.trajectories = trajectories
        
        # Create or use provided preprocessor
        if preprocessor is None:
            from .loader import DataPreprocessor
            self.preprocessor = DataPreprocessor()
            if trajectories:
                self.preprocessor.fit(trajectories)
        else:
            self.preprocessor = preprocessor
    
    def __len__(self) -> int:
        """Return number of trajectories."""
        return len(self.trajectories)
    
    def __getitem__(self, idx: int):
        """Get trajectory at index as list of GraphData.
        
        Args:
            idx: Index
        
        Returns:
            List of GraphData objects (one per timestep)
        """
        traj = self.trajectories[idx]
        # Convert each BeamConfiguration state to GraphData
        graph_data_list = []
        for state in traj.states:
            graph_data = self.preprocessor.transform(state)
            graph_data_list.append(graph_data)
        return graph_data_list
    
    @staticmethod
    def collate_fn(batch):
        """Collate function for batching trajectories.
        
        Args:
            batch: List of trajectory data (each is a list of GraphData)
        
        Returns:
            Batched data dictionary
        """
        if not batch:
            return {'trajectory': None, 'batch_data': None, 'single_data': None}
        
        # batch is a list of trajectories, each trajectory is a list of GraphData
        # For autoregressive loss: use full trajectories
        # For contrastive loss: use first frame from each trajectory
        # For masked particle loss: use first frame from first trajectory
        
        return {
            'trajectory': batch[0] if len(batch) > 0 else None,  # First trajectory for autoregressive
            'batch_data': [traj[0] for traj in batch if len(traj) > 0] if len(batch) > 1 else None,  # First frames for contrastive
            'single_data': batch[0][0] if len(batch) > 0 and len(batch[0]) > 0 else None  # Single frame for masked
        }
