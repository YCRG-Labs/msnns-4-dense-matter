"""MD simulation data loading and parsing."""

import os
import logging
from typing import Optional, List, Tuple
import torch
import torch.nn.functional as F
import numpy as np

from .structures import ParticleState, BeamConfiguration, TrajectoryData, GraphData


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MDDataLoader:
    """Load and preprocess MD simulation data.
    
    Supports multiple file formats:
    - LAMMPS dump files (.dump, .lammpstrj)
    - XYZ format (.xyz)
    - HDF5 format (.h5)
    """
    
    def __init__(self):
        """Initialize MD data loader."""
        pass
    
    def load_trajectory(self, filepath: str, format: str = 'auto') -> Optional[TrajectoryData]:
        """Load MD trajectory from file.
        
        Args:
            filepath: Path to trajectory file
            format: File format ('lammps', 'xyz', 'hdf5', 'pickle', or 'auto')
        
        Returns:
            TrajectoryData object or None if loading fails
        """
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            return None
        
        try:
            if format == 'auto':
                format = self._detect_format(filepath)
            
            if format == 'lammps':
                return self._load_lammps(filepath)
            elif format == 'xyz':
                return self._load_xyz(filepath)
            elif format == 'hdf5':
                return self._load_hdf5(filepath)
            elif format == 'pickle':
                return self._load_pickle(filepath)
            else:
                raise ValueError(f"Unsupported format: {format}")
        
        except FileNotFoundError as e:
            logger.error(f"File not found: {filepath}")
            logger.error(f"Error: {e}")
            return None
        except PermissionError as e:
            logger.error(f"Permission denied: {filepath}")
            logger.error(f"Error: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to load {filepath}:")
            logger.error(f"{type(e).__name__}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _detect_format(self, filepath: str) -> str:
        """Detect file format from extension.
        
        Args:
            filepath: Path to file
        
        Returns:
            Format string ('lammps', 'xyz', 'hdf5', or 'pickle')
        """
        ext = os.path.splitext(filepath)[1].lower()
        
        if ext in ['.dump', '.lammpstrj']:
            return 'lammps'
        elif ext == '.xyz':
            return 'xyz'
        elif ext in ['.h5', '.hdf5']:
            return 'hdf5'
        elif ext in ['.pkl', '.pickle']:
            return 'pickle'
        else:
            # Default to LAMMPS
            logger.warning(f"Unknown extension {ext}, assuming LAMMPS format")
            return 'lammps'
    
    def _load_lammps(self, filepath: str) -> TrajectoryData:
        """Parse LAMMPS dump file.
        
        LAMMPS dump format:
            ITEM: TIMESTEP
            0
            ITEM: NUMBER OF ATOMS
            1000
            ITEM: BOX BOUNDS pp pp pp
            0.0 100.0
            0.0 100.0
            0.0 100.0
            ITEM: ATOMS id type x y z vx vy vz q mass
            1 1 0.5 0.5 0.5 0.1 0.1 0.1 1.0 28.0
            ...
        
        Args:
            filepath: Path to LAMMPS dump file
        
        Returns:
            TrajectoryData object
        """
        states = []
        times = []
        box_bounds = None
        
        with open(filepath, 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                
                # Read timestep
                if 'ITEM: TIMESTEP' in line:
                    timestep_line = f.readline().strip()
                    if timestep_line:
                        timestep = float(timestep_line)
                        times.append(timestep)
                
                # Read number of atoms
                elif 'ITEM: NUMBER OF ATOMS' in line:
                    n_atoms_line = f.readline().strip()
                    if n_atoms_line:
                        n_atoms = int(n_atoms_line)
                
                # Read box bounds
                elif 'ITEM: BOX BOUNDS' in line:
                    box_bounds = []
                    for _ in range(3):
                        bounds_line = f.readline().strip()
                        if bounds_line:
                            parts = bounds_line.split()
                            box_bounds.append((float(parts[0]), float(parts[1])))
                
                # Read atoms
                elif 'ITEM: ATOMS' in line:
                    # Parse header to determine column order
                    header = line.strip().split()[2:]  # Skip "ITEM:" and "ATOMS"
                    
                    particles = []
                    for _ in range(n_atoms):
                        atom_line = f.readline().strip()
                        if not atom_line:
                            break
                        
                        parts = atom_line.split()
                        
                        # Create mapping from header to values
                        atom_data = dict(zip(header, parts))
                        
                        # Extract required fields
                        atom_id = int(atom_data.get('id', 0))
                        atom_type = int(atom_data.get('type', 1))
                        
                        # Position
                        x = float(atom_data.get('x', 0.0))
                        y = float(atom_data.get('y', 0.0))
                        z = float(atom_data.get('z', 0.0))
                        
                        # Velocity
                        vx = float(atom_data.get('vx', 0.0))
                        vy = float(atom_data.get('vy', 0.0))
                        vz = float(atom_data.get('vz', 0.0))
                        
                        # Charge and mass
                        q = float(atom_data.get('q', 0.0))
                        mass = float(atom_data.get('mass', 1.0))
                        
                        # Create particle state
                        # Convert velocity to momentum: p = m * v
                        particle = ParticleState(
                            position=torch.tensor([x, y, z], dtype=torch.float32),
                            momentum=torch.tensor([vx * mass, vy * mass, vz * mass], dtype=torch.float32),
                            charge=q,
                            mass=mass,
                            species=atom_type - 1  # Convert to 0-indexed
                        )
                        particles.append(particle)
                    
                    # Create beam configuration
                    if particles:
                        density = self._compute_density(particles, box_bounds)
                        energy = self._compute_energy(particles)
                        material = self._infer_material(particles)
                        temperature = self._compute_temperature(particles)
                        
                        config = BeamConfiguration(
                            particles=particles,
                            density=density,
                            energy=energy,
                            material=material,
                            temperature=temperature,
                            time=times[-1] if times else 0.0
                        )
                        states.append(config)
        
        if not states:
            raise ValueError(f"No valid timesteps found in {filepath}")
        
        metadata = {
            'source_file': filepath,
            'format': 'lammps',
            'num_timesteps': len(states),
            'num_particles': len(states[0].particles) if states else 0
        }
        
        return TrajectoryData(states=states, times=times, metadata=metadata)
    
    def _load_xyz(self, filepath: str) -> TrajectoryData:
        """Parse XYZ format file.
        
        XYZ format:
            N
            comment line
            element x y z
            ...
        
        Args:
            filepath: Path to XYZ file
        
        Returns:
            TrajectoryData object
        """
        states = []
        times = []
        timestep = 0
        
        with open(filepath, 'r') as f:
            while True:
                # Read number of atoms
                n_atoms_line = f.readline().strip()
                if not n_atoms_line:
                    break
                
                n_atoms = int(n_atoms_line)
                
                # Read comment line (may contain timestep info)
                comment = f.readline().strip()
                
                # Read atoms
                particles = []
                for _ in range(n_atoms):
                    atom_line = f.readline().strip()
                    if not atom_line:
                        break
                    
                    parts = atom_line.split()
                    element = parts[0]
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    
                    # Optional: velocity if present
                    if len(parts) >= 7:
                        vx, vy, vz = float(parts[4]), float(parts[5]), float(parts[6])
                    else:
                        vx, vy, vz = 0.0, 0.0, 0.0
                    
                    # Map element to species and mass
                    species, mass, charge = self._element_to_properties(element)
                    
                    particle = ParticleState(
                        position=torch.tensor([x, y, z], dtype=torch.float32),
                        momentum=torch.tensor([vx * mass, vy * mass, vz * mass], dtype=torch.float32),
                        charge=charge,
                        mass=mass,
                        species=species
                    )
                    particles.append(particle)
                
                if particles:
                    density = self._compute_density(particles, None)
                    energy = self._compute_energy(particles)
                    material = self._infer_material(particles)
                    temperature = self._compute_temperature(particles)
                    
                    config = BeamConfiguration(
                        particles=particles,
                        density=density,
                        energy=energy,
                        material=material,
                        temperature=temperature,
                        time=float(timestep)
                    )
                    states.append(config)
                    times.append(float(timestep))
                    timestep += 1
        
        if not states:
            raise ValueError(f"No valid frames found in {filepath}")
        
        metadata = {
            'source_file': filepath,
            'format': 'xyz',
            'num_timesteps': len(states),
            'num_particles': len(states[0].particles) if states else 0
        }
        
        return TrajectoryData(states=states, times=times, metadata=metadata)
    
    def _load_hdf5(self, filepath: str) -> TrajectoryData:
        """Parse HDF5 format file.
        
        Expected HDF5 structure:
            /timestep_0/positions (N, 3)
            /timestep_0/velocities (N, 3)
            /timestep_0/charges (N,)
            /timestep_0/masses (N,)
            /timestep_0/species (N,)
            ...
        
        Args:
            filepath: Path to HDF5 file
        
        Returns:
            TrajectoryData object
        """
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py is required for HDF5 support. Install with: pip install h5py")
        
        states = []
        times = []
        
        with h5py.File(filepath, 'r') as f:
            # Find all timestep groups
            timestep_keys = sorted([k for k in f.keys() if k.startswith('timestep_')])
            
            for key in timestep_keys:
                group = f[key]
                
                # Extract timestep number
                timestep = int(key.split('_')[1])
                times.append(float(timestep))
                
                # Read particle data
                positions = np.array(group['positions'])
                velocities = np.array(group.get('velocities', np.zeros_like(positions)))
                charges = np.array(group.get('charges', np.zeros(len(positions))))
                masses = np.array(group.get('masses', np.ones(len(positions))))
                species = np.array(group.get('species', np.zeros(len(positions), dtype=int)))
                
                # Create particles
                particles = []
                for i in range(len(positions)):
                    particle = ParticleState(
                        position=torch.tensor(positions[i], dtype=torch.float32),
                        momentum=torch.tensor(velocities[i] * masses[i], dtype=torch.float32),
                        charge=float(charges[i]),
                        mass=float(masses[i]),
                        species=int(species[i])
                    )
                    particles.append(particle)
                
                # Create beam configuration
                density = self._compute_density(particles, None)
                energy = self._compute_energy(particles)
                material = self._infer_material(particles)
                temperature = self._compute_temperature(particles)
                
                config = BeamConfiguration(
                    particles=particles,
                    density=density,
                    energy=energy,
                    material=material,
                    temperature=temperature,
                    time=float(timestep)
                )
                states.append(config)
        
        if not states:
            raise ValueError(f"No valid timesteps found in {filepath}")
        
        metadata = {
            'source_file': filepath,
            'format': 'hdf5',
            'num_timesteps': len(states),
            'num_particles': len(states[0].particles) if states else 0
        }
        
        return TrajectoryData(states=states, times=times, metadata=metadata)
    
    def _compute_density(self, particles: List[ParticleState], 
                        box_bounds: Optional[List[Tuple[float, float]]] = None) -> float:
        """Compute number density from particle positions.
        
        Args:
            particles: List of particles
            box_bounds: Optional box bounds [(xmin, xmax), (ymin, ymax), (zmin, zmax)]
        
        Returns:
            Density in particles/nm³
        """
        if not particles:
            return 0.0
        
        positions = torch.stack([p.position for p in particles])
        
        if box_bounds is not None:
            # Use provided box bounds
            volume = 1.0
            for bounds in box_bounds:
                volume *= (bounds[1] - bounds[0])
        else:
            # Estimate volume from bounding box
            min_pos = positions.min(dim=0)[0]
            max_pos = positions.max(dim=0)[0]
            box_size = max_pos - min_pos
            
            # Add small buffer to avoid zero volume
            box_size = torch.maximum(box_size, torch.ones_like(box_size) * 1.0)
            volume = torch.prod(box_size).item()
        
        return len(particles) / volume if volume > 0 else 0.0
    
    def _compute_energy(self, particles: List[ParticleState]) -> float:
        """Compute average kinetic energy per particle.
        
        Args:
            particles: List of particles
        
        Returns:
            Average kinetic energy in MeV
        """
        if not particles:
            return 0.0
        
        energies = []
        for p in particles:
            # KE = p²/(2m)
            ke = torch.sum(p.momentum ** 2) / (2 * p.mass)
            energies.append(ke)
        
        return torch.tensor(energies).mean().item()
    
    def _compute_temperature(self, particles: List[ParticleState]) -> float:
        """Compute temperature from kinetic energy.
        
        Uses equipartition theorem: <KE> = (3/2) k_B T
        
        Args:
            particles: List of particles
        
        Returns:
            Temperature in Kelvin
        """
        if not particles:
            return 0.0
        
        avg_energy = self._compute_energy(particles)
        
        # Boltzmann constant in eV/K
        k_B = 8.617e-5
        
        # E = (3/2) k_B T => T = 2E / (3 k_B)
        temperature = (2 * avg_energy) / (3 * k_B)
        
        return max(temperature, 0.0)
    
    def _infer_material(self, particles: List[ParticleState]) -> str:
        """Infer material from particle masses.
        
        Args:
            particles: List of particles
        
        Returns:
            Material name
        """
        if not particles:
            return 'Si'
        
        # Compute average mass
        avg_mass = np.mean([p.mass for p in particles])
        
        # Map mass to element (atomic mass units)
        mass_to_element = {
            27.0: 'Al',   # Aluminum
            28.0: 'Si',   # Silicon
            56.0: 'Fe',   # Iron
            64.0: 'Cu',   # Copper
            184.0: 'W'    # Tungsten
        }
        
        # Find closest match
        closest_mass = min(mass_to_element.keys(), key=lambda m: abs(m - avg_mass))
        
        # If mass is too far from any known element, default to Si
        if abs(avg_mass - closest_mass) > 10.0:
            logger.warning(f"Unknown material with average mass {avg_mass:.1f}, defaulting to Si")
            return 'Si'
        
        return mass_to_element[closest_mass]
    
    def _element_to_properties(self, element: str) -> Tuple[int, float, float]:
        """Map element symbol to species index, mass, and charge.
        
        Args:
            element: Element symbol (e.g., 'Si', 'Fe')
        
        Returns:
            Tuple of (species_index, mass, charge)
        """
        element_map = {
            'Si': (0, 28.0, 0.0),
            'Fe': (1, 56.0, 0.0),
            'W': (2, 184.0, 0.0),
            'Cu': (3, 64.0, 0.0),
            'Al': (4, 27.0, 0.0)
        }
        
        # Default to Silicon if unknown
        return element_map.get(element, (0, 28.0, 0.0))
    
    def _load_pickle(self, filepath: str) -> TrajectoryData:
        """Load TrajectoryData from pickle file.
        
        Args:
            filepath: Path to pickle file
        
        Returns:
            TrajectoryData object
        """
        import pickle
        
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            # Verify it's a TrajectoryData object
            if not isinstance(data, TrajectoryData):
                logger.error(f"Pickle file does not contain TrajectoryData object: {type(data)}")
                raise ValueError(f"Pickle file does not contain TrajectoryData object: {type(data)}")
            
            logger.info(f"Successfully loaded pickle file: {filepath}")
            return data
        
        except Exception as e:
            logger.error(f"Failed to load pickle file {filepath}:")
            logger.error(f"{type(e).__name__}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise



class DataPreprocessor:
    """Normalize and prepare data for neural network input.
    
    Computes normalization statistics from training data and transforms
    BeamConfiguration objects into normalized GraphData objects suitable
    for neural network processing.
    
    Attributes:
        pos_mean: Mean position vector (3,)
        pos_std: Position standard deviation (3,)
        mom_mean: Mean momentum vector (3,)
        mom_std: Momentum standard deviation (3,)
        density_mean: Mean density
        density_std: Density standard deviation
        energy_mean: Mean energy
        energy_std: Energy standard deviation
    """
    
    def __init__(self):
        """Initialize data preprocessor with empty statistics."""
        self.pos_mean = None
        self.pos_std = None
        self.mom_mean = None
        self.mom_std = None
        self.density_mean = None
        self.density_std = None
        self.energy_mean = None
        self.energy_std = None
        
        # Material mapping
        self.material_map = {'Si': 0, 'Fe': 1, 'W': 2, 'Cu': 3, 'Al': 4}
        self.num_species = 5  # Si, Fe, W, Cu, Al
        self.num_materials = 5
    
    def fit(self, trajectories: List[TrajectoryData]):
        """Compute normalization statistics from training data.
        
        Computes mean and standard deviation for:
        - Particle positions (3D)
        - Particle momenta (3D)
        - Beam density (scalar)
        - Beam energy (scalar)
        
        Args:
            trajectories: List of trajectory data to compute statistics from
        """
        if not trajectories:
            raise ValueError("Cannot fit on empty trajectory list")
        
        all_pos = []
        all_mom = []
        all_density = []
        all_energy = []
        
        # Collect all data
        for traj in trajectories:
            for state in traj.states:
                for p in state.particles:
                    all_pos.append(p.position)
                    all_mom.append(p.momentum)
                all_density.append(state.density)
                all_energy.append(state.energy)
        
        if not all_pos:
            raise ValueError("No particles found in trajectories")
        
        # Stack into tensors
        all_pos = torch.stack(all_pos)  # (N_total, 3)
        all_mom = torch.stack(all_mom)  # (N_total, 3)
        
        # Compute statistics
        self.pos_mean = all_pos.mean(dim=0)  # (3,)
        self.pos_std = all_pos.std(dim=0)    # (3,)
        self.mom_mean = all_mom.mean(dim=0)  # (3,)
        self.mom_std = all_mom.std(dim=0)    # (3,)
        
        # Add small epsilon to prevent division by zero
        self.pos_std = torch.maximum(self.pos_std, torch.ones_like(self.pos_std) * 1e-8)
        self.mom_std = torch.maximum(self.mom_std, torch.ones_like(self.mom_std) * 1e-8)
        
        # Scalar statistics
        self.density_mean = float(np.mean(all_density))
        self.density_std = float(np.std(all_density))
        self.energy_mean = float(np.mean(all_energy))
        self.energy_std = float(np.std(all_energy))
        
        # Add epsilon for scalar stats
        self.density_std = max(self.density_std, 1e-8)
        self.energy_std = max(self.energy_std, 1e-8)
        
        logger.info(f"Fitted preprocessor on {len(all_pos)} particles from {len(trajectories)} trajectories")
        logger.info(f"Position: mean={self.pos_mean.tolist()}, std={self.pos_std.tolist()}")
        logger.info(f"Momentum: mean={self.mom_mean.tolist()}, std={self.mom_std.tolist()}")
        logger.info(f"Density: mean={self.density_mean:.4f}, std={self.density_std:.4f}")
        logger.info(f"Energy: mean={self.energy_mean:.4f}, std={self.energy_std:.4f}")
    
    def transform(self, state: BeamConfiguration) -> GraphData:
        """Convert BeamConfiguration to normalized GraphData.
        
        Performs the following transformations:
        1. Normalizes positions and momenta using computed statistics
        2. Creates one-hot encodings for species
        3. Concatenates all node features
        4. Normalizes conditioning variables (density, energy)
        5. Creates one-hot encoding for material
        6. Constructs GraphData object
        
        Args:
            state: BeamConfiguration to transform
        
        Returns:
            GraphData object with normalized features
        
        Raises:
            ValueError: If preprocessor has not been fitted
        """
        if self.pos_mean is None:
            raise ValueError("Preprocessor must be fitted before transform. Call fit() first.")
        
        if not state.particles:
            raise ValueError("Cannot transform empty particle list")
        
        # Extract particle data
        positions = torch.stack([p.position for p in state.particles])  # (N, 3)
        momenta = torch.stack([p.momentum for p in state.particles])    # (N, 3)
        charges = torch.tensor([p.charge for p in state.particles], dtype=torch.float32)  # (N,)
        masses = torch.tensor([p.mass for p in state.particles], dtype=torch.float32)     # (N,)
        species = torch.tensor([p.species for p in state.particles], dtype=torch.long)    # (N,)
        
        # Normalize positions and momenta
        pos_norm = (positions - self.pos_mean) / self.pos_std  # (N, 3)
        mom_norm = (momenta - self.mom_mean) / self.mom_std    # (N, 3)
        
        # One-hot encode species
        species_onehot = F.one_hot(species, num_classes=self.num_species).float()  # (N, 5)
        
        # Concatenate node features: [pos_norm, mom_norm, charge, mass, species_onehot]
        # Total dimension: 3 + 3 + 1 + 1 + 5 = 13
        x = torch.cat([
            pos_norm,                    # (N, 3)
            mom_norm,                    # (N, 3)
            charges.unsqueeze(1),        # (N, 1)
            masses.unsqueeze(1),         # (N, 1)
            species_onehot               # (N, 5)
        ], dim=1)  # (N, 13)
        
        # Normalize conditioning variables
        density_norm = (state.density - self.density_mean) / self.density_std
        energy_norm = (state.energy - self.energy_mean) / self.energy_std
        
        # Material one-hot encoding
        material_idx = self.material_map.get(state.material, 0)
        material_onehot = F.one_hot(
            torch.tensor(material_idx, dtype=torch.long), 
            num_classes=self.num_materials
        ).float()  # (5,)
        
        # Create conditioning vector: [density_norm, energy_norm, material_onehot]
        # Total dimension: 1 + 1 + 5 = 7
        conditioning = torch.cat([
            torch.tensor([density_norm], dtype=torch.float32),
            torch.tensor([energy_norm], dtype=torch.float32),
            material_onehot
        ])  # (7,)
        
        # Create GraphData object
        # Note: edge_index and edge_attr will be computed by GNN layer
        data = GraphData(
            x=x,                                                    # (N, 13)
            pos=positions,                                          # (N, 3) - unnormalized for edge construction
            density=torch.tensor([state.density], dtype=torch.float32),
            energy=torch.tensor([state.energy], dtype=torch.float32),
            material=material_onehot.unsqueeze(0)                  # (1, 5)
        )
        
        # Store conditioning for easy access
        data.conditioning = conditioning.unsqueeze(0)  # (1, 7)
        
        # Store original state info as metadata
        data.time = state.time
        data.temperature = state.temperature
        data.num_particles = len(state.particles)
        
        return data
    
    def inverse_transform_positions(self, pos_norm: torch.Tensor) -> torch.Tensor:
        """Convert normalized positions back to original scale.
        
        Args:
            pos_norm: Normalized positions (N, 3)
        
        Returns:
            Original scale positions (N, 3)
        """
        if self.pos_mean is None:
            raise ValueError("Preprocessor must be fitted before inverse transform")
        
        return pos_norm * self.pos_std + self.pos_mean
    
    def inverse_transform_momenta(self, mom_norm: torch.Tensor) -> torch.Tensor:
        """Convert normalized momenta back to original scale.
        
        Args:
            mom_norm: Normalized momenta (N, 3)
        
        Returns:
            Original scale momenta (N, 3)
        """
        if self.mom_mean is None:
            raise ValueError("Preprocessor must be fitted before inverse transform")
        
        return mom_norm * self.mom_std + self.mom_mean
    
    def inverse_transform_density(self, density_norm: float) -> float:
        """Convert normalized density back to original scale.
        
        Args:
            density_norm: Normalized density
        
        Returns:
            Original scale density
        """
        if self.density_mean is None:
            raise ValueError("Preprocessor must be fitted before inverse transform")
        
        return density_norm * self.density_std + self.density_mean
    
    def inverse_transform_energy(self, energy_norm: float) -> float:
        """Convert normalized energy back to original scale.
        
        Args:
            energy_norm: Normalized energy
        
        Returns:
            Original scale energy
        """
        if self.energy_mean is None:
            raise ValueError("Preprocessor must be fitted before inverse transform")
        
        return energy_norm * self.energy_std + self.energy_mean
    
    def save_statistics(self, filepath: str):
        """Save normalization statistics to file.
        
        Args:
            filepath: Path to save statistics
        
        Raises:
            ValueError: If statistics have not been computed
            PermissionError: If file cannot be written
            RuntimeError: If save operation fails
        """
        if self.pos_mean is None:
            logger.error("Cannot save statistics before fitting")
            raise ValueError("Cannot save statistics before fitting")
        
        stats = {
            'pos_mean': self.pos_mean.tolist(),
            'pos_std': self.pos_std.tolist(),
            'mom_mean': self.mom_mean.tolist(),
            'mom_std': self.mom_std.tolist(),
            'density_mean': self.density_mean,
            'density_std': self.density_std,
            'energy_mean': self.energy_mean,
            'energy_std': self.energy_std
        }
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            torch.save(stats, filepath)
            logger.info(f"Saved normalization statistics to {filepath}")
        except PermissionError:
            logger.error(f"Permission denied when writing to: {filepath}")
            raise PermissionError(f"Permission denied when writing to: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save statistics to {filepath}: {e}")
            raise RuntimeError(f"Failed to save statistics to {filepath}: {e}")
    
    def load_statistics(self, filepath: str):
        """Load normalization statistics from file.
        
        Args:
            filepath: Path to load statistics from
        
        Raises:
            FileNotFoundError: If file does not exist
            PermissionError: If file cannot be read
            RuntimeError: If file is corrupted or invalid
        """
        if not os.path.exists(filepath):
            logger.error(f"Statistics file not found: {filepath}")
            raise FileNotFoundError(f"Statistics file not found: {filepath}")
        
        try:
            stats = torch.load(filepath, weights_only=True)
        except PermissionError:
            logger.error(f"Permission denied when reading: {filepath}")
            raise PermissionError(f"Permission denied when reading: {filepath}")
        except Exception as e:
            logger.error(f"Failed to load statistics from {filepath}: {e}")
            raise RuntimeError(f"Failed to load statistics from {filepath}: {e}")
        
        try:
            self.pos_mean = torch.tensor(stats['pos_mean'], dtype=torch.float32)
            self.pos_std = torch.tensor(stats['pos_std'], dtype=torch.float32)
            self.mom_mean = torch.tensor(stats['mom_mean'], dtype=torch.float32)
            self.mom_std = torch.tensor(stats['mom_std'], dtype=torch.float32)
            self.density_mean = stats['density_mean']
            self.density_std = stats['density_std']
            self.energy_mean = stats['energy_mean']
            self.energy_std = stats['energy_std']
        except KeyError as e:
            logger.error(f"Missing key in statistics file: {e}")
            raise RuntimeError(f"Statistics file is missing required key: {e}")
        except Exception as e:
            logger.error(f"Failed to parse statistics: {e}")
            raise RuntimeError(f"Failed to parse statistics: {e}")
        
        logger.info(f"Loaded normalization statistics from {filepath}")


# Convenience function for loading trajectories from a directory
def load_md_trajectories(data_path: str, format: str = 'auto', recursive: bool = True) -> List[TrajectoryData]:
    """Load all MD trajectories from a directory.
    
    Args:
        data_path: Path to directory containing trajectory files
        format: File format ('lammps', 'xyz', 'hdf5', 'pickle', or 'auto')
        recursive: If True, recursively search subdirectories
    
    Returns:
        List of TrajectoryData objects
    """
    loader = MDDataLoader()
    trajectories = []
    
    if not os.path.exists(data_path):
        logger.warning(f"Data path does not exist: {data_path}")
        return trajectories
    
    if os.path.isfile(data_path):
        # Single file
        traj = loader.load_trajectory(data_path, format=format)
        if traj is not None:
            trajectories.append(traj)
    else:
        # Directory of files
        if recursive:
            # Recursively search subdirectories
            for root, dirs, files in os.walk(data_path):
                for filename in files:
                    # Skip metadata files
                    if filename.endswith('_metadata.json') or filename == 'summary.json':
                        continue
                    
                    filepath = os.path.join(root, filename)
                    traj = loader.load_trajectory(filepath, format=format)
                    if traj is not None:
                        trajectories.append(traj)
        else:
            # Only search top-level directory
            for filename in os.listdir(data_path):
                filepath = os.path.join(data_path, filename)
                if os.path.isfile(filepath):
                    traj = loader.load_trajectory(filepath, format=format)
                    if traj is not None:
                        trajectories.append(traj)
    
    logger.info(f"Loaded {len(trajectories)} trajectories from {data_path}")
    return trajectories
