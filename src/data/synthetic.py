"""Synthetic benchmark generation for validating collective phenomena discovery.

This module generates synthetic particle beam data with known collective behavior:
- Plasma oscillations with known plasmon frequency
- Beam instabilities with known growth rates
- Single-particle baselines with no collective effects

Each generator includes ground truth metadata for validation.
"""

import numpy as np
import torch
from typing import List, Dict, Any
from src.data.structures import ParticleState, BeamConfiguration, TrajectoryData


# Physical constants
ELECTRON_CHARGE = 1.602e-19  # C
EPSILON_0 = 8.854e-12  # F/m
ELECTRON_MASS = 9.109e-31  # kg
BOLTZMANN_CONSTANT = 1.381e-23  # J/K
HBAR = 1.054571817e-34  # J·s

# Material properties (atomic mass units)
MATERIAL_MASSES = {
    'Si': 28.0855,
    'Fe': 55.845,
    'W': 183.84,
    'Cu': 63.546,
    'Al': 26.9815
}


class PlasmaGenerator:
    """Generate synthetic plasma with known plasmon frequency.
    
    The plasmon frequency is given by:
        ω_p = sqrt(n e² / (ε_0 m))
    
    where n is the plasma density, e is the electron charge,
    ε_0 is the permittivity of free space, and m is the electron mass.
    """
    
    def generate(self,
                 n_particles: int = 1000,
                 density: float = 1.0,
                 temperature: float = 300.0,
                 duration: float = 100.0,
                 dt: float = 0.1,
                 material: str = 'Si') -> TrajectoryData:
        """Generate plasma oscillation trajectory.
        
        Args:
            n_particles: Number of particles
            density: Plasma density in particles/nm³
            temperature: Temperature in Kelvin
            duration: Simulation duration in femtoseconds
            dt: Timestep in femtoseconds
            material: Material type for particle properties
            
        Returns:
            TrajectoryData with ground truth plasmon frequency in metadata
        """
        # Compute plasmon frequency
        # Convert density from particles/nm³ to particles/m³
        n_SI = density * 1e27
        
        # Use material-specific mass
        mass_amu = MATERIAL_MASSES.get(material, ELECTRON_MASS * 1e3 / 1.66054e-27)
        mass_kg = mass_amu * 1.66054e-27  # Convert AMU to kg
        
        # Plasmon frequency in rad/s
        omega_p = np.sqrt(n_SI * ELECTRON_CHARGE**2 / (EPSILON_0 * mass_kg))
        
        # Convert to THz (cycles per femtosecond)
        freq_THz = omega_p / (2 * np.pi * 1e12)
        
        # Initialize particles in a cubic box
        box_size = (n_particles / density) ** (1/3)
        
        # Random initial positions
        positions = torch.rand(n_particles, 3) * box_size
        
        # Maxwell-Boltzmann velocity distribution
        # v_thermal = sqrt(k_B * T / m)
        v_thermal = np.sqrt(BOLTZMANN_CONSTANT * temperature / mass_kg)
        
        # Convert to nm/fs: 1 m/s = 1e-6 nm/fs
        v_thermal_nm_fs = v_thermal * 1e-6
        
        velocities = torch.randn(n_particles, 3) * v_thermal_nm_fs
        
        # Generate trajectory with collective oscillation
        states = []
        times = np.arange(0, duration, dt)
        
        # Oscillation amplitude (make it visible for testing)
        # Scale amplitude based on box size for detectability
        amplitude = box_size * 0.05  # 5% of box size
        
        for t in times:
            # Add collective oscillation in x direction
            # x(t) = x_0 + A sin(ω_p t)
            # Convert time from fs to seconds for omega_p calculation
            t_seconds = t * 1e-15
            displacement = amplitude * np.sin(omega_p * t_seconds)
            
            # Apply displacement to all particles
            pos_t = positions.clone()
            pos_t[:, 0] += displacement
            
            # Create particle states
            particles = []
            for i in range(n_particles):
                particle = ParticleState(
                    position=pos_t[i],
                    momentum=velocities[i] * mass_amu,  # p = m * v
                    charge=1.0,  # Elementary charge units
                    mass=mass_amu,
                    species=list(MATERIAL_MASSES.keys()).index(material)
                )
                particles.append(particle)
            
            # Compute average kinetic energy
            avg_energy = 0.5 * mass_amu * (velocities**2).sum(dim=1).mean().item()
            
            # Create beam configuration
            config = BeamConfiguration(
                particles=particles,
                density=density,
                energy=avg_energy,
                material=material,
                temperature=temperature,
                time=t
            )
            states.append(config)
        
        # Store ground truth in metadata
        metadata = {
            'type': 'plasma_oscillation',
            'ground_truth_frequency': freq_THz,
            'amplitude': amplitude,
            'omega_p_rad_per_s': omega_p,
            'material': material,
            'n_particles': n_particles,
            'density': density,
            'temperature': temperature
        }
        
        return TrajectoryData(
            states=states,
            times=times.tolist(),
            metadata=metadata
        )


class BeamInstabilityGenerator:
    """Generate beam with known instability growth rate.
    
    Simulates two-stream instability where two counter-propagating beams
    develop a growing perturbation with known growth rate.
    
    Growth rate: γ ≈ (ω_p / 2) * (k v_0 / ω_p)^(1/3)
    """
    
    def generate(self,
                 n_particles: int = 1000,
                 beam_velocity: float = 1.0,
                 density: float = 1.0,
                 duration: float = 100.0,
                 dt: float = 0.1,
                 material: str = 'Si') -> TrajectoryData:
        """Generate two-stream instability trajectory.
        
        Args:
            n_particles: Number of particles (split equally between beams)
            beam_velocity: Beam velocity in nm/fs
            density: Beam density in particles/nm³
            duration: Simulation duration in femtoseconds
            dt: Timestep in femtoseconds
            material: Material type for particle properties
            
        Returns:
            TrajectoryData with ground truth growth rate in metadata
        """
        # Compute plasma frequency for growth rate calculation
        n_SI = density * 1e27
        mass_amu = MATERIAL_MASSES.get(material, 28.0)
        mass_kg = mass_amu * 1.66054e-27
        
        omega_p = np.sqrt(n_SI * ELECTRON_CHARGE**2 / (EPSILON_0 * mass_kg))
        omega_p_normalized = omega_p * 1e-15  # Convert to 1/fs
        
        # Wavenumber (choose characteristic scale)
        box_size = (n_particles / density) ** (1/3)
        k = 2 * np.pi / (box_size / 5)  # Wavelength ~ box_size/5
        
        # Growth rate (simplified two-stream formula)
        # γ ≈ (ω_p / 2) * (k v_0 / ω_p)^(1/3)
        gamma = (omega_p_normalized / 2) * (k * beam_velocity / omega_p_normalized) ** (1/3)
        
        # Initialize two counter-propagating beams
        n_per_beam = n_particles // 2
        
        # Random positions in box
        positions = torch.rand(n_particles, 3) * box_size
        velocities = torch.zeros(n_particles, 3)
        
        # Beam 1: moving in +x direction
        velocities[:n_per_beam, 0] = beam_velocity
        
        # Beam 2: moving in -x direction
        velocities[n_per_beam:, 0] = -beam_velocity
        
        # Generate trajectory with growing instability
        states = []
        times = np.arange(0, duration, dt)
        
        # Initial perturbation amplitude
        initial_amplitude = 0.01  # nm
        
        for t in times:
            # Growing perturbation: δn(t) = δn_0 * exp(γt) * sin(kx)
            amplitude = initial_amplitude * np.exp(gamma * t)
            
            # Apply perturbation to positions
            pos_t = positions.clone()
            for i in range(n_particles):
                perturbation = amplitude * np.sin(k * positions[i, 0].item())
                pos_t[i, 1] += perturbation  # Perturb in y direction
            
            # Update positions (ballistic motion)
            positions = positions + velocities * dt
            
            # Periodic boundary conditions
            positions = positions % box_size
            
            # Create particle states
            particles = []
            for i in range(n_particles):
                particle = ParticleState(
                    position=pos_t[i],
                    momentum=velocities[i] * mass_amu,
                    charge=1.0,
                    mass=mass_amu,
                    species=list(MATERIAL_MASSES.keys()).index(material)
                )
                particles.append(particle)
            
            # Compute average kinetic energy
            avg_energy = 0.5 * mass_amu * (velocities**2).sum(dim=1).mean().item()
            
            # Create beam configuration
            config = BeamConfiguration(
                particles=particles,
                density=density,
                energy=avg_energy,
                material=material,
                temperature=0.0,  # Cold beams
                time=t
            )
            states.append(config)
        
        # Store ground truth in metadata
        metadata = {
            'type': 'beam_instability',
            'ground_truth_growth_rate': gamma,
            'beam_velocity': beam_velocity,
            'wavenumber': k,
            'initial_amplitude': initial_amplitude,
            'material': material,
            'n_particles': n_particles,
            'density': density
        }
        
        return TrajectoryData(
            states=states,
            times=times.tolist(),
            metadata=metadata
        )


class SingleParticleGenerator:
    """Generate non-interacting particles for baseline comparison.
    
    Creates trajectories with pure single-particle dynamics (ballistic motion)
    with no collective effects. Used as a baseline to verify that the model
    can distinguish collective from non-collective behavior.
    """
    
    def generate(self,
                 n_particles: int = 1000,
                 density: float = 0.001,
                 temperature: float = 300.0,
                 duration: float = 100.0,
                 dt: float = 0.1,
                 material: str = 'Si') -> TrajectoryData:
        """Generate independent particle trajectories.
        
        Args:
            n_particles: Number of particles
            density: Low density in particles/nm³ (default very low)
            temperature: Temperature in Kelvin
            duration: Simulation duration in femtoseconds
            dt: Timestep in femtoseconds
            material: Material type for particle properties
            
        Returns:
            TrajectoryData with metadata indicating no collective effects
        """
        # Use material-specific mass
        mass_amu = MATERIAL_MASSES.get(material, 28.0)
        mass_kg = mass_amu * 1.66054e-27
        
        # Initialize particles in a large box (low density)
        box_size = (n_particles / density) ** (1/3)
        
        # Random initial positions
        positions = torch.rand(n_particles, 3) * box_size
        
        # Maxwell-Boltzmann velocity distribution
        v_thermal = np.sqrt(BOLTZMANN_CONSTANT * temperature / mass_kg)
        v_thermal_nm_fs = v_thermal * 1e-6  # Convert to nm/fs
        
        velocities = torch.randn(n_particles, 3) * v_thermal_nm_fs
        
        # Generate trajectory with pure ballistic motion
        states = []
        times = np.arange(0, duration, dt)
        
        for t in times:
            # Simple ballistic motion: r(t) = r_0 + v * t
            # (No interactions, no collective effects)
            pos_t = positions + velocities * t
            
            # Periodic boundary conditions
            pos_t = pos_t % box_size
            
            # Create particle states
            particles = []
            for i in range(n_particles):
                particle = ParticleState(
                    position=pos_t[i],
                    momentum=velocities[i] * mass_amu,
                    charge=1.0,
                    mass=mass_amu,
                    species=list(MATERIAL_MASSES.keys()).index(material)
                )
                particles.append(particle)
            
            # Compute average kinetic energy
            avg_energy = 0.5 * mass_amu * (velocities**2).sum(dim=1).mean().item()
            
            # Create beam configuration
            config = BeamConfiguration(
                particles=particles,
                density=density,
                energy=avg_energy,
                material=material,
                temperature=temperature,
                time=t
            )
            states.append(config)
        
        # Store metadata indicating no collective effects
        metadata = {
            'type': 'single_particle',
            'collective_mode': False,
            'material': material,
            'n_particles': n_particles,
            'density': density,
            'temperature': temperature
        }
        
        return TrajectoryData(
            states=states,
            times=times.tolist(),
            metadata=metadata
        )


def generate_synthetic_benchmarks(output_dir: str = 'data/synthetic',
                                  materials: List[str] = None) -> Dict[str, List[TrajectoryData]]:
    """Generate a complete set of synthetic benchmarks for all materials.
    
    Args:
        output_dir: Directory to save generated data
        materials: List of materials to generate (default: all supported)
        
    Returns:
        Dictionary mapping benchmark type to list of trajectories
    """
    if materials is None:
        materials = list(MATERIAL_MASSES.keys())
    
    benchmarks = {
        'plasma_oscillation': [],
        'beam_instability': [],
        'single_particle': []
    }
    
    plasma_gen = PlasmaGenerator()
    beam_gen = BeamInstabilityGenerator()
    single_gen = SingleParticleGenerator()
    
    for material in materials:
        # Generate plasma oscillation
        plasma_traj = plasma_gen.generate(
            n_particles=1000,
            density=1.0,
            temperature=300.0,
            duration=100.0,
            dt=0.1,
            material=material
        )
        benchmarks['plasma_oscillation'].append(plasma_traj)
        
        # Generate beam instability
        beam_traj = beam_gen.generate(
            n_particles=1000,
            beam_velocity=1.0,
            density=1.0,
            duration=100.0,
            dt=0.1,
            material=material
        )
        benchmarks['beam_instability'].append(beam_traj)
        
        # Generate single-particle baseline
        single_traj = single_gen.generate(
            n_particles=1000,
            density=0.001,
            temperature=300.0,
            duration=100.0,
            dt=0.1,
            material=material
        )
        benchmarks['single_particle'].append(single_traj)
    
    return benchmarks
