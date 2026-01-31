"""Script to generate hypotheses about collective phenomena.

This script loads a trained model, scans density space to detect transitions,
extracts collective modes, predicts experimental signatures, and outputs
CollectiveMode objects with uncertainties.

Usage:
    python scripts/generate_hypotheses.py --checkpoint checkpoints/model.pt --config configs/model/default.yaml --material Si --output results/hypotheses
"""

import argparse
import os
import sys
import json
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional
import torch
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.multi_scale_model import MultiScaleModel
from src.hypothesis.scanning import (
    full_density_scan,
    characterize_collective_mode_with_signatures,
    ScanResult
)
from src.data.structures import BeamConfiguration, ParticleState, CollectiveMode
from src.config.config_loader import load_config


def load_model(checkpoint_path: str, config_path: str = None, device: str = 'cpu') -> MultiScaleModel:
    """Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to model configuration (optional)
        device: Device to load model on
    
    Returns:
        Loaded MultiScaleModel
    """
    print(f"Loading model from {checkpoint_path}...")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load config from file - need the model sub-config
    full_config = load_config('config')
    config = full_config.model  # Extract model config
    
    # Initialize model
    model = MultiScaleModel(config)
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully (epoch {checkpoint.get('epoch', 'unknown')})")
    
    return model


def create_base_configuration(
    material: str = 'Si',
    n_particles: int = 1000,
    base_density: float = 0.1,
    energy: float = 1.0,
    temperature: float = 300.0
) -> BeamConfiguration:
    """Create a base beam configuration for scanning.
    
    Args:
        material: Material type ('Si', 'Fe', 'W', 'Cu', 'Al')
        n_particles: Number of particles
        base_density: Base density (particles/nm³)
        energy: Beam energy (MeV)
        temperature: Temperature (K)
    
    Returns:
        BeamConfiguration object
    """
    print(f"Creating base configuration:")
    print(f"  Material: {material}")
    print(f"  Particles: {n_particles}")
    print(f"  Base density: {base_density} particles/nm³")
    print(f"  Energy: {energy} MeV")
    print(f"  Temperature: {temperature} K")
    
    # Material properties
    material_masses = {
        'Si': 28.0855,
        'Fe': 55.845,
        'W': 183.84,
        'Cu': 63.546,
        'Al': 26.9815
    }
    
    material_species = {
        'Si': 0,
        'Fe': 1,
        'W': 2,
        'Cu': 3,
        'Al': 4
    }
    
    mass = material_masses.get(material, 28.0)
    species = material_species.get(material, 0)
    
    # Initialize particles in a cubic box
    box_size = (n_particles / base_density) ** (1/3)
    
    # Random initial positions
    positions = torch.rand(n_particles, 3) * box_size
    
    # Maxwell-Boltzmann velocity distribution
    # v_thermal = sqrt(k_B * T / m)
    k_B = 1.381e-23  # J/K
    mass_kg = mass * 1.66054e-27  # Convert AMU to kg
    v_thermal = np.sqrt(k_B * temperature / mass_kg)
    v_thermal_nm_fs = v_thermal * 1e-6  # Convert to nm/fs
    
    velocities = torch.randn(n_particles, 3) * v_thermal_nm_fs
    
    # Create particle states
    particles = []
    for i in range(n_particles):
        particle = ParticleState(
            position=positions[i],
            momentum=velocities[i] * mass,  # p = m * v
            charge=1.0,  # Elementary charge units
            mass=mass,
            species=species
        )
        particles.append(particle)
    
    # Create beam configuration
    config = BeamConfiguration(
        particles=particles,
        density=base_density,
        energy=energy,
        material=material,
        temperature=temperature,
        time=0.0
    )
    
    return config


def scan_density_and_detect_transitions(
    model: MultiScaleModel,
    base_config: BeamConfiguration,
    density_min: float = 0.01,
    density_max: float = 10.0,
    num_densities: int = 50,
    detection_threshold: float = 2.0,
    device: str = 'cpu'
) -> ScanResult:
    """Scan density space and detect collective transitions.
    
    Args:
        model: Trained model
        base_config: Base beam configuration
        density_min: Minimum density to scan (particles/nm³)
        density_max: Maximum density to scan (particles/nm³)
        num_densities: Number of density points to scan
        detection_threshold: Z-score threshold for transition detection
        device: Device to run on
    
    Returns:
        ScanResult with densities, latent states, anomaly scores, and critical densities
    """
    print("\nScanning density space...")
    print(f"  Density range: {density_min} to {density_max} particles/nm³")
    print(f"  Number of points: {num_densities}")
    print(f"  Detection threshold: {detection_threshold} σ")
    
    # Create density range
    density_range = np.linspace(density_min, density_max, num_densities)
    
    # Perform full density scan
    scan_result = full_density_scan(
        model=model,
        base_config=base_config,
        density_range=density_range,
        detection_threshold=detection_threshold,
        device=device
    )
    
    print(f"\nScan complete!")
    print(f"  Detected {len(scan_result.critical_densities)} transition(s)")
    
    if scan_result.critical_densities:
        print(f"  Critical densities:")
        for i, rho_c in enumerate(scan_result.critical_densities):
            print(f"    {i+1}. ρ_c = {rho_c:.4f} particles/nm³")
    else:
        print(f"  No transitions detected above threshold")
    
    return scan_result


def extract_collective_modes(
    model: MultiScaleModel,
    base_config: BeamConfiguration,
    critical_densities: List[float],
    device: str = 'cpu'
) -> List[CollectiveMode]:
    """Extract collective mode properties at each critical density.
    
    Args:
        model: Trained model
        base_config: Base beam configuration
        critical_densities: List of critical densities
        device: Device to run on
    
    Returns:
        List of CollectiveMode objects
    """
    print("\nExtracting collective modes...")
    
    modes = []
    
    for i, rho_c in enumerate(critical_densities):
        print(f"\n  Mode {i+1} at ρ_c = {rho_c:.4f} particles/nm³:")
        
        # Characterize collective mode
        mode = characterize_collective_mode_with_signatures(
            model=model,
            base_config=base_config,
            critical_density=rho_c,
            epistemic_uncertainty=0.0,  # Placeholder - would come from ensemble
            device=device
        )
        
        # Print mode properties
        print(f"    Frequency: {mode.frequency:.4f} THz")
        print(f"    Damping rate: {mode.damping_rate:.6f} 1/fs")
        print(f"    Group velocity: {mode.group_velocity:.4f} nm/fs")
        print(f"    Scattering peak: {mode.scattering_peak:.4f} THz")
        print(f"    Energy spread: {mode.energy_spread:.4f} MeV")
        print(f"    Correlation length: {mode.correlation_length:.4f} nm")
        
        modes.append(mode)
    
    return modes


def save_results(
    scan_result: ScanResult,
    modes: List[CollectiveMode],
    base_config: BeamConfiguration,
    output_dir: str
):
    """Save hypothesis generation results.
    
    Args:
        scan_result: Density scan results
        modes: List of extracted collective modes
        base_config: Base beam configuration
        output_dir: Output directory
    """
    print(f"\nSaving results to {output_dir}...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save scan results
    scan_data = {
        'densities': scan_result.densities.tolist(),
        'anomaly_scores': scan_result.anomaly_scores.tolist(),
        'critical_densities': scan_result.critical_densities,
        'material': base_config.material,
        'energy': base_config.energy,
        'temperature': base_config.temperature
    }
    
    with open(output_path / 'density_scan.json', 'w') as f:
        json.dump(scan_data, f, indent=2)
    
    print(f"  Saved density scan to density_scan.json")
    
    # Save latent states
    torch.save(scan_result.latent_states, output_path / 'latent_states.pt')
    print(f"  Saved latent states to latent_states.pt")
    
    # Save collective modes
    for i, mode in enumerate(modes):
        mode_data = {
            'frequency': mode.frequency,
            'damping_rate': mode.damping_rate,
            'critical_density': mode.critical_density,
            'group_velocity': mode.group_velocity,
            'uncertainty': mode.uncertainty,
            'scattering_peak': mode.scattering_peak,
            'energy_spread': mode.energy_spread,
            'correlation_length': mode.correlation_length
        }
        
        with open(output_path / f'mode_{i+1}.json', 'w') as f:
            json.dump(mode_data, f, indent=2)
        
        # Save full mode object as pickle
        with open(output_path / f'mode_{i+1}.pkl', 'wb') as f:
            pickle.dump(mode, f)
        
        print(f"  Saved mode {i+1} to mode_{i+1}.json and mode_{i+1}.pkl")
    
    # Generate summary report
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("HYPOTHESIS GENERATION REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append(f"Material: {base_config.material}")
    report_lines.append(f"Energy: {base_config.energy} MeV")
    report_lines.append(f"Temperature: {base_config.temperature} K")
    report_lines.append("")
    report_lines.append(f"Density scan: {scan_result.densities[0]:.4f} to {scan_result.densities[-1]:.4f} particles/nm³")
    report_lines.append(f"Number of points: {len(scan_result.densities)}")
    report_lines.append("")
    report_lines.append(f"Detected transitions: {len(scan_result.critical_densities)}")
    report_lines.append("")
    
    if modes:
        report_lines.append("COLLECTIVE MODES")
        report_lines.append("-" * 80)
        
        for i, mode in enumerate(modes):
            report_lines.append(f"\nMode {i+1}:")
            report_lines.append(f"  Critical density: {mode.critical_density:.4f} particles/nm³")
            report_lines.append(f"  Frequency: {mode.frequency:.4f} THz")
            report_lines.append(f"  Damping rate: {mode.damping_rate:.6f} 1/fs")
            report_lines.append(f"  Group velocity: {mode.group_velocity:.4f} nm/fs")
            report_lines.append(f"  Uncertainty: {mode.uncertainty:.4f}")
            report_lines.append("")
            report_lines.append(f"  Experimental signatures:")
            report_lines.append(f"    Scattering peak: {mode.scattering_peak:.4f} THz")
            report_lines.append(f"    Energy spread: {mode.energy_spread:.4f} MeV")
            report_lines.append(f"    Correlation length: {mode.correlation_length:.4f} nm")
    else:
        report_lines.append("No collective modes detected")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    
    report_text = "\n".join(report_lines)
    
    with open(output_path / 'hypothesis_report.txt', 'w') as f:
        f.write(report_text)
    
    print(f"  Saved report to hypothesis_report.txt")
    
    # Print report to console
    print("\n" + report_text)


def main():
    parser = argparse.ArgumentParser(
        description='Generate hypotheses about collective phenomena'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to model configuration (optional if checkpoint contains config)'
    )
    
    parser.add_argument(
        '--material',
        type=str,
        default='Si',
        choices=['Si', 'Fe', 'W', 'Cu', 'Al'],
        help='Material to analyze (default: Si)'
    )
    
    parser.add_argument(
        '--n-particles',
        type=int,
        default=1000,
        help='Number of particles in simulation (default: 1000)'
    )
    
    parser.add_argument(
        '--energy',
        type=float,
        default=1.0,
        help='Beam energy in MeV (default: 1.0)'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        default=300.0,
        help='Temperature in Kelvin (default: 300.0)'
    )
    
    parser.add_argument(
        '--density-min',
        type=float,
        default=0.01,
        help='Minimum density to scan in particles/nm³ (default: 0.01)'
    )
    
    parser.add_argument(
        '--density-max',
        type=float,
        default=10.0,
        help='Maximum density to scan in particles/nm³ (default: 10.0)'
    )
    
    parser.add_argument(
        '--num-densities',
        type=int,
        default=50,
        help='Number of density points to scan (default: 50)'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=2.0,
        help='Z-score threshold for transition detection (default: 2.0)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='results/hypotheses',
        help='Output directory for results (default: results/hypotheses)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run on (default: cuda if available, else cpu)'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("HYPOTHESIS GENERATION")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    print(f"Output: {args.output}")
    print()
    
    # Load model
    model = load_model(args.checkpoint, args.config, args.device)
    
    # Create base configuration
    base_config = create_base_configuration(
        material=args.material,
        n_particles=args.n_particles,
        base_density=args.density_min,
        energy=args.energy,
        temperature=args.temperature
    )
    
    # Scan density space and detect transitions
    scan_result = scan_density_and_detect_transitions(
        model=model,
        base_config=base_config,
        density_min=args.density_min,
        density_max=args.density_max,
        num_densities=args.num_densities,
        detection_threshold=args.threshold,
        device=args.device
    )
    
    # Extract collective modes
    modes = []
    if scan_result.critical_densities:
        modes = extract_collective_modes(
            model=model,
            base_config=base_config,
            critical_densities=scan_result.critical_densities,
            device=args.device
        )
    else:
        print("\nNo transitions detected - skipping mode extraction")
    
    # Save results
    save_results(scan_result, modes, base_config, args.output)
    
    print("\nHypothesis generation complete!")


if __name__ == '__main__':
    main()
