"""Script to generate synthetic benchmarks with known collective behavior.

This script generates plasma oscillations, beam instabilities, and single-particle
baselines for all supported materials. The generated data includes ground truth
metadata for validation of the collective phenomena discovery methodology.

Usage:
    python scripts/generate_benchmarks.py --output data/synthetic --materials Si Fe W
"""

import argparse
import os
import sys
import pickle
import json
from pathlib import Path
from typing import List

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.synthetic import (
    PlasmaGenerator,
    BeamInstabilityGenerator,
    SingleParticleGenerator,
    MATERIAL_MASSES
)


def save_trajectory(trajectory, output_path: str):
    """Save trajectory data to disk.
    
    Args:
        trajectory: TrajectoryData object
        output_path: Path to save (without extension)
    """
    # Save as pickle for full Python object
    with open(f"{output_path}.pkl", 'wb') as f:
        pickle.dump(trajectory, f)
    
    # Save metadata as JSON for easy inspection
    with open(f"{output_path}_metadata.json", 'w') as f:
        json.dump(trajectory.metadata, f, indent=2)
    
    print(f"Saved trajectory to {output_path}.pkl")


def generate_all_benchmarks(output_dir: str,
                           materials: List[str],
                           n_particles: int = 1000,
                           duration: float = 100.0,
                           dt: float = 0.1):
    """Generate all synthetic benchmarks.
    
    Args:
        output_dir: Directory to save generated data
        materials: List of materials to generate
        n_particles: Number of particles per simulation
        duration: Simulation duration in femtoseconds
        dt: Timestep in femtoseconds
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize generators
    plasma_gen = PlasmaGenerator()
    beam_gen = BeamInstabilityGenerator()
    single_gen = SingleParticleGenerator()
    
    print(f"Generating synthetic benchmarks for materials: {materials}")
    print(f"Output directory: {output_dir}")
    print(f"Parameters: n_particles={n_particles}, duration={duration}fs, dt={dt}fs")
    print()
    
    # Generate for each material
    for material in materials:
        print(f"Generating benchmarks for {material}...")
        
        # Create material subdirectory
        material_dir = output_path / material
        material_dir.mkdir(exist_ok=True)
        
        # 1. Plasma oscillation
        print(f"  - Plasma oscillation...")
        plasma_traj = plasma_gen.generate(
            n_particles=n_particles,
            density=1.0,
            temperature=300.0,
            duration=duration,
            dt=dt,
            material=material
        )
        save_trajectory(
            plasma_traj,
            str(material_dir / "plasma_oscillation")
        )
        print(f"    Ground truth frequency: {plasma_traj.metadata['ground_truth_frequency']:.4f} THz")
        
        # 2. Beam instability
        print(f"  - Beam instability...")
        beam_traj = beam_gen.generate(
            n_particles=n_particles,
            beam_velocity=1.0,
            density=1.0,
            duration=duration,
            dt=dt,
            material=material
        )
        save_trajectory(
            beam_traj,
            str(material_dir / "beam_instability")
        )
        print(f"    Ground truth growth rate: {beam_traj.metadata['ground_truth_growth_rate']:.6f} 1/fs")
        
        # 3. Single-particle baseline
        print(f"  - Single-particle baseline...")
        single_traj = single_gen.generate(
            n_particles=n_particles,
            density=0.001,  # Very low density
            temperature=300.0,
            duration=duration,
            dt=dt,
            material=material
        )
        save_trajectory(
            single_traj,
            str(material_dir / "single_particle")
        )
        print(f"    No collective effects (baseline)")
        
        print()
    
    # Create summary file
    summary = {
        'materials': materials,
        'n_particles': n_particles,
        'duration': duration,
        'dt': dt,
        'benchmarks': {
            'plasma_oscillation': 'Plasma with known plasmon frequency',
            'beam_instability': 'Two-stream instability with known growth rate',
            'single_particle': 'Non-interacting particles (baseline)'
        }
    }
    
    with open(output_path / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Generation complete! Summary saved to {output_path / 'summary.json'}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic benchmarks for collective phenomena discovery'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/synthetic',
        help='Output directory for generated data (default: data/synthetic)'
    )
    
    parser.add_argument(
        '--materials',
        type=str,
        nargs='+',
        default=list(MATERIAL_MASSES.keys()),
        choices=list(MATERIAL_MASSES.keys()),
        help='Materials to generate (default: all)'
    )
    
    parser.add_argument(
        '--n-particles',
        type=int,
        default=1000,
        help='Number of particles per simulation (default: 1000)'
    )
    
    parser.add_argument(
        '--duration',
        type=float,
        default=100.0,
        help='Simulation duration in femtoseconds (default: 100.0)'
    )
    
    parser.add_argument(
        '--dt',
        type=float,
        default=0.1,
        help='Timestep in femtoseconds (default: 0.1)'
    )
    
    args = parser.parse_args()
    
    generate_all_benchmarks(
        output_dir=args.output,
        materials=args.materials,
        n_particles=args.n_particles,
        duration=args.duration,
        dt=args.dt
    )


if __name__ == '__main__':
    main()
