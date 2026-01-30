"""Generate small synthetic data for quick training."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.synthetic import (
    PlasmaGenerator,
    BeamInstabilityGenerator,
    SingleParticleGenerator
)
import pickle
import json

def main():
    print("Generating small synthetic dataset...")
    print("Parameters: 100 particles, 50 timesteps")
    
    output_dir = Path("data/synthetic_small")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    materials = ['Si', 'Fe', 'W', 'Cu', 'Al']
    
    plasma_gen = PlasmaGenerator()
    beam_gen = BeamInstabilityGenerator()
    single_gen = SingleParticleGenerator()
    
    for material in materials:
        print(f"\nGenerating {material}...")
        material_dir = output_dir / material
        material_dir.mkdir(exist_ok=True)
        
        # Plasma oscillation - 100 particles, 50 timesteps
        print(f"  - Plasma oscillation...")
        plasma_traj = plasma_gen.generate(
            n_particles=100,
            density=1.0,
            temperature=300.0,
            duration=25.0,  # 50 timesteps * 0.5 dt
            dt=0.5,
            material=material
        )
        
        with open(material_dir / "plasma_oscillation.pkl", 'wb') as f:
            pickle.dump(plasma_traj, f)
        with open(material_dir / "plasma_oscillation_metadata.json", 'w') as f:
            json.dump(plasma_traj.metadata, f, indent=2)
        print(f"    Saved ({len(plasma_traj.states)} timesteps)")
        
        # Beam instability
        print(f"  - Beam instability...")
        beam_traj = beam_gen.generate(
            n_particles=100,
            beam_velocity=1.0,
            density=1.0,
            duration=25.0,
            dt=0.5,
            material=material
        )
        
        with open(material_dir / "beam_instability.pkl", 'wb') as f:
            pickle.dump(beam_traj, f)
        with open(material_dir / "beam_instability_metadata.json", 'w') as f:
            json.dump(beam_traj.metadata, f, indent=2)
        print(f"    Saved ({len(beam_traj.states)} timesteps)")
        
        # Single particle
        print(f"  - Single particle...")
        single_traj = single_gen.generate(
            n_particles=100,
            density=1.0,
            duration=25.0,
            dt=0.5,
            material=material
        )
        
        with open(material_dir / "single_particle.pkl", 'wb') as f:
            pickle.dump(single_traj, f)
        with open(material_dir / "single_particle_metadata.json", 'w') as f:
            json.dump(single_traj.metadata, f, indent=2)
        print(f"    Saved ({len(single_traj.states)} timesteps)")
    
    # Save summary
    summary = {
        "materials": materials,
        "n_particles": 100,
        "duration": 25.0,
        "dt": 0.5,
        "num_timesteps": 50,
        "benchmarks": {
            "plasma_oscillation": "Plasma with known plasmon frequency",
            "beam_instability": "Two-stream instability with known growth rate",
            "single_particle": "Non-interacting particles (baseline)"
        }
    }
    
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Generated 15 small trajectories in {output_dir}")
    print("  - 5 materials × 3 scenarios")
    print("  - 100 particles per simulation")
    print("  - 50 timesteps each")
    print("\nReady for training!")


if __name__ == '__main__':
    main()
