#!/usr/bin/env python3
"""Generate synthetic data in CERN high-speed storage location.

Supports resuming interrupted generation via checkpoint files.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.synthetic import (
    PlasmaGenerator,
    BeamInstabilityGenerator,
    SingleParticleGenerator
)
import pickle
import json
import argparse
from datetime import datetime


def load_checkpoint(checkpoint_path: Path) -> dict:
    """Load checkpoint file if it exists."""
    if checkpoint_path.exists():
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    return {"completed": [], "started_at": None, "last_updated": None}


def detect_existing_files(output_dir: Path, materials: list, scenarios: list) -> list:
    """Detect already-generated files and return list of completed tasks."""
    completed = []
    for material in materials:
        material_dir = output_dir / material
        for scenario in scenarios:
            pkl_file = material_dir / f"{scenario}.pkl"
            metadata_file = material_dir / f"{scenario}_metadata.json"
            if pkl_file.exists() and metadata_file.exists():
                completed.append(f"{material}/{scenario}")
    return completed


def load_or_rebuild_checkpoint(checkpoint_path: Path, output_dir: Path, materials: list, scenarios: list) -> dict:
    """Load checkpoint, or rebuild it from existing files if missing."""
    checkpoint = load_checkpoint(checkpoint_path)
    
    # If no checkpoint but files exist, rebuild from existing data
    if not checkpoint["completed"]:
        existing = detect_existing_files(output_dir, materials, scenarios)
        if existing:
            print(f"No checkpoint found, but detected {len(existing)} existing files. Rebuilding checkpoint...")
            checkpoint["completed"] = existing
            checkpoint["started_at"] = datetime.now().isoformat()
            save_checkpoint(checkpoint_path, checkpoint)
    
    return checkpoint


def save_checkpoint(checkpoint_path: Path, checkpoint: dict):
    """Save checkpoint file."""
    checkpoint["last_updated"] = datetime.now().isoformat()
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint, f, indent=2)


def is_completed(checkpoint: dict, material: str, scenario: str) -> bool:
    """Check if a specific material/scenario combination is already done."""
    task_id = f"{material}/{scenario}"
    return task_id in checkpoint["completed"]


def mark_completed(checkpoint: dict, material: str, scenario: str):
    """Mark a material/scenario combination as completed."""
    task_id = f"{material}/{scenario}"
    if task_id not in checkpoint["completed"]:
        checkpoint["completed"].append(task_id)


def generate_small_dataset(output_dir: Path, resume: bool = True):
    """Generate small dataset (100 particles, 50 timesteps)."""
    print("Generating small synthetic dataset...")
    print("Parameters: 100 particles, 50 timesteps")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    materials = ['Si', 'Fe', 'W', 'Cu', 'Al']
    scenarios = ['plasma_oscillation', 'beam_instability', 'single_particle']
    
    checkpoint_path = output_dir / ".checkpoint.json"
    if resume:
        checkpoint = load_or_rebuild_checkpoint(checkpoint_path, output_dir, materials, scenarios)
    else:
        checkpoint = {"completed": [], "started_at": None, "last_updated": None}
    
    if checkpoint["started_at"] is None:
        checkpoint["started_at"] = datetime.now().isoformat()
    
    # Count completed vs total
    total_tasks = len(materials) * len(scenarios)
    completed_count = len(checkpoint["completed"])
    
    if resume and completed_count > 0:
        print(f"\nResuming: {completed_count}/{total_tasks} tasks already completed")
        for task in checkpoint["completed"]:
            print(f"  ✓ {task}")
    
    plasma_gen = PlasmaGenerator()
    beam_gen = BeamInstabilityGenerator()
    single_gen = SingleParticleGenerator()
    
    for material in materials:
        material_dir = output_dir / material
        material_dir.mkdir(exist_ok=True)
        
        # Plasma oscillation
        if not is_completed(checkpoint, material, 'plasma_oscillation'):
            print(f"\n[{material}] Generating plasma oscillation...")
            plasma_traj = plasma_gen.generate(
                n_particles=100,
                density=1.0,
                temperature=300.0,
                duration=25.0,
                dt=0.5,
                material=material
            )
            
            with open(material_dir / "plasma_oscillation.pkl", 'wb') as f:
                pickle.dump(plasma_traj, f)
            with open(material_dir / "plasma_oscillation_metadata.json", 'w') as f:
                json.dump(plasma_traj.metadata, f, indent=2)
            
            mark_completed(checkpoint, material, 'plasma_oscillation')
            save_checkpoint(checkpoint_path, checkpoint)
            print(f"  ✓ Saved ({len(plasma_traj.states)} timesteps)")
        else:
            print(f"\n[{material}] Skipping plasma oscillation (already completed)")
        
        # Beam instability
        if not is_completed(checkpoint, material, 'beam_instability'):
            print(f"[{material}] Generating beam instability...")
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
            
            mark_completed(checkpoint, material, 'beam_instability')
            save_checkpoint(checkpoint_path, checkpoint)
            print(f"  ✓ Saved ({len(beam_traj.states)} timesteps)")
        else:
            print(f"[{material}] Skipping beam instability (already completed)")
        
        # Single particle
        if not is_completed(checkpoint, material, 'single_particle'):
            print(f"[{material}] Generating single particle...")
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
            
            mark_completed(checkpoint, material, 'single_particle')
            save_checkpoint(checkpoint_path, checkpoint)
            print(f"  ✓ Saved ({len(single_traj.states)} timesteps)")
        else:
            print(f"[{material}] Skipping single particle (already completed)")
    
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
        },
        "generation_started": checkpoint["started_at"],
        "generation_completed": datetime.now().isoformat()
    }
    
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Generated 15 small trajectories in {output_dir}")
    print("  - 5 materials × 3 scenarios")
    print("  - 100 particles per simulation")
    print("  - 50 timesteps each")
    print("\nReady for training!")


def generate_medium_dataset(output_dir: Path, resume: bool = True):
    """Generate medium dataset (250 particles, 150 timesteps) - optimized for 24GB GPU."""
    print("Generating medium synthetic dataset...")
    print("Parameters: 250 particles, 150 timesteps")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    materials = ['Si', 'Fe', 'W', 'Cu', 'Al']
    scenarios = ['plasma_oscillation', 'beam_instability', 'single_particle']
    
    checkpoint_path = output_dir / ".checkpoint.json"
    if resume:
        checkpoint = load_or_rebuild_checkpoint(checkpoint_path, output_dir, materials, scenarios)
    else:
        checkpoint = {"completed": [], "started_at": None, "last_updated": None}
    
    if checkpoint["started_at"] is None:
        checkpoint["started_at"] = datetime.now().isoformat()
    
    # Count completed vs total
    total_tasks = len(materials) * len(scenarios)
    completed_count = len(checkpoint["completed"])
    
    if resume and completed_count > 0:
        print(f"\nResuming: {completed_count}/{total_tasks} tasks already completed")
        for task in checkpoint["completed"]:
            print(f"  ✓ {task}")
    
    plasma_gen = PlasmaGenerator()
    beam_gen = BeamInstabilityGenerator()
    single_gen = SingleParticleGenerator()
    
    for material in materials:
        material_dir = output_dir / material
        material_dir.mkdir(exist_ok=True)
        
        # Plasma oscillation - 250 particles, 150 timesteps
        if not is_completed(checkpoint, material, 'plasma_oscillation'):
            print(f"\n[{material}] Generating plasma oscillation...")
            plasma_traj = plasma_gen.generate(
                n_particles=250,
                density=1.0,
                temperature=300.0,
                duration=37.5,  # 150 timesteps * 0.25 dt
                dt=0.25,
                material=material
            )
            
            with open(material_dir / "plasma_oscillation.pkl", 'wb') as f:
                pickle.dump(plasma_traj, f)
            with open(material_dir / "plasma_oscillation_metadata.json", 'w') as f:
                json.dump(plasma_traj.metadata, f, indent=2)
            
            mark_completed(checkpoint, material, 'plasma_oscillation')
            save_checkpoint(checkpoint_path, checkpoint)
            print(f"  ✓ Saved ({len(plasma_traj.states)} timesteps)")
        else:
            print(f"\n[{material}] Skipping plasma oscillation (already completed)")
        
        # Beam instability
        if not is_completed(checkpoint, material, 'beam_instability'):
            print(f"[{material}] Generating beam instability...")
            beam_traj = beam_gen.generate(
                n_particles=250,
                beam_velocity=1.0,
                density=1.0,
                duration=37.5,
                dt=0.25,
                material=material
            )
            
            with open(material_dir / "beam_instability.pkl", 'wb') as f:
                pickle.dump(beam_traj, f)
            with open(material_dir / "beam_instability_metadata.json", 'w') as f:
                json.dump(beam_traj.metadata, f, indent=2)
            
            mark_completed(checkpoint, material, 'beam_instability')
            save_checkpoint(checkpoint_path, checkpoint)
            print(f"  ✓ Saved ({len(beam_traj.states)} timesteps)")
        else:
            print(f"[{material}] Skipping beam instability (already completed)")
        
        # Single particle
        if not is_completed(checkpoint, material, 'single_particle'):
            print(f"[{material}] Generating single particle...")
            single_traj = single_gen.generate(
                n_particles=250,
                density=0.01,
                duration=37.5,
                dt=0.25,
                material=material
            )
            
            with open(material_dir / "single_particle.pkl", 'wb') as f:
                pickle.dump(single_traj, f)
            with open(material_dir / "single_particle_metadata.json", 'w') as f:
                json.dump(single_traj.metadata, f, indent=2)
            
            mark_completed(checkpoint, material, 'single_particle')
            save_checkpoint(checkpoint_path, checkpoint)
            print(f"  ✓ Saved ({len(single_traj.states)} timesteps)")
        else:
            print(f"[{material}] Skipping single particle (already completed)")
    
    # Save summary
    summary = {
        "materials": materials,
        "n_particles": 250,
        "duration": 37.5,
        "dt": 0.25,
        "num_timesteps": 150,
        "benchmarks": {
            "plasma_oscillation": "Plasma with known plasmon frequency",
            "beam_instability": "Two-stream instability with known growth rate",
            "single_particle": "Non-interacting particles (baseline)"
        },
        "generation_started": checkpoint["started_at"],
        "generation_completed": datetime.now().isoformat()
    }
    
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Generated 15 medium trajectories in {output_dir}")
    print("  - 5 materials × 3 scenarios")
    print("  - 250 particles per simulation")
    print("  - 150 timesteps each")
    print("\nReady for training!")


def generate_comprehensive_dataset(output_dir: Path, resume: bool = True):
    """Generate comprehensive dataset (1000 particles, 1000 timesteps)."""
    print("Generating comprehensive synthetic dataset...")
    print("Parameters: 1000 particles, 1000 timesteps")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    materials = ['Si', 'Fe', 'W', 'Cu', 'Al']
    scenarios = ['plasma_oscillation', 'beam_instability', 'single_particle']
    
    checkpoint_path = output_dir / ".checkpoint.json"
    if resume:
        checkpoint = load_or_rebuild_checkpoint(checkpoint_path, output_dir, materials, scenarios)
    else:
        checkpoint = {"completed": [], "started_at": None, "last_updated": None}
    
    if checkpoint["started_at"] is None:
        checkpoint["started_at"] = datetime.now().isoformat()
    
    # Count completed vs total
    total_tasks = len(materials) * len(scenarios)
    completed_count = len(checkpoint["completed"])
    
    if resume and completed_count > 0:
        print(f"\nResuming: {completed_count}/{total_tasks} tasks already completed")
        for task in checkpoint["completed"]:
            print(f"  ✓ {task}")
    
    plasma_gen = PlasmaGenerator()
    beam_gen = BeamInstabilityGenerator()
    single_gen = SingleParticleGenerator()
    
    for material in materials:
        material_dir = output_dir / material
        material_dir.mkdir(exist_ok=True)
        
        # Plasma oscillation
        if not is_completed(checkpoint, material, 'plasma_oscillation'):
            print(f"\n[{material}] Generating plasma oscillation...")
            plasma_traj = plasma_gen.generate(
                n_particles=1000,
                density=1.0,
                temperature=300.0,
                duration=100.0,
                dt=0.1,
                material=material
            )
            
            with open(material_dir / "plasma_oscillation.pkl", 'wb') as f:
                pickle.dump(plasma_traj, f)
            with open(material_dir / "plasma_oscillation_metadata.json", 'w') as f:
                json.dump(plasma_traj.metadata, f, indent=2)
            
            mark_completed(checkpoint, material, 'plasma_oscillation')
            save_checkpoint(checkpoint_path, checkpoint)
            print(f"  ✓ Saved ({len(plasma_traj.states)} timesteps)")
        else:
            print(f"\n[{material}] Skipping plasma oscillation (already completed)")
        
        # Beam instability
        if not is_completed(checkpoint, material, 'beam_instability'):
            print(f"[{material}] Generating beam instability...")
            beam_traj = beam_gen.generate(
                n_particles=1000,
                beam_velocity=1.0,
                density=1.0,
                duration=100.0,
                dt=0.1,
                material=material
            )
            
            with open(material_dir / "beam_instability.pkl", 'wb') as f:
                pickle.dump(beam_traj, f)
            with open(material_dir / "beam_instability_metadata.json", 'w') as f:
                json.dump(beam_traj.metadata, f, indent=2)
            
            mark_completed(checkpoint, material, 'beam_instability')
            save_checkpoint(checkpoint_path, checkpoint)
            print(f"  ✓ Saved ({len(beam_traj.states)} timesteps)")
        else:
            print(f"[{material}] Skipping beam instability (already completed)")
        
        # Single particle
        if not is_completed(checkpoint, material, 'single_particle'):
            print(f"[{material}] Generating single particle...")
            single_traj = single_gen.generate(
                n_particles=1000,
                density=0.001,
                duration=100.0,
                dt=0.1,
                material=material
            )
            
            with open(material_dir / "single_particle.pkl", 'wb') as f:
                pickle.dump(single_traj, f)
            with open(material_dir / "single_particle_metadata.json", 'w') as f:
                json.dump(single_traj.metadata, f, indent=2)
            
            mark_completed(checkpoint, material, 'single_particle')
            save_checkpoint(checkpoint_path, checkpoint)
            print(f"  ✓ Saved ({len(single_traj.states)} timesteps)")
        else:
            print(f"[{material}] Skipping single particle (already completed)")
    
    # Save summary
    summary = {
        "materials": materials,
        "n_particles": 1000,
        "duration": 100.0,
        "dt": 0.1,
        "num_timesteps": 1000,
        "benchmarks": {
            "plasma_oscillation": "Plasma with known plasmon frequency",
            "beam_instability": "Two-stream instability with known growth rate",
            "single_particle": "Non-interacting particles (baseline)"
        },
        "generation_started": checkpoint["started_at"],
        "generation_completed": datetime.now().isoformat()
    }
    
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Generated 15 comprehensive trajectories in {output_dir}")
    print("  - 5 materials × 3 scenarios")
    print("  - 1000 particles per simulation")
    print("  - 1000 timesteps each")
    print("\nReady for training!")


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic data for CERN training')
    parser.add_argument('--output-dir', type=str, default='/root/highspeedstorage/CERN/data',
                        help='Output directory for generated data')
    parser.add_argument('--size', type=str, choices=['small', 'medium', 'comprehensive'], default='small',
                        help='Dataset size: small (100p/50t), medium (250p/150t), comprehensive (1000p/1000t)')
    parser.add_argument('--resume', action='store_true', default=True,
                        help='Resume from checkpoint if interrupted (default: True)')
    parser.add_argument('--no-resume', action='store_true',
                        help='Start fresh, ignoring any existing checkpoint')
    
    args = parser.parse_args()
    
    resume = args.resume and not args.no_resume
    
    output_base = Path(args.output_dir)
    
    if args.size == 'small':
        output_dir = output_base / 'synthetic_small'
        generate_small_dataset(output_dir, resume=resume)
    elif args.size == 'medium':
        output_dir = output_base / 'synthetic_medium'
        generate_medium_dataset(output_dir, resume=resume)
    else:
        output_dir = output_base / 'synthetic_comprehensive'
        generate_comprehensive_dataset(output_dir, resume=resume)


if __name__ == '__main__':
    main()
