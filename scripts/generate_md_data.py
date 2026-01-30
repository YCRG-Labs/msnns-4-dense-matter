#!/usr/bin/env python3
"""
Generate comprehensive MD simulation data for training.

This script creates LAMMPS input files and manages batch MD simulations
for multiple materials, densities, energies, and temperatures.

Usage:
    python scripts/generate_md_data.py --output_dir data/md_simulations --num_configs 1000
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np


# Material properties (atomic mass in amu, charge in e)
MATERIALS = {
    'Si': {'mass': 28.0855, 'charge': 14, 'lattice_constant': 5.43},  # Angstroms
    'Fe': {'mass': 55.845, 'charge': 26, 'lattice_constant': 2.87},
    'W': {'mass': 183.84, 'charge': 74, 'lattice_constant': 3.16},
    'Cu': {'mass': 63.546, 'charge': 29, 'lattice_constant': 3.61},
    'Al': {'mass': 26.9815, 'charge': 13, 'lattice_constant': 4.05},
}


def generate_lammps_input(
    material: str,
    num_particles: int,
    density: float,  # particles/nm^3
    energy: float,  # MeV
    temperature: float,  # K
    timesteps: int,
    output_dir: Path,
    config_id: int
) -> Tuple[Path, Dict]:
    """
    Generate LAMMPS input script for a specific configuration.
    
    Args:
        material: Material name (Si, Fe, W, Cu, Al)
        num_particles: Number of particles
        density: Particle density in particles/nm^3
        energy: Beam energy in MeV
        temperature: Temperature in K
        timesteps: Number of simulation timesteps
        output_dir: Output directory
        config_id: Configuration ID for naming
    
    Returns:
        Tuple of (input_file_path, metadata_dict)
    """
    mat_props = MATERIALS[material]
    
    # Calculate box size from density
    # density is in particles/nm^3, convert to particles/Angstrom^3
    density_angstrom = density / 1000.0  # 1 nm^3 = 1000 Angstrom^3
    volume_angstrom = num_particles / density_angstrom
    box_length = volume_angstrom ** (1/3)
    
    # Convert energy to velocity (non-relativistic)
    # E = 0.5 * m * v^2
    # v = sqrt(2 * E / m)
    # E in MeV, m in amu, need v in Angstrom/fs
    energy_joules = energy * 1.60218e-13  # MeV to Joules
    mass_kg = mat_props['mass'] * 1.66054e-27  # amu to kg
    velocity_ms = np.sqrt(2 * energy_joules / mass_kg)  # m/s
    velocity_angstrom_fs = velocity_ms * 1e-5  # m/s to Angstrom/fs
    
    # Create output directory
    config_dir = output_dir / f"{material}_{config_id:04d}"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # LAMMPS input script
    input_file = config_dir / "input.lammps"
    
    lammps_script = f"""# LAMMPS input script for {material} beam simulation
# Configuration ID: {config_id}
# Material: {material}
# Particles: {num_particles}
# Density: {density:.3f} particles/nm^3
# Energy: {energy:.2f} MeV
# Temperature: {temperature:.1f} K

# Initialization
units metal
atom_style charge
dimension 3
boundary p p p

# Create simulation box
lattice fcc {mat_props['lattice_constant']}
region box block 0 {box_length/mat_props['lattice_constant']:.2f} 0 {box_length/mat_props['lattice_constant']:.2f} 0 {box_length/mat_props['lattice_constant']:.2f}
create_box 1 box
create_atoms 1 random {num_particles} 12345 box

# Set atom properties
mass 1 {mat_props['mass']}
set type 1 charge {mat_props['charge']}

# Interatomic potential (Lennard-Jones for simplicity)
# For production, use EAM or other appropriate potential
pair_style lj/cut 10.0
pair_coeff 1 1 0.1 3.0

# Initialize velocities
velocity all create {temperature} 87287 dist gaussian
velocity all set {velocity_angstrom_fs:.6f} 0.0 0.0 sum yes units box

# Thermostat (NVT ensemble)
fix 1 all nvt temp {temperature} {temperature} 0.1

# Output
dump 1 all custom 10 {config_dir}/trajectory.dump id type x y z vx vy vz q mass
dump_modify 1 sort id

thermo 100
thermo_style custom step temp ke pe etotal press vol density

# Run simulation
timestep 0.001  # 1 fs
run {timesteps}

# Write final configuration
write_data {config_dir}/final.data
"""
    
    with open(input_file, 'w') as f:
        f.write(lammps_script)
    
    # Metadata
    metadata = {
        'config_id': config_id,
        'material': material,
        'num_particles': num_particles,
        'density': density,
        'energy': energy,
        'temperature': temperature,
        'timesteps': timesteps,
        'box_length': box_length,
        'initial_velocity': velocity_angstrom_fs,
        'mass': mat_props['mass'],
        'charge': mat_props['charge'],
        'lattice_constant': mat_props['lattice_constant'],
    }
    
    metadata_file = config_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return input_file, metadata


def sample_parameters(num_configs: int) -> List[Dict]:
    """
    Sample diverse parameter combinations for comprehensive dataset.
    
    Args:
        num_configs: Number of configurations to generate
    
    Returns:
        List of parameter dictionaries
    """
    np.random.seed(42)
    
    configs = []
    
    # Ensure coverage of all materials
    materials_list = list(MATERIALS.keys())
    
    for i in range(num_configs):
        # Cycle through materials to ensure balance
        material = materials_list[i % len(materials_list)]
        
        # Sample parameters with log-uniform for density and energy
        config = {
            'material': material,
            'num_particles': np.random.choice([500, 1000, 2000, 5000]),
            'density': 10 ** np.random.uniform(-2, 1),  # 0.01 to 10 particles/nm^3
            'energy': 10 ** np.random.uniform(-1, 2),  # 0.1 to 100 MeV
            'temperature': 10 ** np.random.uniform(2, 4),  # 100 to 10000 K
            'timesteps': np.random.choice([500, 1000, 2000]),
        }
        configs.append(config)
    
    return configs


def create_batch_script(configs: List[Dict], output_dir: Path, batch_size: int = 10) -> List[Path]:
    """
    Create batch scripts for parallel execution.
    
    Args:
        configs: List of configuration dictionaries
        output_dir: Output directory
        batch_size: Number of simulations per batch
    
    Returns:
        List of batch script paths
    """
    batch_scripts = []
    
    for batch_idx in range(0, len(configs), batch_size):
        batch_configs = configs[batch_idx:batch_idx + batch_size]
        batch_script = output_dir / f"batch_{batch_idx//batch_size:04d}.sh"
        
        with open(batch_script, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write(f"# Batch script for simulations {batch_idx} to {batch_idx + len(batch_configs) - 1}\n\n")
            
            for i, config in enumerate(batch_configs):
                config_id = batch_idx + i
                config_dir = output_dir / f"{config['material']}_{config_id:04d}"
                input_file = config_dir / "input.lammps"
                log_file = config_dir / "log.lammps"
                
                f.write(f"echo 'Running simulation {config_id}...'\n")
                f.write(f"lammps -in {input_file} -log {log_file}\n")
                f.write(f"echo 'Completed simulation {config_id}'\n\n")
        
        # Make executable
        os.chmod(batch_script, 0o755)
        batch_scripts.append(batch_script)
    
    return batch_scripts


def main():
    parser = argparse.ArgumentParser(description='Generate MD simulation data')
    parser.add_argument('--output_dir', type=str, default='data/md_simulations',
                        help='Output directory for MD data')
    parser.add_argument('--num_configs', type=int, default=1000,
                        help='Number of configurations to generate')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Number of simulations per batch script')
    parser.add_argument('--generate_only', action='store_true',
                        help='Only generate input files, do not create batch scripts')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {args.num_configs} MD simulation configurations...")
    print(f"Materials: {', '.join(MATERIALS.keys())}")
    print(f"Output directory: {output_dir}")
    
    # Sample parameters
    configs = sample_parameters(args.num_configs)
    
    # Generate LAMMPS input files
    print("\nGenerating LAMMPS input files...")
    all_metadata = []
    
    for i, config in enumerate(configs):
        input_file, metadata = generate_lammps_input(
            material=config['material'],
            num_particles=config['num_particles'],
            density=config['density'],
            energy=config['energy'],
            temperature=config['temperature'],
            timesteps=config['timesteps'],
            output_dir=output_dir,
            config_id=i
        )
        all_metadata.append(metadata)
        
        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{args.num_configs} configurations")
    
    print(f"✓ Generated {len(configs)} LAMMPS input files")
    
    # Save summary metadata
    summary_file = output_dir / "dataset_summary.json"
    summary = {
        'num_configs': len(configs),
        'materials': list(MATERIALS.keys()),
        'density_range': [0.01, 10.0],
        'energy_range': [0.1, 100.0],
        'temperature_range': [100.0, 10000.0],
        'configurations': all_metadata
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Saved dataset summary to {summary_file}")
    
    # Create batch scripts
    if not args.generate_only:
        print(f"\nCreating batch scripts (batch size: {args.batch_size})...")
        batch_scripts = create_batch_script(configs, output_dir, args.batch_size)
        print(f"✓ Created {len(batch_scripts)} batch scripts")
        
        # Create master script
        master_script = output_dir / "run_all.sh"
        with open(master_script, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("# Master script to run all MD simulations\n\n")
            f.write("# Run batches in parallel (adjust based on available cores)\n")
            f.write("PARALLEL_JOBS=10\n\n")
            
            for batch_script in batch_scripts:
                f.write(f"{batch_script} &\n")
                f.write("# Limit parallel jobs\n")
                f.write("if [[ $(jobs -r -p | wc -l) -ge $PARALLEL_JOBS ]]; then\n")
                f.write("    wait -n\n")
                f.write("fi\n\n")
            
            f.write("# Wait for all jobs to complete\n")
            f.write("wait\n")
            f.write("echo 'All simulations completed!'\n")
        
        os.chmod(master_script, 0o755)
        print(f"✓ Created master script: {master_script}")
        
        print("\n" + "="*60)
        print("NEXT STEPS:")
        print("="*60)
        print(f"1. Install LAMMPS on your cloud instance")
        print(f"2. Run simulations: {master_script}")
        print(f"3. Monitor progress in {output_dir}")
        print(f"4. Convert to training format: python scripts/convert_md_to_training.py")
        print("="*60)


if __name__ == '__main__':
    main()
