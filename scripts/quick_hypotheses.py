#!/usr/bin/env python3
"""Quick hypothesis generation - scans density space without expensive mode characterization."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import torch
import numpy as np

from src.config.config_loader import load_config, create_model_from_config
from src.data.structures import BeamConfiguration, ParticleState


def create_config(material, n_particles, density, energy, temperature):
    """Create a beam configuration."""
    material_masses = {'Si': 28.0855, 'Fe': 55.845, 'W': 183.84, 'Cu': 63.546, 'Al': 26.9815}
    material_species = {'Si': 0, 'Fe': 1, 'W': 2, 'Cu': 3, 'Al': 4}
    
    mass = material_masses.get(material, 28.0)
    species = material_species.get(material, 0)
    box_size = (n_particles / density) ** (1/3)
    
    positions = torch.rand(n_particles, 3) * box_size
    k_B = 1.381e-23
    mass_kg = mass * 1.66054e-27
    v_thermal = np.sqrt(k_B * temperature / mass_kg) * 1e-6
    velocities = torch.randn(n_particles, 3) * v_thermal
    
    particles = []
    for i in range(n_particles):
        particles.append(ParticleState(
            position=positions[i],
            momentum=velocities[i] * mass,
            charge=1.0,
            mass=mass,
            species=species
        ))
    
    return BeamConfiguration(
        particles=particles,
        density=density,
        energy=energy,
        material=material,
        temperature=temperature,
        time=0.0
    )


def config_to_graph_data(config, device):
    """Convert config to GraphData."""
    from src.data.structures import GraphData
    
    positions = torch.stack([p.position for p in config.particles]).float().to(device)
    momenta = torch.stack([p.momentum for p in config.particles]).float().to(device)
    charges = torch.tensor([p.charge for p in config.particles], dtype=torch.float32, device=device).unsqueeze(1)
    masses = torch.tensor([p.mass for p in config.particles], dtype=torch.float32, device=device).unsqueeze(1)
    species = torch.tensor([p.species for p in config.particles], dtype=torch.long, device=device)
    
    species_onehot = torch.nn.functional.one_hot(species, num_classes=5).float()
    x = torch.cat([positions, momenta, charges, masses, species_onehot], dim=1)
    
    material_map = {'Si': 0, 'Fe': 1, 'W': 2, 'Cu': 3, 'Al': 4}
    material_idx = material_map.get(config.material, 0)
    material_onehot = torch.nn.functional.one_hot(
        torch.tensor([material_idx], device=device), num_classes=5
    ).float()
    
    return GraphData(
        x=x,
        pos=positions,
        density=torch.tensor([config.density], dtype=torch.float32, device=device),
        energy=torch.tensor([config.energy], dtype=torch.float32, device=device),
        material=material_onehot
    )


def main():
    parser = argparse.ArgumentParser(description='Quick hypothesis scan')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--materials', type=str, nargs='+', default=['Si', 'Fe', 'W', 'Cu', 'Al'])
    parser.add_argument('--n-particles', type=int, default=50)
    parser.add_argument('--density-min', type=float, default=0.5)
    parser.add_argument('--density-max', type=float, default=2.0)
    parser.add_argument('--num-densities', type=int, default=10)
    parser.add_argument('--output', type=str, default='results/quick_hypotheses')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    print("=" * 60)
    print("QUICK HYPOTHESIS SCAN")
    print("=" * 60)
    
    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    full_config = load_config('config')
    model = create_model_from_config(full_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded (epoch {checkpoint.get('epoch', 'unknown')})")
    
    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Density range
    densities = np.linspace(args.density_min, args.density_max, args.num_densities)
    
    results = {}
    
    for material in args.materials:
        print(f"\n--- Scanning {material} ---")
        
        latent_states = []
        
        with torch.no_grad():
            for i, density in enumerate(densities):
                config = create_config(material, args.n_particles, density, 1.0, 300.0)
                data = config_to_graph_data(config, args.device)
                z = model.encode(data)
                latent_states.append(z.cpu().numpy())
                
                if args.device == 'cuda':
                    torch.cuda.empty_cache()
                
                print(f"  Density {density:.2f}: latent norm = {z.norm().item():.4f}")
        
        latent_states = np.array(latent_states).squeeze()
        
        # Compute latent space changes (simple anomaly detection)
        if len(latent_states) > 1:
            diffs = np.linalg.norm(np.diff(latent_states, axis=0), axis=1)
            max_change_idx = np.argmax(diffs)
            critical_density = (densities[max_change_idx] + densities[max_change_idx + 1]) / 2
            
            print(f"\n  Largest latent change at density ≈ {critical_density:.3f}")
            print(f"  Change magnitude: {diffs[max_change_idx]:.4f}")
        else:
            critical_density = densities[0]
        
        results[material] = {
            'densities': densities.tolist(),
            'latent_norms': [np.linalg.norm(z) for z in latent_states],
            'critical_density_estimate': float(critical_density)
        }
    
    # Save results
    with open(output_path / 'scan_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for material, data in results.items():
        print(f"{material}: Critical density ≈ {data['critical_density_estimate']:.3f} particles/nm³")
    
    print(f"\nResults saved to {output_path / 'scan_results.json'}")


if __name__ == '__main__':
    main()
