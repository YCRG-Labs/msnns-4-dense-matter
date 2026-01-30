#!/usr/bin/env python3
"""
Convert LAMMPS MD simulation output to training format.

This script processes LAMMPS dump files and converts them to the format
expected by the training pipeline.

Usage:
    python scripts/convert_md_to_training.py \
        --input_dir data/md_simulations \
        --output_dir data/training_data \
        --train_split 0.8
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict
import pickle
import numpy as np
from tqdm import tqdm

# Import from your data loader
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.loader import load_lammps_dump, compute_beam_properties
from src.data.structures import TrajectoryData, BeamConfiguration


def process_simulation(config_dir: Path) -> TrajectoryData:
    """
    Process a single MD simulation directory.
    
    Args:
        config_dir: Path to simulation directory
    
    Returns:
        TrajectoryData object
    """
    # Load metadata
    metadata_file = config_dir / "metadata.json"
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Load trajectory
    dump_file = config_dir / "trajectory.dump"
    if not dump_file.exists():
        raise FileNotFoundError(f"Trajectory file not found: {dump_file}")
    
    trajectory = load_lammps_dump(str(dump_file))
    
    # Add metadata
    trajectory.metadata.update(metadata)
    
    return trajectory


def split_dataset(
    trajectories: List[TrajectoryData],
    train_split: float = 0.8,
    val_split: float = 0.1
) -> Dict[str, List[TrajectoryData]]:
    """
    Split dataset into train/val/test sets.
    
    Args:
        trajectories: List of trajectories
        train_split: Fraction for training
        val_split: Fraction for validation
    
    Returns:
        Dictionary with 'train', 'val', 'test' keys
    """
    np.random.seed(42)
    indices = np.random.permutation(len(trajectories))
    
    n_train = int(len(trajectories) * train_split)
    n_val = int(len(trajectories) * val_split)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    return {
        'train': [trajectories[i] for i in train_indices],
        'val': [trajectories[i] for i in val_indices],
        'test': [trajectories[i] for i in test_indices],
    }


def save_split(trajectories: List[TrajectoryData], output_dir: Path, split_name: str):
    """
    Save a dataset split.
    
    Args:
        trajectories: List of trajectories
        output_dir: Output directory
        split_name: Name of split ('train', 'val', 'test')
    """
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    
    # Save each trajectory
    for i, traj in enumerate(trajectories):
        traj_file = split_dir / f"trajectory_{i:05d}.pkl"
        with open(traj_file, 'wb') as f:
            pickle.dump(traj, f)
    
    # Save summary
    summary = {
        'num_trajectories': len(trajectories),
        'materials': list(set(traj.metadata.get('material', 'unknown') for traj in trajectories)),
        'total_timesteps': sum(len(traj.states) for traj in trajectories),
    }
    
    summary_file = split_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"  {split_name}: {len(trajectories)} trajectories")


def main():
    parser = argparse.ArgumentParser(description='Convert MD data to training format')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input directory with MD simulations')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for training data')
    parser.add_argument('--train_split', type=float, default=0.8,
                        help='Fraction of data for training')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Fraction of data for validation')
    parser.add_argument('--max_trajectories', type=int, default=None,
                        help='Maximum number of trajectories to process (for testing)')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Converting MD Data to Training Format")
    print("="*60)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Train/Val/Test split: {args.train_split:.1%}/{args.val_split:.1%}/{1-args.train_split-args.val_split:.1%}")
    
    # Find all simulation directories
    config_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir() and (d / "metadata.json").exists()])
    
    if args.max_trajectories:
        config_dirs = config_dirs[:args.max_trajectories]
    
    print(f"\nFound {len(config_dirs)} simulation directories")
    
    # Process all simulations
    print("\nProcessing simulations...")
    trajectories = []
    failed = []
    
    for config_dir in tqdm(config_dirs, desc="Loading trajectories"):
        try:
            traj = process_simulation(config_dir)
            trajectories.append(traj)
        except Exception as e:
            failed.append((config_dir.name, str(e)))
    
    print(f"✓ Successfully processed {len(trajectories)} trajectories")
    if failed:
        print(f"✗ Failed to process {len(failed)} trajectories:")
        for name, error in failed[:5]:  # Show first 5 failures
            print(f"  - {name}: {error}")
    
    # Split dataset
    print("\nSplitting dataset...")
    splits = split_dataset(trajectories, args.train_split, args.val_split)
    
    # Save splits
    print("\nSaving splits...")
    for split_name, split_trajectories in splits.items():
        save_split(split_trajectories, output_dir, split_name)
    
    # Save overall summary
    overall_summary = {
        'total_trajectories': len(trajectories),
        'train': len(splits['train']),
        'val': len(splits['val']),
        'test': len(splits['test']),
        'failed': len(failed),
        'materials': list(set(traj.metadata.get('material', 'unknown') for traj in trajectories)),
    }
    
    summary_file = output_dir / "dataset_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(overall_summary, f, indent=2)
    
    print("\n" + "="*60)
    print("Conversion Complete!")
    print("="*60)
    print(f"Total trajectories: {len(trajectories)}")
    print(f"  Train: {len(splits['train'])}")
    print(f"  Val: {len(splits['val'])}")
    print(f"  Test: {len(splits['test'])}")
    print(f"\nData ready for training!")
    print(f"Use: python scripts/train_pretrain.py data.train_path={output_dir}/train")
    print("="*60)


if __name__ == '__main__':
    main()
