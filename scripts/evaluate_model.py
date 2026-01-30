"""Script to evaluate trained model on test data and synthetic benchmarks.

This script loads a trained model, evaluates it on test data, computes accuracy
metrics, evaluates on synthetic benchmarks, and generates a comprehensive
evaluation report.

Usage:
    python scripts/evaluate_model.py --checkpoint checkpoints/model.pt --test-data data/test --benchmarks data/synthetic --output results/evaluation
"""

import argparse
import os
import sys
import json
import pickle
from pathlib import Path
from typing import Dict, List, Any
import torch
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.multi_scale_model import MultiScaleModel
from src.evaluation.metrics import (
    PredictionMetrics,
    RegimeMetrics,
    CollectiveModeMetrics,
    evaluate_model
)
from src.data.structures import TrajectoryData, GraphData
from src.data.loader import MDDataLoader
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
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load config if provided, otherwise use config from checkpoint
    if config_path:
        config = load_config(config_path)
    elif 'config' in checkpoint:
        config = checkpoint['config']
    else:
        raise ValueError("No configuration found. Provide config_path or ensure checkpoint contains config.")
    
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


def load_test_data(test_data_path: str) -> List[GraphData]:
    """Load test data from directory.
    
    Args:
        test_data_path: Path to test data directory
    
    Returns:
        List of GraphData samples
    """
    print(f"Loading test data from {test_data_path}...")
    
    test_data = []
    test_path = Path(test_data_path)
    
    # Look for pickle files
    for pkl_file in test_path.glob("*.pkl"):
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
                
                # Handle different data formats
                if isinstance(data, GraphData):
                    test_data.append(data)
                elif isinstance(data, list):
                    test_data.extend([d for d in data if isinstance(d, GraphData)])
                elif isinstance(data, TrajectoryData):
                    # Convert trajectory to GraphData samples
                    # This is a simplified conversion - actual implementation may vary
                    print(f"  Skipping trajectory data in {pkl_file.name} (conversion not implemented)")
        except Exception as e:
            print(f"  Warning: Failed to load {pkl_file.name}: {e}")
    
    print(f"Loaded {len(test_data)} test samples")
    
    return test_data


def load_benchmarks(benchmark_path: str) -> Dict[str, List[TrajectoryData]]:
    """Load synthetic benchmarks from directory.
    
    Args:
        benchmark_path: Path to benchmark directory
    
    Returns:
        Dictionary mapping benchmark type to list of trajectories
    """
    print(f"Loading benchmarks from {benchmark_path}...")
    
    benchmarks = {
        'plasma_oscillation': [],
        'beam_instability': [],
        'single_particle': []
    }
    
    benchmark_path = Path(benchmark_path)
    
    # Iterate through material subdirectories
    for material_dir in benchmark_path.iterdir():
        if not material_dir.is_dir():
            continue
        
        print(f"  Loading {material_dir.name} benchmarks...")
        
        # Load each benchmark type
        for benchmark_type in benchmarks.keys():
            pkl_file = material_dir / f"{benchmark_type}.pkl"
            
            if pkl_file.exists():
                try:
                    with open(pkl_file, 'rb') as f:
                        traj = pickle.load(f)
                        benchmarks[benchmark_type].append(traj)
                except Exception as e:
                    print(f"    Warning: Failed to load {pkl_file.name}: {e}")
    
    # Print summary
    for benchmark_type, trajs in benchmarks.items():
        print(f"  {benchmark_type}: {len(trajs)} trajectories")
    
    return benchmarks


def evaluate_on_test_data(model: MultiScaleModel,
                          test_data: List[GraphData],
                          device: str = 'cpu',
                          regime_split: bool = True) -> Dict[str, Any]:
    """Evaluate model on test data.
    
    Args:
        model: Trained model
        test_data: List of test samples
        device: Device to run on
        regime_split: Whether to split by regime
    
    Returns:
        Dictionary of evaluation metrics
    """
    print("\nEvaluating on test data...")
    
    if not test_data:
        print("  No test data available")
        return {}
    
    # Prepare targets (simplified - in practice, targets should be loaded with data)
    targets = []
    for sample in test_data:
        target = {}
        
        # Extract target positions and momenta if available
        if hasattr(sample, 'target_pos'):
            target['positions'] = sample.target_pos
        if hasattr(sample, 'target_mom'):
            target['momenta'] = sample.target_mom
        if hasattr(sample, 'target_obs'):
            target['observables'] = sample.target_obs
        
        targets.append(target)
    
    # Evaluate
    results = evaluate_model(
        model=model,
        data=test_data,
        targets=targets,
        device=device,
        regime_split=regime_split
    )
    
    # Print summary
    print("\nTest Data Results:")
    for regime, metrics in results.items():
        if regime.endswith('_count'):
            continue
        
        print(f"\n  {regime.upper()}:")
        if isinstance(metrics, dict):
            for metric_name, value in metrics.items():
                if not metric_name.endswith('_std'):
                    std_key = f"{metric_name}_std"
                    if std_key in metrics:
                        print(f"    {metric_name}: {value:.6f} ± {metrics[std_key]:.6f}")
                    else:
                        print(f"    {metric_name}: {value:.6f}")
    
    return results


def evaluate_on_benchmarks(model: MultiScaleModel,
                           benchmarks: Dict[str, List[TrajectoryData]],
                           device: str = 'cpu') -> Dict[str, Any]:
    """Evaluate model on synthetic benchmarks.
    
    Args:
        model: Trained model
        benchmarks: Dictionary of benchmark trajectories
        device: Device to run on
    
    Returns:
        Dictionary of benchmark evaluation results
    """
    print("\nEvaluating on synthetic benchmarks...")
    
    results = {}
    
    for benchmark_type, trajectories in benchmarks.items():
        if not trajectories:
            continue
        
        print(f"\n  {benchmark_type.upper()}:")
        
        type_results = []
        
        for i, traj in enumerate(trajectories):
            material = traj.metadata.get('material', 'unknown')
            
            # Extract ground truth
            if benchmark_type == 'plasma_oscillation':
                true_freq = traj.metadata.get('ground_truth_frequency', None)
                print(f"    {material}: Ground truth frequency = {true_freq:.4f} THz")
                
                # Placeholder for actual frequency prediction
                # In practice, this would use hypothesis generation
                type_results.append({
                    'material': material,
                    'true_frequency': true_freq,
                    'has_collective_mode': True
                })
            
            elif benchmark_type == 'beam_instability':
                true_growth = traj.metadata.get('ground_truth_growth_rate', None)
                print(f"    {material}: Ground truth growth rate = {true_growth:.6f} 1/fs")
                
                type_results.append({
                    'material': material,
                    'true_growth_rate': true_growth,
                    'has_collective_mode': True
                })
            
            elif benchmark_type == 'single_particle':
                print(f"    {material}: No collective effects (baseline)")
                
                type_results.append({
                    'material': material,
                    'has_collective_mode': False
                })
        
        results[benchmark_type] = type_results
    
    print("\n  Note: Full benchmark evaluation requires hypothesis generation integration")
    
    return results


def generate_report(test_results: Dict[str, Any],
                   benchmark_results: Dict[str, Any],
                   output_path: str):
    """Generate evaluation report.
    
    Args:
        test_results: Results from test data evaluation
        benchmark_results: Results from benchmark evaluation
        output_path: Path to save report
    """
    print(f"\nGenerating evaluation report...")
    
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save full results as JSON
    full_results = {
        'test_data': test_results,
        'benchmarks': benchmark_results
    }
    
    with open(output_path / 'evaluation_results.json', 'w') as f:
        json.dump(full_results, f, indent=2, default=str)
    
    print(f"  Saved results to {output_path / 'evaluation_results.json'}")
    
    # Generate human-readable report
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("MODEL EVALUATION REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Test data results
    report_lines.append("TEST DATA EVALUATION")
    report_lines.append("-" * 80)
    
    if test_results:
        for regime, metrics in test_results.items():
            if regime.endswith('_count'):
                continue
            
            report_lines.append(f"\n{regime.upper()}:")
            
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    if not metric_name.endswith('_std'):
                        std_key = f"{metric_name}_std"
                        if std_key in metrics:
                            report_lines.append(f"  {metric_name}: {value:.6f} ± {metrics[std_key]:.6f}")
                        else:
                            report_lines.append(f"  {metric_name}: {value:.6f}")
    else:
        report_lines.append("\nNo test data results available")
    
    report_lines.append("")
    report_lines.append("")
    
    # Benchmark results
    report_lines.append("SYNTHETIC BENCHMARK EVALUATION")
    report_lines.append("-" * 80)
    
    if benchmark_results:
        for benchmark_type, results in benchmark_results.items():
            report_lines.append(f"\n{benchmark_type.upper()}:")
            
            if results:
                for result in results:
                    material = result.get('material', 'unknown')
                    report_lines.append(f"  {material}:")
                    
                    for key, value in result.items():
                        if key != 'material':
                            report_lines.append(f"    {key}: {value}")
            else:
                report_lines.append("  No results")
    else:
        report_lines.append("\nNo benchmark results available")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    
    # Save report
    report_text = "\n".join(report_lines)
    
    with open(output_path / 'evaluation_report.txt', 'w') as f:
        f.write(report_text)
    
    print(f"  Saved report to {output_path / 'evaluation_report.txt'}")
    
    # Print report to console
    print("\n" + report_text)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate trained model on test data and synthetic benchmarks'
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
        '--test-data',
        type=str,
        default=None,
        help='Path to test data directory (optional)'
    )
    
    parser.add_argument(
        '--benchmarks',
        type=str,
        default=None,
        help='Path to synthetic benchmarks directory (optional)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='results/evaluation',
        help='Output directory for evaluation results (default: results/evaluation)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run evaluation on (default: cuda if available, else cpu)'
    )
    
    parser.add_argument(
        '--no-regime-split',
        action='store_true',
        help='Disable regime-specific evaluation'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    print(f"Output: {args.output}")
    print()
    
    # Load model
    model = load_model(args.checkpoint, args.config, args.device)
    
    # Evaluate on test data
    test_results = {}
    if args.test_data:
        test_data = load_test_data(args.test_data)
        if test_data:
            test_results = evaluate_on_test_data(
                model,
                test_data,
                args.device,
                regime_split=not args.no_regime_split
            )
    else:
        print("\nNo test data provided (use --test-data)")
    
    # Evaluate on benchmarks
    benchmark_results = {}
    if args.benchmarks:
        benchmarks = load_benchmarks(args.benchmarks)
        if any(benchmarks.values()):
            benchmark_results = evaluate_on_benchmarks(
                model,
                benchmarks,
                args.device
            )
    else:
        print("\nNo benchmarks provided (use --benchmarks)")
    
    # Generate report
    if test_results or benchmark_results:
        generate_report(test_results, benchmark_results, args.output)
    else:
        print("\nNo evaluation results to report")
    
    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()
