"""Hypothesis generation module for collective phenomena discovery.

This module provides tools for:
- Latent space scanning to identify collective transitions
- Collective mode extraction and characterization
- Observable signature prediction for experimental validation
"""

from .scanning import (
    scan_density_space,
    mahalanobis_distance,
    fit_gaussian_to_low_density,
    compute_mahalanobis_distances,
    compute_temporal_variance,
    compute_temporal_variances_for_scan,
    compute_effective_dimensionality,
    compute_effective_dimensionality_for_density,
    compute_effective_dimensionalities_for_scan,
    compute_anomaly_score,
    compute_anomaly_scores_for_scan,
    detect_transitions,
    full_density_scan,
    ScanResult,
)

__all__ = [
    # Main scanning functions
    'scan_density_space',
    'full_density_scan',
    'detect_transitions',
    
    # Anomaly score components
    'mahalanobis_distance',
    'fit_gaussian_to_low_density',
    'compute_mahalanobis_distances',
    'compute_temporal_variance',
    'compute_temporal_variances_for_scan',
    'compute_effective_dimensionality',
    'compute_effective_dimensionality_for_density',
    'compute_effective_dimensionalities_for_scan',
    'compute_anomaly_score',
    'compute_anomaly_scores_for_scan',
    
    # Data structures
    'ScanResult',
]
