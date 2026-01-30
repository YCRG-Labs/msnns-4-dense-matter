"""Configuration loading utilities using Hydra.

This module provides utilities for loading and managing hierarchical configurations
with support for command-line overrides.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
import yaml
import torch


def load_config(
    config_name: str = "config",
    config_path: Optional[str] = None,
    overrides: Optional[list] = None,
) -> DictConfig:
    """Load configuration using Hydra.
    
    Args:
        config_name: Name of the config file (without .yaml extension)
        config_path: Path to config directory. If None, uses default configs/ directory
        overrides: List of config overrides in the form ["key=value", ...]
    
    Returns:
        DictConfig object containing the configuration
    
    Example:
        >>> cfg = load_config("config", overrides=["model.gnn.hidden_dim=256"])
        >>> print(cfg.model.gnn.hidden_dim)
        256
    """
    if overrides is None:
        overrides = []
    
    # Determine config path
    if config_path is None:
        # Get absolute path to configs directory
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent
        config_path = str(project_root / "configs")
    
    # Clear any existing Hydra instance
    GlobalHydra.instance().clear()
    
    # Initialize Hydra with config directory
    initialize_config_dir(config_dir=config_path, version_base=None)
    
    # Compose configuration with overrides
    cfg = compose(config_name=config_name, overrides=overrides)
    
    return cfg


def get_model_config(cfg: DictConfig) -> Dict[str, Any]:
    """Extract model configuration from full config.
    
    Args:
        cfg: Full configuration object
    
    Returns:
        Dictionary containing model configuration
    """
    return OmegaConf.to_container(cfg.model, resolve=True)


def get_training_config(cfg: DictConfig) -> Dict[str, Any]:
    """Extract training configuration from full config.
    
    Args:
        cfg: Full configuration object
    
    Returns:
        Dictionary containing training configuration
    """
    return OmegaConf.to_container(cfg.training, resolve=True)


def save_config(cfg: DictConfig, save_path: str) -> None:
    """Save configuration to YAML file.
    
    Args:
        cfg: Configuration object to save
        save_path: Path where to save the config file
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Convert to container and save
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    
    with open(save_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


def print_config(cfg: DictConfig, resolve: bool = True) -> None:
    """Pretty print configuration.
    
    Args:
        cfg: Configuration object to print
        resolve: Whether to resolve interpolations
    """
    print(OmegaConf.to_yaml(cfg, resolve=resolve))


def setup_experiment(cfg: DictConfig) -> None:
    """Setup experiment environment based on configuration.
    
    This function:
    - Sets random seeds for reproducibility
    - Creates necessary directories
    - Configures device (CPU/GPU)
    
    Args:
        cfg: Configuration object
    """
    # Set random seeds
    seed = cfg.experiment.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Set deterministic mode if requested
    if cfg.experiment.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
    
    # Create directories
    for path_key in ['data_root', 'checkpoint_root', 'log_root', 'output_root']:
        if path_key in cfg.paths:
            os.makedirs(cfg.paths[path_key], exist_ok=True)
    
    # Configure device
    if cfg.model.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Using CPU instead.")
        cfg.model.device = 'cpu'


def merge_configs(*configs: DictConfig) -> DictConfig:
    """Merge multiple configurations.
    
    Later configs override earlier ones.
    
    Args:
        *configs: Variable number of DictConfig objects to merge
    
    Returns:
        Merged configuration
    """
    if not configs:
        return DictConfig({})
    
    merged = configs[0]
    for cfg in configs[1:]:
        merged = OmegaConf.merge(merged, cfg)
    
    return merged


def validate_config(cfg: DictConfig) -> None:
    """Validate configuration for consistency.
    
    Checks:
    - Dimension compatibility between components
    - Valid hyperparameter ranges
    - Required fields are present
    
    Args:
        cfg: Configuration to validate
    
    Raises:
        ValueError: If configuration is invalid
    """
    # Check dimension compatibility
    if cfg.transformer.input_dim != cfg.gnn.hidden_dim:
        raise ValueError(
            f"Transformer input_dim ({cfg.transformer.input_dim}) must match "
            f"GNN hidden_dim ({cfg.gnn.hidden_dim})"
        )
    
    if cfg.encoder.input_dim != cfg.transformer.hidden_dim:
        raise ValueError(
            f"Encoder input_dim ({cfg.encoder.input_dim}) must match "
            f"Transformer hidden_dim ({cfg.transformer.hidden_dim})"
        )
    
    if cfg.neural_ode.latent_dim != cfg.encoder.latent_dim:
        raise ValueError(
            f"Neural ODE latent_dim ({cfg.neural_ode.latent_dim}) must match "
            f"Encoder latent_dim ({cfg.encoder.latent_dim})"
        )
    
    if cfg.decoder.latent_dim != cfg.encoder.latent_dim:
        raise ValueError(
            f"Decoder latent_dim ({cfg.decoder.latent_dim}) must match "
            f"Encoder latent_dim ({cfg.encoder.latent_dim})"
        )
    
    if cfg.decoder.particle_dim != cfg.transformer.hidden_dim:
        raise ValueError(
            f"Decoder particle_dim ({cfg.decoder.particle_dim}) must match "
            f"Transformer hidden_dim ({cfg.transformer.hidden_dim})"
        )
    
    # Check conditioning dimensions match
    if cfg.neural_ode.conditioning_dim != cfg.encoder.conditioning_dim:
        raise ValueError(
            f"Neural ODE conditioning_dim ({cfg.neural_ode.conditioning_dim}) must match "
            f"Encoder conditioning_dim ({cfg.encoder.conditioning_dim})"
        )
    
    if cfg.decoder.conditioning_dim != cfg.encoder.conditioning_dim:
        raise ValueError(
            f"Decoder conditioning_dim ({cfg.decoder.conditioning_dim}) must match "
            f"Encoder conditioning_dim ({cfg.encoder.conditioning_dim})"
        )
    
    # Validate hyperparameter ranges
    if cfg.gnn.cutoff_radius <= 0:
        raise ValueError(f"GNN cutoff_radius must be positive, got {cfg.gnn.cutoff_radius}")
    
    if cfg.transformer.num_heads <= 0:
        raise ValueError(f"Transformer num_heads must be positive, got {cfg.transformer.num_heads}")
    
    if cfg.transformer.hidden_dim % cfg.transformer.num_heads != 0:
        raise ValueError(
            f"Transformer hidden_dim ({cfg.transformer.hidden_dim}) must be divisible by "
            f"num_heads ({cfg.transformer.num_heads})"
        )
    
    if cfg.encoder.latent_dim <= 0:
        raise ValueError(f"Encoder latent_dim must be positive, got {cfg.encoder.latent_dim}")
    
    # Validate training config if present
    if 'training' in cfg:
        if cfg.training.training.epochs <= 0:
            raise ValueError(f"Training epochs must be positive, got {cfg.training.training.epochs}")
        
        if cfg.training.data.batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {cfg.training.data.batch_size}")
        
        if cfg.training.optimizer.lr <= 0:
            raise ValueError(f"Learning rate must be positive, got {cfg.training.optimizer.lr}")


def create_model_from_config(cfg: DictConfig):
    """Create model instance from configuration.
    
    Args:
        cfg: Configuration object
    
    Returns:
        Instantiated MultiScaleModel
    """
    from src.models.multi_scale_model import MultiScaleModel, ModelConfig
    
    # Validate config first
    validate_config(cfg.model)
    
    # Create ModelConfig from Hydra config
    model_config = ModelConfig(
        input_dim=cfg.model.gnn.input_dim,
        num_species=cfg.model.materials.num_species,
        gnn_hidden_dim=cfg.model.gnn.hidden_dim,
        gnn_num_layers=cfg.model.gnn.num_layers,
        gnn_cutoff_radius=cfg.model.gnn.cutoff_radius,
        transformer_hidden_dim=cfg.model.transformer.hidden_dim,
        transformer_num_heads=cfg.model.transformer.num_heads,
        transformer_dropout=cfg.model.transformer.dropout,
        encoder_hidden_dims=list(cfg.model.encoder.hidden_dims),
        latent_dim=cfg.model.encoder.latent_dim,
        ode_hidden_dims=list(cfg.model.neural_ode.hidden_dims),
        ode_solver=cfg.model.neural_ode.solver,
        ode_rtol=cfg.model.neural_ode.rtol,
        ode_atol=cfg.model.neural_ode.atol,
        num_observables=cfg.model.decoder.num_observables,
        conditioning_dim=cfg.model.encoder.conditioning_dim,
    )
    
    # Create model
    model = MultiScaleModel(model_config)
    
    # Move to device
    device = torch.device(cfg.model.device)
    model = model.to(device)
    
    return model
