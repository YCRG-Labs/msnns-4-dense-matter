"""Training module for multi-scale neural network.

This module provides training infrastructure including:
- Pretraining losses (autoregressive, contrastive, masked particle)
- Pretraining loop with AdamW optimizer and cosine annealing
- Fine-tuning losses (prediction + conservation + physics limits)
- Fine-tuning loop with physics constraints
- Uncertainty calibration (deep ensembles, MC dropout, aleatoric uncertainty)
- Temperature scaling for calibration
- Checkpointing and logging utilities
"""

from .pretraining_losses import (
    AutoregressiveLoss,
    ContrastiveLoss,
    MaskedParticleLoss,
    CombinedPretrainingLoss,
    PretrainingLossWeights
)

from .pretraining import (
    PretrainingConfig,
    PretrainingTrainer,
    create_pretraining_dataloader
)

from .finetuning_losses import (
    PredictionLoss,
    FineTuningLoss,
    FineTuningLossWeights,
    create_finetuning_loss
)

from .finetuning import (
    FineTuningConfig,
    FineTuningTrainer,
    create_finetuning_dataloader
)

from .uncertainty import (
    UncertaintyConfig,
    DeepEnsemble,
    MCDropoutModel,
    AleatoricDecoder,
    AleatoricModel,
    aleatoric_loss,
    calibrate_temperature,
    TemperatureScaledModel,
    UncertaintyReport,
    CombinedUncertaintyModel,
    print_uncertainty_report
)

from .gpu_utils import (
    DeviceManager,
    GraphBatcher,
    GPUMemoryMonitor,
    DistributedTrainingManager,
    MixedPrecisionTrainer,
    setup_training_environment
)

from .logging import (
    TrainingLogger,
    create_logger
)

# Try to import wandb logger, but don't fail if wandb is broken
try:
    from .wandb_logger import (
        WandbLogger,
        create_wandb_logger,
        WANDB_AVAILABLE
    )
except (ImportError, AttributeError) as e:
    # W&B not available or broken
    WandbLogger = None
    create_wandb_logger = None
    WANDB_AVAILABLE = False
    print(f"Warning: Weights & Biases not available: {e}")

from .checkpointing import (
    CheckpointManager,
    save_checkpoint_simple,
    load_checkpoint_simple
)

from .stability import (
    clip_gradients,
    check_model_outputs,
    check_loss_value,
    safe_divide,
    safe_log,
    safe_sqrt,
    GradientMonitor
)

__all__ = [
    # Pretraining losses
    'AutoregressiveLoss',
    'ContrastiveLoss',
    'MaskedParticleLoss',
    'CombinedPretrainingLoss',
    'PretrainingLossWeights',
    
    # Pretraining
    'PretrainingConfig',
    'PretrainingTrainer',
    'create_pretraining_dataloader',
    
    # Fine-tuning losses
    'PredictionLoss',
    'FineTuningLoss',
    'FineTuningLossWeights',
    'create_finetuning_loss',
    
    # Fine-tuning
    'FineTuningConfig',
    'FineTuningTrainer',
    'create_finetuning_dataloader',
    
    # Uncertainty calibration
    'UncertaintyConfig',
    'DeepEnsemble',
    'MCDropoutModel',
    'AleatoricDecoder',
    'AleatoricModel',
    'aleatoric_loss',
    'calibrate_temperature',
    'TemperatureScaledModel',
    'UncertaintyReport',
    'CombinedUncertaintyModel',
    'print_uncertainty_report',
    
    # GPU utilities
    'DeviceManager',
    'GraphBatcher',
    'GPUMemoryMonitor',
    'DistributedTrainingManager',
    'MixedPrecisionTrainer',
    'setup_training_environment',
    
    # Logging
    'TrainingLogger',
    'create_logger',
    
    # W&B logging
    'WandbLogger',
    'create_wandb_logger',
    'WANDB_AVAILABLE',
    
    # Checkpointing
    'CheckpointManager',
    'save_checkpoint_simple',
    'load_checkpoint_simple',
    
    # Numerical stability
    'clip_gradients',
    'check_model_outputs',
    'check_loss_value',
    'safe_divide',
    'safe_log',
    'safe_sqrt',
    'GradientMonitor',
]
