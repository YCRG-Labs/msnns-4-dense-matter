# Configuration Management

This directory contains YAML configuration files for the multi-scale neural network system. The configuration system uses [Hydra](https://hydra.cc/) for hierarchical configuration management with command-line overrides.

## Directory Structure

```
configs/
├── config.yaml              # Main configuration entry point
├── model/
│   └── default.yaml         # Model architecture hyperparameters
└── training/
    ├── pretraining.yaml     # Self-supervised pretraining config
    ├── finetuning.yaml      # Physics-constrained fine-tuning config
    └── uncertainty.yaml     # Uncertainty calibration config
```

## Configuration Files

### Main Config (`config.yaml`)

The main configuration file that imports model and training configs. It defines:
- Experiment settings (name, seed, deterministic mode)
- Path configurations (data, checkpoints, logs, outputs)
- Hydra runtime settings

### Model Config (`model/default.yaml`)

Defines hyperparameters for all model components:
- **GNN Layer**: Hidden dimensions, number of layers, cutoff radius
- **Transformer Layer**: Hidden dimensions, number of heads, dropout
- **Encoder**: Latent dimension, hidden layer sizes
- **Neural ODE**: Dynamics network architecture, solver settings
- **Decoder**: Observable prediction settings
- **Device**: CPU/GPU configuration, mixed precision

### Training Configs

#### Pretraining (`training/pretraining.yaml`)
Self-supervised pretraining on diverse MD simulations:
- Learning rate: 1e-3
- Epochs: 100
- Batch size: 32
- Loss weights: autoregressive (1.0), contrastive (0.5), masked particle (0.5)
- Cosine annealing scheduler with warmup

#### Fine-tuning (`training/finetuning.yaml`)
Physics-constrained fine-tuning with conservation laws:
- Learning rate: 1e-4 (lower than pretraining)
- Epochs: 50
- Batch size: 16
- Conservation loss weight: 0.1
- Physics limit loss weight: 0.05
- Loads pretrained checkpoint

#### Uncertainty (`training/uncertainty.yaml`)
Bayesian uncertainty calibration:
- Methods: Deep ensembles, MC dropout, aleatoric uncertainty
- Ensemble size: 5 models
- MC dropout samples: 50
- Temperature scaling for calibration

## Usage

### Basic Usage

```python
from src.config import load_config

# Load default configuration
cfg = load_config()

# Access configuration values
print(cfg.model.gnn.hidden_dim)  # 128
print(cfg.training.optimizer.lr)  # 1e-3
```

### Command-Line Overrides

Override configuration values from command line:

```python
# Override specific parameters
cfg = load_config(overrides=[
    "model.gnn.hidden_dim=256",
    "training.data.batch_size=64",
])
```

### Different Training Modes

Switch between training configurations:

```python
# Pretraining
cfg = load_config(overrides=["training=pretraining"])

# Fine-tuning
cfg = load_config(overrides=["training=finetuning"])

# Uncertainty calibration
cfg = load_config(overrides=["training=uncertainty"])
```

### Creating Models from Config

```python
from src.config.config_loader import create_model_from_config

cfg = load_config()
model = create_model_from_config(cfg)
```

### Validation

Validate configuration for consistency:

```python
from src.config.config_loader import validate_config

cfg = load_config()
validate_config(cfg.model)  # Raises ValueError if invalid
```

### Saving Configurations

Save configuration to file:

```python
from src.config import save_config

cfg = load_config(overrides=["model.gnn.hidden_dim=512"])
save_config(cfg, "outputs/my_config.yaml")
```

## Command-Line Usage

When using Hydra-decorated scripts, you can override configs directly:

```bash
# Override model parameters
python train.py model.gnn.hidden_dim=256 model.transformer.num_heads=16

# Switch training mode
python train.py training=finetuning

# Override multiple parameters
python train.py \
    model.gnn.hidden_dim=512 \
    training.data.batch_size=64 \
    training.optimizer.lr=5e-4
```

## Configuration Validation

The system automatically validates:
- **Dimension compatibility**: Ensures dimensions match between components
  - Transformer input_dim = GNN hidden_dim
  - Encoder input_dim = Transformer hidden_dim
  - Neural ODE latent_dim = Encoder latent_dim
  - Decoder dimensions match encoder and transformer
- **Hyperparameter ranges**: Positive values, valid ranges
- **Divisibility constraints**: Hidden_dim divisible by num_heads

## Adding New Configurations

### Adding a New Model Config

Create `configs/model/my_model.yaml`:

```yaml
# Inherit from default and override
defaults:
  - default

gnn:
  hidden_dim: 512
  num_layers: 6

transformer:
  hidden_dim: 512
  num_heads: 16
```

Load with:
```python
cfg = load_config(overrides=["model=my_model"])
```

### Adding a New Training Config

Create `configs/training/my_training.yaml`:

```yaml
data:
  batch_size: 128
  
optimizer:
  lr: 2e-3
  
training:
  epochs: 200
```

Load with:
```python
cfg = load_config(overrides=["training=my_training"])
```

## Best Practices

1. **Use overrides for experiments**: Keep base configs stable, use overrides for experiments
2. **Validate before training**: Always validate config before starting training
3. **Save configs with checkpoints**: Save configuration alongside model checkpoints
4. **Document custom configs**: Add comments explaining non-standard choices
5. **Version control configs**: Track configuration changes in git

## Examples

See `examples/demo_config.py` for comprehensive usage examples.

## Testing

Run configuration tests:

```bash
pytest tests/test_config.py -v
```

## References

- [Hydra Documentation](https://hydra.cc/)
- [OmegaConf Documentation](https://omegaconf.readthedocs.io/)
