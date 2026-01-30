#!/bin/bash
# Setup CERN high-speed storage structure

set -e

CERN_ROOT="/root/highspeedstorage/CERN"

echo "=========================================="
echo "Setting up CERN Storage Structure"
echo "=========================================="
echo "Root: ${CERN_ROOT}"
echo ""

# Create directory structure
echo "Creating directories..."

mkdir -p "${CERN_ROOT}/data/synthetic_small"
mkdir -p "${CERN_ROOT}/data/synthetic_comprehensive"
mkdir -p "${CERN_ROOT}/data/md_simulations"
mkdir -p "${CERN_ROOT}/checkpoints/small_training"
mkdir -p "${CERN_ROOT}/checkpoints/comprehensive_training"
mkdir -p "${CERN_ROOT}/checkpoints/md_training"
mkdir -p "${CERN_ROOT}/logs/pretraining"
mkdir -p "${CERN_ROOT}/logs/finetuning"
mkdir -p "${CERN_ROOT}/logs/uncertainty"
mkdir -p "${CERN_ROOT}/outputs/evaluation"
mkdir -p "${CERN_ROOT}/outputs/hypotheses"
mkdir -p "${CERN_ROOT}/outputs/benchmarks"

echo "✓ Directory structure created"
echo ""

# Create README
cat > "${CERN_ROOT}/README.md" << 'EOF'
# CERN High-Speed Storage

This directory contains all data, checkpoints, and outputs for the CERN beam physics project.

## Directory Structure

```
/root/highspeedstorage/CERN/
├── data/                           # Training data
│   ├── synthetic_small/            # Small synthetic dataset (100 particles, 50 timesteps)
│   ├── synthetic_comprehensive/    # Large synthetic dataset (1000 particles, 1000 timesteps)
│   └── md_simulations/             # Molecular dynamics simulation data
├── checkpoints/                    # Model checkpoints
│   ├── small_training/             # Checkpoints from small dataset training
│   ├── comprehensive_training/     # Checkpoints from comprehensive dataset training
│   └── md_training/                # Checkpoints from MD data training
├── logs/                           # Training logs
│   ├── pretraining/                # Pretraining logs
│   ├── finetuning/                 # Fine-tuning logs
│   └── uncertainty/                # Uncertainty quantification logs
└── outputs/                        # Analysis outputs
    ├── evaluation/                 # Model evaluation results
    ├── hypotheses/                 # Generated hypotheses
    └── benchmarks/                 # Benchmark results

## Usage

### Generate Data
```bash
# Small dataset (quick testing)
python scripts/generate_cern_data.py --output-dir /root/highspeedstorage/CERN/data --size small

# Comprehensive dataset (for paper)
python scripts/generate_cern_data.py --output-dir /root/highspeedstorage/CERN/data --size comprehensive
```

### Train Model
```bash
# Quick training (small dataset)
bash scripts/train_cern.sh small 50 16 cuda

# Full training (comprehensive dataset)
bash scripts/train_cern.sh comprehensive 50 16 cuda
```

### Evaluate Model
```bash
python scripts/evaluate_model.py \
    --checkpoint /root/highspeedstorage/CERN/checkpoints/comprehensive_training/best_model.pth \
    --test-data /root/highspeedstorage/CERN/data/synthetic_comprehensive \
    --output-dir /root/highspeedstorage/CERN/outputs/evaluation
```

## Storage Usage

Monitor storage usage:
```bash
du -sh /root/highspeedstorage/CERN/*
```

Expected sizes:
- Small dataset: ~150 MB
- Comprehensive dataset: ~12 GB
- Checkpoints: ~500 MB per training run
- Logs: ~100 MB per training run
EOF

echo "✓ README created"
echo ""

# Display structure
echo "Directory structure:"
tree -L 2 "${CERN_ROOT}" 2>/dev/null || find "${CERN_ROOT}" -maxdepth 2 -type d

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Storage root: ${CERN_ROOT}"
echo ""
echo "Next steps:"
echo "1. Generate data:"
echo "   python scripts/generate_cern_data.py --size small"
echo ""
echo "2. Start training:"
echo "   bash scripts/train_cern.sh small 50 16 cuda"
echo ""
echo "=========================================="
