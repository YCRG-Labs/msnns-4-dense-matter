#!/bin/bash
# Training script for CERN high-speed storage
# All data and checkpoints stored in /root/highspeedstorage/CERN

set -e

# Configuration
CERN_ROOT="/root/highspeedstorage/CERN"
DATA_DIR="${CERN_ROOT}/data"
CHECKPOINT_DIR="${CERN_ROOT}/checkpoints"
LOG_DIR="${CERN_ROOT}/logs"

# Parse arguments
DATASET_SIZE="${1:-small}"  # small or comprehensive
EPOCHS="${2:-50}"
BATCH_SIZE="${3:-16}"
DEVICE="${4:-cuda}"

echo "=========================================="
echo "CERN Training Pipeline"
echo "=========================================="
echo "Dataset size: ${DATASET_SIZE}"
echo "Epochs: ${EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Device: ${DEVICE}"
echo "Storage root: ${CERN_ROOT}"
echo "=========================================="
echo ""

# Create directories
echo "Creating directories..."
mkdir -p "${DATA_DIR}"
mkdir -p "${CHECKPOINT_DIR}"
mkdir -p "${LOG_DIR}"

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "Warning: Virtual environment not found"
fi

# Generate data if not exists
if [ "${DATASET_SIZE}" = "small" ]; then
    DATA_PATH="${DATA_DIR}/synthetic_small"
    CHECKPOINT_PATH="${CHECKPOINT_DIR}/small_training"
else
    DATA_PATH="${DATA_DIR}/synthetic_comprehensive"
    CHECKPOINT_PATH="${CHECKPOINT_DIR}/comprehensive_training"
fi

if [ ! -d "${DATA_PATH}" ]; then
    echo "Generating ${DATASET_SIZE} dataset..."
    python scripts/generate_cern_data.py \
        --output-dir "${DATA_DIR}" \
        --size "${DATASET_SIZE}"
    echo "✓ Data generation complete"
else
    echo "✓ Dataset already exists: ${DATA_PATH}"
fi

# Test GPU
echo ""
echo "Testing GPU..."
python scripts/test_gpu.py

# Start training
echo ""
echo "Starting training..."
echo "Data: ${DATA_PATH}"
echo "Checkpoints: ${CHECKPOINT_PATH}"
echo ""

python scripts/train_pretrain.py \
    --train-data "${DATA_PATH}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --device "${DEVICE}" \
    --checkpoint-dir "${CHECKPOINT_PATH}" \
    --log-interval 10

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo "Checkpoints saved to: ${CHECKPOINT_PATH}"
echo "Logs saved to: ${LOG_DIR}"
echo ""
echo "Next steps:"
echo "1. Evaluate model:"
echo "   python scripts/evaluate_model.py \\"
echo "       --checkpoint ${CHECKPOINT_PATH}/best_model.pth \\"
echo "       --test-data ${DATA_PATH} \\"
echo "       --output-dir ${CERN_ROOT}/outputs/evaluation"
echo ""
echo "2. Generate hypotheses:"
echo "   python scripts/generate_hypotheses.py \\"
echo "       --checkpoint ${CHECKPOINT_PATH}/best_model.pth \\"
echo "       --data ${DATA_PATH} \\"
echo "       --output-dir ${CERN_ROOT}/outputs/hypotheses"
echo "=========================================="
