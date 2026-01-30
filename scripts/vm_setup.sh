#!/bin/bash
# VM Setup Script for Training
# Sets up environment on Linux VM (local or cloud)

set -e  # Exit on error

echo "=========================================="
echo "Multi-Scale Neural Network - VM Setup"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo -e "${YELLOW}Warning: This script is designed for Linux. You're running: $OSTYPE${NC}"
    echo "Continuing anyway..."
fi

# Update system packages
echo -e "${GREEN}[1/8] Updating system packages...${NC}"
sudo apt-get update -qq

# Install system dependencies
echo -e "${GREEN}[2/8] Installing system dependencies...${NC}"
sudo apt-get install -y -qq \
    python3 \
    python3-pip \
    python3-venv \
    git \
    wget \
    curl \
    build-essential \
    libhdf5-dev \
    pkg-config

# Check Python version
echo -e "${GREEN}[3/8] Checking Python version...${NC}"
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python version: $PYTHON_VERSION"

if (( $(echo "$PYTHON_VERSION < 3.8" | bc -l) )); then
    echo -e "${RED}Error: Python 3.8+ required. Found: $PYTHON_VERSION${NC}"
    exit 1
fi

# Create virtual environment
echo -e "${GREEN}[4/8] Creating virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Virtual environment created"
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo -e "${GREEN}[5/8] Upgrading pip...${NC}"
pip install --upgrade pip setuptools wheel -q

# Install PyTorch (with CUDA if available)
echo -e "${GREEN}[6/8] Installing PyTorch...${NC}"
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected, installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -q
else
    echo "No GPU detected, installing CPU-only PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu -q
fi

# Install requirements
echo -e "${GREEN}[7/8] Installing Python packages...${NC}"
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt -q
    echo "Requirements installed"
else
    echo -e "${YELLOW}Warning: requirements.txt not found${NC}"
fi

# Verify installation
echo -e "${GREEN}[8/8] Verifying installation...${NC}"

# Check PyTorch
python3 << EOF
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
EOF

# Check other packages
python3 << EOF
try:
    import numpy
    import scipy
    import hydra
    import tqdm
    print("\nAll required packages installed successfully!")
except ImportError as e:
    print(f"\nWarning: Missing package: {e}")
EOF

# Run tests
echo ""
echo -e "${GREEN}Running tests...${NC}"
if pytest tests/ -q --tb=no; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
else
    echo -e "${YELLOW}⚠ Some tests failed. Check output above.${NC}"
fi

# Create necessary directories
echo ""
echo -e "${GREEN}Creating directories...${NC}"
mkdir -p data/synthetic_small
mkdir -p data/synthetic_large
mkdir -p checkpoints/vm_training
mkdir -p logs/pretraining_vm
mkdir -p outputs/evaluation
mkdir -p outputs/hypotheses
echo "Directories created"

# Print GPU info if available
echo ""
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}GPU Information:${NC}"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
fi

# Print summary
echo ""
echo "=========================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Activate environment: source venv/bin/activate"
echo "2. Generate data: python scripts/generate_small_data.py"
echo "3. Start training: python scripts/train_pretrain.py --train-data data/synthetic_small --epochs 50 --batch-size 16 --device cuda"
echo ""
echo "For detailed instructions, see: VM_SETUP.md"
echo ""

# Save environment info
cat > vm_info.txt << EOF
Setup completed: $(date)
OS: $(uname -a)
Python: $(python3 --version)
PyTorch: $(python3 -c "import torch; print(torch.__version__)")
CUDA Available: $(python3 -c "import torch; print(torch.cuda.is_available())")
EOF

if command -v nvidia-smi &> /dev/null; then
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)" >> vm_info.txt
fi

echo "Environment info saved to: vm_info.txt"
