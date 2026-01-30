#!/bin/bash
# Quick start script for cloud MD generation
# Run this after cloud_setup.sh completes

set -e

echo "========================================="
echo "Quick Start: MD Data Generation"
echo "========================================="
echo ""

# Configuration
NUM_CONFIGS=${1:-1000}
BATCH_SIZE=${2:-10}
OUTPUT_DIR="data/md_simulations"

echo "Configuration:"
echo "  Number of configs: $NUM_CONFIGS"
echo "  Batch size: $BATCH_SIZE"
echo "  Output directory: $OUTPUT_DIR"
echo ""

# Step 1: Generate input files
echo "Step 1/3: Generating LAMMPS input files..."
python scripts/generate_md_data.py \
    --output_dir $OUTPUT_DIR \
    --num_configs $NUM_CONFIGS \
    --batch_size $BATCH_SIZE

echo "✓ Input files generated"
echo ""

# Step 2: Run simulations
echo "Step 2/3: Running MD simulations..."
echo "This will take 2-3 days. You can:"
echo "  - Monitor: watch -n 10 'ls -1d $OUTPUT_DIR/*/log.lammps | wc -l'"
echo "  - Check logs: tail -f $OUTPUT_DIR/Si_0001/log.lammps"
echo "  - Detach: Use screen or tmux"
echo ""
read -p "Start simulations now? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    cd $OUTPUT_DIR
    
    # Use screen if available
    if command -v screen &> /dev/null; then
        echo "Starting in screen session 'md-sims'..."
        screen -dmS md-sims bash -c "./run_all.sh; echo 'Simulations complete!' > DONE"
        echo "✓ Simulations started in background"
        echo "  Attach: screen -r md-sims"
        echo "  Detach: Ctrl+A, then D"
    else
        echo "Starting simulations (this will take a while)..."
        ./run_all.sh
        echo "✓ Simulations complete"
    fi
    
    cd -
else
    echo "Skipping simulation run."
    echo "To run later: cd $OUTPUT_DIR && ./run_all.sh"
fi

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Wait for simulations to complete (2-3 days)"
echo "2. Convert to training format:"
echo "   python scripts/convert_md_to_training.py \\"
echo "       --input_dir $OUTPUT_DIR \\"
echo "       --output_dir data/training_data"
echo "3. Download data to local machine"
echo "4. Start training!"
echo ""
echo "Monitor progress:"
echo "  watch -n 10 'ls -1d $OUTPUT_DIR/*/log.lammps | wc -l'"
echo "  Expected: $NUM_CONFIGS completed simulations"
echo "========================================="
