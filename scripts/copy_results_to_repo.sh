#!/bin/bash
# Copy important results from high-speed storage to repo for committing

CERN_ROOT="/root/highspeedstorage/CERN"
REPO_ROOT="$(dirname "$0")/.."

echo "Copying results from $CERN_ROOT to $REPO_ROOT"

# Create output directories
mkdir -p "$REPO_ROOT/outputs/evaluation"
mkdir -p "$REPO_ROOT/outputs/hypotheses"
mkdir -p "$REPO_ROOT/checkpoints/medium_training"

# Copy evaluation results (small JSON/text files)
echo "Copying evaluation results..."
cp -v "$CERN_ROOT/outputs/evaluation/evaluation_results.json" "$REPO_ROOT/outputs/evaluation/" 2>/dev/null || echo "  Not found"
cp -v "$CERN_ROOT/outputs/evaluation/evaluation_report.txt" "$REPO_ROOT/outputs/evaluation/" 2>/dev/null || echo "  Not found"

# Copy hypothesis results
echo "Copying hypothesis results..."
cp -v "$CERN_ROOT/outputs/quick_hypotheses/scan_results.json" "$REPO_ROOT/outputs/hypotheses/" 2>/dev/null || echo "  Not found"

# Copy training history (not the full checkpoint - too large)
echo "Copying training history..."
cp -v "$CERN_ROOT/checkpoints/medium_training/training_history.json" "$REPO_ROOT/checkpoints/medium_training/" 2>/dev/null || echo "  Not found"

# Copy data summaries (not the actual data - too large)
echo "Copying data summaries..."
mkdir -p "$REPO_ROOT/outputs/data_summaries"
cp -v "$CERN_ROOT/data/synthetic_medium/summary.json" "$REPO_ROOT/outputs/data_summaries/medium_dataset_summary.json" 2>/dev/null || echo "  Not found"
cp -v "$CERN_ROOT/data/synthetic_small/summary.json" "$REPO_ROOT/outputs/data_summaries/small_dataset_summary.json" 2>/dev/null || echo "  Not found"

echo ""
echo "Done! Files copied to repo:"
find "$REPO_ROOT/outputs" "$REPO_ROOT/checkpoints" -name "*.json" -o -name "*.txt" 2>/dev/null | head -20

echo ""
echo "Note: Large files (model checkpoints, training data) were NOT copied."
echo "To push to git:"
echo "  git add outputs/ checkpoints/"
echo "  git commit -m 'Add training results and evaluation outputs'"
echo "  git push origin main"
