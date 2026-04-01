#!/bin/bash
# TFLE Server Setup — Run this first on vast.ai instance
# Usage: bash setup.sh

set -e

echo "=== TFLE Server Setup ==="

# Install deps
pip install torch torchvision tqdm

# Clone repo
cd /workspace
if [ ! -d "tfle" ]; then
    git clone https://github.com/joe51111jwd/tfle.git
fi
cd tfle

# Apply the task-loss fitness fix (if not already in repo)
# These patches add FitnessType.TASK_LOSS to config, layers, and model
python -c "from tfle.config import FitnessType; print(f'TASK_LOSS available: {hasattr(FitnessType, \"TASK_LOSS\")}')"

# Copy server scripts
mkdir -p server checkpoints

echo ""
echo "=== Setup complete ==="
echo "Run experiment: python server/run_100k.py"
echo "Monitor: tail -f nohup.out"
