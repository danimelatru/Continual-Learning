#!/bin/bash
#SBATCH --job-name=cl_lora
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --exclude=ruche-gpu02,ruche-gpu11,ruche-gpu16,ruche-gpu17,ruche-gpu19

module purge
source ~/.bashrc
conda activate research

# Automatically detect the project root (assuming script is in scripts/)
PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Create logs directory if it doesn't exist
mkdir -p "$PROJECT_ROOT/logs"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "[DEBUG SLURM] Hostname: $(hostname)"
echo "[DEBUG SLURM] Working directory: $(pwd)"
echo "[DEBUG SLURM] Python: $(which python)"
echo "[DEBUG SLURM] CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "[DEBUG SLURM] Arguments: $@"
echo "=========================================="

nvidia-smi || echo "[DEBUG SLURM] nvidia-smi failed"

# Run the main script with Hydra arguments passed from sbatch
# Usage: sbatch scripts/run_slurm.sh train.epochs=10 lora.r=64
python -u main.py "$@"