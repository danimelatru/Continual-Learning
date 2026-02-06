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

# Handle Project Root detection for both SLURM (sbatch) and local execution
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    # When running via sbatch, use the submission directory
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    # When running locally, assume script is in scripts/ relative to root
    PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
fi

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# WandB Cluster Fixes: 
# Ensure WandB uses project directory for temp files and cache (avoiding /scratch or /tmp permission issues)
export WANDB_DIR="$PROJECT_ROOT"
export WANDB_CACHE_DIR="$PROJECT_ROOT/.cache/wandb"
export WANDB_CONFIG_DIR="$PROJECT_ROOT/.config/wandb"
export WANDB_START_METHOD="thread"

# Create necessary directories
mkdir -p "$PROJECT_ROOT/logs"
mkdir -p "$WANDB_CACHE_DIR"
mkdir -p "$WANDB_CONFIG_DIR"

cd "$PROJECT_ROOT"

# Run the main script with Hydra arguments passed from sbatch
# Usage: sbatch scripts/run_slurm.sh train.epochs=10 lora.r=64
python -u main.py "$@"