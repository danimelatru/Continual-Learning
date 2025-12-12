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
conda activate continual_learning

PROJECT_ROOT=/gpfs/workdir/fernandeda/continual_learning
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

mkdir -p logs
cd "$PROJECT_ROOT"

echo "=========================================="
echo "[DEBUG SLURM] Hostname: $(hostname)"
echo "[DEBUG SLURM] Working directory: $(pwd)"
echo "[DEBUG SLURM] Python: $(which python)"
echo "[DEBUG SLURM] CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

nvidia-smi || echo "[DEBUG SLURM] nvidia-smi failed"

python -u - << 'EOF'
import torch, os
print("[DEBUG PY] torch:", torch.__file__)
print("[DEBUG PY] version:", torch.__version__)
print("[DEBUG PY] cuda available:", torch.cuda.is_available())
print("[DEBUG PY] cuda count:", torch.cuda.device_count())
print("[DEBUG PY] visible:", os.environ.get("CUDA_VISIBLE_DEVICES"))
EOF

python -u main.py