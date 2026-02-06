# Continual Learning with LoRA: Mitigating Catastrophic Forgetting

This project evaluates **LoRA (Low-Rank Adaptation)** as a method to mitigate **Catastrophic Forgetting** in a Continual Learning setting on CIFAR-10.

## 🚀 Key Features
- **Professional Configs**: Powered by [Hydra](https://hydra.cc/).
- **Experiment Tracking**: Real-time logging with [Weights & Biases](https://wandb.ai/).
- **Cluster Ready**: Optimized for SLURM execution.
- **Reproducibility**: Parameter pinning and seeded execution.

## 🛠 Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/cifar-lora-forgetting.git
    cd cifar-lora-forgetting
    ```

2.  **Install dependencies**:
    ```bash
    conda create -n cl-research python=3.10
    conda activate cl-research
    pip install -r requirements.txt
    ```

## ⚙️ Usage

### Local Execution
Run the experiment using **Hydra**:

```bash
python main.py
```

### Cluster Execution (SLURM)
Use the provided script to submit jobs to a specific managed environment (e.g., `research`):

```bash
# Standard run
sbatch scripts/run_slurm.sh

# Override parameters on the fly
sbatch scripts/run_slurm.sh train.epochs=50 lora.r=32 wandb.name="experiment-50epochs"
```

## 👥 Configuration
You can modify any parameter from the command line:

```bash
# Change epochs and learning rate
python main.py train.epochs=10 train.learning_rate=1e-4

# Disable WandB for debugging
python main.py wandb.project=null

# Change model checkpoint
python main.py model.checkpoint="google/vit-base-patch32-224-in21k"
```

## 📊 Results & WandB
The script automatically logs:
- **Loss curves** (Training & Eval)
- **Backward Transfer (BWT)**: Measures how much the model forgot Task A after learning Task B.
- **Comparison Plots**: Saved locally in `media/` and uploaded to WandB.

### Experiment Structure
The code sequentially:
1.  Trains on **Task A** (CIFAR-10 classes 0-4).
2.  Evaluates on Task A ("Ideal" baseline).
3.  Trains on **Task B** (CIFAR-10 classes 5-9).
4.  Evaluates on Task A again ("Forgetting").
5.  Calculates metrics.

This is repeated for both **Full Fine-Tuning** and **LoRA**.

## 📂 Project Structure
- `conf/`: Hydra configuration files (`base.yaml`, etc.).
- `src/`: Source code (`model.py`, `train.py`, `data.py`, `utils.py`).
- `main.py`: Entry point.
- `scripts/`: Execution scripts (`run_slurm.sh`).
- `tests/`: Unit tests.

## 📄 License
MIT License.