# Continual Learning with LoRA: Mitigating Catastrophic Forgetting

This project implements a **Continual Learning (CL)** experiment to demonstrate how **Parameter-Efficient Fine-Tuning (PEFT)** techniques, specifically **LoRA (Low-Rank Adaptation)**, help mitigate Catastrophic Forgetting compared to traditional Full Fine-Tuning.

The base model used is a pre-trained **Vision Transformer (ViT)** (`google/vit-base-patch16-224`) trained on the **CIFAR-10** dataset, which is split into two sequential tasks.

## 🧪 Experiment Overview

The goal is to evaluate the model's ability to **remember Task A** (Classes 0-4) after being re-trained to learn **Task B** (Classes 5-9).

### The Dataset Split
* **Task A:** CIFAR-10 Classes 0–4 (Airplane, Automobile, Bird, Cat, Deer)
* **Task B:** CIFAR-10 Classes 5–9 (Dog, Frog, Horse, Ship, Truck)

### The Comparison scenarios
1.  **Baseline: Full Fine-Tuning**
    * All model parameters are updated during training.
    * **Hypothesis:** The model will suffer from severe **Catastrophic Forgetting** (high loss on Task A after learning Task B).
2.  **Solution: LoRA (PEFT)**
    * The pre-trained backbone is frozen. Only small low-rank adapter matrices are injected and trained.
    * **Hypothesis:** The frozen backbone preserves general knowledge, resulting in significantly **lower forgetting** on Task A.

## 📂 Project Structure

The project is organized into a modular structure for scalability and readability:

```text
cifar-lora-forgetting/
│
├── media/               # Generated comparison plots
├── src/                 # Source code package
│   ├── config.py        # Hyperparameters (Batch size, LR, Model ID)
│   ├── data.py          # Data loading, splitting, and preprocessing
│   ├── model.py         # Model initialization (Base ViT & LoRA config)
│   ├── train.py         # Training loop and evaluation wrappers
│   └── utils.py         # Helper functions (collators, plotting)
├── main.py              # Entry point to run the experiments
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
````

## 🚀 Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/cifar-lora-forgetting.git](https://github.com/your-username/cifar-lora-forgetting.git)
    cd cifar-lora-forgetting
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

## ⚙️ Usage

To run the full experiment pipeline (Data Split -> Fine-Tuning -> LoRA -> Plotting):

```bash
python main.py

### Configuration
You can modify hyperparameters in `src/config.py`:
* `BATCH_SIZE`: Default 32
* `EPOCHS`: Default 5
* `MODEL_CHECKPOINT`: Default `google/vit-base-patch16-224`

## 📊 Results

After execution, the script saves a bar chart to `media/comparison_graph.png`.

**Interpretation:**
* **Ideal (Initial):** The loss on Task A immediately after training on Task A (the "Upper Bound").
* **Fine-Tuning:** The loss on Task A increases dramatically after training on Task B (Forgetting).
* **LoRA:** The loss on Task A remains much closer to the Ideal state, proving effective mitigation of forgetting.

## 🛠 Tech Stack

* **PyTorch**: Deep Learning Framework.
* **Hugging Face Transformers**: For the Vision Transformer (ViT) architecture.
* **PEFT (Parameter-Efficient Fine-Tuning)**: For LoRA implementation.
* **Datasets**: For easy loading of CIFAR-10.
* **Matplotlib**: For visualization.

## 📄 License

This project is open-source and available under the MIT License.