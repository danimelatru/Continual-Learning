import torch
import numpy as np
import random
import os
import matplotlib.pyplot as plt

def set_seed(seed=42):
    """Sets the seed for reproducibility across torch, numpy, and random."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Global seed set to: {seed}")


def print_section(title):
    print(f"\n{'='*20} {title} {'='*20}")

def collate_fn(examples):
    """
    Custom collator to prevent 'input_ids' from crashing CLIP/ViT models 
    that expect only pixel_values.
    """
    return {
        'pixel_values': torch.stack([example['pixel_values'] for example in examples]),
        'labels': torch.tensor([example['labels'] for example in examples])
    }

def plot_results(loss_ideal, loss_ft, loss_lora, save_path):
    """Generates and saves the comparison bar chart."""
    print("Generating comparison graph...")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    labels = ['Ideal (Initial)', 'Fine-Tuning (Forgot)', 'LoRA (Mitigated)']
    values = [loss_ideal, loss_ft, loss_lora]
    colors = ['lightgray', 'salmon', 'lightgreen']

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, color=colors, edgecolor='black', alpha=0.7)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.4f}',
                 ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.ylabel('Eval Loss (Lower is Better)')
    plt.title('Catastrophic Forgetting Mitigation: LoRA vs Full Fine-Tuning')
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    plt.savefig(save_path)
    print(f"Graph saved as {save_path}")

def calculate_bwt(acc_matrix):
    """
    Calculates Backward Transfer (BWT) from an accuracy matrix R.
    R[i, j] = accuracy on task j after training on task i.
    BWT = 1 / (T-1) * sum_{i=1}^{T-1} (R_{T,i} - R_{i,i})
    
    For 2 tasks (A, B):
    BWT = R_{B,A} - R_{A,A}
    """
    # Assuming acc_matrix is a dictionary or simple list for this specific 2-task experiment
    # { 'A': {'A': val, 'B': val}, 'B': {'A': val, 'B': val} }
    # But for now, let's keep it simple as requested in the plan
    pass

def simple_bwt(acc_after_a, acc_after_b):
    """
    Simple BWT for 2-task scenario.
    acc_after_a: Accuracy on Task A after training on Task A
    acc_after_b: Accuracy on Task A after training on Task B
    """
    return acc_after_b - acc_after_a