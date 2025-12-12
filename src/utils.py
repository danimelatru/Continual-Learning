# src/utils.py
import torch
import matplotlib.pyplot as plt
import os

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