# main.py
from src.config import Config
from src.data import DataHandler
from src.model import get_model
from src.train import train_model, evaluate_model
from src.utils import print_section, plot_results

def main():
    # --- 1. DATA PREPARATION ---
    data_handler = DataHandler()
    train_ds_a, train_ds_b, test_ds_a = data_handler.load_and_split_data()

    # --- EXPERIMENT 1: NAIVE FINE-TUNING ---
    print_section("EXPERIMENT 1: Full Fine-Tuning (The Baseline)")
    
    model_ft = get_model(use_lora=False)
    
    # Step 1: Train A
    print("Training Full Model on Task A...")
    train_model(model_ft, train_ds_a, Config.OUTPUT_DIR_FT_A)
    
    # Evaluate Initial Performance
    acc_after_a = evaluate_model(model_ft, test_ds_a, "Test A (After Training A)")
    loss_ideal = acc_after_a['eval_loss']

    # Step 2: Train B (Induce Forgetting)
    print("Training Full Model on Task B...")
    train_model(model_ft, train_ds_b, Config.OUTPUT_DIR_FT_B)
    
    # Evaluate Forgetting
    acc_after_b = evaluate_model(model_ft, test_ds_a, "Test A (After Training B)")
    loss_ft = acc_after_b['eval_loss']


    # --- EXPERIMENT 2: LoRA ---
    print_section("EXPERIMENT 2: PEFT / LoRA")
    
    model_lora = get_model(use_lora=True)

    # Step 1: Train LoRA on A
    print("Training LoRA on Task A...")
    train_model(model_lora, train_ds_a, Config.OUTPUT_DIR_FT_A)

    # Step 2: Train LoRA on B
    print("Training LoRA on Task B...")
    train_model(model_lora, train_ds_b, Config.OUTPUT_DIR_FT_B)

    # Evaluate Mitigation
    acc_lora_after_b = evaluate_model(model_lora, test_ds_a, "Test A (LoRA - After Training B)")
    loss_lora = acc_lora_after_b['eval_loss']


    # --- RESULTS ---
    print_section("FINAL RESULTS")
    print(f"Baseline - Loss on A after learning B: {loss_ft:.4f}")
    print(f"LoRA     - Loss on A after learning B: {loss_lora:.4f}")

    plot_results(loss_ideal, loss_ft, loss_lora, Config.PLOT_FILENAME)

if __name__ == "__main__":
    main()