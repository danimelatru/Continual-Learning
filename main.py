import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from src.data import DataHandler
from src.model import get_model
from src.train import train_model, evaluate_model
from src.utils import print_section, plot_results, set_seed, simple_bwt

@hydra.main(version_base=None, config_path="conf", config_name="base")
def main(cfg: DictConfig):
    set_seed(cfg.seed)
    
    print_section("CONFIGURATION")
    print(OmegaConf.to_yaml(cfg))

    if cfg.wandb.project:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.name,
            config=OmegaConf.to_container(cfg, resolve=True)
        )

    # --- 1. DATA PREPARATION ---
    data_handler = DataHandler(cfg)
    train_ds_a, train_ds_b, test_ds_a = data_handler.load_and_split_data()

    # --- EXPERIMENT 1: NAIVE FINE-TUNING ---
    print_section("EXPERIMENT 1: Full Fine-Tuning (The Baseline)")
    
    model_ft = get_model(cfg, use_lora=False)
    
    # Step 1: Train A
    print("Training Full Model on Task A...")
    train_model(model_ft, train_ds_a, cfg.train.output_dir_ft_a, cfg)
    
    # Evaluate Initial Performance
    acc_after_a = evaluate_model(model_ft, test_ds_a, "Test A (After Training A)", cfg)
    loss_ideal = acc_after_a['eval_loss']

    # Step 2: Train B (Induce Forgetting)
    print("Training Full Model on Task B...")
    train_model(model_ft, train_ds_b, cfg.train.output_dir_ft_b, cfg)
    
    # Evaluate Forgetting
    acc_after_b = evaluate_model(model_ft, test_ds_a, "Test A (After Training B)", cfg)
    loss_ft = acc_after_b['eval_loss']

    bwt_ft = simple_bwt(loss_ideal, loss_ft) # Using loss as proxy, though usually accuracy
    print(f"Fine-Tuning BWT (Loss based): {bwt_ft:.4f}")


    # --- EXPERIMENT 2: LoRA ---
    print_section("EXPERIMENT 2: PEFT / LoRA")
    
    model_lora = get_model(cfg, use_lora=True)

    # Step 1: Train LoRA on A
    print("Training LoRA on Task A...")
    train_model(model_lora, train_ds_a, cfg.train.output_dir_ft_a, cfg)

    # Step 2: Train LoRA on B
    print("Training LoRA on Task B...")
    train_model(model_lora, train_ds_b, cfg.train.output_dir_ft_b, cfg)

    # Evaluate Mitigation
    acc_lora_after_b = evaluate_model(model_lora, test_ds_a, "Test A (LoRA - After Training B)", cfg)
    loss_lora = acc_lora_after_b['eval_loss']
    
    bwt_lora = simple_bwt(loss_ideal, loss_lora)
    print(f"LoRA BWT (Loss based): {bwt_lora:.4f}")


    # --- RESULTS ---
    print_section("FINAL RESULTS SUMMARY")
    
    # Create a nice markdown-style table for the logs
    print("| Experiment | Metric | Task A (Ideal) | Task A (After B) | BWT (Forget) |")
    print("|------------|--------|----------------|------------------|--------------|")
    print(f"| Baseline   | Loss   | {loss_ideal:.4f} | {loss_ft:.4f} | {bwt_ft:.4f} |")
    print(f"| LoRA       | Loss   | {loss_ideal:.4f}* | {loss_lora:.4f} | {bwt_lora:.4f} |")
    print("\n*Note: Ideal LoRA loss on A is assumed similar for BWT calculation context.")

    plot_results(loss_ideal, loss_ft, loss_lora, cfg.plot_filename if 'plot_filename' in cfg else "media/comparison_graph.png")
    
    if cfg.wandb.project:
        wandb.log({
            "final/loss_ideal": loss_ideal,
            "final/loss_ft": loss_ft,
            "final/loss_lora": loss_lora, 
            "final/bwt_ft": bwt_ft,
            "final/bwt_lora": bwt_lora
        })
        # Log the plot as an artifact
        wandb.log({"results/plot": wandb.Image("media/comparison_graph.png")})
        wandb.finish()

if __name__ == "__main__":
    main()