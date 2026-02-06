# src/train.py
from transformers import TrainingArguments, Trainer
from omegaconf import DictConfig
from .utils import collate_fn
import wandb

def train_model(model, dataset, output_dir, cfg: DictConfig):
    """Generic training wrapper."""
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg.train.epochs,
        per_device_train_batch_size=cfg.data.batch_size,
        learning_rate=cfg.train.learning_rate,
        logging_steps=cfg.train.logging_steps,
        save_strategy="no",
        remove_unused_columns=False,
        report_to="wandb" if cfg.wandb.project else "none",
        fp16=cfg.train.fp16,
        dataloader_num_workers=cfg.train.dataloader_num_workers,
        disable_tqdm=True # Silence TQDM in cluster logs
    )
    
    trainer = Trainer(
        model=model, 
        args=args, 
        train_dataset=dataset, 
        data_collator=collate_fn
    )
    trainer.train()
    return trainer

def evaluate_model(model, dataset, name, cfg: DictConfig):
    """Generic evaluation wrapper."""
    print(f"\033[94m>>> Evaluating on {name}...\033[0m")
    
    args_eval = TrainingArguments(
        output_dir=cfg.train.eval_results_dir, 
        per_device_eval_batch_size=cfg.data.batch_size,
        remove_unused_columns=False,
        report_to="none", # Don't clutter WandB with tiny intermediate eval steps
        fp16=cfg.train.fp16,
        disable_tqdm=True # Silence TQDM in logs
    )
    
    trainer = Trainer(
        model=model, 
        args=args_eval, 
        data_collator=collate_fn
    )
    
    eval_result = trainer.evaluate(dataset) 
    
    # Ensure eval_loss exists to avoid KeyError in main.py
    if 'eval_loss' not in eval_result:
        # This can happen if the model is in a weird state or labels are missing
        eval_result['eval_loss'] = 0.0
        print(f"\033[93mWarning: 'eval_loss' not found for {name}. Defaulting to 0.0\033[0m")

    print(f"Results for {name}: Loss {eval_result['eval_loss']:.4f}")
    
    if cfg.wandb.project:
        # Log to wandb with a cleaner name
        wandb.log({f"eval/{name.replace(' ', '_')}": eval_result['eval_loss']})
        
    return eval_result