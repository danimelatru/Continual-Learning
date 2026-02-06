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
        report_to="wandb" if cfg.wandb.project else "none"
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
    print(f"Evaluating on {name}...")
    
    args_eval = TrainingArguments(
        output_dir=cfg.train.eval_results_dir, 
        remove_unused_columns=False,
        report_to="wandb" if cfg.wandb.project else "none"
    )
    
    trainer = Trainer(
        model=model, 
        args=args_eval, 
        data_collator=collate_fn
    )
    
    eval_result = trainer.evaluate(dataset) 
    print(f"Accuracy/Loss on {name}: {eval_result}")
    
    if cfg.wandb.project:
        wandb.log({f"eval/{name}": eval_result})
        
    return eval_result