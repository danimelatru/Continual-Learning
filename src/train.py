# src/train.py
from transformers import TrainingArguments, Trainer
from .config import Config
from .utils import collate_fn

def train_model(model, dataset, output_dir):
    """Generic training wrapper."""
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=Config.EPOCHS,
        per_device_train_batch_size=Config.BATCH_SIZE,
        logging_steps=50,
        save_strategy="no",
        remove_unused_columns=False
    )
    
    trainer = Trainer(
        model=model, 
        args=args, 
        train_dataset=dataset, 
        data_collator=collate_fn
    )
    trainer.train()
    return trainer

def evaluate_model(model, dataset, name):
    """Generic evaluation wrapper."""
    print(f"Evaluating on {name}...")
    
    args_eval = TrainingArguments(
        output_dir=Config.EVAL_RESULTS_DIR, 
        remove_unused_columns=False 
    )
    
    trainer = Trainer(
        model=model, 
        args=args_eval, 
        data_collator=collate_fn
    )
    
    eval_result = trainer.evaluate(dataset) 
    print(f"Accuracy/Loss on {name}: {eval_result}")
    return eval_result