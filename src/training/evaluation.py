from transformers import TrainingArguments, Trainer

def evaluate(model, dataset, collate_fn):
    args = TrainingArguments(output_dir="./eval", remove_unused_columns=False)
    trainer = Trainer(model=model, args=args, data_collator=collate_fn)
    return trainer.evaluate(dataset)