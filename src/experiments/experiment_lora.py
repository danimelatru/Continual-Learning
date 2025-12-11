from transformers import TrainingArguments, Trainer

from src.data.dataset_cifar import load_cifar10_tasks
from src.models.clip_classifier import load_clip_classifier
from src.models.lora_wrapper import add_lora
from src.training.collate import collate_clip
from src.training.evaluation import evaluate

CHECKPOINT = "openai/clip-vit-base-patch32"
EPOCHS = 5
BATCH_SIZE = 32

train_a, train_b, test_a = load_cifar10_tasks(CHECKPOINT)

model = load_clip_classifier(CHECKPOINT)
model = add_lora(model)

args = TrainingArguments(
    output_dir="./results_lora_A",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    remove_unused_columns=False
)

trainer = Trainer(model=model, args=args, train_dataset=train_a, data_collator=collate_clip)
trainer.train()

print("Eval after Task A")
eval_a = evaluate(model, test_a, collate_clip)

args_b = TrainingArguments(
    output_dir="./results_lora_B",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    remove_unused_columns=False
)

trainer = Trainer(model=model, args=args_b, train_dataset=train_b, data_collator=collate_clip)
trainer.train()

print("Eval after Task B")
eval_b = evaluate(model, test_a, collate_clip)