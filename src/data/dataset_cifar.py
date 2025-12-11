import torch
from datasets import load_dataset
from transformers import AutoImageProcessor

def load_cifar10_tasks(model_checkpoint):
    dataset = load_dataset("cifar10")

    task_a_indices = [i for i, y in enumerate(dataset["train"]["label"]) if y < 5]
    task_b_indices = [i for i, y in enumerate(dataset["train"]["label"]) if y >= 5]
    test_a_indices = [i for i, y in enumerate(dataset["test"]["label"]) if y < 5]

    train_a = dataset["train"].select(task_a_indices)
    train_b = dataset["train"].select(task_b_indices)
    test_a = dataset["test"].select(test_a_indices)

    processor = AutoImageProcessor.from_pretrained(model_checkpoint)

    def transform(batch):
        imgs = batch["image"]
        inputs = processor(imgs, return_tensors="pt")
        inputs["labels"] = batch["label"]
        return inputs

    train_a.set_transform(transform)
    train_b.set_transform(transform)
    test_a.set_transform(transform)

    return train_a, train_b, test_a