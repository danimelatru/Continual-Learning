# src/data.py
from datasets import load_dataset
from transformers import AutoImageProcessor
from .config import Config

class DataHandler:
    def __init__(self):
        self.processor = AutoImageProcessor.from_pretrained(Config.MODEL_CHECKPOINT)

    def _transform(self, example_batch):
        key = 'image' if 'image' in example_batch else 'img'
        inputs = self.processor([x for x in example_batch[key]], return_tensors='pt')
        inputs['labels'] = example_batch['label']
        return inputs

    def load_and_split_data(self):
        print("Loading and Splitting CIFAR-10...")
        dataset = load_dataset("cifar10")

        # Define split logic
        # Task A: 0-4, Task B: 5-9
        task_a_indices = [i for i, label in enumerate(dataset['train']['label']) if label < 5]
        task_b_indices = [i for i, label in enumerate(dataset['train']['label']) if label >= 5]
        test_a_indices = [i for i, label in enumerate(dataset['test']['label']) if label < 5]

        train_ds_a = dataset['train'].select(task_a_indices)
        train_ds_b = dataset['train'].select(task_b_indices)
        test_ds_a = dataset['test'].select(test_a_indices)

        print(f"Task A (Classes 0-4): {len(train_ds_a)} samples")
        print(f"Task B (Classes 5-9): {len(train_ds_b)} samples")

        # Apply transforms
        train_ds_a.set_transform(self._transform)
        train_ds_b.set_transform(self._transform)
        test_ds_a.set_transform(self._transform)

        return train_ds_a, train_ds_b, test_ds_a