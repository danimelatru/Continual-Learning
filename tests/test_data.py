import pytest
import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import DataHandler

class TestDataHandler:
    def test_data_splitting_disjoint(self):
        """Verify that Task A and Task B have no overlapping classes."""
        handler = DataHandler()
        # Mocking or using a small subset would be better, but for now we test the logic on the real load
        # Warning: This might be slow if it downloads cifar every time, but datasets caches it.
        train_ds_a, train_ds_b, test_ds_a = handler.load_and_split_data()
        
        # Check Task A classes (0-4)
        labels_a = train_ds_a['label']
        assert all(l < 5 for l in labels_a), "Task A contains classes >= 5"
        
        # Check Task B classes (5-9)
        labels_b = train_ds_b['label']
        assert all(l >= 5 for l in labels_b), "Task B contains classes < 5"
        
        # Check leakage
        set_a = set(labels_a)
        set_b = set(labels_b)
        assert set_a.isdisjoint(set_b), "Task A and Task B have overlapping classes!"
        
    def test_transform_shape(self):
        """Verify that transforms produce correctly shaped tensors."""
        handler = DataHandler()
        train_ds_a, _, _ = handler.load_and_split_data()
        
        # Get one sample
        sample = train_ds_a[0] # This triggers the transform
        
        assert 'pixel_values' in sample
        assert 'labels' in sample
        # ViT expects [3, 224, 224] usually
        assert sample['pixel_values'].shape == (3, 224, 224)
