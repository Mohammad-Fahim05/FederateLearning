import pytest
import torch
from src.data.dataset import Camelyon17HospitalDataset
from src.data.client_splitter import HospitalClientSplitter

def test_dataset_loading():
    dataset = Camelyon17HospitalDataset(use_synthetic=True, seed=42)
    assert len(dataset) > 0, "Dataset should not be empty"
    
    img, label, center_id, slide_id = dataset[0]
    assert isinstance(img, torch.Tensor), "Image must be a PyTorch Tensor"
    assert img.shape == (3, 96, 96), f"Expected shape (3, 96, 96), got {img.shape}"
    assert label in [0, 1], f"Label must be 0 or 1, got {label}"
    assert center_id in range(5), f"Center ID must be in 0-4, got {center_id}"

def test_client_splitter_no_leakage():
    dataset = Camelyon17HospitalDataset(use_synthetic=True, seed=42)
    splitter = HospitalClientSplitter(dataset, train_centers=[0, 1, 2], seed=42)
    client_indices = splitter.get_natural_split()
    
    assert len(client_indices) == 3, "Expected 3 training hospital clients"
    
    # Check zero overlap across hospital client index sets
    set0 = set(client_indices[0])
    set1 = set(client_indices[1])
    set2 = set(client_indices[2])
    
    assert len(set0.intersection(set1)) == 0, "Client 0 and Client 1 indices overlap!"
    assert len(set1.intersection(set2)) == 0, "Client 1 and Client 2 indices overlap!"
    assert len(set0.intersection(set2)) == 0, "Client 0 and Client 2 indices overlap!"
    print("Zero slide/patch leakage test PASSED!")

if __name__ == "__main__":
    test_dataset_loading()
    test_client_splitter_no_leakage()
