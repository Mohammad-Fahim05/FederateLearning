import pytest
import yaml
import torch
from src.data.dataset import Camelyon17HospitalDataset
from src.data.client_splitter import HospitalClientSplitter
from src.federated.trainer import FederatedTrainer

def test_federated_training_loop():
    with open('./configs/camelyon17_wilds.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    dataset = Camelyon17HospitalDataset(use_synthetic=True, seed=42)
    splitter = HospitalClientSplitter(dataset, train_centers=[0, 1, 2], seed=42)
    client_indices = splitter.get_natural_split()
    
    trainer = FederatedTrainer(
        dataset=dataset,
        client_indices=client_indices,
        config=config,
        method_name='DP-WHFedDG',
        device='cpu'
    )
    
    history = trainer.run_training(rounds=2)
    assert len(history['rounds']) == 2, "Expected 2 rounds executed"
    assert history['rounds'] == [1, 2]
    assert len(history['client_losses']) == 2
    print("Integration Test: Federated Training Loop PASSED!")

if __name__ == "__main__":
    test_federated_training_loop()
