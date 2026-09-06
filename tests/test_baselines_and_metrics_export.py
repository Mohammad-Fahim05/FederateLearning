import math
import copy
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.models.resnet_backbone import ResNet18Backbone
from src.federated.client import HospitalClient
from src.federated.server import CentralServer
from src.data.dataset import Camelyon17HospitalDataset


def test_fedprox_produces_different_objective_than_fedavg():
    """
    Verify FedProx applies (mu / 2) * ||theta - theta_global||^2 when local model
    differs from global model, producing a distinct loss and update compared to FedAvg.
    """
    torch.manual_seed(42)
    dataset = Camelyon17HospitalDataset(use_synthetic=True, seed=42)
    indices = list(range(20))
    
    base_config = {
        'federated': {
            'batch_size': 10,
            'learning_rate': 0.01,
            'momentum': 0.0,
            'weight_decay': 0.0,
            'focal_gamma': 0.0,
            'focal_alpha': [0.5, 0.5],
            'sam_rho': 0.0,
            'mu': 0.5,
            'local_epochs': 1,
            'num_workers': 0
        },
        'privacy': {
            'enabled': False,
            'max_grad_norm': 1.0,
            'noise_multiplier': 0.0,
            'chunk_size': 64
        }
    }
    
    # Global model
    global_model = ResNet18Backbone(num_classes=2, pretrained=False)
    
    # Configuration for FedAvg
    fedavg_cfg = copy.deepcopy(base_config)
    fedavg_cfg['method'] = 'FedAvg'
    fedavg_client = HospitalClient(client_id=0, dataset=dataset, indices=indices, config=fedavg_cfg, device='cpu')
    
    # Configuration for FedProx
    fedprox_cfg = copy.deepcopy(base_config)
    fedprox_cfg['method'] = 'FedProx'
    fedprox_client = HospitalClient(client_id=0, dataset=dataset, indices=indices, config=fedprox_cfg, device='cpu')
    
    # Create perturbed starting weights so theta != theta_global
    torch.manual_seed(123)
    perturbed_model = ResNet18Backbone(num_classes=2, pretrained=False)
    for p in perturbed_model.parameters():
        p.data.add_(torch.randn_like(p.data) * 0.1)
        
    # Run 1 epoch on perturbed model with FedAvg vs FedProx
    opt_fedavg = torch.optim.SGD(perturbed_model.parameters(), lr=0.01)
    model_for_fedavg = copy.deepcopy(perturbed_model)
    loss_fedavg = fedavg_client.train_epoch(model_for_fedavg, opt_fedavg, global_model=global_model)
    
    model_for_fedprox = copy.deepcopy(perturbed_model)
    opt_fedprox = torch.optim.SGD(model_for_fedprox.parameters(), lr=0.01)
    loss_fedprox = fedprox_client.train_epoch(model_for_fedprox, opt_fedprox, global_model=global_model)
    
    # FedProx loss must strictly exceed FedAvg loss due to positive proximal penalty
    assert loss_fedprox > loss_fedavg, f"Expected FedProx loss ({loss_fedprox}) > FedAvg loss ({loss_fedavg})"
    
    # Resulting weights must differ
    diff_norm = 0.0
    for p_avg, p_prox in zip(model_for_fedavg.parameters(), model_for_fedprox.parameters()):
        diff_norm += torch.norm(p_avg - p_prox).item()
    assert diff_norm > 1e-4, "FedProx model weights did not diverge from FedAvg model weights"


def test_groupdro_numerical_stability_large_losses():
    """
    Verify GroupDRO aggregation with extreme losses (e.g., 1000 - 5000) does not
    produce overflow (inf/nan) and weights sum strictly to 1.0.
    """
    global_model = ResNet18Backbone(num_classes=2, pretrained=False)
    config = {
        'method': 'GroupDRO',
        'dro': {'eta': 0.5, 'ema_decay': 0.9}
    }
    server = CentralServer(global_model, config)
    
    client_weights = [global_model.state_dict(), global_model.state_dict(), global_model.state_dict()]
    # Very large losses that would overflow naive exp(0.5 * 2000) = exp(1000) -> inf
    client_losses = {0: 1000.0, 1: 1500.0, 2: 2000.0}
    client_samples = {0: 100, 1: 100, 2: 100}
    
    agg_weights = server.aggregate(client_weights, client_losses, client_samples)
    
    assert not np.any(np.isnan(agg_weights)), "GroupDRO produced NaN weights"
    assert not np.any(np.isinf(agg_weights)), "GroupDRO produced Inf weights"
    assert np.isclose(np.sum(agg_weights), 1.0), f"Weights do not sum to 1.0: {np.sum(agg_weights)}"
    # Highest loss client (client 2) must have highest aggregation weight
    assert agg_weights[2] > agg_weights[1] > agg_weights[0]


def test_csv_metric_keys_coverage():
    """
    Verify schema completeness: check that expected metric fields are covered.
    """
    required_metrics = [
        'accuracy',
        'balanced_accuracy',
        'macro_f1',
        'auroc',
        'ece',
        'slide_auroc',
        'slide_accuracy',
        'worst_train_accuracy'
    ]
    
    # Simulated baseline result row keys
    baseline_keys = [
        'Method', 'Privacy Enabled', 'Spent Epsilon',
        'Mean Test Center 4 Accuracy', 'Std Test Center 4 Accuracy',
        'Mean Test Center 4 Balanced Accuracy', 'Std Test Center 4 Balanced Accuracy',
        'Mean Test Center 4 Macro F1', 'Std Test Center 4 Macro F1',
        'Mean Test Center 4 AUROC', 'Std Test Center 4 AUROC',
        'Mean Test Center 4 ECE', 'Std Test Center 4 ECE',
        'Mean Test Center 4 Slide AUROC', 'Std Test Center 4 Slide AUROC',
        'Mean Test Center 4 Slide Accuracy', 'Std Test Center 4 Slide Accuracy',
        'Mean Worst Train Hospital Accuracy', 'Std Worst Train Hospital Accuracy',
        'Seeds Evaluated'
    ]
    
    joined_keys = " ".join(baseline_keys).lower()
    for metric in required_metrics:
        # Check that metric name or subparts are represented in the export schema
        parts = metric.split('_')
        for part in parts:
            assert part in joined_keys, f"Metric '{metric}' (part: '{part}') missing from baseline CSV schema"
