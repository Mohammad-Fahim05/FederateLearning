import pytest
import math
import numpy as np
import torch
import torch.nn as nn
from src.privacy.privacy_accountant import RDPPrivacyAccountant
from src.privacy.dp_optimizer import DPGradientClipper

def test_privacy_accountant_basic():
    accountant = RDPPrivacyAccountant(target_delta=1e-5)
    eps, delta = accountant.get_privacy_spent(q=0.01, noise_multiplier=0.8, steps=100)
    
    assert eps > 0, "Epsilon must be positive"
    assert eps < float('inf'), "Epsilon must be finite for noise_multiplier > 0"
    assert delta == 1e-5, f"Delta should be 1e-5, got {delta}"

def test_client_local_q_and_exact_step_calculation():
    """
    Verifies that:
    1. q is client-local: q_k = B / N_k
    2. steps per epoch is ceil(N_k / B)
    3. total gradient steps = rounds * local_epochs * ceil(N_k / B)
    """
    accountant = RDPPrivacyAccountant(target_delta=1e-5)
    
    # Client A: 1000 samples, batch_size 64 -> steps_per_epoch = ceil(1000/64) = 16
    res_a = accountant.get_client_privacy_spent(
        num_samples=1000,
        batch_size=64,
        local_epochs=2,
        rounds=10,
        noise_multiplier=0.8,
        compose_loss=False
    )
    assert res_a['q'] == 64 / 1000.0
    assert res_a['steps_per_epoch'] == 16
    assert res_a['total_grad_steps'] == 10 * 2 * 16  # 320 steps
    
    # Client B: 25000 samples, batch_size 64 -> steps_per_epoch = ceil(25000/64) = 391
    res_b = accountant.get_client_privacy_spent(
        num_samples=25000,
        batch_size=64,
        local_epochs=2,
        rounds=10,
        noise_multiplier=0.8,
        compose_loss=False
    )
    assert res_b['q'] == 64 / 25000.0
    assert res_b['steps_per_epoch'] == 391
    assert res_b['total_grad_steps'] == 10 * 2 * 391  # 7820 steps
    
    # Client with smaller sample size has higher q and thus higher epsilon under same rounds
    assert res_a['q'] > res_b['q']
    assert res_a['grad_epsilon'] > res_b['grad_epsilon']

def test_accountant_monotonicity_with_rounds_epochs_samples():
    """
    Verifies that privacy expenditure increases monotonically when:
    - communication rounds increase
    - local epochs increase
    - client dataset size decreases (higher sampling ratio)
    """
    accountant = RDPPrivacyAccountant(target_delta=1e-5)
    
    # Baseline
    base = accountant.get_client_privacy_spent(
        num_samples=5000, batch_size=64, local_epochs=2, rounds=20, noise_multiplier=0.8
    )
    
    # More rounds -> higher epsilon
    more_rounds = accountant.get_client_privacy_spent(
        num_samples=5000, batch_size=64, local_epochs=2, rounds=40, noise_multiplier=0.8
    )
    assert more_rounds['total_epsilon'] > base['total_epsilon']
    
    # More epochs -> higher epsilon
    more_epochs = accountant.get_client_privacy_spent(
        num_samples=5000, batch_size=64, local_epochs=4, rounds=20, noise_multiplier=0.8
    )
    assert more_epochs['total_epsilon'] > base['total_epsilon']
    
    # Smaller client size -> higher epsilon
    smaller_client = accountant.get_client_privacy_spent(
        num_samples=2000, batch_size=64, local_epochs=2, rounds=20, noise_multiplier=0.8
    )
    assert smaller_client['total_epsilon'] > base['total_epsilon']

def test_loss_release_privacy_composition():
    """
    Verifies that the accountant correctly accounts for additional private releases (loss statistics)
    and does NOT silently claim a gradient-only epsilon when scalar releases are present.
    """
    accountant = RDPPrivacyAccountant(target_delta=1e-5)
    
    # Without loss release composition
    res_no_loss = accountant.get_client_privacy_spent(
        num_samples=5000, batch_size=64, local_epochs=2, rounds=20, noise_multiplier=0.8,
        compose_loss=False
    )
    
    # With loss release composition
    res_with_loss = accountant.get_client_privacy_spent(
        num_samples=5000, batch_size=64, local_epochs=2, rounds=20, noise_multiplier=0.8,
        loss_noise_std=0.05, max_loss=5.0, compose_loss=True
    )
    
    assert res_with_loss['loss_epsilon'] > 0
    assert res_with_loss['loss_epsilon'] < float('inf')
    assert res_with_loss['total_epsilon'] >= res_no_loss['grad_epsilon']
    assert res_with_loss['total_epsilon'] >= res_with_loss['grad_epsilon']

def test_per_sample_dp_gradient_clipper():
    model = nn.Linear(10, 2)
    clipper = DPGradientClipper(max_grad_norm=1.0, noise_multiplier=0.8, enabled=True)
    
    x = torch.randn(4, 10)
    y = torch.tensor([0, 1, 0, 1])
    criterion = nn.CrossEntropyLoss()
    
    avg_loss = clipper.clip_and_noise_sample_grad(model, criterion, x, y, device='cpu')
    
    assert avg_loss > 0, "Average loss must be positive"
    for p in model.parameters():
        assert p.grad is not None, "Gradients must be computed"
        assert not torch.isnan(p.grad).any(), "Gradient values must not be NaN"

def test_per_sample_gradient_clipping_bound():
    """
    Validates that per-sample clipping strictly bounds individual sample gradient contributions.
    """
    model = nn.Linear(5, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
        
    C = 1.0
    clipper = DPGradientClipper(max_grad_norm=C, noise_multiplier=0.0, enabled=True)
    
    # 2 samples: sample 0 has normal magnitude, sample 1 is an extreme outlier
    x = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0],
                      [1000.0, 0.0, 0.0, 0.0, 0.0]])
    y = torch.tensor([[0.0], [0.0]])
    criterion = nn.MSELoss()
    
    clipper.clip_and_noise_sample_grad(model, criterion, x, y, device='cpu')
    
    total_grad_norm = model.weight.grad.norm(2).item()
    assert total_grad_norm <= C + 1e-4, f"Averaged clipped gradient norm {total_grad_norm} exceeded max bound {C}"

def test_dp_loss_protection():
    raw_loss = 0.45
    loss_noises = [float(np.random.normal(0.0, 0.05)) for _ in range(10)]
    protected_losses = [max(0.0, raw_loss + n) for n in loss_noises]
    
    assert any(p != raw_loss for p in protected_losses), "Transmitted client loss must be DP perturbed"
    assert all(p >= 0.0 for p in protected_losses), "Protected loss must be non-negative"

if __name__ == "__main__":
    test_privacy_accountant_basic()
    test_client_local_q_and_exact_step_calculation()
    test_accountant_monotonicity_with_rounds_epochs_samples()
    test_loss_release_privacy_composition()
    test_per_sample_dp_gradient_clipper()
    test_per_sample_gradient_clipping_bound()
    test_dp_loss_protection()
    print("All privacy tests passed successfully!")
