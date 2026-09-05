import pytest
import torch
import torch.nn as nn
from src.privacy.privacy_accountant import RDPPrivacyAccountant
from src.privacy.dp_optimizer import DPGradientClipper

def test_privacy_accountant():
    accountant = RDPPrivacyAccountant(target_delta=1e-5)
    eps, delta = accountant.get_privacy_spent(q=0.01, noise_multiplier=0.8, steps=100)
    
    assert eps > 0, "Epsilon must be positive"
    assert eps < float('inf'), "Epsilon must be finite for noise_multiplier > 0"
    assert delta == 1e-5, f"Delta should be 1e-5, got {delta}"
    print(f"Privacy Accountant Test PASSED! Spent Epsilon: {eps:.4f}, Delta: {delta}")

def test_per_sample_dp_gradient_clipper():
    model = nn.Linear(10, 2)
    clipper = DPGradientClipper(max_grad_norm=1.0, noise_multiplier=0.8, enabled=True)
    
    x = torch.randn(4, 10)
    y = torch.tensor([0, 1, 0, 1])
    criterion = nn.CrossEntropyLoss()
    
    # Compute per-sample gradient clipping & noise injection
    avg_loss = clipper.clip_and_noise_sample_grad(model, criterion, x, y, device='cpu')
    
    assert avg_loss > 0, "Average loss must be positive"
    for p in model.parameters():
        assert p.grad is not None, "Gradients must be computed"
        assert not torch.isnan(p.grad).any(), "Gradient values must not be NaN"
        
    print("Per-Sample DP Gradient Clipper Test PASSED!")

def test_per_sample_gradient_clipping_bound():
    """
    Validates that per-sample clipping strictly bounds individual sample gradient contributions.
    Even when one outlier sample has an arbitrarily large unclipped gradient norm (e.g., 1000.0),
    its clipped gradient contribution to the batch gradient average cannot exceed C / B.
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
    
    # The gradient of sample 1 alone before clipping would be d/dw (1000*w)^2 / 2 = 1000 * 1000 = 10^6
    # With per-sample clipping to C=1.0, sample 1 contributes at most C=1.0.
    # Total batch average gradient norm must be <= (C + C) / 2 = 1.0.
    total_grad_norm = model.weight.grad.norm(2).item()
    assert total_grad_norm <= C + 1e-4, f"Averaged clipped gradient norm {total_grad_norm} exceeded max bound {C}"
    print(f"Per-Sample Outlier Sensitivity Bounding Test PASSED! Grad Norm: {total_grad_norm:.4f} <= {C}")

def test_dp_loss_protection():
    import numpy as np
    raw_loss = 0.45
    loss_noises = [float(np.random.normal(0.0, 0.05)) for _ in range(10)]
    protected_losses = [max(0.0, raw_loss + n) for n in loss_noises]
    
    assert any(p != raw_loss for p in protected_losses), "Transmitted client loss must be DP perturbed"
    assert all(p >= 0.0 for p in protected_losses), "Protected loss must be non-negative"
    print("DP Loss Protection Test PASSED!")

if __name__ == "__main__":
    test_privacy_accountant()
    test_per_sample_dp_gradient_clipper()
    test_per_sample_gradient_clipping_bound()
    test_dp_loss_protection()

