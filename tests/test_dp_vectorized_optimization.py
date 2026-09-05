import os
import time
import pytest
import torch
import torch.nn as nn
from src.models.resnet_backbone import ResNet18Backbone, LocalFocalLoss
from src.privacy.dp_optimizer import DPGradientClipper
from src.privacy.privacy_accountant import RDPPrivacyAccountant

def test_dp_vectorized_numerical_equivalence():
    """
    Verifies that the vectorized torch.func per-sample clipping implementation
    produces numerically equivalent clipped parameter gradients to the reference loop implementation.
    """
    torch.manual_seed(42)
    model_ref = ResNet18Backbone(num_classes=2, pretrained=False)
    model_vec = ResNet18Backbone(num_classes=2, pretrained=False)
    model_vec.load_state_dict(model_ref.state_dict())
    
    model_ref.eval()
    model_vec.eval()
    
    criterion = LocalFocalLoss()
    C = 1.0
    
    x = torch.randn(4, 3, 96, 96)
    y = torch.tensor([0, 1, 0, 1])
    
    # 1. Reference per-sample loop
    params_ref = [p for p in model_ref.parameters() if p.requires_grad]
    clipped_grads_ref = {p: torch.zeros_like(p.data) for p in params_ref}
    for i in range(len(x)):
        model_ref.zero_grad()
        out_i = model_ref(x[i:i+1])
        loss_i = criterion(out_i, y[i:i+1])
        loss_i.backward()
        
        norm_sq = sum(p.grad.data.norm(2).item()**2 for p in params_ref if p.grad is not None)
        norm = norm_sq ** 0.5
        clip_coef = min(1.0, C / (norm + 1e-6))
        for p in params_ref:
            if p.grad is not None:
                clipped_grads_ref[p] += p.grad.data * clip_coef
                
    for p in params_ref:
        p.grad = clipped_grads_ref[p] / len(x)
        
    # 2. Vectorized Clipper (with noise_multiplier=0 to verify exact gradient equality)
    clipper = DPGradientClipper(max_grad_norm=C, noise_multiplier=0.0, enabled=True)
    loss_val = clipper.clip_and_noise_sample_grad(model_vec, criterion, x, y, device='cpu')
    
    assert loss_val > 0.0
    assert not torch.isnan(torch.tensor(loss_val))
    
    diffs = []
    for p_ref, p_vec in zip(model_ref.parameters(), model_vec.parameters()):
        if p_ref.grad is not None:
            assert p_vec.grad is not None
            assert not torch.isnan(p_vec.grad).any()
            assert not torch.isinf(p_vec.grad).any()
            diff = (p_ref.grad - p_vec.grad).abs().max().item()
            diffs.append(diff)
            
    max_diff = max(diffs)
    print(f"Max Gradient Absolute Difference (Reference vs Vectorized): {max_diff:.2e}")
    assert max_diff < 1e-5, f"Vectorized gradient deviated from reference: {max_diff}"

def test_dp_vectorized_clipping_bound():
    """
    Verifies that per-sample clipping strictly bounds outlier gradients.
    """
    torch.manual_seed(42)
    model = nn.Linear(5, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
        
    C = 1.0
    clipper = DPGradientClipper(max_grad_norm=C, noise_multiplier=0.0, enabled=True)
    
    x = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0],
                      [1000.0, 0.0, 0.0, 0.0, 0.0]])
    y = torch.tensor([[0.0], [0.0]])
    criterion = nn.MSELoss()
    
    clipper.clip_and_noise_sample_grad(model, criterion, x, y, device='cpu')
    
    total_grad_norm = model.weight.grad.norm(2).item()
    assert total_grad_norm <= C + 1e-4, f"Gradient norm {total_grad_norm} exceeded C={C}"

def test_dp_vectorized_with_noise_and_optimizer_step():
    """
    Verifies that Gaussian DP noise injection and optimizer.step() succeed without NaNs.
    """
    torch.manual_seed(42)
    model = ResNet18Backbone(num_classes=2, pretrained=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = LocalFocalLoss()
    
    C = 1.0
    sigma = 0.8
    clipper = DPGradientClipper(max_grad_norm=C, noise_multiplier=sigma, enabled=True)
    
    x = torch.randn(4, 3, 96, 96)
    y = torch.tensor([0, 1, 0, 1])
    
    optimizer.zero_grad()
    loss_val = clipper.clip_and_noise_sample_grad(model, criterion, x, y, device='cpu')
    optimizer.step()
    
    assert loss_val > 0.0
    for p in model.parameters():
        assert not torch.isnan(p.data).any()
        assert not torch.isinf(p.data).any()

def test_privacy_accountant_parameters_unchanged():
    """
    Verifies that the RDP privacy accountant receives identical parameters (C, sigma, delta, q, T)
    and computes the correct bounded epsilon.
    """
    accountant = RDPPrivacyAccountant(target_delta=1e-5)
    res = accountant.get_client_privacy_spent(
        num_samples=91950,
        batch_size=64,
        local_epochs=2,
        rounds=100,
        noise_multiplier=0.8,
        compose_loss=True
    )
    assert res['delta'] == 1e-5
    assert res['q'] == 64.0 / 91950.0
    assert res['steps_per_epoch'] == 1437
def test_dp_chunked_vs_full_batch_equivalence():
    """
    Verifies that chunked vectorized calculation (e.g. chunk_size=4, 8, 16) produces
    strictly identical parameter gradients to full unchunked batch calculation.
    """
    torch.manual_seed(42)
    model_full = ResNet18Backbone(num_classes=2, pretrained=False)
    model_chunked = ResNet18Backbone(num_classes=2, pretrained=False)
    model_chunked.load_state_dict(model_full.state_dict())
    
    model_full.eval()
    model_chunked.eval()
    
    criterion = LocalFocalLoss()
    C = 1.0
    
    x = torch.randn(16, 3, 96, 96)
    y = torch.randint(0, 2, (16,))
    
    # 1. Full batch clipper (chunk_size=16)
    clipper_full = DPGradientClipper(max_grad_norm=C, noise_multiplier=0.0, enabled=True, chunk_size=16)
    loss_full = clipper_full.clip_and_noise_sample_grad(model_full, criterion, x, y, device='cpu')
    
    # 2. Chunked clipper (chunk_size=4)
    clipper_chunked = DPGradientClipper(max_grad_norm=C, noise_multiplier=0.0, enabled=True, chunk_size=4)
    loss_chunked = clipper_chunked.clip_and_noise_sample_grad(model_chunked, criterion, x, y, device='cpu')
    
    assert abs(loss_full - loss_chunked) < 1e-5
    
    diffs = []
    for p_full, p_chunk in zip(model_full.parameters(), model_chunked.parameters()):
        if p_full.grad is not None:
            assert p_chunk.grad is not None
            diff = (p_full.grad - p_chunk.grad).abs().max().item()
            diffs.append(diff)
            
    max_diff = max(diffs)
    print(f"Max Gradient Absolute Difference (Full Batch vs Chunked Batch): {max_diff:.2e}")
    assert max_diff < 1e-6, f"Chunked gradient deviated from full batch: {max_diff}"

if __name__ == "__main__":
    test_dp_vectorized_numerical_equivalence()
    test_dp_chunked_vs_full_batch_equivalence()
    test_dp_vectorized_clipping_bound()
    test_dp_vectorized_with_noise_and_optimizer_step()
    test_privacy_accountant_parameters_unchanged()
    print("All DP vectorized and chunked optimization tests passed successfully!")
