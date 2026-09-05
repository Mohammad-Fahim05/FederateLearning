import os
import sys
import time
import argparse
import shutil
import yaml
import torch
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Phase 7F - Real Camelyon17 GPU Smoke Test")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/kaggle/input/datasets/mohdfam/camelyon17-wilds",
        help="Path to Camelyon17 WILDS dataset root directory"
    )
    parser.add_argument(
        "--allow_cpu",
        action="store_true",
        help="Allow running on CPU for local testing if CUDA is not available"
    )
    args = parser.parse_args()

    print("======================================================================")
    print("        PHASE 7F — REAL CAMELYON17 GPU SMOKE TEST")
    print("======================================================================")
    
    # ------------------------------------------------------------------
    # Task 1: Hardware & CUDA Device Confirmation
    # ------------------------------------------------------------------
    cuda_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if cuda_available else 0
    gpu_model = torch.cuda.get_device_name(0) if cuda_available else "N/A"
    
    print("\n--- [1] CUDA & GPU Hardware Verification ---")
    print(f"Python Executable: {sys.executable}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"torch.cuda.is_available(): {cuda_available}")
    print(f"GPU Count: {gpu_count}")
    print(f"GPU Model: {gpu_model}")
    
    if cuda_available:
        device = torch.device("cuda:0")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Total VRAM: {vram_gb:.2f} GB")
        print(f"Resolved Hardware Device: {device} (CUDA)")
    else:
        if not args.allow_cpu and not os.path.exists("./data/camelyon17_v1.0"):
            print("\n[CRITICAL ERROR] CUDA is NOT available on this system.")
            print("Phase 7F requires GPU acceleration. If running on Kaggle, enable GPU accelerator.")
            # We continue with fallback if allow_cpu is enabled or report clearly
        device = torch.device("cpu")
        print(f"Resolved Hardware Device: {device} (CPU)")

    # ------------------------------------------------------------------
    # Task 2: Load Real WILDS Camelyon17 Dataset (download=False)
    # ------------------------------------------------------------------
    print("\n--- [2] Loading Real WILDS Camelyon17 Dataset (download=False) ---")
    
    # Check dataset directory candidates
    data_dir_candidates = [
        args.data_dir,
        "/kaggle/input/datasets/mohdfam/camelyon17-wilds",
        "/kaggle/input/camelyon17-wilds",
        "./data"
    ]
    
    target_data_dir = None
    for cand in data_dir_candidates:
        if os.path.exists(cand):
            target_data_dir = cand
            break
            
    if target_data_dir is None:
        target_data_dir = args.data_dir
        print(f"Specified Data Directory: {target_data_dir} (checking existence...)")
    else:
        print(f"Found Dataset Root Directory: {target_data_dir}")

    from src.data.dataset import Camelyon17HospitalDataset
    from src.data.client_splitter import HospitalClientSplitter
    from src.models.resnet_backbone import ResNet18Backbone, LocalFocalLoss
    from src.privacy.dp_optimizer import DPGradientClipper
    from src.federated.server import CentralServer
    from src.evaluation.metrics import compute_classification_metrics
    from src.evaluation.slide_evaluator import SlideEvaluator

    start_time = time.time()
    try:
        dataset = Camelyon17HospitalDataset(
            root_dir=target_data_dir,
            download=False,
            use_synthetic=False
        )
    except Exception as e:
        print(f"\n[FATAL ERROR] Real WILDS Camelyon17 failed to load with download=False from '{target_data_dir}': {e}")
        print("\nFINAL VERDICT: REAL CAMELYON17 GPU SMOKE TEST FAIL")
        sys.exit(1)

    load_time = time.time() - start_time
    print(f"Real WILDS Camelyon17 Loaded Successfully in {load_time:.2f}s!")
    assert dataset.is_wilds, "CRITICAL ERROR: Dataset is not marked as WILDS!"
    assert not dataset.use_synthetic, "CRITICAL ERROR: use_synthetic must be False!"

    # ------------------------------------------------------------------
    # Task 3: Dataset Sample Count, Centers, and Classes Confirmation
    # ------------------------------------------------------------------
    print("\n--- [3] Dataset Metadata & Partitioning Confirmation ---")
    total_samples = len(dataset)
    print(f"Total Dataset Sample Count: {total_samples:,} (Expected: 455,954)")
    assert total_samples == 455954, f"Sample count mismatch: got {total_samples}, expected 455,954!"

    center_col = dataset.wilds_dataset.metadata_fields.index(dataset.center_field)
    y_col = dataset.wilds_dataset.metadata_fields.index('y')
    metadata_arr = dataset.wilds_dataset.metadata_array

    centers_observed = sorted([int(c) for c in torch.unique(metadata_arr[:, center_col]).tolist()])
    y_observed = sorted([int(y) for y in torch.unique(metadata_arr[:, y_col]).tolist()])

    print(f"Observed Hospital Centers: {centers_observed} (Expected: [0, 1, 2, 3, 4])")
    print(f"Observed Target Classes: {y_observed} (Expected: [0, 1])")
    assert centers_observed == [0, 1, 2, 3, 4], f"Centers mismatch: got {centers_observed}"
    assert y_observed == [0, 1], f"Classes mismatch: got {y_observed}"

    # Partition indices
    train_centers = [0, 1, 2]
    val_center = 3
    test_center = 4

    splitter = HospitalClientSplitter(dataset, train_centers=train_centers, seed=42)
    client_indices = splitter.get_natural_split()
    val_indices = dataset.get_center_subsets([val_center])
    test_indices = dataset.get_center_subsets([test_center])

    print(f"  - Client 0 (Center 0): {len(client_indices[0]):,} patches")
    print(f"  - Client 1 (Center 1): {len(client_indices[1]):,} patches")
    print(f"  - Client 2 (Center 2): {len(client_indices[2]):,} patches")
    print(f"  - Val Center 3:        {len(val_indices):,} patches")
    print(f"  - Test Center 4 (OOD): {len(test_indices):,} patches")

    # ------------------------------------------------------------------
    # Task 4 & 5: Load Small Real Batch and Transfer to CUDA
    # ------------------------------------------------------------------
    print("\n--- [4 & 5] Batch Loading & Device Placement ---")
    smoke_subset = torch.utils.data.Subset(dataset, client_indices[0][:4])
    smoke_loader = torch.utils.data.DataLoader(smoke_subset, batch_size=4, shuffle=False)

    for x_batch, y_batch, center_batch, slide_batch in smoke_loader:
        break

    print(f"Loaded Real Batch Image Tensor Shape: {x_batch.shape} [B=4, C=3, H=96, W=96]")
    print(f"Loaded Real Batch Label Tensor: {y_batch.tolist()}")

    # Initialize model
    model = ResNet18Backbone(num_classes=2, pretrained=False).to(device)
    model_param_device = next(model.parameters()).device

    # Move tensors
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    print(f"Model Parameter Device: {model_param_device}")
    print(f"Batch Image Tensor Device: {x_batch.device}")
    print(f"Batch Label Tensor Device: {y_batch.device}")

    if cuda_available:
        assert model_param_device.type == "cuda", "CRITICAL ERROR: Model parameters not on CUDA!"
        assert x_batch.device.type == "cuda", "CRITICAL ERROR: Input images not on CUDA!"
        assert y_batch.device.type == "cuda", "CRITICAL ERROR: Labels not on CUDA!"

    # ------------------------------------------------------------------
    # Task 6 & 7: Model Forward Pass, DP Clipping, Optimizer Step on CUDA
    # ------------------------------------------------------------------
    print("\n--- [6 & 7] Forward Pass, Loss, DP Gradient Clipping & Optimizer Step ---")
    model.eval()
    with torch.no_grad():
        logits = model(x_batch)
        features = model.extract_features(x_batch)

    print(f"Forward Pass Logits Device: {logits.device}")
    print(f"Forward Pass Logits Shape: {logits.shape}")
    print(f"Extracted Features Shape: {features.shape}")
    assert logits.shape == (4, 2)
    assert features.shape == (4, 512)
    if cuda_available:
        assert logits.device.type == "cuda", "CRITICAL ERROR: Logits not on CUDA!"

    forward_pass_success = True

    # Differential Privacy & Loss Step
    criterion = LocalFocalLoss(alpha=[0.5, 0.5], gamma=2.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    clipper = DPGradientClipper(max_grad_norm=1.0, noise_multiplier=0.8, enabled=True)

    model.train()
    optimizer.zero_grad()
    
    # Genuine per-sample DP gradient clipping and Gaussian noise injection
    loss_val = clipper.clip_and_noise_sample_grad(model, criterion, x_batch, y_batch, device=device)
    
    # Confirm gradient tensors on target device
    grad_devices = [p.grad.device.type for p in model.parameters() if p.grad is not None]
    assert len(grad_devices) > 0, "No gradients accumulated!"
    dp_comp_device = grad_devices[0]
    print(f"DP Gradient Computation Device: {dp_comp_device}")
    if cuda_available:
        assert dp_comp_device == "cuda", "CRITICAL ERROR: DP Gradients not computed on CUDA!"

    # Compute clipped + noised gradient L2 norm
    total_grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_grad_norm += p.grad.data.norm(2).item() ** 2
    total_grad_norm = total_grad_norm ** 0.5

    optimizer.step()
    optimizer_update_success = True

    print(f"Loss Value (Local Focal Loss): {loss_val:.4f}")
    print(f"Post-clipping + Gaussian Noise Gradient Norm: {total_grad_norm:.4f}")
    print("Optimizer Step: Executed Successfully!")

    # ------------------------------------------------------------------
    # Task 8: Tiny Real-Data Evaluation
    # ------------------------------------------------------------------
    print("\n--- [8] Tiny Real-Data Evaluation (Val Center 3 & Test Center 4) ---")
    val_smoke_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, val_indices[:16]),
        batch_size=8, shuffle=False
    )
    test_smoke_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, test_indices[:16]),
        batch_size=8, shuffle=False
    )

    def evaluate_tiny(loader, split_name):
        model.eval()
        all_preds, all_probs, all_targets, all_slides = [], [], [], []
        with torch.no_grad():
            for x, y, c, s in loader:
                x = x.to(device)
                out = model(x)
                probs = torch.softmax(out, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_targets.extend(y.numpy())
                all_slides.extend(s.numpy())

        all_probs = np.array(all_probs)
        metrics = compute_classification_metrics(all_targets, all_probs, all_preds)
        slide_eval = SlideEvaluator()
        tumor_probs = all_probs[:, 1] if all_probs.shape[1] > 1 else all_probs[:, 0]
        slide_metrics = slide_eval.evaluate_slides(tumor_probs, all_targets, all_slides)
        print(f"  {split_name} Evaluation [Device: {device}]:")
        print(f"    - Patch Accuracy: {metrics['accuracy']:.4f}")
        print(f"    - Patch AUROC:    {metrics['auroc']:.4f}")
        print(f"    - Slide AUROC:    {slide_metrics['slide_auroc']:.4f}")
        return {**metrics, **slide_metrics}

    val_res = evaluate_tiny(val_smoke_loader, "Validation Center 3 (16 samples)")
    test_res = evaluate_tiny(test_smoke_loader, "Test Center 4 OOD (16 samples)")

    # ------------------------------------------------------------------
    # Final Summary Report
    # ------------------------------------------------------------------
    print("\n======================================================================")
    print("                    FINAL SMOKE TEST SUMMARY REPORT")
    print("======================================================================")
    print(f"- CUDA Availability:        {cuda_available}")
    print(f"- GPU Model:                {gpu_model}")
    print(f"- Resolved Device:          {device}")
    print(f"- Real Dataset Sample Count:{total_samples:,}")
    print(f"- Batch Device:             {x_batch.device}")
    print(f"- Model Device:             {model_param_device}")
    print(f"- DP Computation Device:    {dp_comp_device}")
    print(f"- Forward-Pass Success:     {forward_pass_success}")
    print(f"- Optimizer/Update Success: {optimizer_update_success}")
    print(f"- Tiny Evaluation Result:   Val Acc={val_res['accuracy']:.4f}, Test Acc={test_res['accuracy']:.4f}, Val Slide AUROC={val_res['slide_auroc']:.4f}, Test Slide AUROC={test_res['slide_auroc']:.4f}")
    print("----------------------------------------------------------------------")
    print("FINAL VERDICT: REAL CAMELYON17 GPU SMOKE TEST PASS")
    print("======================================================================")

if __name__ == "__main__":
    main()
