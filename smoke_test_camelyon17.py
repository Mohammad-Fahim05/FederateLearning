import os
import sys
import time
import copy
import shutil
import yaml
import torch
import numpy as np

print("==================================================")
print("   PHASE 7: REAL WILDS CAMELYON17 SMOKE TEST")
print("==================================================")
print(f"Python Executable: {sys.executable}")
print(f"PyTorch Version: {torch.__version__}")

# Measure initial disk space
total_disk, used_disk, free_disk = shutil.disk_usage(".")
print(f"Initial Free Disk Space: {free_disk / (1024**3):.2f} GB")

from src.data.dataset import Camelyon17HospitalDataset
from src.data.client_splitter import HospitalClientSplitter
from src.models.resnet_backbone import ResNet18Backbone, LocalFocalLoss
from src.privacy.dp_optimizer import DPGradientClipper
from src.privacy.privacy_accountant import RDPPrivacyAccountant
from src.federated.client import HospitalClient
from src.federated.server import CentralServer
from src.evaluation.metrics import compute_classification_metrics
from src.evaluation.slide_evaluator import SlideEvaluator

# Step 1: Load Real WILDS Camelyon17 Dataset
print("\n[Step 1] Loading Real WILDS Camelyon17 Dataset (download=True, use_synthetic=False)...")
start_time = time.time()
try:
    dataset = Camelyon17HospitalDataset(
        root_dir="./data",
        download=True,
        use_synthetic=False
    )
except Exception as e:
    print(f"\n[FATAL ERROR] Real WILDS Camelyon17 failed to load: {e}")
    sys.exit(1)

load_time = time.time() - start_time
print(f"Real WILDS Camelyon17 Loaded Successfully in {load_time:.2f} seconds!")
assert dataset.is_wilds, "CRITICAL ERROR: Dataset is not marked as WILDS!"
assert not dataset.use_synthetic, "CRITICAL ERROR: use_synthetic must be False!"

total_len = len(dataset)
print(f"Total Dataset Size: {total_len:,} samples")

# Step 2: Metadata and Domain Inspection
print("\n[Step 2] Inspecting Metadata and Domain Partitioning...")
center_col = dataset.wilds_dataset.metadata_fields.index(dataset.center_field)
slide_col = dataset.wilds_dataset.metadata_fields.index('slide')
y_col = dataset.wilds_dataset.metadata_fields.index('y')
metadata_arr = dataset.wilds_dataset.metadata_array

centers_observed = torch.unique(metadata_arr[:, center_col]).tolist()
y_observed = torch.unique(metadata_arr[:, y_col]).tolist()
print(f"Observed Hospital Center IDs: {centers_observed}")
print(f"Observed Label Classes: {y_observed}")

center_counts = {}
for c in centers_observed:
    indices = dataset.get_center_subsets([c])
    center_counts[c] = len(indices)
    print(f"  - Hospital Center {c}: {len(indices):,} patches")

# Step 3: Train / Val / Test Partitioning
print("\n[Step 3] Verifying Federated Hospital Partitioning...")
train_centers = [0, 1, 2]
val_center = 3
test_center = 4

splitter = HospitalClientSplitter(dataset, train_centers=train_centers, seed=42)
client_indices = splitter.get_natural_split()
val_indices = dataset.get_center_subsets([val_center])
test_indices = dataset.get_center_subsets([test_center])

print(f"Training Client 0 (Center 0): {len(client_indices[0]):,} samples")
print(f"Training Client 1 (Center 1): {len(client_indices[1]):,} samples")
print(f"Training Client 2 (Center 2): {len(client_indices[2]):,} samples")
print(f"Validation Center 3: {len(val_indices):,} samples")
print(f"Unseen Test Center 4: {len(test_indices):,} samples")

assert len(client_indices[0]) > 0 and len(client_indices[1]) > 0 and len(client_indices[2]) > 0
assert len(val_indices) > 0 and len(test_indices) > 0

# Step 4: Small Batch Loading and Forward Pass
print("\n[Step 4] Testing Real Batch Loading & Model Forward Pass...")
test_subset = torch.utils.data.Subset(dataset, client_indices[0][:4])
smoke_loader = torch.utils.data.DataLoader(test_subset, batch_size=4, shuffle=False)

for x_batch, y_batch, center_batch, slide_batch in smoke_loader:
    break

print(f"Batch Image Tensor Shape: {x_batch.shape} (Expected [4, 3, 96, 96])")
print(f"Batch Labels: {y_batch.tolist()}")
print(f"Batch Center IDs: {center_batch.tolist()}")
print(f"Batch Slide IDs: {slide_batch.tolist()}")

assert x_batch.shape == (4, 3, 96, 96), f"Unexpected shape {x_batch.shape}"

model = ResNet18Backbone(num_classes=2, pretrained=False)
model.eval()
with torch.no_grad():
    logits = model(x_batch)
    features = model.extract_features(x_batch)
print(f"Model Forward Output Logits Shape: {logits.shape} (Expected [4, 2])")
print(f"Features Tensor Shape: {features.shape} (Expected [4, 512])")
assert logits.shape == (4, 2)
assert features.shape == (4, 512)

# Step 5: Training Step with DP-SGD Gradient Clipping & Noise
print("\n[Step 5] Testing 1 Training Step with Focal Loss and DP Gradient Clipping...")
criterion = LocalFocalLoss(alpha=[0.5, 0.5], gamma=2.0)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
clipper = DPGradientClipper(max_grad_norm=1.0, noise_multiplier=0.8, enabled=True)

model.train()
optimizer.zero_grad()
loss_val = clipper.clip_and_noise_sample_grad(model, criterion, x_batch, y_batch, device='cpu')

# Record clipped & noised gradient norm
total_grad_norm = 0.0
for p in model.parameters():
    if p.grad is not None:
        total_grad_norm += p.grad.data.norm(2).item() ** 2
total_grad_norm = total_grad_norm ** 0.5

optimizer.step()

print(f"Step Mean Loss: {loss_val:.4f}")
print(f"Post-clipping + Noise Gradient Norm: {total_grad_norm:.4f}")
print(f"DP Per-Sample Clipping & Gaussian Noise Injection: EXECUTED")

# Step 6: 1 Minimal Federated Aggregation Step (Smooth DRO)
print("\n[Step 6] Testing 1 Federated Aggregation Step...")
with open('./configs/camelyon17_wilds.yaml', 'r') as f:
    config = yaml.safe_load(f)

server = CentralServer(global_model=model, config=config)
client_weights_list = [copy.deepcopy(model.state_dict()) for _ in range(3)]
client_losses_dict = {0: loss_val, 1: loss_val * 1.1, 2: loss_val * 0.9}
client_sample_counts = {0: len(client_indices[0]), 1: len(client_indices[1]), 2: len(client_indices[2])}

dro_weights = server.aggregate(
    client_weights_list=client_weights_list,
    client_losses_dict=client_losses_dict,
    client_sample_counts=client_sample_counts
)
print(f"Smooth DRO Aggregation Weights: {dro_weights}")
assert len(dro_weights) == 3
assert abs(sum(dro_weights) - 1.0) < 1e-4

# Step 7: Validation and OOD Evaluation Step on Real Patches
print("\n[Step 7] Testing Real Evaluation on Validation (Center 3) and Test (Center 4) Subsets...")
val_smoke_loader = torch.utils.data.DataLoader(
    torch.utils.data.Subset(dataset, val_indices[:16]),
    batch_size=8, shuffle=False
)
test_smoke_loader = torch.utils.data.DataLoader(
    torch.utils.data.Subset(dataset, test_indices[:16]),
    batch_size=8, shuffle=False
)

def evaluate_smoke(eval_loader, name):
    model.eval()
    all_preds, all_probs, all_targets, all_slides = [], [], [], []
    with torch.no_grad():
        for x, y, c, s in eval_loader:
            out = model(x)
            probs = torch.softmax(out, dim=1).numpy()
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
    print(f"  {name} - Accuracy: {metrics['accuracy']:.4f}, AUROC: {metrics['auroc']:.4f}, Slide AUROC: {slide_metrics['slide_auroc']:.4f}")
    return {**metrics, **slide_metrics}

val_res = evaluate_smoke(val_smoke_loader, "Val Center 3 (Smoke Subset)")
test_res = evaluate_smoke(test_smoke_loader, "Test Center 4 Unseen (Smoke Subset)")

# Final Disk Check
_, _, final_free = shutil.disk_usage(".")
print(f"\nRemaining Free Disk Space: {final_free / (1024**3):.2f} GB")
print("\n==================================================")
print("   REAL CAMELYON17 SMOKE TEST COMPLETED SUCCESSFULLY!")
print("==================================================")
