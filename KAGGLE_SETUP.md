# Kaggle GPU Environment Setup & Execution Guide

This document specifies the exact instructions for setting up and executing the **DP-WHFedDG** federated learning research pipeline on a **Kaggle GPU Notebook**.

---

## 1. Kaggle Notebook Creation & Hardware Settings

1. On [Kaggle](https://www.kaggle.com/), create a new Notebook (**Code** $\rightarrow$ **New Notebook**).
2. In the right-hand **Notebook settings** sidebar:
   - **Accelerator**: Select **GPU T4 x2** or **GPU P100** (Recommended).
   - **Language**: `Python`.
   - **Internet**: Toggle **ON** (Required for package installation and WILDS dataset download).
   - **Environment**: Use default latest Kaggle environment.

---

## 2. Notebook Execution Cells

### Cell 1: Install Project Dependencies
Kaggle GPU environments provide PyTorch and CUDA out-of-the-box. Install only the required domain-specific libraries:

```python
!pip install -q wilds opacus
```

---

### Cell 2: Obtain Project Code & Set Working Directory
Clone or extract the project repository into the Kaggle environment:

```python
import os
import sys

# Option A: If cloning from Git repository:
# !git clone <YOUR_REPO_URL> /kaggle/working/FederateLearning

# Set working directory to project root and add to sys.path
PROJECT_DIR = '/kaggle/working/FederateLearning'
if not os.path.exists(PROJECT_DIR):
    PROJECT_DIR = '/kaggle/working'

sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)
print(f"Active Working Directory: {os.getcwd()}")
```

---

### Cell 3: Verify CUDA & GPU Acceleration
Ensure PyTorch is properly recognizing the Kaggle GPU accelerator:

```python
import torch

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Model: {torch.cuda.get_device_name(0)}")
    print(f"Device Count: {torch.cuda.device_count()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
else:
    raise SystemError("CUDA GPU not detected! Please ensure GPU accelerator is enabled in notebook settings.")
```

---

### Cell 4: Verify Project Module Imports
Verify that all core project modules load without dependency or path issues:

```python
import src.models.resnet_backbone
import src.privacy.dp_optimizer
import src.privacy.privacy_accountant
import src.federated.client
import src.federated.server
import src.federated.trainer
import src.evaluation.metrics
import src.evaluation.slide_evaluator
import src.data.dataset
import src.data.client_splitter
import src.data.preprocessing

print("All 11 project modules imported successfully on Kaggle environment!")
```

---

### Cell 5: Real WILDS Camelyon17 GPU Smoke Test (Phase 7F)
Execute the end-to-end minimal real dataset GPU smoke test on Kaggle:

```bash
!python smoke_test_camelyon17.py --data_dir /kaggle/input/datasets/mohdfam/camelyon17-wilds
```

---

## 3. Dataset Path Configuration

In `configs/camelyon17_wilds.yaml`:
- **Pre-Mounted Kaggle Dataset (Recommended)**:
  ```yaml
  dataset:
    name: "camelyon17"
    root_dir: "/kaggle/input/datasets/mohdfam/camelyon17-wilds"
    download: false
    use_synthetic: false
  ```
- **Fallback Automatic Download**:
  ```yaml
  dataset:
    name: "camelyon17"
    root_dir: "./data"
    download: true
    use_synthetic: false
  ```

---

## 4. Research Experiment Commands (After Smoke Test Passes)

Once the GPU smoke test passes:

- **Main Proposed Method Benchmark (DP-WHFedDG across 5 seeds & 3 privacy budgets)**:
  ```bash
  !python experiments/run_proposed.py
  ```

- **Comparative Baselines (FedAvg, FedProx, GroupDRO, DP-FedAvg)**:
  ```bash
  !python experiments/run_baselines.py
  ```

- **Ablation Studies**:
  ```bash
  !python experiments/run_ablations.py
  ```

---

## 5. Artifacts & Results

All output evaluation tables and checkpoints will be written to:
- Results: `/kaggle/working/results/`
- Checkpoints: `/kaggle/working/checkpoints/`
- Download results via Kaggle notebook output file manager.
