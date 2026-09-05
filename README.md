# DP-WHFedDG: Domain-Generalized Federated Learning for Robust Privacy-Preserving Medical Diagnosis Across Hospitals

**Project Title:** Domain-Generalized Federated Learning for Robust Privacy-Preserving Medical Diagnosis Across Hospitals  
**Target Dataset:** Camelyon17 (Histopathology Metastasis Detection Benchmark)  
**Proposed Framework:** DP-WHFedDG (Differentially Private Worst-Hospital Risk-Aware Federated Domain Generalization)  
**Target Venue:** Peer-Reviewed Medical AI / ML Conference (e.g., MICCAI, IEEE TMI, NeurIPS/ICLR Workshop)  

---

## 📌 Executive Overview

Centralized deep learning models for medical image analysis fail when deployed across new hospitals due to severe distribution shifts (variations in tissue stain intensity, slide preparation protocols, scanner hardware, and institutional patient demographics). While Federated Learning (FL) enables collaborative model training across hospitals without sharing raw patient data, standard FL (FedAvg) does not generalize to unseen hospitals and is vulnerable to patient-level gradient leakage attacks. Existing Federated Domain Generalization (FedDG) algorithms rely on sharing style statistics or frequency spectra across clients, violating privacy compliance.

**DP-WHFedDG** solves this fundamental bottleneck by introducing:
1. **Local Focal + SAM-lite Sharpness Regularization:** Combines focal loss for local class imbalance with sharpness awareness to stabilize local representation learning under differential privacy clipping.
2. **Server-Side Noise-Bounded Smooth DRO Aggregation:** Applies exponential risk-reweighting over historical DP-perturbed hospital losses, preventing worst-hospital optimization collapse under DP noise.
3. **Formal Differential Privacy Guarantees:** Enforces $(\epsilon, \delta)$-Differential Privacy (Rényi DP accountant) against an honest-but-curious server and external update eavesdroppers.

---

## 📁 Repository Architecture

```text
FederateLearning/
├── configs/
│   └── camelyon17_wilds.yaml      # Master configuration file
├── src/
│   ├── data/
│   │   ├── dataset.py            # Unified Camelyon17 PyTorch / WILDS dataset loader
│   │   ├── preprocessing.py      # Normalization & stain augmentation
│   │   └── client_splitter.py    # Hospital partitioning & stress test splitter
│   ├── models/
│   │   └── resnet_backbone.py    # ResNet-18 feature extractor & local focal loss
│   ├── privacy/
│   │   ├── dp_optimizer.py       # Per-sample gradient clipper & Gaussian DP noise
│   │   └── privacy_accountant.py # RDP Epsilon/Delta accountant
│   ├── federated/
│   │   ├── client.py             # Hospital client trainer (Focal + SAM + DP)
│   │   ├── server.py             # Central server aggregator (Smooth DRO / FedAvg / GroupDRO)
│   │   └── trainer.py            # Communication round coordinator
│   └── evaluation/
│       ├── metrics.py            # Patch accuracy, AUROC, F1, ECE, worst-hospital calculator
│       └── slide_evaluator.py    # WSI patch-to-slide clinical AUROC evaluator
├── experiments/
│   ├── run_baselines.py          # Baseline suite runner (FedAvg, FedProx, GroupDRO, DP-FedAvg)
│   ├── run_proposed.py           # Multi-seed proposed DP-WHFedDG runner
│   └── run_ablations.py          # Full component ablation suite runner
├── deployment/
│   └── predict.py                # Standalone inference & clinical WSI risk score prediction
├── tests/
│   ├── test_data_pipeline.py     # Unit test for dataset loader & zero slide leakage
│   ├── test_privacy.py           # Unit test for RDP accountant & DP clipper
│   └── test_federated_loop.py    # Integration test for federated loop
├── env_check.py                  # System & GPU environment inspection script
├── requirements.txt              # Pinned Python package dependencies
└── README.md                     # Comprehensive project documentation
```

---

## 🚀 Quickstart & Reproduction Guide

### 1. Environment Setup
Ensure Anaconda Python 3.10+ is installed. Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```
To verify system dependencies, GPU availability, and package status:
```bash
python env_check.py
```

### 2. Run Automated Verification Tests
Run the unit and integration test suite to verify data loading, zero slide leakage, privacy accountant calculations, and federated loop execution:
```bash
$env:PYTHONPATH="."
python tests/test_data_pipeline.py
python tests/test_privacy.py
python tests/test_federated_loop.py
```

### 3. Execute Baseline Experiments
Run training and evaluation for baseline models (FedAvg, FedProx, Group DRO, DP-FedAvg):
```bash
python experiments/run_baselines.py
```

### 4. Execute Proposed Method (`DP-WHFedDG`)
Run multi-seed evaluation for the proposed DP-WHFedDG framework under targeted privacy budgets ($\epsilon \in \{1.0, 3.0, 8.0\}$):
```bash
python experiments/run_proposed.py
```

### 5. Execute Ablation Study
Run the full ablation matrix to quantify individual component contributions:
```bash
python experiments/run_ablations.py
```

---

## 🔬 Clinical Diagnostic Inference (Deployment Prototype)

To run diagnostic prediction on new whole slide image patches using trained model weights:
```bash
python deployment/predict.py
```

Example Python API usage:
```python
from deployment.predict import Camelyon17Predictor
from PIL import Image

predictor = Camelyon17Predictor(checkpoint_path="checkpoints/dp_whfeddg_best.pt")

# Single Patch Diagnosis
patch_img = Image.open("sample_patch.png")
res = predictor.predict_patch(patch_img)
print("Patch Prediction:", res)

# WSI Slide Risk Score
slide_res = predictor.predict_wsi_slide([patch_img] * 10)
print("WSI Slide Diagnosis:", slide_res)
```

---

## 📊 Scientific Metrics & Evaluation Protocol
* **Patch-Level Metrics:** Worst-Hospital Accuracy (%), Average Accuracy (%), Out-of-Domain AUROC, Macro-F1, Expected Calibration Error (ECE).
* **Clinical Slide-Level Metrics:** Slide-Level AUROC, Slide Classification Accuracy (aggregated top-5 patch mean per WSI).
* **Privacy Bounds:** Formal $(\epsilon \le 3.0, \delta = 10^{-5})$ Differential Privacy guaranteed via Rényi DP composition.

---

## 🔒 Citation & Research Context
* **Dataset:** Camelyon17 (WILDS Benchmark / Grand Challenge)
* **Research Specification:** See [FINAL_RESEARCH_SPECIFICATION.md](file:///C:/Users/fahim/.gemini/antigravity-ide/brain/3889cfd2-6c28-42e1-90aa-9dfde206cd6d/FINAL_RESEARCH_SPECIFICATION.md)
* **Research Phase Report:** See [research_phase_report.md](file:///C:/Users/fahim/.gemini/antigravity-ide/brain/3889cfd2-6c28-42e1-90aa-9dfde206cd6d/research_phase_report.md)
