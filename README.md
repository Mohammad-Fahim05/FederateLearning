# DP-WHFedDG: Domain-Generalized Federated Learning for Robust Privacy-Preserving Medical Diagnosis Across Hospitals

**Project Title:** Domain-Generalized Federated Learning for Robust Privacy-Preserving Medical Diagnosis Across Hospitals
**Dataset:** Camelyon17 / WILDS Histopathology Metastasis Detection Benchmark
**Proposed Framework:** DP-WHFedDG — Differentially Private Worst-Hospital Risk-Aware Federated Domain Generalization
**Task:** Binary histopathology patch classification with evaluation on an unseen hospital domain
**Research Focus:** Privacy-preserving federated learning under hospital-level domain shift

---

## 📌 Executive Overview

Medical image models can experience substantial performance degradation when applied to data from hospitals that differ in staining procedures, scanners, tissue preparation, patient populations, and other acquisition factors.

**Federated Learning (FL)** allows multiple hospitals to collaboratively train a machine-learning model without directly sharing their local image data. However, conventional federated optimization methods such as **FedAvg** do not explicitly prioritize hospitals where the current model performs poorly.

At the same time, **Differential Privacy (DP)** introduces controlled noise into the training process to reduce the information that can be inferred from individual training examples. This privacy mechanism can make optimization more difficult and may affect generalization.

This project investigates whether **risk-aware federated aggregation combined with differential privacy** can improve robustness across heterogeneous hospital domains.

The proposed framework is:

> **DP-WHFedDG — Differentially Private Worst-Hospital Risk-Aware Federated Domain Generalization**

The implemented framework combines:

1. **Local Focal Loss**
2. **Per-Sample Gradient Clipping**
3. **Gaussian Gradient Noise**
4. **DP-Protected Client Loss**
5. **Worst-Hospital Risk-Aware Aggregation**
6. **Unseen-Hospital Domain-Generalization Evaluation**

---

# 🎯 Research Objective

The main research question is:

> **Can risk-aware federated aggregation improve robustness across heterogeneous hospital domains when client updates and client loss information are protected using differential privacy?**

The project evaluates this question by comparing standard, proximal, risk-aware, and differentially private federated-learning approaches on Camelyon17.

The key experimental setting is:

```text
Training Hospitals
      │
      ├── Center 0
      ├── Center 1
      └── Center 2
              │
              ▼
       Federated Training
              │
              ▼
        Validation Domain
           Center 3
              │
              ▼
        Unseen Test Domain
           Center 4
```

**Center 4 is not used for federated training, aggregation, or model tuning.**

This provides an evaluation setting for measuring generalization to an unseen hospital domain.

---

# 🏥 Dataset

The project uses the **Camelyon17/WILDS** histopathology benchmark for binary metastasis detection.

The dataset contains histopathology image patches collected from multiple hospital centers. These hospital centers provide naturally occurring domain differences that can be used to study cross-hospital generalization.

## Hospital Split

| Center    |     Samples | Role                      |
| --------- | ----------: | ------------------------- |
| 0         |      59,436 | Federated training client |
| 1         |      34,904 | Federated training client |
| 2         |      85,054 | Federated training client |
| 3         |     129,838 | Validation domain         |
| 4         |     146,722 | Unseen test domain        |
| **Total** | **455,954** |                           |

## Experimental Roles

```text
Training Clients:
    Center 0
    Center 1
    Center 2

Validation:
    Center 3

Unseen Test:
    Center 4
```

The unseen test center is excluded from:

* Local client training
* Federated aggregation
* Hyperparameter tuning
* Model selection

This allows the final evaluation to measure generalization to an unseen hospital domain.

---

# 🧠 Proposed DP-WHFedDG Framework

The implemented DP-WHFedDG pipeline is:

```text
Hospital Dataset
       │
       ▼
Local Model Training
       │
       ├── Focal Loss
       │
       ├── Per-Sample Gradients
       │
       ├── Gradient Clipping
       │
       └── Gaussian Noise
       │
       ▼
DP-Protected Client Update
       │
       ▼
DP-Protected Client Loss
       │
       ▼
Worst-Hospital Risk-Aware Aggregation
       │
       ▼
Global Model
       │
       ▼
Unseen-Hospital Evaluation
```

---

## 1. Local Focal Loss

Each federated hospital trains a local model using focal loss.

The configured parameters are:

```text
focal_gamma = 2.0
focal_alpha = [0.5, 0.5]
```

Focal loss is used as the local classification objective.

---

## 2. Per-Sample Differential Privacy

The local training process computes gradients for individual samples.

The implemented process is:

```text
Input Batch
     ↓
Per-Sample Gradients
     ↓
Per-Sample Gradient Norm
     ↓
Gradient Clipping
     ↓
Gradient Aggregation
     ↓
Gaussian Noise
     ↓
Noisy Gradient
     ↓
Optimizer Update
```

The configured privacy parameters are:

```text
max_grad_norm = 1.0
noise_multiplier = 0.8
delta = 1e-5
```

The per-sample clipping operation limits the contribution of an individual training example before Gaussian noise is added.

---

## 3. DP-Protected Client Loss

The server uses client loss information for risk-aware aggregation.

Because transmitting an unprotected client loss would expose additional information about the client's local data, the implementation perturbs the client loss with Gaussian noise before it is transmitted to the server.

Therefore, the server uses a **DP-protected client loss** rather than the raw local loss.

---

## 4. Worst-Hospital Risk-Aware Aggregation

Standard FedAvg weights client models primarily according to their dataset sizes.

DP-WHFedDG instead uses the current DP-protected client losses to determine risk-aware aggregation weights.

For client `k`:

```math
w_k =
\frac{
\exp\left(
\eta(\tilde L_k-\max_j\tilde L_j)
\right)
}{
\sum_i
\exp\left(
\eta(\tilde L_i-\max_j\tilde L_j)
\right)
}
```

Where:

* `\tilde L_k` is the DP-protected client loss.
* `\eta` controls the strength of risk reweighting.
* Clients with higher losses receive larger aggregation weights.

The configured value is:

```text
eta = 0.5
```

The objective is to give greater influence to hospitals where the current global model performs poorly.

---

# 🧪 Baseline Methods

The project implements the following federated-learning methods.

## FedAvg

Standard sample-count-weighted Federated Averaging.

## FedProx

FedAvg with a proximal regularization term intended to reduce client drift caused by heterogeneous local data.

Configured proximal coefficient:

```text
mu = 0.01
```

## GroupDRO

A risk-aware federated aggregation approach based on client losses.

## DP-FedAvg

FedAvg combined with the implemented differentially private local optimization mechanism.

## DP-WHFedDG

The proposed method:

```text
Focal Loss
     +
Per-Sample Gradient Clipping
     +
Gaussian Gradient Noise
     +
DP-Protected Client Loss
     +
Worst-Hospital Risk-Aware Aggregation
```

---

# ⚙️ Configuration

The main configuration file is:

```text
configs/camelyon17_wilds.yaml
```

Important configuration parameters include:

```yaml
federated:
  rounds: 100
  local_epochs: 2
  batch_size: 64
  learning_rate: 0.001
  momentum: 0.9
  weight_decay: 0.0001
  focal_gamma: 2.0
  focal_alpha: [0.5, 0.5]
  mu: 0.01

privacy:
  enabled: true
  target_epsilon: 3.0
  target_delta: 0.00001
  max_grad_norm: 1.0
  noise_multiplier: 0.8

dro:
  eta: 0.5
```

The configuration also contains multiple random seeds for future multi-seed experiments.

---

# 📁 Repository Architecture

```text
FederateLearning/
│
├── configs/
│   └── camelyon17_wilds.yaml
│
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   ├── preprocessing.py
│   │   └── client_splitter.py
│   │
│   ├── models/
│   │   └── resnet_backbone.py
│   │
│   ├── privacy/
│   │   ├── dp_optimizer.py
│   │   └── privacy_accountant.py
│   │
│   ├── federated/
│   │   ├── client.py
│   │   ├── server.py
│   │   └── trainer.py
│   │
│   └── evaluation/
│       ├── metrics.py
│       └── slide_evaluator.py
│
├── experiments/
│   ├── run_baselines.py
│   ├── run_proposed.py
│   └── run_ablations.py
│
├── deployment/
│   ├── predict.py
│   └── app.py
│
├── tests/
│   ├── test_data_pipeline.py
│   ├── test_privacy.py
│   └── test_federated_loop.py
│
├── env_check.py
├── requirements.txt
└── README.md
```

---

# 🚀 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/Mohammad-Fahim05/FederateLearning.git
cd FederateLearning
```

> The repository can remain private. Authentication is required if the repository is configured as private.

---

## 2. Create the Python Environment

Python 3.10+ is recommended.

Using Anaconda:

```bash
conda create -n federated-medical python=3.10
conda activate federated-medical
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Check the Environment

Run:

```bash
python env_check.py
```

This checks the available Python environment, PyTorch installation, device availability, and required project packages.

---

# 🧪 Verification Tests

Set the repository root as the Python path.

### Windows PowerShell

```powershell
$env:PYTHONPATH="."
```

Run the test suite:

```bash
python tests/test_data_pipeline.py
python tests/test_privacy.py
python tests/test_federated_loop.py
```

The tests cover:

* Dataset loading
* Data partitioning
* Slide-level leakage safeguards
* Privacy components
* Federated training functionality

---

# 🏃 Running Experiments

## Baseline Experiments

Run the baseline experiment runner:

```bash
python experiments/run_baselines.py
```

Available baselines include:

```text
FedAvg
FedProx
GroupDRO
DP-FedAvg
```

---

## Proposed DP-WHFedDG

Run:

```bash
python experiments/run_proposed.py
```

The experiment runner supports the configured seeds and privacy-budget settings.

Configured target privacy budgets include:

```text
epsilon = 1.0
epsilon = 3.0
epsilon = 8.0
```

> These are available experimental configurations and should not be interpreted as completed results unless the corresponding experiments have actually been executed.

---

## Ablation Experiments

Run:

```bash
python experiments/run_ablations.py
```

The ablation runner is intended to measure the contribution of individual components.

For example:

```text
DP-FedAvg
    ↓
DP + Focal Loss
    ↓
DP + Risk-Aware Aggregation
    ↓
DP + Focal Loss + Risk-Aware Aggregation
```

The complete real-data ablation matrix has not yet been completed.

---

# 📊 Evaluation Metrics

The project evaluates model performance using several metrics.

## Patch-Level Metrics

* Accuracy
* Balanced Accuracy
* Macro F1
* AUROC
* Expected Calibration Error (ECE)

## Hospital-Level Metrics

The evaluation reports:

* Average hospital performance
* Worst-hospital performance
* Validation-center performance
* Unseen-hospital performance

The primary domain-generalization metric of interest is performance on the unseen **Center 4**.

---

# 🧬 Slide-Level Evaluation

The repository contains a slide-level evaluation component that converts patch-level predictions into slide-level predictions.

The implemented aggregation uses the **top-5 patch tumor probabilities**.

Supported slide-level metrics include:

* Slide AUROC
* Slide Classification Accuracy

If AUROC is mathematically undefined because the evaluated slide labels contain only one class, the implementation reports:

```text
NaN
```

rather than incorrectly replacing the value with `0.5`.

---

# 🔐 Differential Privacy Accounting

The repository includes:

```text
src/privacy/privacy_accountant.py
```

which implements RDP-based privacy accounting.

The accounting procedure considers parameters such as:

* Number of federated rounds
* Local epochs
* Client dataset size
* Batch size
* Sampling ratio
* Gradient clipping norm
* Gaussian noise multiplier
* Target delta

The configured target delta is:

```text
delta = 1e-5
```

The accounting implementation reports estimated privacy expenditure in terms of `epsilon`.

## Important Privacy Limitation

The current accountant uses an analytical approximation for the subsampled Gaussian mechanism.

The actual training DataLoader uses shuffled mini-batches rather than an explicitly implemented Poisson-sampling mechanism.

Therefore, the reported epsilon values should currently be treated as **privacy-accounting estimates produced by the implementation**, not as an independently verified formal privacy certificate.

Before making a publication-level formal `(epsilon, delta)`-DP claim, the accounting assumptions should be independently verified against the exact sampling and release mechanism.

---

# 📈 Completed Real-Data Results

Real Camelyon17 experiments were performed using the verified dataset and the hospital split described above.

The recorded unseen-hospital **Center 4 accuracy** results are:

| Method     | Rounds | Seed | Center 4 Accuracy |
| ---------- | -----: | ---: | ----------------: |
| FedAvg     |     20 |   42 |        **88.66%** |
| GroupDRO   |     10 |   42 |        **81.31%** |
| FedProx    |     10 |   42 |        **71.08%** |
| DP-FedAvg  |     10 |   42 |        **62.57%** |
| DP-WHFedDG |     10 |   42 |        **58.67%** |

For the completed DP-WHFedDG run:

| Metric                           |     Result |
| -------------------------------- | ---------: |
| Center 4 Accuracy                | **58.67%** |
| Center 4 Balanced Accuracy       | **58.67%** |
| Center 4 Macro F1                | **0.5022** |
| Center 4 AUROC                   | **0.5616** |
| Training-Center Average Accuracy | **75.05%** |
| Training-Center Worst Accuracy   | **62.47%** |
| Reported Total Epsilon           | **≈ 0.83** |
| Delta                            |   **1e-5** |

---

# 📌 Current Result Interpretation

The current results **do not demonstrate that DP-WHFedDG outperforms the tested baselines**.

In the controlled 10-round comparison:

```text
DP-WHFedDG = 58.67%
DP-FedAvg  = 62.57%
```

Therefore, under the tested configuration, adding the current worst-hospital risk-aware aggregation did not recover the unseen-hospital performance lost under differential privacy.

This is an important experimental result rather than a result that should be hidden or replaced.

The current study therefore investigates the trade-off between:

```text
Differential Privacy
        ↕
Optimization Difficulty
        ↕
Hospital-Level Robustness
        ↕
Unseen-Hospital Generalization
```

---

# ⚠️ Experimental Comparison Caveat

The recorded FedAvg result used:

```text
20 rounds
```

while the other listed real-data experiments used:

```text
10 rounds
```

Therefore, the current table is **not a perfectly matched computational-budget comparison**.

This limitation should be explicitly considered in any scientific interpretation or publication.

---

# 🔬 Current Research Status

## Completed

* Real Camelyon17/WILDS dataset verification
* Hospital-based federated partitioning
* Unseen-hospital evaluation protocol
* Federated training infrastructure
* FedAvg implementation
* FedProx implementation
* GroupDRO implementation
* DP-FedAvg implementation
* DP-WHFedDG implementation
* Per-sample gradient clipping
* Gaussian gradient noise
* DP-protected client losses
* RDP accounting implementation
* Patch-level evaluation
* Hospital-level evaluation
* Slide-level evaluation framework
* Automated verification tests
* Deployment inference pipeline
* Gradio deployment interface

## Remaining Research Work

* Complete multi-seed real-data evaluation
* Complete privacy-budget sweep
* Complete real-data ablation study
* Independently verify privacy accounting assumptions
* Perform final statistical analysis
* Obtain and verify final trained checkpoint
* Connect the verified trained checkpoint to deployment
* Finalize research paper/report

---

# 🚀 Deployment Prototype

The repository includes an inference pipeline:

```text
deployment/predict.py
```

Run:

```bash
python deployment/predict.py
```

Example:

```python
from deployment.predict import Camelyon17Predictor
from PIL import Image

predictor = Camelyon17Predictor(
    checkpoint_path="checkpoints/dp_whfeddg_seed42.pt"
)

patch_img = Image.open("sample_patch.png")

result = predictor.predict_patch(patch_img)

print("Patch Prediction:", result)
```

The predictor returns:

* Tumor probability
* Normal probability
* Predicted diagnosis

The WSI prediction interface can aggregate multiple patch predictions to produce a slide-level risk score.

---

# 🖥️ Gradio Deployment Interface

A Gradio-based interface is provided in:

```text
deployment/app.py
```

Run:

```bash
python deployment/app.py
```

The interface accepts an image patch and displays:

```text
Tumor Probability
Normal Probability
Diagnosis
```

## Deployment Status

The deployment pipeline and Gradio interface have been tested successfully as an inference pipeline.

However, a verified final trained DP-WHFedDG checkpoint is currently not available in the local deployment environment.

Therefore:

> **The current deployment interface is a research prototype and must not be considered a clinically validated diagnostic system.**

A trained checkpoint must be connected and verified before using the interface for model-based inference.

---

# ⚠️ Research Limitations

The current study has several limitations.

## 1. Limited Real-Data Seeds

The currently recorded real-data experiments use seed 42.

The configuration supports multiple seeds, but a complete multi-seed study has not yet been completed.

## 2. Different Training Budgets

The recorded FedAvg experiment used 20 rounds, while several other methods used 10 rounds.

## 3. Privacy Accountant Verification

The current RDP accountant uses an analytical approximation for the sampling mechanism.

Independent verification is required before making a strong formal privacy claim.

## 4. Current Proposed-Method Performance

DP-WHFedDG does not currently outperform the tested baselines on unseen Center 4 accuracy.

Therefore, this project does not claim state-of-the-art performance.

## 5. Missing Final Deployment Checkpoint

The final trained research checkpoint is currently unavailable in the deployment environment.

## 6. Clinical Validation

This project is an academic research prototype.

The model has not undergone prospective clinical validation and must not be used to make medical decisions.

---

# 🔬 Research Contribution

This project investigates the combination of:

```text
Hospital-Level Domain Shift
            +
Federated Learning
            +
Differential Privacy
            +
Worst-Hospital Risk-Aware Aggregation
```

The central contribution is an experimental framework for studying whether **privacy-preserving risk-aware federated optimization** can improve robustness when participating hospitals have heterogeneous data distributions and the final model must generalize to an unseen hospital.

Rather than assuming that the proposed method will outperform existing methods, the project evaluates it against both private and non-private federated baselines.

---

# 📚 Reproducibility

The repository provides:

* Configuration files
* Federated-learning implementations
* Baseline implementations
* Privacy mechanisms
* Privacy accounting
* Dataset split definitions
* Evaluation metrics
* Experiment runners
* Automated tests
* Deployment code

The primary domain-generalization split is:

```text
Training:
    Center 0
    Center 1
    Center 2

Validation:
    Center 3

Unseen Test:
    Center 4
```

The unseen test center is excluded from federated training and aggregation.

---

# 📖 Research Documentation

Additional research documentation is maintained in:

```text
FINAL_RESEARCH_SPECIFICATION.md
research_phase_report.md
```

These documents contain the detailed research specification, methodology, experimental planning, and project progress information.

---

# 📌 Project Status

**Status: Research Prototype / Experimental Study**

The core federated-learning, differential-privacy, evaluation, and deployment infrastructure has been implemented and tested.

Real Camelyon17 experiments have been completed for the principal baselines and DP-WHFedDG under the currently available compute budget.

The current experimental results do not establish superiority of DP-WHFedDG. Further controlled experiments, privacy-accounting verification, and statistical analysis are required before making strong research claims.

The next research steps are:

1. Verify the privacy-accounting methodology.
2. Complete the necessary controlled experiments when compute resources are available.
3. Perform statistical analysis.
4. Finalize the scientific conclusions.
5. Connect and verify a trained checkpoint for deployment.
6. Prepare the final research paper/report.

---

# ⚖️ Disclaimer

This repository is intended for **academic research and experimentation**.

The models and deployment interface are **not clinically validated** and must not be used for medical diagnosis, treatment decisions, or other clinical decision-making.
