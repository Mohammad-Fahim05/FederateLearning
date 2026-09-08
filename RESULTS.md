# Experimental Results

## 1. Experimental Objective

The objective of the experiments is to evaluate privacy-preserving federated domain generalization for medical image classification across heterogeneous hospital domains.

The primary evaluation setting uses Camelyon17/WILDS with:

- Centers 0, 1, and 2 as federated training clients
- Center 3 as the validation domain
- Center 4 as the unseen test domain

Center 4 is excluded from federated training, aggregation, and model tuning.

---

## 2. Methods Evaluated

The following federated-learning methods were evaluated:

1. FedAvg
2. FedProx
3. GroupDRO
4. DP-FedAvg
5. DP-WHFedDG

DP-WHFedDG is the proposed framework combining:

- Local focal loss
- Per-sample gradient clipping
- Gaussian gradient noise
- DP-protected client loss
- Worst-hospital risk-aware aggregation

---

## 3. Unseen-Hospital Results

The primary domain-generalization evaluation is performed on unseen Center 4.

| Method | Privacy | Rounds | Seed | Center 4 Accuracy |
|---|---|---:|---:|---:|
| FedAvg | No | 20 | 42 | 88.66% |
| GroupDRO | No | 10 | 42 | 81.31% |
| FedProx | No | 10 | 42 | 71.08% |
| DP-FedAvg | Yes | 10 | 42 | 62.57% |
| DP-WHFedDG | Yes | 10 | 42 | 58.67% |

---

## 4. DP-WHFedDG Detailed Results

The completed real-data DP-WHFedDG experiment produced:

| Metric | Result |
|---|---:|
| Unseen Center 4 Accuracy | 58.67% |
| Unseen Center 4 Balanced Accuracy | 58.67% |
| Unseen Center 4 Macro F1 | 0.5022 |
| Unseen Center 4 AUROC | 0.5616 |
| Training-Center Average Accuracy | 75.05% |
| Training-Center Worst Accuracy | 62.47% |
| Reported Total Epsilon | ≈ 0.83 |
| Delta | 1 × 10⁻⁵ |

The reported privacy value is the result of the project's implemented RDP accounting procedure and should be treated as an accounting estimate rather than an independently verified formal privacy certificate.

---

## 5. Controlled Private Comparison

The most directly relevant comparison is between DP-FedAvg and DP-WHFedDG because both were evaluated using:

- Seed 42
- 10 federated rounds
- 1 local epoch
- Differentially private local optimization

Results:

| Method | Center 4 Accuracy |
|---|---:|
| DP-FedAvg | 62.57% |
| DP-WHFedDG | 58.67% |

Difference:

```text
58.67% - 62.57% = -3.90 percentage points