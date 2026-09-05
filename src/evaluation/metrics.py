import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, balanced_accuracy_score

def compute_classification_metrics(y_true, y_probs, y_preds):
    """
    Computes Patch-level evaluation metrics.
    """
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    y_preds = np.array(y_preds)
    
    acc = accuracy_score(y_true, y_preds)
    balanced_acc = balanced_accuracy_score(y_true, y_preds)
    f1 = f1_score(y_true, y_preds, average='macro', zero_division=0)
    
    try:
        auroc = roc_auc_score(y_true, y_probs[:, 1])
    except Exception:
        auroc = 0.5
        
    return {
        'accuracy': float(acc),
        'balanced_accuracy': float(balanced_acc),
        'macro_f1': float(f1),
        'auroc': float(auroc)
    }

def compute_worst_hospital_metric(hospital_metrics):
    """
    Computes worst-hospital accuracy and average accuracy across hospitals.
    hospital_metrics: Dict of {center_id: {'accuracy': acc, 'auroc': auroc, ...}}
    """
    accuracies = [m['accuracy'] for m in hospital_metrics.values()]
    aurocs = [m['auroc'] for m in hospital_metrics.values()]
    
    worst_acc = min(accuracies)
    avg_acc = sum(accuracies) / len(accuracies)
    worst_auroc = min(aurocs)
    avg_auroc = sum(aurocs) / len(aurocs)
    
    return {
        'worst_hospital_accuracy': worst_acc,
        'average_hospital_accuracy': avg_acc,
        'worst_hospital_auroc': worst_auroc,
        'average_hospital_auroc': avg_auroc
    }

def compute_ece(y_true, y_probs, n_bins=10):
    """
    Computes Expected Calibration Error (ECE).
    """
    y_true = np.array(y_true)
    y_probs = np.max(np.array(y_probs), axis=1)
    y_preds = np.argmax(np.array(y_probs), axis=1) if len(y_probs.shape) > 1 else (y_probs >= 0.5).astype(int)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (y_probs > bin_lower) & (y_probs <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin] == y_preds[in_bin])
            avg_confidence_in_bin = np.mean(y_probs[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
            
    return float(ece)
