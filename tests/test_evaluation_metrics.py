import math
import warnings
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from src.evaluation.metrics import compute_classification_metrics, compute_ece, compute_worst_hospital_metric
from src.evaluation.slide_evaluator import SlideEvaluator
from src.federated.trainer import FederatedTrainer

def test_compute_ece_known_values_2d():
    """
    Test ECE on known synthetic 2D probability matrices.
    """
    # Case 1: Overconfident and completely wrong
    # 4 samples: true label = 1
    # Model predicts class 0 with 90% confidence: [0.9, 0.1]
    y_true = np.array([1, 1, 1, 1])
    y_probs = np.array([
        [0.9, 0.1],
        [0.9, 0.1],
        [0.9, 0.1],
        [0.9, 0.1]
    ])
    # Predicted class = 0, accuracy = 0.0, confidence = 0.9 -> ECE = |0.0 - 0.9| = 0.9
    ece = compute_ece(y_true, y_probs, n_bins=10)
    assert abs(ece - 0.9) < 1e-5

    # Case 2: Perfectly calibrated
    # 10 samples in bin (0.7, 0.8]: confidence 0.8, exactly 8 correct (acc = 0.8)
    y_true = np.array([1]*8 + [0]*2)
    y_probs = np.array([[0.2, 0.8]]*8 + [[0.2, 0.8]]*2) # predicts class 1 with conf 0.8
    ece = compute_ece(y_true, y_probs, n_bins=10)
    assert abs(ece - 0.0) < 1e-5

def test_compute_ece_1d_input():
    """
    Test ECE on 1D probabilities representing P(Y=1).
    """
    # 4 samples: true label = 1, predicted P(Y=1) = 0.1 (so conf = 0.9 for class 0, acc = 0)
    y_true = np.array([1, 1, 1, 1])
    y_probs = np.array([0.1, 0.1, 0.1, 0.1])
    ece = compute_ece(y_true, y_probs, n_bins=10)
    assert abs(ece - 0.9) < 1e-5

def test_compute_classification_metrics_includes_ece():
    """
    Test that compute_classification_metrics computes and returns valid non-zero ECE.
    """
    y_true = [1, 1, 0, 0]
    y_probs = np.array([
        [0.1, 0.9],
        [0.8, 0.2],  # wrong prediction with conf 0.8
        [0.7, 0.3],  # correct prediction with conf 0.7
        [0.4, 0.6]   # wrong prediction with conf 0.6
    ])
    y_preds = [1, 0, 0, 1]
    
    metrics = compute_classification_metrics(y_true, y_probs, y_preds)
    
    assert 'accuracy' in metrics
    assert 'balanced_accuracy' in metrics
    assert 'macro_f1' in metrics
    assert 'auroc' in metrics
    assert 'ece' in metrics
    assert isinstance(metrics['ece'], float)
    assert metrics['ece'] > 0.0  # Must not be hardcoded to 0.0

def test_slide_evaluator_one_class_returns_nan():
    """
    When all slides belong to only one class, Slide AUROC must return NaN and emit a warning.
    """
    evaluator = SlideEvaluator()
    # 10 patches across 2 slides; all patches and slides are tumor-positive (class 1)
    patch_preds = [0.8, 0.7, 0.9, 0.85, 0.9, 0.75, 0.8, 0.65, 0.7, 0.8]
    patch_targets = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    slide_ids = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    
    with pytest.warns(UserWarning, match="mathematically undefined"):
        results = evaluator.evaluate_slides(patch_preds, patch_targets, slide_ids)
        
    assert math.isnan(results['slide_auroc']), f"Expected NaN slide_auroc, got {results['slide_auroc']}"
    assert results['slide_accuracy'] == 1.0
    assert results['num_slides'] == 2

def test_slide_evaluator_two_classes_returns_finite_auroc():
    """
    When both classes are present at slide level, Slide AUROC must return a finite numeric score.
    """
    evaluator = SlideEvaluator()
    # Slide 0: negative (all patch targets 0), Slide 1: positive (patch targets 1)
    patch_preds = [0.1, 0.2, 0.05, 0.15, 0.9, 0.85, 0.95, 0.8]
    patch_targets = [0, 0, 0, 0, 1, 1, 1, 1]
    slide_ids = [0, 0, 0, 0, 1, 1, 1, 1]
    
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        results = evaluator.evaluate_slides(patch_preds, patch_targets, slide_ids)
        
    assert not math.isnan(results['slide_auroc'])
    assert results['slide_auroc'] == 1.0
    assert results['slide_accuracy'] == 1.0
    assert results['num_slides'] == 2

def test_evaluate_on_subset_returns_ece():
    """
    Verify evaluate_on_subset() returns the computed ECE along with classification metrics.
    """
    import yaml
    with open('./configs/camelyon17_wilds.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    from src.data.dataset import Camelyon17HospitalDataset
    dataset = Camelyon17HospitalDataset(use_synthetic=True, seed=42)
    client_indices = {0: list(range(10))}
    trainer = FederatedTrainer(dataset, client_indices, config, device='cpu')
    
    eval_indices = list(range(20))
    metrics = trainer.evaluate_on_subset(eval_indices)
    
    assert 'ece' in metrics, "evaluate_on_subset() must return 'ece' in metrics dict"
    assert isinstance(metrics['ece'], float)
    assert not math.isnan(metrics['ece'])
    assert 'accuracy' in metrics
    assert 'balanced_accuracy' in metrics
    assert 'macro_f1' in metrics
    assert 'auroc' in metrics
    assert 'slide_auroc' in metrics
    assert 'slide_accuracy' in metrics
