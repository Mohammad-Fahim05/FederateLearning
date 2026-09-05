import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

class SlideEvaluator:
    """
    Aggregates patch-level predictions to clinical Whole Slide Image (WSI) level.
    Computes Slide-Level AUROC and Slide Classification Accuracy.
    """
    def __init__(self):
        pass

    def evaluate_slides(self, patch_predictions, patch_targets, slide_ids):
        """
        patch_predictions: List of tumor probability values for patches
        patch_targets: List of patch binary ground truth labels (1 = tumor patch)
        slide_ids: List of slide IDs corresponding to patches
        """
        slide_data = {}
        
        for pred, target, sid in zip(patch_predictions, patch_targets, slide_ids):
            if sid not in slide_data:
                slide_data[sid] = {'probs': [], 'targets': []}
            slide_data[sid]['probs'].append(pred)
            slide_data[sid]['targets'].append(target)
            
        slide_targets = []
        slide_max_probs = []
        slide_top5_mean_probs = []
        
        for sid, data in slide_data.items():
            # A slide is positive if any patch contains tumor cells
            is_slide_tumor = 1 if max(data['targets']) == 1 else 0
            slide_targets.append(is_slide_tumor)
            
            # WSI aggregation methods: Max patch probability & Top-5 patch mean
            sorted_probs = sorted(data['probs'], reverse=True)
            max_prob = sorted_probs[0]
            top5_mean = np.mean(sorted_probs[:min(5, len(sorted_probs))])
            
            slide_max_probs.append(max_prob)
            slide_top5_mean_probs.append(top5_mean)
            
        slide_targets = np.array(slide_targets)
        slide_top5_mean_probs = np.array(slide_top5_mean_probs)
        
        # Calculate Slide AUROC
        try:
            slide_auroc = roc_auc_score(slide_targets, slide_top5_mean_probs)
        except Exception:
            slide_auroc = 0.5
            
        # Calculate Slide Accuracy at 0.5 threshold
        slide_preds = (slide_top5_mean_probs >= 0.5).astype(int)
        slide_acc = accuracy_score(slide_targets, slide_preds)
        
        return {
            'slide_auroc': float(slide_auroc),
            'slide_accuracy': float(slide_acc),
            'num_slides': len(slide_targets)
        }
