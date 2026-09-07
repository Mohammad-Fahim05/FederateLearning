import sys
import os
import torch
import numpy as np
from PIL import Image

# Add root directory to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.resnet_backbone import ResNet18Backbone
from src.data.preprocessing import get_transforms
from src.evaluation.slide_evaluator import SlideEvaluator

class Camelyon17Predictor:
    """
    Research Deployment Prototype for Cross-Hospital Histopathology Metastasis Diagnosis.
    Loads trained DP-WHFedDG model weights and performs patch-level and slide-level diagnostic inference.
    """
    def __init__(self, checkpoint_path=None, device=None):
        if device is None or device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        self.transform = get_transforms(is_train=False)
        self.slide_evaluator = SlideEvaluator()
        
        # Instantiate model architecture
        self.model = ResNet18Backbone(num_classes=2, pretrained=False).to(self.device)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
    print(f"Loading trained model weights from {checkpoint_path}...")

    checkpoint = torch.load(checkpoint_path, map_location=self.device)

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    self.model.load_state_dict(state_dict)
        else:
            print("Warning: No checkpoint provided or file not found. Running with initialized model weights.")
            
        self.model.eval()

    def predict_patch(self, pil_image):
        """
        Predicts tumor metastasis probability for a single 96x96 RGB tissue patch.
        """
        tensor_img = self.transform(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(tensor_img)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            
        tumor_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
        prediction = "Tumor Metastasis Detected" if tumor_prob >= 0.5 else "Normal Tissue"
        return {
            'probability_tumor': tumor_prob,
            'probability_normal': 1.0 - tumor_prob,
            'prediction': prediction
        }

    def predict_wsi_slide(self, patch_list):
        """
        Predicts slide-level metastasis risk from a list of WSI tissue patches.
        """
        patch_probs = []
        for patch in patch_list:
            res = self.predict_patch(patch)
            patch_probs.append(res['probability_tumor'])
            
        sorted_probs = sorted(patch_probs, reverse=True)
        top5_mean = float(np.mean(sorted_probs[:min(5, len(sorted_probs))]))
        
        slide_prediction = "METASTASIS POSITIVE WSI" if top5_mean >= 0.5 else "NORMAL WSI"
        return {
            'slide_metastasis_risk_score': top5_mean,
            'slide_diagnosis': slide_prediction,
            'num_patches_analyzed': len(patch_list),
            'top_patch_probabilities': sorted_probs[:5]
        }

if __name__ == "__main__":
    print("=== CAMELYON17 RESEARCH DEPLOYMENT INFERENCE TEST ===")
    predictor = Camelyon17Predictor()
    
    # Create test synthetic patch
    synthetic_patch = Image.fromarray(np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8))
    res = predictor.predict_patch(synthetic_patch)
    print("Single Patch Inference Output:")
    print(res)
    
    slide_res = predictor.predict_wsi_slide([synthetic_patch] * 10)
    print("\nSlide-Level WSI Inference Output:")
    print(slide_res)
