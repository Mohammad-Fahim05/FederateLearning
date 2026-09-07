import sys
import os
import torch
import numpy as np
from PIL import Image

# Add project root directory to Python path
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.resnet_backbone import ResNet18Backbone
from src.data.preprocessing import get_transforms
from src.evaluation.slide_evaluator import SlideEvaluator


class Camelyon17Predictor:
    """
    Research Deployment Prototype for Cross-Hospital Histopathology
    Metastasis Diagnosis.

    Loads trained DP-WHFedDG model weights and performs:
    1. Patch-level inference
    2. Slide-level inference
    """

    def __init__(self, checkpoint_path=None, device=None):

        # ---------------------------------------------------------
        # 1. Select device
        # ---------------------------------------------------------
        if device is None or device == 'auto':
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'
            )
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # ---------------------------------------------------------
        # 2. Load inference preprocessing
        # ---------------------------------------------------------
        self.transform = get_transforms(is_train=False)

        # Slide evaluator
        self.slide_evaluator = SlideEvaluator()

        # ---------------------------------------------------------
        # 3. Create model architecture
        # ---------------------------------------------------------
        self.model = ResNet18Backbone(
            num_classes=2,
            pretrained=False
        ).to(self.device)

        # ---------------------------------------------------------
        # 4. Load trained checkpoint
        # ---------------------------------------------------------
        if checkpoint_path and os.path.exists(checkpoint_path):

            print(
                f"Loading trained model weights from "
                f"{checkpoint_path}..."
            )

            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.device
            )

            # Research checkpoints created by
            # run_single_experiment.py contain:
            #
            # {
            #     'model_state_dict': ...,
            #     'config': ...,
            #     'history': ...,
            #     ...
            # }
            #
            # Plain state_dict checkpoints are also supported.

            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            self.model.load_state_dict(state_dict)

            print("Trained model weights loaded successfully.")

        else:

            print(
                "WARNING: No valid checkpoint was provided. "
                "The model is using initialized weights."
            )

        # ---------------------------------------------------------
        # 5. Evaluation mode
        # ---------------------------------------------------------
        self.model.eval()

    # =============================================================
    # PATCH-LEVEL PREDICTION
    # =============================================================

    def predict_patch(self, pil_image):
        """
        Predict tumor metastasis probability for one
        96x96 RGB Camelyon17 tissue patch.

        Parameters
        ----------
        pil_image : PIL.Image
            RGB tissue image.

        Returns
        -------
        dict
            Tumor probability, normal probability,
            and predicted diagnosis.
        """

        # Ensure RGB
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        # Apply inference preprocessing
        tensor_img = self.transform(
            pil_image
        ).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():

            outputs = self.model(tensor_img)

            probs = torch.softmax(
                outputs,
                dim=1
            ).cpu().numpy()[0]

        # Class 1 = tumor metastasis
        tumor_prob = float(probs[1])

        normal_prob = float(
            1.0 - tumor_prob
        )

        # Classification threshold
        if tumor_prob >= 0.5:
            prediction = "Tumor Metastasis Detected"
        else:
            prediction = "Normal Tissue"

        return {
            "probability_tumor": tumor_prob,
            "probability_normal": normal_prob,
            "prediction": prediction
        }

    # =============================================================
    # SLIDE-LEVEL PREDICTION
    # =============================================================

    def predict_wsi_slide(self, patch_list):
        """
        Predict slide-level metastasis risk from a list
        of WSI tissue patches.

        The current research aggregation uses the mean
        probability of the top 5 highest-risk patches.
        """

        if not patch_list:
            raise ValueError(
                "patch_list must contain at least one image."
            )

        patch_probs = []

        # Predict every patch
        for patch in patch_list:

            result = self.predict_patch(patch)

            patch_probs.append(
                result["probability_tumor"]
            )

        # Sort highest-risk patches first
        sorted_probs = sorted(
            patch_probs,
            reverse=True
        )

        # Top-5 mean aggregation
        top_k = min(
            5,
            len(sorted_probs)
        )

        top5_mean = float(
            np.mean(sorted_probs[:top_k])
        )

        # Slide-level decision
        if top5_mean >= 0.5:
            slide_prediction = "METASTASIS POSITIVE WSI"
        else:
            slide_prediction = "NORMAL WSI"

        return {
            "slide_metastasis_risk_score": top5_mean,
            "slide_diagnosis": slide_prediction,
            "num_patches_analyzed": len(patch_list),
            "top_patch_probabilities": sorted_probs[:5]
        }


# ================================================================
# BASIC LOCAL TEST
# ================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("CAMELYON17 RESEARCH DEPLOYMENT INFERENCE TEST")
    print("=" * 60)

    # ------------------------------------------------------------
    # IMPORTANT:
    # Put the actual trained checkpoint path here when available.
    #
    # Example:
    #
    # checkpoint_path = "../checkpoints/"
    #     "controlled_run_DP-WHFedDG_rounds10_epochs1_"
    #     "seed42_eps3.0.pt"
    #
    # Currently None means the test uses randomly initialized
    # model weights.
    # ------------------------------------------------------------

    checkpoint_path = None

    predictor = Camelyon17Predictor(
        checkpoint_path=checkpoint_path,
        device="auto"
    )

    # ------------------------------------------------------------
    # Create a synthetic 96x96 RGB patch for code testing
    # ------------------------------------------------------------

    synthetic_patch = Image.fromarray(
        np.random.randint(
            0,
            255,
            (96, 96, 3),
            dtype=np.uint8
        )
    )

    # ------------------------------------------------------------
    # Patch inference test
    # ------------------------------------------------------------

    patch_result = predictor.predict_patch(
        synthetic_patch
    )

    print("\nSingle Patch Inference Output:")
    print(patch_result)

    # ------------------------------------------------------------
    # Slide inference test
    # ------------------------------------------------------------

    slide_result = predictor.predict_wsi_slide(
        [synthetic_patch] * 10
    )

    print("\nSlide-Level WSI Inference Output:")
    print(slide_result)

    print("\n" + "=" * 60)
    print("DEPLOYMENT INFERENCE TEST COMPLETED")
    print("=" * 60)
