import os
import sys
import gradio as gr
from PIL import Image

# ---------------------------------------------------------
# Project root
# ---------------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from deployment.predict import Camelyon17Predictor


# ---------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------

CHECKPOINT_PATH = os.environ.get(
    "CAMELYON_CHECKPOINT",
    None
)


# ---------------------------------------------------------
# Predictor
# ---------------------------------------------------------

predictor = Camelyon17Predictor(
    checkpoint_path=CHECKPOINT_PATH,
    device="auto"
)


# ---------------------------------------------------------
# Patch prediction
# ---------------------------------------------------------

def predict_image(image):

    if image is None:
        return {
            "probability_tumor": 0.0,
            "probability_normal": 0.0,
            "prediction": "No image provided"
        }

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    image = image.convert("RGB")

    result = predictor.predict_patch(image)

    return (
    result["probability_tumor"],
    result["probability_normal"],
    result["prediction"]
)


# ---------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------

with gr.Blocks(
    title="Camelyon17 Medical Diagnosis"
) as demo:

    gr.Markdown(
        """
        # 🧬 Camelyon17 Tumor Metastasis Detection

        **Research prototype for federated cross-hospital
        histopathology diagnosis.**

        Upload a **96×96 RGB tissue patch** to obtain
        the model's predicted metastasis probability.
        """
    )

    with gr.Row():

        image_input = gr.Image(
            type="pil",
            label="Histopathology Tissue Patch"
        )

        with gr.Column():

            tumor_probability = gr.Number(
                label="Tumor Probability"
            )

            normal_probability = gr.Number(
                label="Normal Probability"
            )

            diagnosis = gr.Textbox(
                label="Diagnosis"
            )

    predict_button = gr.Button(
        "Analyze Tissue Patch",
        variant="primary"
    )

    predict_button.click(
        fn=predict_image,
        inputs=image_input,
        outputs=[
            tumor_probability,
            normal_probability,
            diagnosis
        ]
    )

    gr.Markdown(
        """
        ---
        **Important:** This is a research prototype and
        should not be used as a standalone clinical diagnostic
        system.
        """
    )


# ---------------------------------------------------------
# Start application
# ---------------------------------------------------------

if __name__ == "__main__":
    demo.launch()
