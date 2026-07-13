# ==========================================================
# IMPORTS
# ==========================================================

import numpy as np
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from model import model
from preprocess import preprocess_image

# ==========================================================
# TARGET LAYER
# ==========================================================

target_layers = [

    model.features[-1]

]

cam = GradCAM(

    model=model,

    target_layers=target_layers

)

# ==========================================================
# GENERATE GRAD-CAM
# ==========================================================

import cv2

def generate_gradcam(image):

    """
    Input:
        PIL Image

    Output:
        PIL Image (Grad-CAM)
    """

    # --------------------------------------------------
    # ORIGINAL IMAGE
    # --------------------------------------------------

    rgb_image = image.convert("RGB")

    rgb_image = rgb_image.resize(

        (320, 320)

    )

    original = np.array(

        rgb_image

    ).astype(

        np.float32

    ) / 255.0

    # --------------------------------------------------
    # MODEL INPUT
    # --------------------------------------------------

    input_tensor = preprocess_image(

        rgb_image

    )

    # --------------------------------------------------
    # GENERATE CAM
    # --------------------------------------------------

    grayscale_cam = cam(

        input_tensor=input_tensor

    )[0]

    # --------------------------------------------------
    # RESIZE CAM
    # --------------------------------------------------

    grayscale_cam = cv2.resize(

        grayscale_cam,

        (

            original.shape[1],

            original.shape[0]

        )

    )

    # --------------------------------------------------
    # OVERLAY
    # --------------------------------------------------

    visualization = show_cam_on_image(

        original,

        grayscale_cam,

        use_rgb=True

    )

    # --------------------------------------------------
    # RETURN IMAGE
    # --------------------------------------------------

    return Image.fromarray(

        visualization

    )