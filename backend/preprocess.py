# ==========================================================
# IMPORTS
# ==========================================================

from PIL import Image

import torch

from torchvision import transforms

from config import (

    IMAGE_SIZE,

    MEAN,

    STD,

    DEVICE

)

# ==========================================================
# IMAGE TRANSFORM
# ==========================================================

transform = transforms.Compose([

    transforms.Resize(

        (

            IMAGE_SIZE,

            IMAGE_SIZE

        )

    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=MEAN,

        std=STD

    )

])

# ==========================================================
# PREPROCESS IMAGE
# ==========================================================

def preprocess_image(image):

    """
    Input:
        PIL Image

    Output:
        Tensor
        Shape:
        (1, 3, IMAGE_SIZE, IMAGE_SIZE)
    """

    if not isinstance(

        image,

        Image.Image

    ):

        raise TypeError(

            "Input must be a PIL Image."

        )

    image = image.convert(

        "RGB"

    )

    image = transform(

        image

    )

    image = image.unsqueeze(

        0

    )

    image = image.to(

        DEVICE

    )

    return image

# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    print()

    print("=" * 60)

    print("PREPROCESS MODULE READY")

    print("=" * 60)

    print(f"Image Size : {IMAGE_SIZE}")

    print(f"Device     : {DEVICE}")

    print("=" * 60)