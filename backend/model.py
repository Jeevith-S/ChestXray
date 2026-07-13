# ==========================================================
# IMPORTS
# ==========================================================

import torch
import torch.nn as nn

from torchvision.models import (

    densenet121,

    DenseNet121_Weights

)

from config import (

    MODEL_PATH,

    DEVICE,

    NUM_CLASSES

)

# ==========================================================
# LOAD MODEL
# ==========================================================

def load_model():

    print()

    print("=" * 60)
    print("LOADING DENSENET121")
    print("=" * 60)

    # ------------------------------------------------------
    # CREATE MODEL
    # ------------------------------------------------------

    model = densenet121(

        weights=None

    )

    # ------------------------------------------------------
    # REPLACE CLASSIFIER
    # ------------------------------------------------------

    num_features = model.classifier.in_features

    model.classifier = nn.Linear(

    num_features,

    NUM_CLASSES

)

    # ------------------------------------------------------
    # LOAD TRAINED WEIGHTS
    # ------------------------------------------------------

    model.load_state_dict(

        torch.load(

            MODEL_PATH,

            map_location=DEVICE

        )

    )

    # ------------------------------------------------------
    # MOVE TO DEVICE
    # ------------------------------------------------------

    model.to(

        DEVICE

    )

    model.eval()

    print()

    print("✓ Best DenseNet121 Loaded")

    print("Device :", DEVICE)

    print()

    print("=" * 60)

    return model


# ==========================================================
# LOAD MODEL ONLY ONCE
# ==========================================================

model = load_model()