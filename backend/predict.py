# ==========================================================
# IMPORTS
# ==========================================================

import torch
import pandas as pd

from model import model

from config import (

    DEVICE,

    DISEASES,

    THRESHOLD_PATH

)

# ==========================================================
# LOAD THRESHOLDS
# ==========================================================

threshold_df = pd.read_csv(

    THRESHOLD_PATH

)

best_thresholds = {}

for _, row in threshold_df.iterrows():

    best_thresholds[

        row["Disease"]

    ] = float(

        row["Best Threshold"]

    )

print()

print("=" * 60)

print("THRESHOLDS LOADED")

print("=" * 60)

for disease, threshold in best_thresholds.items():

    print(f"{disease:<22} : {threshold:.2f}")

print("=" * 60)

# ==========================================================
# PREDICTION FUNCTION
# ==========================================================

def predict_image(image_tensor):

    """
    Input
    -----
    image_tensor

    Output
    ------
    probabilities

    predicted diseases

    thresholds
    """

    with torch.no_grad():

        outputs = model(

            image_tensor

        )

        probs = torch.sigmoid(

            outputs

        ).cpu().numpy()[0]

    probabilities = {}

    predicted_diseases = []

    used_thresholds = {}

    # ------------------------------------------------------
    # APPLY PER-DISEASE THRESHOLDS
    # ------------------------------------------------------

    for i, disease in enumerate(DISEASES):

        probability = float(

            probs[i]

        )

        threshold = float(

            best_thresholds[disease]

        )

        probabilities[disease] = round(

            probability,

            4

        )

        used_thresholds[disease] = threshold

        if probability >= threshold:

            predicted_diseases.append({

                "disease": disease,

                "probability": round(

                    probability,

                    4

                ),

                "threshold": threshold

            })

    # ------------------------------------------------------
    # SORT BY PROBABILITY
    # ------------------------------------------------------

    predicted_diseases = sorted(

        predicted_diseases,

        key=lambda x: x["probability"],

        reverse=True

    )

    # ------------------------------------------------------
    # NO DISEASE FOUND
    # ------------------------------------------------------

    if len(predicted_diseases) == 0:

        predicted_diseases.append({

            "disease": "No Disease Detected",

            "probability": 0.0,

            "threshold": 0.0

        })

    return (

        probabilities,

        predicted_diseases,

        used_thresholds

    )