from fastapi import FastAPI, UploadFile, File
from PIL import Image
from dotenv import load_dotenv
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import io
import google.generativeai as genai
load_dotenv()

# ==========================================================
# GEMINI API
# ==========================================================

genai.configure(
    api_key=os.getenv(
        "GEMINI_API_KEY"
    )
)

llm = genai.GenerativeModel(
    "gemini-2.5-flash"
)

# ==========================================================
# FASTAPI APP
# ==========================================================

app = FastAPI(
    title="Chest X-ray AI API"
)

# ==========================================================
# DISEASE LABELS
# ==========================================================

disease_cols = [

    'Atelectasis',
    'Cardiomegaly',
    'Consolidation',
    'Edema',
    'Effusion',
    'Emphysema',
    'Fibrosis',
    'Hernia',
    'Infiltration',
    'Mass',
    'No Finding',
    'Nodule',
    'Pleural_Thickening',
    'Pneumonia',
    'Pneumothorax'
]

# ==========================================================
# DEVICE
# ==========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using Device:", device)

# ==========================================================
# LOAD MODEL
# ==========================================================

model = models.resnet18(
    weights=None
)

num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    len(disease_cols)
)

model.load_state_dict(
    torch.load(
        "best_resnet18.pth",
        map_location=device
    )
)

model = model.to(device)

model.eval()

print("✅ Model Loaded")

# ==========================================================
# IMAGE TRANSFORM
# ==========================================================

transform = transforms.Compose([

    transforms.Resize((224,224)),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[0.485,0.456,0.406],

        std=[0.229,0.224,0.225]
    )
])

# ==========================================================
# HOME ROUTE
# ==========================================================

@app.get("/")
def home():

    return {
        "message":
        "Chest X-ray API Running Successfully"
    }

# ==========================================================
# PREDICT ROUTE
# ==========================================================

@app.post("/predict")
async def predict(
        file: UploadFile = File(...)
):

    # Read uploaded image

    contents = await file.read()

    image = Image.open(
        io.BytesIO(contents)
    ).convert("RGB")

    # Preprocess

    img = transform(image)

    img = img.unsqueeze(0)

    img = img.to(device)

    # Prediction

    with torch.no_grad():

        outputs = model(img)

        probs = torch.sigmoid(
            outputs
        ).cpu().numpy()[0]

    threshold = 0.15

    predicted_diseases = []

    for i, disease in enumerate(
            disease_cols
    ):

        if probs[i] > threshold:

            predicted_diseases.append(
                disease
            )

    if len(predicted_diseases) == 0:

        predicted_diseases = [
            "No Disease Detected"
        ]

    probabilities = {}

    for i, disease in enumerate(
            disease_cols
    ):

        probabilities[disease] = round(
            float(probs[i]),
            4
        )

    # ======================================================
    # GEMINI EXPLANATION
    # ======================================================

    prompt = f"""
    You are an expert radiology assistant.

    Predicted diseases:

    {predicted_diseases}

    Explain in simple language:

    1. Disease overview
    2. Common symptoms
    3. Causes
    4. Treatments
    5. Prevention

    Mention clearly that this is not
    a final medical diagnosis.
    """

    response = llm.generate_content(
        prompt
    )

    explanation = response.text

    # ======================================================
    # RETURN JSON
    # ======================================================

    return {

        "predicted_diseases":
            predicted_diseases,

        "probabilities":
            probabilities,

        "explanation":
            explanation
    }