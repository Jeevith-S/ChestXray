# ==========================================================
# IMPORTS
# ==========================================================

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from PIL import Image

import io
import os
import base64

from io import BytesIO

from dotenv import load_dotenv

import google.generativeai as genai

# ==========================================================
# IMPORT YOUR MODULES
# ==========================================================

from config import (

    API_TITLE,

    GEMINI_MODEL

)

from preprocess import preprocess_image

from predict import predict_image

from gradcam import generate_gradcam

from rag import generate_explanation

# ==========================================================
# LOAD ENVIRONMENT
# ==========================================================

load_dotenv()

genai.configure(

    api_key=os.getenv(

        "GEMINI_API_KEY"

    )

)

llm = genai.GenerativeModel(

    GEMINI_MODEL

)

# ==========================================================
# FASTAPI
# ==========================================================

app = FastAPI(

    title=API_TITLE,

    version="1.0.0"

)

# ==========================================================
# HOME ROUTE
# ==========================================================

@app.get("/")

def home():

    return {

        "message": "Chest X-ray AI API Running Successfully",

        "model": "DenseNet121",

        "status": "Ready"

    }

# ==========================================================
# HEALTH CHECK
# ==========================================================

@app.get("/health")

def health():

    return {

        "status": "healthy"

    }
# ==========================================================
# PREDICT ROUTE
# ==========================================================

@app.post("/predict")

async def predict(

    file: UploadFile = File(...)

):

    try:

        # --------------------------------------------------
        # READ IMAGE
        # --------------------------------------------------

        contents = await file.read()

        image = Image.open(

            io.BytesIO(contents)

        ).convert(

            "RGB"

        )

        # --------------------------------------------------
        # PREPROCESS
        # --------------------------------------------------

        image_tensor = preprocess_image(

            image

        )

        # --------------------------------------------------
        # MODEL PREDICTION
        # --------------------------------------------------

        probabilities, predicted_diseases, thresholds = predict_image(

            image_tensor

        )

        # --------------------------------------------------
        # GENERATE MEDICAL EXPLANATION (RAG + GEMINI)
        # --------------------------------------------------

        explanation = generate_explanation(

            predicted_diseases

        )

        # --------------------------------------------------
        # GENERATE GRAD-CAM
        # --------------------------------------------------

        gradcam_image = generate_gradcam(

            image

        )

        # --------------------------------------------------
        # CONVERT TO BASE64
        # --------------------------------------------------

        buffer = BytesIO()

        gradcam_image.save(

            buffer,

            format="PNG"

        )

        gradcam_base64 = base64.b64encode(

            buffer.getvalue()

        ).decode()

        # --------------------------------------------------
        # SUCCESS RESPONSE
        # --------------------------------------------------

        return {

            "success": True,

            "predicted_diseases": predicted_diseases,

            "probabilities": probabilities,

            "thresholds": thresholds,

            "explanation": explanation,

            "gradcam": gradcam_base64

        }

    except Exception as e:

        return JSONResponse(

            status_code=500,

            content={

                "success": False,

                "error": str(e)

            }

        )
    # ==========================================================
# STARTUP EVENT
# ==========================================================

@app.on_event("startup")

async def startup_event():

    print()

    print("=" * 70)

    print("CHEST X-RAY AI API STARTED")

    print("=" * 70)

    print("✓ DenseNet121 Loaded")

    print("✓ Thresholds Loaded")

    print("✓ Grad-CAM Ready")

    print("✓ FAISS Ready")

    print("✓ Gemini Ready")

    print("✓ API Ready")

    print("=" * 70)

# ==========================================================
# SHUTDOWN EVENT
# ==========================================================

@app.on_event("shutdown")

async def shutdown_event():

    print()

    print("=" * 70)

    print("CHEST X-RAY AI API STOPPED")

    print("=" * 70)

# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(

        "api:app",

        host="0.0.0.0",

        port=8000,

        reload=True

    )