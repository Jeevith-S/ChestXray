# ==========================================================
# CONFIGURATION
# ==========================================================

import os
import torch

# ==========================================================
# PROJECT PATHS
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================================
# MODEL FILES
# ==========================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "best_densenet121.pth"
)

THRESHOLD_PATH = os.path.join(
    BASE_DIR,
    "1densenet121_best_thresholds.csv"
)

# ==========================================================
# RAG FILES
# ==========================================================

FAISS_INDEX_PATH = os.path.join(
    BASE_DIR,
    "medical_index.faiss"
)

CHUNKS_PATH = os.path.join(
    BASE_DIR,
    "medical_chunks.pkl"
)

PDF_PATH = os.path.join(
    BASE_DIR,
    "Chest_Diseases_Medical_Reference.pdf"
)

# ==========================================================
# IMAGE SETTINGS
# ==========================================================

IMAGE_SIZE = 320

# ImageNet Normalization

MEAN = [0.485, 0.456, 0.406]

STD = [0.229, 0.224, 0.225]

# ==========================================================
# DEVICE
# ==========================================================

DEVICE = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else

    "cpu"

)

# ==========================================================
# DISEASE LABELS
# ==========================================================

DISEASES = [

    "Atelectasis",

    "Cardiomegaly",

    "Consolidation",

    "Edema",

    "Effusion",

    "Emphysema",

    "Fibrosis",

    "Infiltration",

    "Mass",

    "No Finding",

    "Nodule",

    "Pleural_Thickening",

    "Pneumonia",

    "Pneumothorax"

]

NUM_CLASSES = len(DISEASES)

# ==========================================================
# API SETTINGS
# ==========================================================

API_TITLE = "Chest X-ray Disease Detection API"

API_VERSION = "1.0.0"

# ==========================================================
# GEMINI MODEL
# ==========================================================

GEMINI_MODEL = "gemini-2.5-flash"

# ==========================================================
# PRINT CONFIGURATION
# ==========================================================

print("=" * 60)

print("CONFIGURATION LOADED")

print("=" * 60)

print(f"Device            : {DEVICE}")

print(f"Image Size        : {IMAGE_SIZE}")

print(f"Number of Classes : {NUM_CLASSES}")

print(f"Model Path        : {MODEL_PATH}")

print("=" * 60)