# ==========================================================
# IMPORTS
# ==========================================================

import os
import pickle
import faiss
import numpy as np

from dotenv import load_dotenv
import google.generativeai as genai

from sentence_transformers import SentenceTransformer

# ==========================================================
# LOAD GEMINI
# ==========================================================

load_dotenv()

genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

llm = genai.GenerativeModel(
    "gemini-2.5-flash"
)

# ==========================================================
# LOAD EMBEDDING MODEL
# ==========================================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding Model Loaded")

# ==========================================================
# LOAD FAISS INDEX
# ==========================================================

index = faiss.read_index(
    "medical_index.faiss"
)

print("FAISS Loaded")

# ==========================================================
# LOAD CHUNKS
# ==========================================================

with open(
    "medical_chunks.pkl",
    "rb"
) as f:

    chunks = pickle.load(f)

print("Chunks Loaded")

# ==========================================================
# GENERATE EXPLANATION
# ==========================================================
# ==========================================================
# HANDLE NO FINDING
# ==========================================================


def generate_explanation(predicted_diseases):
    if len(predicted_diseases) == 1 and predicted_diseases[0]["disease"] == "No Finding":

        return """
        No abnormal chest disease was detected by the AI model.

        This means the chest X-ray appears normal based on the model's prediction.

        However, this is only an AI-assisted result and should not replace a professional radiologist's diagnosis.

        If the patient has symptoms such as chest pain, fever, cough, or difficulty breathing, consult a qualified healthcare professional.
        """

    # ------------------------------------------------------
    # HANDLE EMPTY PREDICTIONS
    # ------------------------------------------------------

    if len(predicted_diseases) == 0:

        return "No disease detected."

    # ------------------------------------------------------
    # EXTRACT DISEASE NAMES
    # ------------------------------------------------------

    disease_names = []

    for item in predicted_diseases:

        if isinstance(item, dict):

            disease_names.append(
                item["disease"]
            )

        else:

            disease_names.append(
                str(item)
            )

    # ------------------------------------------------------
    # CREATE QUERY
    # ------------------------------------------------------

    query = (

        "Chest X-ray diseases: "

        + ", ".join(disease_names)

    )

    # ------------------------------------------------------
    # QUERY EMBEDDING
    # ------------------------------------------------------

    query_embedding = embedding_model.encode(

        [query]

    ).astype(

        np.float32

    )

    # ------------------------------------------------------
    # SEARCH FAISS
    # ------------------------------------------------------

    distances, indices = index.search(

        query_embedding,

        3

    )

    # ------------------------------------------------------
    # BUILD CONTEXT
    # ------------------------------------------------------

    context = ""

    for idx in indices[0]:

        context += chunks[idx]

        context += "\n\n"

    # ------------------------------------------------------
    # FORMAT PREDICTIONS
    # ------------------------------------------------------

    disease_text = ""

    for item in predicted_diseases:

        if isinstance(item, dict):

            disease_text += (

                f"- {item['disease']}\n"

                f"  Probability : {item['probability']:.4f}\n"

                f"  Threshold   : {item['threshold']:.2f}\n\n"

            )

        else:

            disease_text += f"- {item}\n\n"

    # ------------------------------------------------------
    # PROMPT
    # ------------------------------------------------------

    prompt = f"""
You are an expert radiologist.

Use ONLY the medical reference below.

==========================
MEDICAL REFERENCE
==========================

{context}

==========================
MODEL PREDICTIONS
==========================

{disease_text}

Explain in simple language.

Include:

1. Disease Overview

2. Symptoms

3. Causes

4. Treatment

5. Prevention

6. When should the patient consult a doctor?

At the end write:

"This AI prediction is only an assistive tool and not a final medical diagnosis."
"""

    # ------------------------------------------------------
    # GEMINI RESPONSE
    # ------------------------------------------------------

    try:

        response = llm.generate_content(

            prompt

        )

        return response.text

    except Exception as e:

        return f"Gemini Error: {str(e)}"