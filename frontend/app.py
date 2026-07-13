import streamlit as st
import requests
import pandas as pd

from PIL import Image

from io import BytesIO
import base64

import os

from datetime import datetime
from config import API_URL
st.set_page_config(

    page_title="AI Chest X-ray Assistant",

    page_icon="🩺",

    layout="wide"

)
st.title(

    "🩺 AI Chest X-ray Disease Detection"

)

st.markdown("""

### Deep Learning + FastAPI + RAG

✅ DenseNet121

✅ Grad-CAM

✅ Per Disease Thresholds

✅ Gemini AI

✅ Medical RAG

""")
with st.sidebar:

    st.header("Project")

    st.write("""

Frontend : Streamlit

Backend : FastAPI

Model : DenseNet121

LLM : Gemini

RAG : FAISS

""")

    st.info("Upload a Chest X-ray image.")

# ============================================================================
# FILE UPLOADER - MAIN PAGE
# ============================================================================
uploaded_file = st.file_uploader(
    "📤 Upload Chest X-ray",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    
    # Create centered layout with image on left and results on center-right
    col_empty1, col_image, col_spacer, col_results, col_empty2 = st.columns([0.1, 1.2, 0.2, 1.5, 0.1])

    with col_image:
        st.image(image, use_container_width=True, caption="Uploaded X-ray Image")

    # ==========================================================
    # SEND IMAGE TO API
    # ==========================================================
    with st.spinner("🔍 Analyzing Chest X-ray..."):
        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type
            )
        }

        try:
            response = requests.post(
                f"{API_URL}/predict",
                files=files,
                timeout=120
            )
            response.raise_for_status()
            result = response.json()
        except Exception as e:
            st.error(f"Cannot connect to FastAPI.\n\n{e}")
            st.stop()

    # ==========================================================
    # CHECK API RESPONSE
    # ==========================================================
    if result.get("success") is False:
        st.error(result.get("error", "Prediction Failed."))
        st.stop()

    # ==========================================================
    # GET RESPONSE
    # ==========================================================
    predicted_diseases = result.get("predicted_diseases", [])
    probabilities = result.get("probabilities", {})
    thresholds = result.get("thresholds", {})
    explanation = result.get("explanation", "")

    # ==========================================================
    # PROBABILITY TABLE
    # ==========================================================
    prob_df = pd.DataFrame({
        "Disease": list(probabilities.keys()),
        "Probability (%)": [round(p * 100, 2) for p in probabilities.values()],
        "Threshold": [thresholds.get(d, "-") for d in probabilities.keys()]
    })
    prob_df = prob_df.sort_values("Probability (%)", ascending=False)

    with col_results:
        st.subheader("📊 Disease Probabilities")
        st.dataframe(prob_df, use_container_width=True)
        
        # ==========================================================
        # PREDICTED DISEASES
        # ==========================================================
        st.subheader("🩺 Predicted Diseases")
        for item in predicted_diseases:
            st.success(f"""
Disease: {item["disease"]}
Probability: {item["probability"]:.4f}
Threshold: {item["threshold"]:.2f}
""")

    # ==========================================================
    # AI EXPLANATION - CENTERED BELOW
    # ==========================================================
    col_exp_empty1, col_exp, col_exp_empty2 = st.columns([0.15, 1.7, 0.15])
    with col_exp:
        st.subheader("🤖 AI Medical Explanation")
        st.write(explanation)

    # ==========================================================
    # GRADCAM - CENTERED BELOW
    # ==========================================================
    if "gradcam" in result:
        col_grad_empty1, col_grad, col_grad_empty2 = st.columns([0.15, 1.7, 0.15])
        with col_grad:
            st.subheader("🔥 Grad-CAM")
            gradcam = Image.open(BytesIO(base64.b64decode(result["gradcam"])))
            st.image(gradcam, use_container_width=True)

    # ==========================================================
    # SAVE PREDICTION HISTORY
    # ==========================================================
    history = pd.DataFrame({
        "Timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "Image": [uploaded_file.name],
        "Prediction": [", ".join([item["disease"] for item in predicted_diseases])]
    })

    if os.path.exists("prediction_history.csv"):
        old = pd.read_csv("prediction_history.csv")
        history = pd.concat([old, history], ignore_index=True)

    history.to_csv("prediction_history.csv", index=False)

    # ==========================================================
    # DOWNLOAD REPORT - CENTERED
    # ==========================================================
    col_btn_empty1, col_btn, col_btn_empty2 = st.columns([0.3, 1.4, 0.3])
    with col_btn:
        csv = prob_df.to_csv(index=False)
        st.download_button(
            label="⬇ Download Prediction Report",
            data=csv,
            file_name="prediction_report.csv",
            mime="text/csv",
            key="download_report",
            use_container_width=True
        )


# ==========================================================
# DISCLAIMER
# ==========================================================
st.warning("""
⚠️ Disclaimer

This AI system is intended for educational and assistive purposes only.

It should not be considered a final medical diagnosis.

Always consult a qualified radiologist or healthcare professional.

""")

# ==========================================================
# FOOTER
# ==========================================================
st.markdown("---")
st.markdown("""
Developed using

- Streamlit
- FastAPI
- DenseNet121
- Grad-CAM
- FAISS
- Gemini AI
""")