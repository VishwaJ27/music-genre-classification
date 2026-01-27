import streamlit as st
import joblib
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

from feature_extraction import extract_features

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(
    page_title="Music Genre Classification",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------------------- GLOBAL CSS (THEME + ANIMATION) ----------------------
st.markdown("""
<style>
:root {
    --accent-color: #6D28D9;
}

.fade-in {
    animation: fadeIn 1.2s ease-in;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
}

.gradient-text {
    color: var(--accent-color);
    font-weight: 700;
}

.result-card {
    background-color: #f8f9fa;
    color: var(--accent-color);
    padding: 22px;
    border-radius: 8px;
    text-align: center;
    font-size: 22px;
    font-weight: 600;
    letter-spacing: 1px;
    margin-top: 10px;
    border: 1px solid rgba(109, 40, 217, 0.2);
}

/* Button styling */
.stButton > button {
    background-color: var(--accent-color);
    color: white;
    border-radius: 6px;
    border: none;
    padding: 0.6rem 1rem;
    font-weight: 500;
}

.stButton > button:hover {
    background-color: #5b21b6;
}

/* Section headings */
h3 {
    color: var(--accent-color);
}
</style>
""", unsafe_allow_html=True)

# ---------------------- LOAD MODEL ----------------------
@st.cache_resource
def load_assets():
    model = joblib.load("music_genre_model.pkl")
    encoder = joblib.load("label_encoder.pkl")
    return model, encoder

if not (os.path.exists("music_genre_model.pkl") and os.path.exists("label_encoder.pkl")):
    st.error("Model files not found. Please train the model first.")
    st.stop()

model, label_encoder = load_assets()

# ---------------------- HEADER ----------------------
st.markdown(
    """
    <div class="fade-in">
        <h2 class="gradient-text" style="text-align:center;">
            Music Genre Classification
        </h2>
        <p style="text-align:center; color:#6c757d;">
            AI-based music genre recognition using machine learning
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ---------------------- FILE UPLOAD ----------------------
uploaded_file = st.file_uploader(
    "Upload WAV Audio File",
    type=["wav"]
)

if uploaded_file:
    st.audio(uploaded_file)

    st.markdown("---")

    if st.button("Analyze Audio", use_container_width=True):

        with st.spinner("Analyzing audio..."):

            temp_file = f"temp_{uploaded_file.name}"
            with open(temp_file, "wb") as f:
                f.write(uploaded_file.getbuffer())

            features = extract_features(temp_file)
            os.remove(temp_file)

            if features is None:
                st.error("Unable to process this audio file.")
                st.stop()

            features = features.reshape(1, -1)

            prediction = model.predict(features)
            probabilities = model.predict_proba(features)[0]

            predicted_genre = label_encoder.inverse_transform(prediction)[0]

        # ---------------------- RESULT ----------------------
        st.markdown("### Prediction Result")

        st.markdown(
            f"""
            <div class="result-card fade-in">
                {predicted_genre.upper()}
            </div>
            """,
            unsafe_allow_html=True
        )

        # ---------------------- PROBABILITY CHART ----------------------
        st.markdown("### Genre Probability Distribution")

        prob_df = pd.DataFrame({
            "Genre": label_encoder.classes_,
            "Probability (%)": probabilities * 100
        }).sort_values(by="Probability (%)", ascending=False)

        st.markdown("<div class='fade-in'>", unsafe_allow_html=True)

        fig, ax = plt.subplots()
        ax.barh(prob_df["Genre"], prob_df["Probability (%)"])
        ax.invert_yaxis()
        ax.set_xlabel("Probability (%)")
        ax.set_ylabel("Genre")

        for bar in ax.patches:
            bar.set_color("#6D28D9")

        st.pyplot(fig)

        st.markdown("</div>", unsafe_allow_html=True)

        # ---------------------- EXPLANATION ----------------------
        with st.expander("How this prediction works"):
            st.write("""
            - Audio features are extracted from the waveform
            - Spectral and temporal characteristics are analyzed
            - A trained machine learning classifier predicts genre probabilities
            - The genre with the highest probability is selected
            """)

else:
    # ---------------------- INSTRUCTIONS ----------------------
    st.markdown(
        """
        ### How to Use
        - Upload a WAV audio file
        - Click **Analyze Audio**
        - View the predicted genre and probability distribution

        ### Notes
        - WAV format recommended
        - Audio duration ≥ 10 seconds
        - Full tracks perform better than short clips
        """
    )

st.markdown("---")
st.caption("Music Genre Classification System | Machine Learning Application")
