# app.py — minimal, stable Streamlit app

from pathlib import Path
import re
import sys
import types
import joblib
import numpy as np
import streamlit as st

# ✅ Must be the first Streamlit call
st.set_page_config(page_title="News Classifier", page_icon="🗞️", layout="centered")

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ------------------
# Load classic TF-IDF pipeline
# ------------------
logreg_model = joblib.load("model/logistic_regression_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

# ------------------
# Load NN (.keras only)
# ------------------
KERAS_PATH = Path("model/nn_model.keras")

@st.cache_resource(show_spinner=False)
def load_nn():
    if not KERAS_PATH.exists():
        raise FileNotFoundError(f"Missing neural net at {KERAS_PATH}")
    # compile=False → inference only
    return load_model(KERAS_PATH, compile=False)

nn_model = load_nn()

# ------------------
# Tokenizer (pickle) compat shim, in case it was created
# with keras.preprocessing.text.Tokenizer on older Keras
# ------------------
try:
    import keras_preprocessing.text as kp_text
    sys.modules["keras.preprocessing"] = types.ModuleType("keras.preprocessing")
    shim = types.ModuleType("keras.preprocessing.text")
    shim.Tokenizer = kp_text.Tokenizer
    sys.modules["keras.preprocessing.text"] = shim
except Exception:
    pass

tokenizer = joblib.load("model/tokenizer.pkl")

# ------------------
# Labels + preprocessing
# ------------------
label_names = {
    0: "🌍 World",
    1: "🏅 Sports",
    2: "💼 Business",
    3: "🧪 Sci/Tech",
}

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)   # remove punctuation
    text = re.sub(r"\d+", " ", text)       # remove numbers
    return " ".join(text.split())

# ------------------
# UI
# ------------------
st.title("🗞️ AG News Category Classifier")
st.markdown(
    """
Enter a **news article** or **headline**, choose a model, and predict its category 🚀

**Categories**
- 🌍 World  
- 🏅 Sports  
- 💼 Business  
- 🧪 Sci/Tech  
"""
)

user_input = st.text_area(
    "✍️ Your news text:",
    height=150,
    placeholder="E.g. Global markets saw a sharp decline after..."
)

model_choice = st.radio(
    "🤖 Choose a model:",
    ["Logistic Regression (TF-IDF)", "Neural Network (Tokenizer + Embedding)"]
)

if st.button("🔍 Classify"):
    if not user_input.strip():
        st.warning("Please enter some text before classifying.")
    else:
        if model_choice == "Logistic Regression (TF-IDF)":
            cleaned = preprocess_text(user_input)
            feats = vectorizer.transform([cleaned])
            pred = int(logreg_model.predict(feats)[0])
            probs = logreg_model.predict_proba(feats)[0]
            conf = round(float(probs[pred]) * 100, 2)
        else:
            seq = tokenizer.texts_to_sequences([user_input])
            padded = pad_sequences(seq, maxlen=100, padding="post")
            probs = nn_model.predict(padded, verbose=0)[0]
            pred = int(np.argmax(probs))
            conf = round(float(probs[pred]) * 100, 2)

        st.markdown("---")
        st.subheader("📊 Prediction Result")
        st.markdown(f"**Category:** {label_names[pred]}")
        st.markdown(f"**Confidence:** `{conf}%`")
        st.caption("Confidence shows how sure the model is about its prediction.")
        st.markdown("---")

st.markdown(
    "<div style='text-align: center;'>Made with ❤️ for Elevvo Internship Task 2</div>",
    unsafe_allow_html=True,
)
