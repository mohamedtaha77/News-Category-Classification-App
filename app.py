import streamlit as st
import joblib
import numpy as np
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =======================
# Load Models & Tools
# =======================

logreg_model = joblib.load("model/logistic_regression_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

nn_model = load_model("model/nn_model.keras")
tokenizer = joblib.load("model/tokenizer.pkl")

label_names = {0: "🌍 World", 1: "🏅 Sports", 2: "💼 Business", 3: "🧪 Sci/Tech"}

# =======================
# Preprocessing (for TF-IDF model)
# =======================

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)   # Remove punctuation
    text = re.sub(r"\d+", "", text)       # Remove numbers
    tokens = text.split()                 # Simple whitespace tokenizer
    return " ".join(tokens)

# =======================
# Streamlit UI
# =======================

st.set_page_config(page_title="News Classifier", page_icon="🗞️", layout="centered")

st.title("🗞️ AG News Category Classifier")

st.markdown(
    """
    Enter a **news article** or **headline**,  
    choose a model, and predict its category instantly 🚀  

    **Possible Categories:**
    - 🌍 **World** — international affairs, politics, global news  
    - 🏅 **Sports** — events, teams, scores, tournaments  
    - 💼 **Business** — markets, companies, stocks, economy  
    - 🧪 **Sci/Tech** — science, innovations, gadgets, AI  
    """
)


# Text input
user_input = st.text_area("✍️ Your news text:", height=150, placeholder="E.g. Global markets saw a sharp decline after...")

# Model selector
model_choice = st.radio("🤖 Choose a model:", ["Logistic Regression (TF-IDF)", "Neural Network (Tokenizer + Embedding)"])

# Predict button
if st.button("🔍 Classify"):
    if not user_input.strip():
        st.warning("Please enter some text before classifying.")
    else:
        if model_choice == "Logistic Regression (TF-IDF)":
            cleaned_text = preprocess_text(user_input)
            features = vectorizer.transform([cleaned_text])
            prediction = logreg_model.predict(features)[0]
            probabilities = logreg_model.predict_proba(features)[0]
            confidence_pct = round(probabilities[prediction] * 100, 2)

        else:  # Neural Network
            sequence = tokenizer.texts_to_sequences([user_input])
            padded = pad_sequences(sequence, maxlen=100, padding='post')
            prediction_probs = nn_model.predict(padded)[0]
            prediction = np.argmax(prediction_probs)
            confidence_pct = round(prediction_probs[prediction] * 100, 2)

        # Output
        st.markdown("---")
        st.subheader("📊 Prediction Result")
        st.markdown(f"**Category:** {label_names[prediction]}")
        st.markdown(f"**Confidence:** `{confidence_pct}%`")
        st.markdown("---")
        st.caption("Confidence shows how sure the model is about its prediction.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center;'>Made with ❤️ for Elevvo Internship Task 2</div>",
    unsafe_allow_html=True
)
