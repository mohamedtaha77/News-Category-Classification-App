# train_and_save.py — trains both pipelines and saves to /model
import os
import re
import joblib
import numpy as np
import pandas as pd
from datasets import load_dataset

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)   # remove punctuation
    text = re.sub(r"\d+", " ", text)       # remove numbers
    return " ".join(text.split())

def main():
    # 1) Load AG News (Hugging Face)
    ds = load_dataset("ag_news")
    train_df = pd.DataFrame(ds["train"])[["text", "label"]]
    test_df  = pd.DataFrame(ds["test"])[["text", "label"]]

    train_df["proc"] = train_df["text"].apply(preprocess_text)
    test_df["proc"]  = test_df["text"].apply(preprocess_text)

    # 2) TF‑IDF + Logistic Regression
    tfidf = TfidfVectorizer(max_features=5000)
    X_train = tfidf.fit_transform(train_df["proc"])
    X_test  = tfidf.transform(test_df["proc"])

    logreg = LogisticRegression(max_iter=1000, n_jobs=None)
    logreg.fit(X_train, train_df["label"])

    y_pred = logreg.predict(X_test)
    print("\n[TF‑IDF + LogReg] accuracy:", accuracy_score(test_df["label"], y_pred))
    print(classification_report(test_df["label"], y_pred, digits=4))

    joblib.dump(logreg, f"{MODEL_DIR}/logistic_regression_model.pkl")
    joblib.dump(tfidf,  f"{MODEL_DIR}/tfidf_vectorizer.pkl")

    # 3) Tokenizer + Simple NN
    vocab_size = 10000
    max_len = 100

    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_df["proc"])

    X_train_seq = tokenizer.texts_to_sequences(train_df["proc"])
    X_test_seq  = tokenizer.texts_to_sequences(test_df["proc"])

    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding="post")
    X_test_pad  = pad_sequences(X_test_seq,  maxlen=max_len, padding="post")

    y_train = tf.keras.utils.to_categorical(train_df["label"], num_classes=4)
    y_test  = tf.keras.utils.to_categorical(test_df["label"],  num_classes=4)

    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=128, input_length=max_len),
        GlobalAveragePooling1D(),
        Dense(128, activation="relu"),
        Dropout(0.4),
        Dense(4, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train_pad, y_train, validation_split=0.1, epochs=5, batch_size=64, verbose=1)

    loss, acc = model.evaluate(X_test_pad, y_test, verbose=0)
    print(f"\n[NN] accuracy: {acc:.4f}")

    # Save NN and tokenizer (zip-based .keras format)
    model.save(f"{MODEL_DIR}/nn_model.keras", include_optimizer=False, save_format="keras")
    joblib.dump(tokenizer, f"{MODEL_DIR}/tokenizer.pkl")

    print("\nSaved:")
    print(f"  - {MODEL_DIR}/logistic_regression_model.pkl")
    print(f"  - {MODEL_DIR}/tfidf_vectorizer.pkl")
    print(f"  - {MODEL_DIR}/nn_model.keras")
    print(f"  - {MODEL_DIR}/tokenizer.pkl")

if __name__ == "__main__":
    main()
