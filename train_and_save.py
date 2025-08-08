# train_and_save.py

import os
import pandas as pd
import numpy as np
import joblib
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import nltk

nltk.download('punkt_tab')

# Keras imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Ensure NLTK downloads
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Create model directory
os.makedirs("model", exist_ok=True)

# Label map
label_names = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

# 1. Load Dataset
dataset = load_dataset("ag_news")
df_train = pd.DataFrame(dataset["train"])
df_test = pd.DataFrame(dataset["test"])

df_train = df_train[["text", "label"]]
df_test = df_test[["text", "label"]]


# 2. Preprocessing function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text, language="english")
    processed = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(processed)

df_train['processed_text'] = df_train['text'].apply(preprocess_text)
df_test['processed_text'] = df_test['text'].apply(preprocess_text)

# 3. TF-IDF + Logistic Regression
tfidf = TfidfVectorizer(max_features=5000)
X_train_vec = tfidf.fit_transform(df_train['processed_text'])
X_test_vec = tfidf.transform(df_test['processed_text'])

logreg = LogisticRegression()
logreg.fit(X_train_vec, df_train['label'])

# Save model + vectorizer
joblib.dump(logreg, "model/logistic_regression_model.pkl")
joblib.dump(tfidf, "model/tfidf_vectorizer.pkl")
print(" Logistic Regression and TF-IDF saved.")

# 4. Tokenizer + NN
vocab_size = 10000
sequence_length = 100

tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
tokenizer.fit_on_texts(df_train['processed_text'])

X_train_seq = tokenizer.texts_to_sequences(df_train['processed_text'])
X_test_seq = tokenizer.texts_to_sequences(df_test['processed_text'])

X_train_pad = pad_sequences(X_train_seq, maxlen=sequence_length, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=sequence_length, padding='post')

y_train_nn = to_categorical(df_train['label'], num_classes=4)
y_test_nn = to_categorical(df_test['label'], num_classes=4)

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=200, input_length=sequence_length),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train_pad, y_train_nn, validation_split=0.2, epochs=5, batch_size=64, verbose=1)

# Save model + tokenizer
model.save("model/news_classification_nn.keras", include_optimizer=False, save_format='keras')
joblib.dump(tokenizer, "model/tokenizer.pkl")
print(" Neural network and tokenizer saved.")