
# 🗞️ News Category Classification Web App - AG News

This is a clean and interactive **News Classifier App** built with **Streamlit**, based on the **AG News dataset**.  
It classifies short news articles or headlines into one of **four categories**:
- 🌍 **World**
- 🏅 **Sports**
- 💼 **Business**
- 🧪 **Sci/Tech**

Two models are implemented and switchable within the UI:
- **Logistic Regression (TF-IDF based)**
- **Neural Network (Tokenizer + Embedding Layer)**

---

## 📌 Features

✅ Clean & lemmatize text input  
✅ TF-IDF and Tokenizer-based pipelines  
✅ Supports two distinct model architectures  
✅ Displays predicted **category** and **confidence level**  
✅ User-friendly interface built in **Streamlit**  
✅ Includes notebook for full training & analysis

---

## 🔗 Live Demo

Try the app live here:  
👉 [https://news-category-classification-77.streamlit.app/]

---

## 📁 Files Included

| File | Description |
|------|-------------|
| `app.py` | Streamlit UI for classification |
| `train_and_save.py` | Trains and saves both models |
| `Elevvo_NLP_Internship_Task2.ipynb` | Full notebook for experimentation |
| `model/logistic_regression_model.pkl` | TF-IDF + Logistic Regression |
| `model/tfidf_vectorizer.pkl` | Fitted TF-IDF vectorizer |
| `model/news_classification_nn.keras` | Neural Network model |
| `model/tokenizer.pkl` | Keras tokenizer |
| `requirements.txt` | All dependencies |

---

## 🚀 How to Run Locally

1. **Clone the repo**:

```bash
git clone https://github.com/mohamedtaha77/News-Category-Classification-App.git
cd News-Category-Classification-App
```

2. **(Optional) Create a virtual environment**:

```bash
python -m venv venv
venv\Scripts\activate  # on Windows
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

4. **Train and save the models**:

```bash
python train_and_save.py
```

5. **Run the app**:

```bash
streamlit run app.py
```

---

## 🌍 Deployment (Optional)

To deploy to **Streamlit Cloud**:

- Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
- Connect your GitHub repo
- Set the main file to: `app.py`

---

## 📓 Notebook Workflow

All preprocessing, training, evaluation, and visualization steps are included in the notebook:

> [`Elevvo_NLP_Internship_Task2.ipynb`](./Elevvo_NLP_Internship_Task2.ipynb)

View it in Colab:  
👉 [https://colab.research.google.com/drive/1TR-0EZUNryriSE7_akl2Vbsx8hXWEnZN?usp=sharing]

Covers:
- Dataset loading (AG News from HuggingFace)
- Text preprocessing (NLTK)
- TF-IDF vectorization and Keras Tokenization
- Training both models
- WordClouds + bar plots per class
- Saving models and visualizations


