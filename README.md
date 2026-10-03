# 🛍️ Flipkart Reviews Sentiment AI: Enterprise NLP Suite

An end-to-end, multi-class Natural Language Processing and Machine Learning application designed to classify customer reviews into **Positive**, **Neutral**, and **Negative** polarities.

---

## 🚀 Key Features

- **Cyberpunk / Glassmorphic UI**: High-contrast, dark-theme interface with neon gradient borders, glass cards, and interactive Plotly figures.
- **Real-Time Sentiment Analyzer**: Predicts polarity with normalized confidence metrics derived from Linear SVM decision margins.
- **Session History Tracker**: In-memory inspection log allowing review comparison and session CSV export.
- **Exploratory Analytics Dashboard**: Visualizes category distributions, discount vs. sentiment density, brand rankings, and delivery timelines.
- **Model Evaluation Suite**: Side-by-side performance comparison against Naive Bayes, Logistic Regression, LSTM, and Bidirectional LSTM models.
- **Linguistic Feature Attribution**: Direct inspection of SVM n-gram coefficient weights driving predictions.

---
## Deployed URL:
https://flipkart-sentiment-app-3ygjfs2hqckifrnzlk8jjs.streamlit.app/
---
## 📂 Project Structure

```text
Flipkart-Sentiment-AI/
│
├── app.py                             # Streamlit application entry point
├── requirements.txt                   # Dependency specifications
├── README.md                          # Project documentation
│
├── models/
│   ├── sentiment_svm.pkl              # Trained Linear Support Vector Classifier
│   ├── tfidf_vectorizer.pkl          # Fitted TF-IDF N-Gram Vectorizer
│   └── label_encoder.pkl             # Fitted LabelEncoder (Optional)
│
├── data/
│   └── dataset.csv                    # (Optional) 30k Flipkart review dataset
│
├── notebooks/
│   └── Flipkart_Sentiment_Analysis.ipynb # Experimental notebook
│
└── assets/
    └── logo.png                       # UI branding assets
