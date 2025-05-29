# Fake-News Detection with NLP & Machine Learning

Detect misinformation in news headlines with a fully reproducible, end-to-end NLP pipeline built on the LIAR benchmark dataset.

---

## 📊 Project Highlights
| Stage | Key Techniques |
|-------|----------------|
| **Data Engineering** | TSV ingest ➜ pandas cleaning, tokenisation, lemmatisation, stop-word filtering |
| **Feature Engineering** | TF-IDF 1–3 gram vectors · χ² feature selection · (prototyped) Word2Vec & POS-tag embeddings |
| **Modelling** | Multinomial NB · Logistic Regression · Linear SVM · SGD · Random Forest — hyper-tuned with GridSearchCV |
| **Evaluation** | Macro-F1, precision/recall, confusion matrices, learning curves |
| **Serving** | Model serialised with `joblib` for instant CLI predictions; minimal Flask API demo |

The best model (Logistic Regression) achieves **≈ 0.78 macro-F1** on the held-out test set.

---


