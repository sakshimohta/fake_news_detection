# Fake News Detection Using BERT

This repository implements a fake news detection system using a multilingual BERT model fine-tuned on a labeled news dataset. The model classifies news articles as Real or Fake.

---

## Features

- Data loading and preprocessing (cleaning, stopword removal, lemmatization)
- Visualization of dataset distribution and word clouds for real and fake news
- Tokenization using Hugging Face's `bert-base-multilingual-cased` tokenizer
- Fine-tuning a BERT-based neural network classifier with TensorFlow/Keras
- Model evaluation with accuracy, confusion matrix, and classification report
- Prediction on custom news text (including Hindi)
- Model interpretability using LIME (Local Interpretable Model-agnostic Explanations)

---

## Requirements

- Python 3.x
- TensorFlow
- Transformers
- scikit-learn
- pandas, numpy
- matplotlib, seaborn
- nltk
- wordcloud
- Pillow (PIL)
- lime
- mlxtend

---

## Setup Instructions

1. **Mount Google Drive** (if using Google Colab) to access the dataset:

2. **Install required packages:**

3. **Download NLTK data:**


---

## Dataset

- CSV file named `news_dataset.csv` containing news articles labeled as `REAL` or `FAKE`.
- Remove null values before processing.

---

## Data Preprocessing

- Clean text by removing URLs, special characters, punctuation, and converting to lowercase.
- Remove stopwords and apply lemmatization using NLTK.
- Encode labels (`REAL` → 1, `FAKE` → 0).

---

## Tokenization

- Use `BertTokenizer` from Hugging Face with `bert-base-multilingual-cased`.
- Maximum sequence length set to 128 tokens.
- Tokenize text data with padding and truncation.

---

## Training

- Split data into train and test sets (80% train, 20% test).
- Train model for 5 epochs with batch size 32.
- Use 20% of training data as validation.

---

## Evaluation

- Evaluate accuracy on test data.
- Plot confusion matrix and print classification report.

---

---

## Explainability with LIME

- Install LIME:

- Use `LimeTextExplainer` to explain predictions on sample texts.

---

## File Structure

- `FakeNews.py` — main script with full pipeline (data loading, preprocessing, training, evaluation, prediction).
- `news_dataset.csv` — dataset file (should be placed in the specified directory).

---

## Notes

- The model supports multilingual news articles, including Hindi.
- The trained model is saved as `bert.pkl` using pickle.
- Visualization includes accuracy/loss plots and word clouds.

---

## References

- Hugging Face Transformers: `bert-base-multilingual-cased`
- LIME for interpretability
- Dataset source: news_dataset.csv

---

Feel free to clone this repo and run the notebook/script to train and test the fake news detection model.

