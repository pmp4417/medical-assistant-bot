# Medical Assistant Bot 

A simple but effective question-answering system trained on a medical dataset. It uses NLP techniques to retrieve accurate answers about medical conditions such as glaucoma, high blood pressure, and Paget's disease.

## Problem Statement

The objective was to create a question-answering model based on a dataset of medical questions and answers. The model must be capable of handling queries related to medical diseases and return precise answers.

## Approach and Methodology

I used a **TF-IDF vectorizer** combined with **cosine similarity** to compare user queries with known medical questions and return the most relevant answer.

- **Language Processing**: Performed with spaCy (tokenization, lemmatization, stopword removal)
- **Vectorization**: TF-IDF (term-frequency inverse document frequency)
- **Similarity Metric**: Cosine similarity
- **Fallback Handling**: If similarity score is below 0.3, a default message is returned

## Data Preprocessing

- Removed punctuation and lowercased all text
- Tokenized and lemmatized using `spaCy`
- Dataset cleaned and indexed with a new column `question_clean` for semantic matching

## Model Training

- No traditional training phase; used unsupervised similarity search
- Vector space created from all preprocessed questions in the dataset
- New queries are matched against this space in real time

## Evaluation

Since the model is similarity-based:
- Evaluation was **qualitative**
- Sample queries were tested and matched correctly to original answers
- Strength: Accurate recall of existing medical information
- Limitation: Does not generalize beyond known data

## Example Interactions

Q: What is Glaucoma? A: Glaucoma is a group of diseases that can damage the eye's optic nerve...

Q: How to treat high blood pressure? A: High blood pressure is treated with lifestyle changes and medication...

Q: What are the symptoms of Paget's disease? A: Symptoms may include pain, bone deformities, and fractures...