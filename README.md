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

## Model Performance

The current model performs well in retrieving relevant responses based on user questions. Since it uses TF-IDF with cosine similarity, it ensures that all responses are grounded in the medical dataset provided — preserving accuracy and preventing hallucinations.

### Strengths:
- Precise retrieval of known answers
- Fully grounded in a verified dataset
- Easy to implement and scale with more Q&A pairs

### Limitations:
- Can't generate new answers or infer knowledge outside the dataset
- Sensitive to exact wording of user queries
- Lower semantic understanding compared to transformer-based models

## Example Interactions

Q: What is Glaucoma? A: Glaucoma is a group of diseases that can damage the eye's optic nerve...

Q: How to treat high blood pressure? A: High blood pressure is treated with lifestyle changes and medication...

Q: What are the symptoms of Paget's disease? A: Symptoms may include pain, bone deformities, and fractures...


## Suggested Improvements

To enhance both performance and user experience, the following improvements are suggested:

### 1. Replace `pandas` with `polars`
`polars` offers faster performance and better memory efficiency, especially for large datasets. It supports parallel execution and optimized queries compared to `pandas`.

### 2. Use semantic embeddings instead of TF-IDF
Swap TF-IDF for contextual embeddings using models like:
- `SentenceTransformers` (e.g. `all-MiniLM-L6-v2`)
- `scispaCy` (optimized for biomedical text)
- `BioWordVec` or domain-specific medical embeddings

This would enable more intelligent matching of user input to relevant answers.

### 3. Add fallback handling
When no good match is found (low similarity score), consider:
- Providing a list of related or suggested questions
- Adding a “Did you mean…” feature via fuzzy matching

### 4. Streamlit or Flask Web Interface
Building a simple web app with Streamlit would allow:
- Better user experience
- Easy demo for stakeholders or non-technical users

### 5. API integration with FastAPI
Deploy the model as a REST API using FastAPI and Docker, allowing external systems or apps to access the model.

### 6. Evaluation with semantic metrics
In addition to qualitative testing, evaluate using:
- ROUGE / BLEU / BERTScore (for generated answers or ranking quality)
- User testing and feedback loops

### 7. Expand dataset and add category classification
Structure the dataset by specialty (e.g. Cardiology, Neurology) and optionally add a first-stage classifier to route questions to sub-models for better accuracy.


## Declaration of Independent Work

I hereby declare that this project was completed independently without the use of any third-party AI systems (such as OpenAI, Claude, or similar tools) in any part of the solution.  
All code, model design, and documentation were developed manually.


## How to Run

```bash
# Create virtual environment
python -m venv venv
.\venv\Scripts\activate      # On Windows
source venv/Scripts/activate # On Git Bash
source venv/bin/activate     # On Linux/macOS

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Run the bot
python src/interface.py


## Project Structure

medical-assistant-bot/ ├── data/ │ └── medical_qa.csv ├── src/ │ ├── interface.py │ ├── model.py │ └── preprocess.py ├── tests/ │ └── test_similarity.py ├── requirements.txt ├── .gitignore ├── README.md └── setup_project.py