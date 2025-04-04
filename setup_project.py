import os

# Criação das pastas, se ainda não existirem
os.makedirs("src", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("tests", exist_ok=True)

# requirements.txt
with open("requirements.txt", "w", encoding="utf-8") as f:
    f.write("pandas\nscikit-learn\nspacy\n")

# README.md
with open("README.md", "w", encoding="utf-8") as f:
    f.write(
        "# Medical Assistant Bot\n\n"
        "This is a medical question-answering system using NLP and TF-IDF similarity search.\n\n"
        "## How to Run\n\n"
        "```bash\n"
        "python -m venv venv\n"
        "venv\\Scripts\\activate\n"
        "pip install -r requirements.txt\n"
        "python -m spacy download en_core_web_sm\n"
        "python src/interface.py\n"
        "```\n\n"
        "## Example\n\n"
        "**Q:** What is Glaucoma?\n"
        "**A:** Glaucoma is a group of diseases that can damage the optic nerve...\n"
    )

# src/preprocess.py
with open("src/preprocess.py", "w", encoding="utf-8") as f:
    f.write("""import pandas as pd
import re
import spacy

nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\\s]", "", text)
    return text

def preprocess(text):
    doc = nlp(clean_text(text))
    return " ".join([token.lemma_ for token in doc if not token.is_stop])

def load_dataset(path):
    df = pd.read_csv(path)
    df = df.dropna()
    df["question_clean"] = df["Question"].apply(preprocess)
    return df
""")

# src/model.py
with open("src/model.py", "w", encoding="utf-8") as f:
    f.write("""from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class MedicalQAModel:
    def __init__(self, questions, answers):
        self.vectorizer = TfidfVectorizer()
        self.questions = questions
        self.answers = answers
        self.q_vectors = self.vectorizer.fit_transform(questions)

    def get_answer(self, user_question, threshold=0.3):
        vec = self.vectorizer.transform([user_question])
        similarities = cosine_similarity(vec, self.q_vectors).flatten()
        idx = np.argmax(similarities)
        score = similarities[idx]
        if score < threshold:
            return "Sorry, I don't have an accurate answer for that question."
        return self.answers[idx]
""")

# src/interface.py
with open("src/interface.py", "w", encoding="utf-8") as f:
    f.write("""from preprocess import load_dataset, preprocess
from model import MedicalQAModel

if __name__ == "__main__":
    df = load_dataset("data/medical_qa.csv")
    model = MedicalQAModel(df["question_clean"].tolist(), df["Answer"].tolist())

    print("Welcome to the Medical Assistant Bot!")
    while True:
        question = input("\\nEnter your medical question (type 'exit' to quit): ")
        if question.lower() == "exit":
            break
        cleaned = preprocess(question)
        answer = model.get_answer(cleaned)
        print("\\nAnswer:", answer)
""")

print("✅ Projeto recriado com sucesso! Verifique a pasta 'src/'.")
