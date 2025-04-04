import pandas as pd
import re
import spacy

nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text

def preprocess(text):
    doc = nlp(clean_text(text))
    return " ".join([token.lemma_ for token in doc if not token.is_stop])

def load_dataset(path):
    df = pd.read_csv(path)
    df = df.dropna()
    df["question_clean"] = df["question"].apply(preprocess)
    return df
