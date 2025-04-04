from sklearn.feature_extraction.text import TfidfVectorizer
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
