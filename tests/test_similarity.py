import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from model import MedicalQAModel

def test_answer_matching():
    questions = [
        "What is Glaucoma?",
        "How to treat high blood pressure?"
    ]
    answers = [
        "Glaucoma is a group of diseases that can damage the eye's optic nerve...",
        "High blood pressure is treated with lifestyle changes and medication..."
    ]

    model = MedicalQAModel(questions, answers)
    result = model.get_answer("What is Glaucoma?")
    assert "glaucoma" in result.lower()
