from preprocess import load_dataset, preprocess
from model import MedicalQAModel

if __name__ == "__main__":
    df = load_dataset("data/medical_qa.csv")
    model = MedicalQAModel(df["question_clean"].tolist(), df["answer"].tolist())

    print("Welcome to the Medical Assistant Bot!")
    while True:
        question = input("\nEnter your medical question (type 'exit' to quit): ")
        if question.lower() == "exit":
            break
        cleaned = preprocess(question)
        answer = model.get_answer(cleaned)
        print("\nanswer:", answer)
