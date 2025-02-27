from transformers import pipeline

def run_qa():
    # Load the question-answering model
    model_name = "Jasu/bert-finetuned-squad-legalbert"  # Ensure this model exists
    try:
        qa_pipeline = pipeline("question-answering", model=model_name)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Read the sample legal document
    file_path = r"C:\Users\mehna\Downloads\legal_document.txt"
    try:
        with open(file_path, 'r', encoding="utf-8") as f:
            document = f.read()
    except Exception as e:
        print(f"Error reading document: {e}")
        return

    # Define a test question
    question = "What is it about?"

    # Get the model's answer
    try:
        result = qa_pipeline(question=question, context=document)
        print(f"Question: {question}")
        print(f"Answer: {result['answer']}")
        print(f"Confidence: {result['score']:.2f}")
    except Exception as e:
        print(f"Error during inference: {e}")

if __name__ == "__main__":
    run_qa()
