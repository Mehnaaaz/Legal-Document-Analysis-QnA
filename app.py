from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import pdfplumber
import docx2txt
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import re
import tempfile

def load_model():
    """Load the fine-tuned BERT model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained("Jasu/bert-finetuned-squad-legalbert")
    model = AutoModelForQuestionAnswering.from_pretrained("Jasu/bert-finetuned-squad-legalbert")
    return pipeline("question-answering", model=model, tokenizer=tokenizer)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load model once during startup
qa_pipeline = load_model()

# Global variables to store document text and sections
document_text = ""
sections = []

def save_file(file):
    """Save the uploaded file securely using a temporary file."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp_file:
        file.save(tmp_file.name)
        return tmp_file.name

def extract_text(file_path):
    """Extract text from supported file types, return None if unsupported."""
    if file_path.endswith('.pdf'):
        with pdfplumber.open(file_path) as pdf:
            text = ''.join(page.extract_text() for page in pdf.pages if page.extract_text())
    elif file_path.endswith('.docx'):
        text = docx2txt.process(file_path)
    elif file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        return None
    return text.strip() if text else None

def identify_sections(text):
    """Identify section markers in the text."""
    return re.findall(r'\§\d+\.\d+', text)

def answer_question(question, context):
    """Answer a question based on the document context."""
    result = qa_pipeline(question=question, context=context)
    answer = result['answer']
    confidence = result['score'] * 100  # Convert to percentage
    answer_start = result['start']
    answer_end = result['end']
    nearby_text = context[max(0, answer_start-100):answer_end+100]
    relevant_sections = [section for section in sections if section in nearby_text]
    return answer, confidence, relevant_sections

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file uploads and return identified sections."""
    global document_text, sections
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    file_path = save_file(file)
    document_text = extract_text(file_path)
    if document_text is None:
        os.remove(file_path)
        return jsonify({'error': 'Unsupported file type. Please upload PDF, DOCX, or TXT.'}), 400
    sections = identify_sections(document_text)
    os.remove(file_path)
    return jsonify({'sections': sections})

@app.route('/ask', methods=['POST'])
def ask_question():
    """Answer questions based on the uploaded document."""
    global document_text
    data = request.json
    question = data.get('question', '')
    if not document_text:
        return jsonify({'error': 'No document uploaded'}), 400
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    answer, confidence, relevant_sections = answer_question(question, document_text)
    return jsonify({
        'answer': answer,
        'confidence': confidence,
        'sections': relevant_sections
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)