from flask import Flask, request, jsonify, render_template, redirect, url_for
import spacy
import random
import pdfplumber
from collections import Counter
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def extract_pdf_text(file_path):
    try:
        with pdfplumber.open(file_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
        return text
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def generate_mcqs(text, num_questions=20):
    nlp = spacy.load('en_core_web_sm')
    if text is None:
        return []

    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    selected_sentences = random.sample(sentences, min(num_questions, len(sentences)))
    mcqs = []

    for sentence in selected_sentences:
        sent_doc = nlp(sentence)
        nouns = [token.text for token in sent_doc if token.pos_ == "NOUN"]
        if len(nouns) < 2:
            continue
        noun_counts = Counter(nouns)
        if noun_counts:
            subject = noun_counts.most_common(1)[0][0]
            question_stem = sentence.replace(subject, "_______")
            answer_choices = [subject]
            distractors = list(set(nouns) - set([subject]))
            while len(distractors) < 3:
                distractors.append("[Distractor]")
            random.shuffle(distractors)
            for distractor in distractors[:3]:
                answer_choices.append(distractor)
            random.shuffle(answer_choices)
            correct_answer = chr(64 + answer_choices.index(subject) + 1)
            mcqs.append((question_stem, answer_choices, correct_answer))

    return mcqs

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'pdf_file' not in request.files:
        return redirect(request.url)
    file = request.files['pdf_file']
    if file.filename == '':
        return redirect(request.url)
    if file and file.filename.endswith('.pdf'):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        return redirect(url_for('questions', file_path=file_path))
    return redirect(request.url)

@app.route('/questions')
def questions():
    file_path = request.args.get('file_path')
    text = extract_pdf_text(file_path)
    mcqs = generate_mcqs(text, num_questions=20)
    mcqs_with_index = [(i + 1, mcq) for i, mcq in enumerate(mcqs)]
    return render_template('questions.html', mcqs=mcqs_with_index, enumerate=enumerate, chr=chr)

if __name__ == '__main__':
    app.run(debug=True)
