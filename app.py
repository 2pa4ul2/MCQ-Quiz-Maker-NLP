from flask import Flask, request, jsonify, render_template, redirect, url_for
import spacy
import random
import pdfplumber
from collections import Counter
import os
import nltk
from nltk.corpus import wordnet
nltk.download('wordnet')

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

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ')
            if synonym != word:
                synonyms.add(synonym)
    return list(synonyms)

def generate_mcqs(text, num_questions=20):
    if text is None:
        return []
    
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    
    sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 15 and not any(char.isdigit() for char in sent.text.strip())]
    
    generated_questions = set()
    mcqs = []
    
    while len(mcqs) < num_questions:
        sentence = random.choice(sentences)
        
        if len(sentence) > 200:
            continue
        
        sent_doc = nlp(sentence)
        # nouns = [token.text for token in sent_doc if token.pos_ == "NOUN"]
        nouns = [token.text for token in sent_doc if token.pos_ in ["NOUN", "PROPN"]]

        
        if len(nouns) < 1:
            continue
        
        subject = random.choice(nouns)
        question_stem = sentence.replace(subject, "_______", 1)
        
        if (question_stem, subject) in generated_questions:
            continue
        
        answer_choices = [subject]
        
        synonyms = get_synonyms(subject)
        similar_words = [token.text for token in nlp.vocab if token.is_alpha and token.has_vector and token.is_lower and token.similarity(nlp(subject)) > 0.5][:3]
        
        distractors = list(set(synonyms + similar_words))
        distractors = [d for d in distractors if d.lower() != subject.lower()]  # Ensure different words
        
        while len(distractors) < 3:
            new_distractor = random.choice([token.text for token in nlp(text) if token.pos_ in ["NOUN", "PROPN"] and token.text.lower() != subject.lower() and token.text.lower() not in [d.lower() for d in distractors]])
            distractors.append(new_distractor)
        
        answer_choices.extend(random.sample(distractors, 3))
        random.shuffle(answer_choices)
        
        trivial_answer = True
        for option in answer_choices:
            if len(option) > 1:
                trivial_answer = False
                break
        
        if trivial_answer:
            continue
        
        # Check for similarity among choices
        similar_choices = True
        for i in range(len(answer_choices)):
            for j in range(i + 1, len(answer_choices)):
                if answer_choices[i].lower() == answer_choices[j].lower():
                    similar_choices = False
                    break
            if not similar_choices:
                break
        
        if not similar_choices:
            continue
        
        correct_answer = chr(64 + answer_choices.index(subject) + 1)
        mcqs.append((question_stem, answer_choices, correct_answer))
        generated_questions.add((question_stem, subject))
    
    return mcqs



@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/how')
def howto():
    return render_template('howto.html')

@app.route('/index')
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
        # Retrieve the number of questions from the form
        num_questions = int(request.form.get('num_questions', 5))  # Default to 5 if not provided
        # Redirect to the questions route with file path and number of questions
        return redirect(url_for('questions', file_path=file_path, num_questions=num_questions))
    return redirect(request.url)

@app.route('/questions')
def questions():
    file_path = request.args.get('file_path')
    num_questions = int(request.args.get('num_questions', 5))  # Default to 5 if not provided
    text = extract_pdf_text(file_path)
    mcqs = generate_mcqs(text, num_questions=num_questions)
    mcqs_with_index = [(i + 1, mcq) for i, mcq in enumerate(mcqs)]
    return render_template('questions.html', mcqs=mcqs_with_index, enumerate=enumerate, chr=chr)


if __name__ == '__main__':
    app.run(debug=True)