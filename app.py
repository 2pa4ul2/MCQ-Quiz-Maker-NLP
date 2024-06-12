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
    # Process the text with spaCy
    doc = nlp(text)

    # Extract sentences from the text and filter out unwanted sentences
    sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10 and not any(char.isdigit() for char in sent.text.strip())]

    # Initialize set to store generated questions and answers
    generated_questions = set()

    # Initialize list to store generated MCQs
    mcqs = []

    # Generate MCQs until we reach the desired number
    while len(mcqs) < num_questions:
        # Randomly select a sentence to form a question
        sentence = random.choice(sentences)

        # Skip sentences that are too long
        if len(sentence) > 200:
            continue

        # Process the sentence with spaCy
        sent_doc = nlp(sentence)

        # Extract entities (nouns) from the sentence
        nouns = [token.text for token in sent_doc if token.pos_ == "NOUN"]

        # Ensure there are enough nouns to generate MCQs
        if len(nouns) < 1:
            continue

        # Select a random noun as the subject of the question
        subject = random.choice(nouns)

        # Generate the question stem
        question_stem = sentence.replace(subject, "_______", 1)

        # Check if the question has already been generated
        if (question_stem, subject) in generated_questions:
            continue

        # Generate answer choices
        answer_choices = [subject]

        # Get synonyms of the correct answer and similar words
        synonyms = get_synonyms(subject)
        similar_words = [token.text for token in nlp.vocab if token.is_alpha and token.has_vector and token.is_lower and token.similarity(nlp(subject)) > 0.5][:3]

        # Combine synonyms and similar words for distractors
        distractors = list(set(synonyms + similar_words))

        # Remove the correct answer from distractors
        distractors = [d for d in distractors if d != subject]

        # Ensure there are at least 3 distractors
        while len(distractors) < 3:
            new_distractor = random.choice([token.text for token in nlp(text) if token.pos_ == "NOUN" and token.text != subject])
            if new_distractor not in distractors:
                distractors.append(new_distractor)

        # Add distractors to answer choices
        answer_choices.extend(random.sample(distractors, 3))

        # Shuffle the answer choices
        random.shuffle(answer_choices)

        # Check if the correct answer is trivial (e.g., "a.")
        trivial_answer = True
        for option in answer_choices:
            if len(option) > 1:
                trivial_answer = False
                break

        if trivial_answer:
            continue

        # Append the generated MCQ to the list
        correct_answer = chr(64 + answer_choices.index(subject) + 1)  # Convert index to letter
        mcqs.append((question_stem, answer_choices, correct_answer))

        # Add the generated question to the set
        generated_questions.add((question_stem, subject))

    return mcqs


@app.route('/')
def landing():
    return render_template('landing.html')

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
