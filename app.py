from flask import Flask, request, jsonify, render_template
import spacy
import random
from PyPDF2 import PdfReader 
from collections import Counter

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')