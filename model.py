from PyPDF2 import PdfReader
from gensim import corpora
from gensim.models import TfidfModel
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

def read_pdf(pdf_file_path):
    # Open PDF file
    with open(pdf_file_path, 'rb') as f:
        pdf = PdfReader(f)
        text = ""
        for page in range(len(pdf.pages)):
            text += pdf.pages[page].extract_text()
    return text

def process_text(text):
    # Tokenize the text
    tokens = [word_tokenize(t) for t in text.split('\n') if t]
    # Create a dictionary representation of the documents
    dictionary = corpora.Dictionary(tokens)
    # Create a bag of words corpus
    corpus = [dictionary.doc2bow(token) for token in tokens]
    # Create a TF-IDF model from the corpus
    tfidf = TfidfModel(corpus)
    # Store the TF-IDF vectors in a list
    vectors = [tfidf[c] for c in corpus]
    return vectors, dictionary, tfidf
