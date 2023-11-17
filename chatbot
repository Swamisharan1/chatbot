import streamlit as st
from transformers import BartTokenizer, BartForConditionalGeneration, pipeline
from PyPDF2 import PdfReader
from gensim import corpora
from gensim.models import TfidfModel
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

# Load pre-trained model and tokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

# Create a question-answering pipeline
nlp = pipeline('question-answering', model='distilbert-base-uncased-distilled-squad')

# Streamlit code to upload PDF file
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Open PDF file
    pdf = PdfReader(uploaded_file)
    text = ""
    for page in range(len(pdf.pages)):
        text += pdf.pages[page].extract_text()

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

    # Take questions from user
    question = st.text_input("Enter your question:")

    if st.button('Get Answer'):
        # Use the pipeline to answer the question
        answer = nlp(question=question, context=text)

        st.write(f"Question: {question}")
        st.write(f"Answer: {answer}\n")
