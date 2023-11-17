from transformers import BartTokenizer, BartForConditionalGeneration, pipeline
from PyPDF2 import PdfReader
from gensim import corpora
from gensim.models import TfidfModel
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import numpy as np



def load_model():
    # Load pre-trained model and tokenizer
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    # Create a question-answering pipeline
    nlp = pipeline('question-answering', model='distilbert-base-uncased-distilled-squad')
    return nlp


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

def ask_questions(nlp, vectors, dictionary, tfidf, text, questions):
    # Convert questions to vectors
    question_vectors = [tfidf[dictionary.doc2bow(word_tokenize(q))] for q in questions]
    # Convert sparse vectors to dense vectors and reshape to 2D
    vectors_2d = [np.reshape(np.array(v).mean(axis=0), (1, -1)) for v in vectors]
    question_vectors_2d = [np.reshape(np.array(qv).mean(axis=0), (1, -1)) for qv in question_vectors]
    # Find the most similar document for each question
    answers = []
    for i, qv in enumerate(question_vectors_2d):
        # Ensure the question vector has the same number of features as the document vectors
        if qv.shape[1] < vectors_2d[0].shape[1]:
            qv = np.pad(qv, ((0, 0), (0, vectors_2d[0].shape[1] - qv.shape[1])), 'constant')
        similarities = [cosine_similarity(qv, dv) for dv in vectors_2d]
        most_similar_index = similarities.index(max(similarities))
        # Use the pre-trained model to answer the question based on the most similar document
        answer = nlp(question=questions[i], context=text)
        answers.append(answer['answer'])
    return answers





def main():
    nlp = load_model()
    pdf_file_path = input("Enter the path of the PDF file: ")
    text = read_pdf(pdf_file_path)
    vectors, dictionary, tfidf = process_text(text)
    questions = []
    while True:
        question = input("Enter a question (or 'quit' to stop): ")
        if question.lower() == 'quit':
            break
        questions.append(question)
    answers = ask_questions(nlp, vectors, dictionary, tfidf, text, questions)
    for question, answer in zip(questions, answers):
        print(f"Question: {question}")
        print(f"Answer: {' '.join(answer)}\n")

if __name__ == '__main__':
    main()

