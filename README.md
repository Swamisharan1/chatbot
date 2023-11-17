
# Chatbot: PDF Question Answering System

This project is a question-answering system that can answer questions based on the content of a given PDF file. It uses a pre-trained model from the transformers library for question answering, PyPDF2 for reading the PDF file, and Gensim for creating a TF-IDF model.

## Installation

Before running the code, you need to install the necessary Python packages. You can do this by running the following command:

```bash
pip install transformers PyPDF2 gensim sklearn nltk numpy
```

## Usage

To use the question-answering system, run the main.py script. The script will prompt you to enter the path of the PDF file. After you’ve entered the path, you can start asking questions. To stop asking questions, type ‘quit’.

Here’s an example of how to use the system:

```bash
python main.py
Enter the path of the PDF file: /path/to/your/file.pdf
Enter a question: What is the capital of France?
Question: What is the capital of France?
Answer: Paris

Enter a question: quit
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request.

## License

This project is licensed under the terms of the MIT license.
