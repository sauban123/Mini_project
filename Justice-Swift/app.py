from flask import Flask, render_template, request
from transformers import pipeline
import requests
from PyPDF2 import PdfReader
from io import BytesIO
import textwrap
import spacy
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Initialize the summarization pipeline
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Load the tokenizer and label encoder for sentiment analysis
with open('tokenizer.pkl', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)
with open('label_encoder.pkl', 'rb') as label_encoder_file:
    label_encoder = pickle.load(label_encoder_file)

# Load the spaCy model for preprocessing
nlp = spacy.load('en_core_web_sm')

# Load the machine learning model using Keras
model = load_model('my_model.h5')

# Function to extract text from a PDF URL
def extract_text_from_pdf_url(url):
    response = requests.get(url)
    pdf_file = BytesIO(response.content)
    pdf_reader = PdfReader(pdf_file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Replace non-breaking space with regular space
    text = text.replace('\xa0', ' ')

    return text

# Function to split text into lines with line breaks after every n words
def split_text_with_line_break(text, words_per_line=10):
    words = text.split()
    lines = [" ".join(words[i:i+words_per_line]) for i in range(0, len(words), words_per_line)]
    return "\n".join(lines)

# Define preprocessing function
def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop]
    return ' '.join(tokens)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result = None
    summarized_content = None
    pdf_url=""
    if request.method == 'POST':
        pdf_url = request.form['pdf_url']

        if not pdf_url.startswith('/supremecourt'):
            error_message = "Invalid URL. The URL should start with '/supremecourt'."
            return render_template('index.html', error_message=error_message)

        full_url = 'https://main.sci.gov.in' + pdf_url

        # Extract the text
        text = extract_text_from_pdf_url(full_url)

        # Convert text to lowercase
        lower_text = text.lower()

        # Preprocess the input text for sentiment analysis
        user_input = preprocess_text(lower_text)
        first_50_words = lower_text.split()[:50]

        if 'civil' in first_50_words and 'jurisdiction' in first_50_words:
            priority_message = 'Priority: Low since civil case'
        elif 'criminal' in first_50_words and 'jurisdiction' in first_50_words:
            priority_message = 'Priority: High since criminal case'
        else:
            priority_message = 'Priority: Not specified'

        user_input_sequence = tokenizer.texts_to_sequences([user_input])
        user_input_padded = pad_sequences(user_input_sequence, maxlen=100, padding='post')

        # Make prediction using the trained model
        prediction = model.predict(user_input_padded)

        # If your model is trained for binary classification (sigmoid activation in the output layer)
        # convert the prediction to a binary label (0 or 1) using a threshold (e.g., 0.5)
        threshold = 0.5
        binary_prediction = (prediction > threshold).astype(int)[0, 0]

        # Inverse transform the binary prediction using the label_encoder
        inverted_prediction = label_encoder.inverse_transform([binary_prediction])[0]

        # Convert prediction to a human-readable format
        prediction_result = f'The  prediction is:  ({inverted_prediction})'

        # Split the text into chunks of approximately 1024 tokens
        chunks = textwrap.wrap(text, width=1024)

        # Summarize each chunk and combine the summaries
        summary = []
        for chunk in chunks:
            if len(chunk.split()) > 50:
                summary.append(summarizer(chunk, max_length=50, min_length=25, do_sample=False)[0]['summary_text'])
            else:
                summary.append(chunk)

        # Prepare the summarized content for rendering
        summarized_content = []
        for sentence in summary:
            lines = split_text_with_line_break(sentence)
            summarized_content.extend(lines.split('\n'))
        summarized_content.insert(0, priority_message)
    return render_template('index.html', pdf_url=pdf_url, prediction_result=prediction_result, summarized_content=summarized_content)

if __name__ == '__main__':
    app.run(debug=True)
