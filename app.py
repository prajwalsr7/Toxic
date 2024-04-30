from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import tensorflow as tf
import urllib.request
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import TextVectorization

app = Flask(__name__)
url = "https://22228811toxic.s3.amazonaws.com/train.csv"
model_url = 'https://toxicmodel.s3.amazonaws.com/toxicity.h5'
model_file = 'toxicity.h5'

# Read the CSV file directly into a pandas DataFrame
df = pd.read_csv(url)

X = df['comment_text']

# Download the model file from the HTTPS URL

urllib.request.urlretrieve(model_url, model_file)

# Load the model from the local file
model = tf.keras.models.load_model(model_file)

# Define the maximum number of words in the vocabulary
MAX_FEATURES = 200000

# Initialize the TextVectorization layer
vectorizer = TextVectorization(max_tokens=MAX_FEATURES, output_sequence_length=1800, output_mode='int')

# Adapt the vectorizer to the input data
vectorizer.adapt(X.values)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    input_text = request.form['input_text']
    input_text = vectorizer(input_text)
    prediction = model.predict(np.expand_dims(input_text, 0))
    result = (prediction > 0.5).astype(int)
    
    toxicity_result = "Toxic" if result[0][0] == 1 else "Not Toxic"
    severe_toxic_result = "Severe Toxic" if result[0][1] == 1 else "Not Severe Toxic"
    obscene_result = "Obscene" if result[0][2] == 1 else "Not Obscene"
    threat_result = "Threat" if result[0][3] == 1 else "Not Threat"
    insult_result = "Insult" if result[0][4] == 1 else "Not Insult"
    identity_hate_result = "Identity Hate" if result[0][5] == 1 else "Not Identity Hate"
    
    combined_result = f" {toxicity_result},  {severe_toxic_result},  {obscene_result},  {threat_result},  {insult_result},  {identity_hate_result}"
    
    return render_template('result.html', result=combined_result)

if __name__ == '__main__':
    app.run(debug=True)