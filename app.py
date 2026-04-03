from flask import Flask, render_template, request, redirect, url_for, session

import pickle
import string
from nltk.corpus import stopwords
import nltk
import os
from nltk.stem.porter import PorterStemmer 

app = Flask(__name__)
app.secret_key = "spam_detector_secret_key"


# Load model and vectorizer
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, 'model.pkl'), 'rb'))
tfidf = pickle.load(open(os.path.join(BASE_DIR, 'vectorizer.pkl'), 'rb'))

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


@app.route('/')
def home():
    prediction = session.pop('prediction', None)  # remove after showing
    return render_template('index.html', prediction_text=prediction)



@app.route('/predict', methods=['POST'])
def predict():
    input_sms = request.form['message']
    
    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms]).toarray()
    result = model.predict(vector_input)[0]

    session['prediction'] = "Spam" if result == 1 else "Not Spam"
    
    return redirect(url_for('home'))


if __name__ == '__main__':
    app.run(debug=True)
