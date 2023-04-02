from transformers import RobertaTokenizerFast, TFRobertaForSequenceClassification, pipeline
from flask import Flask, render_template, url_for, request
import operator
# runing models

tokenizer = RobertaTokenizerFast.from_pretrained("arpanghoshal/EmoRoBERTa")
model = TFRobertaForSequenceClassification.from_pretrained("arpanghoshal/EmoRoBERTa")
emotion = pipeline('sentiment-analysis',
model='arpanghoshal/EmoRoBERTa')

app = Flask(__name__)

# home

@app.route('/')
def home():
    return render_template('home.html')

# Emotion

@app.route('/emotion_', methods=['GET', 'POST'])
def emotion_():
    if request.method == 'POST':
        data = request.form['fulltextarea']
        emotions = emotion(data)
        return render_template('emotion.html', fulltext=data,emotion = emotions[0]['label'])

    return render_template('home.html')



if __name__ == "__main__":
    app.secret_key = '0b579d376dc5dde856e0a0ddca6f403cc8707924ff8d6d31'
    app.run(debug=True)
