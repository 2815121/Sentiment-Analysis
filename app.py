from flask import Flask, render_template, request
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

app = Flask(__name__)
sia = SentimentIntensityAnalyzer()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/review', methods=['GET', 'POST'])
def review():
    sentiment = None
    if request.method == 'POST':
        text = request.form['review']
        score = sia.polarity_scores(text)['compound']
        if score >= 0.05:
            sentiment = 'Positive'
        elif score <= -0.05:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'
    return render_template('review.html', sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
