from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load dataset
df = pd.read_csv("naive_bayes.csv")

# Split features and labels
X = df['text']
y = df['label']

# Vectorize text
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Train model
model = MultinomialNB()
model.fit(X_vectorized, y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news_text = request.form['news']
    news_vectorized = vectorizer.transform([news_text])
    prediction = model.predict(news_vectorized)[0]
    return render_template('index.html', prediction_text=f"The news is predicted as: {prediction}")

if __name__ == "__main__":
    app.run(debug=True)
