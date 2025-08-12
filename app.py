from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load model & vectorizer
model = joblib.load("fake_news_nb_model.pkl")
vectorizer = joblib.load("fake_news_vectorizer.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        news_text = request.form["news_text"]
        text_vec = vectorizer.transform([news_text])
        pred = model.predict(text_vec)[0]
        prediction = "Real" if pred == 1 else "Fake"
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
