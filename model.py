import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("ds.csv")

# Features & target
X = df['text']
y = df['label'].map({'Real': 1, 'Fake': 0})

# Vectorize text
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_vec = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Save model & vectorizer
joblib.dump(model, "fake_news_nb_model.pkl")
joblib.dump(vectorizer, "fake_news_vectorizer.pkl")

print("Model and vectorizer saved successfully!")
