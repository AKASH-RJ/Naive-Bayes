# Fake News Detection using Naive Bayes & Flask

## Overview

This project detects whether a news article is **Fake** or **Real** using the **Naive Bayes** algorithm. The model is trained on a labeled dataset of news headlines/content and deployed as a **Flask web application** with HTML & CSS.

-----

## Features

  - **Naive Bayes classifier** for text classification.
  - **Flask backend** to serve predictions.
  - **Interactive HTML form** for entering news text.
  - **CSS styling** for better UI.
  - **CSV dataset** with labeled news data for training.

-----

## Project Structure

```
naive_bayes_fake_news_app/
│
├── model.py             # Trains and saves the Naive Bayes model
├── app.py               # Flask application for predictions
├── templates/
│   ├── index.html       # Main input form
│   └── result.html      # Displays prediction
├── static/
│   └── style.css        # CSS for styling
├── dataset.csv          # Dataset (training data)
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

-----

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file contains:

```
Flask==3.0.0
pandas==2.1.4
scikit-learn==1.3.2
numpy==1.26.2
```

-----

## Dataset

The dataset (`dataset.csv`) contains news headlines/content with labels.

Example:

```
text,label
"Breaking: Government announces new policy",Real
"Aliens have landed in New York City",Fake
```

Columns:

  - `text`: News content or headline.
  - `label`: Real or Fake.

-----

## How It Works

### Model Training (`model.py`)

  - Loads dataset from `dataset.csv`.
  - Cleans and processes text using TF-IDF Vectorizer.
  - Trains a Multinomial Naive Bayes classifier.
  - Saves trained model as `model.pkl`.

### Web Application (`app.py`)

  - Loads `model.pkl`.
  - Accepts news text from an HTML form.
  - Predicts whether news is Fake or Real.
  - Displays the prediction result.

-----

## Running the Project

1.  **Train the Model**
    ```bash
    python model.py
    ```
2.  **Run Flask App**
    ```bash
    python app.py
    ```
3.  **Open in Browser**
    Go to: `http://127.0.0.1:5000/`

-----

## Screenshots
---
Home Page

<img width="556" height="391" alt="Screenshot 2025-08-12 121219" src="https://github.com/user-attachments/assets/24fa38d2-3a68-40aa-9e99-22c8b3758640" />

---
Prediction Result

<img width="573" height="449" alt="Screenshot 2025-08-12 121230" src="https://github.com/user-attachments/assets/ddc53189-cbd5-4326-9199-a4071589e634" />
