import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
import string
import pickle
import os

# Download NLTK data
nltk.download('stopwords')

def preprocess_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    stop_words = set(stopwords.words('english'))
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# Load the dataset
try:
    df = pd.read_csv('data/mail_data.csv')
    print("Dataset loaded successfully.")
    print(f"Total samples: {len(df)}")
    
    # Check if necessary columns exist
    if 'Category' in df.columns and 'Message' in df.columns:
        df['label'] = df['Category'].map({'spam': 1, 'ham': 0})
        df = df.dropna(subset=['label', 'Message'])
        df['cleaned_text'] = df['Message'].apply(preprocess_text)
        print(f"Spam: {(df['label'] == 1).sum()}, Ham: {(df['label'] == 0).sum()}")
    else:
        print("Error: Dataset must contain 'Category' and 'Message' columns.")
        exit()

    # Vectorization — use bigrams + sublinear TF for sharper feature separation
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),      # unigrams + bigrams (catches phrases like "free offer")
        sublinear_tf=True,       # apply log normalization to term frequencies
        min_df=2,                # ignore extremely rare words (noise)
        max_df=0.95,             # ignore words that appear in 95%+ of emails (too common)
    )
    X = vectorizer.fit_transform(df['cleaned_text'])
    y = df['label']
    print(f"Feature count: {X.shape[1]}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model training — balanced class weights + higher C for decisive boundaries
    model = LogisticRegression(
        C=10,                    # stronger regularization inverse = sharper decision boundary
        class_weight='balanced', # handles class imbalance (fewer spam than ham)
        max_iter=1000,           # ensure convergence
        solver='liblinear',      # works well for small-to-medium datasets
    )
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

    # Save model and vectorizer
    os.makedirs('app/models', exist_ok=True)
    with open('app/models/spam_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('app/models/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    print("Model and vectorizer saved successfully.")
except FileNotFoundError:
    print("Error: 'data/mail_data.csv' not found. Please provide the dataset.")
except Exception as e:
    print(f"An error occurred: {e}")