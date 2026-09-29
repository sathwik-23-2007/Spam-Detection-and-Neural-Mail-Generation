import pandas as pd
import pickle
import nltk
from nltk.corpus import stopwords
import string
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

nltk.download('stopwords', quiet=True)

def preprocess_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    stop_words = set(stopwords.words('english'))
    text = ' '.join([w for w in text.split() if w not in stop_words])
    return text

df = pd.read_csv('data/mail_data.csv')
df['label'] = df['Category'].map({'spam': 1, 'ham': 0})
df['cleaned_text'] = df['Message'].apply(preprocess_text)

with open('app/models/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open('app/models/spam_model.pkl', 'rb') as f:
    model = pickle.load(f)

X = vectorizer.transform(df['cleaned_text'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)

train_prec = precision_score(y_train, train_pred)
test_prec = precision_score(y_test, test_pred)

train_rec = recall_score(y_train, train_pred)
test_rec = recall_score(y_test, test_pred)

train_f1 = f1_score(y_train, train_pred)
test_f1 = f1_score(y_test, test_pred)

spam_count = int((df['label'] == 1).sum())
ham_count = int((df['label'] == 0).sum())

print(f"Total samples: {len(df)}")
print(f"Training samples: {len(y_train)}")
print(f"Testing samples: {len(y_test)}")
print(f"Spam count: {spam_count}")
print(f"Ham count: {ham_count}")
print(f"\n{'Metric':<12} {'Training':>10} {'Testing':>10}")
print(f"{'-'*34}")
print(f"{'Accuracy':<12} {train_acc*100:>9.2f}% {test_acc*100:>9.2f}%")
print(f"{'Precision':<12} {train_prec*100:>9.2f}% {test_prec*100:>9.2f}%")
print(f"{'Recall':<12} {train_rec*100:>9.2f}% {test_rec*100:>9.2f}%")
print(f"{'F1-Score':<12} {train_f1*100:>9.2f}% {test_f1*100:>9.2f}%")
