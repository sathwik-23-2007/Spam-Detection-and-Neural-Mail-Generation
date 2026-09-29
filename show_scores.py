import pickle
import string
from nltk.corpus import stopwords

with open('app/models/spam_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('app/models/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

stop_words = set(stopwords.words('english'))
coefs = model.coef_[0]
names = vectorizer.get_feature_names_out()

def show_decision(email):
    text = email.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    text = ' '.join([w for w in text.split() if w not in stop_words])
    vec = vectorizer.transform([text])
    features = vec.tocoo()

    print(f'Email: "{email}"')
    print(f'Step 1 - Start with intercept: {model.intercept_[0]:.4f}')
    print(f'Step 2 - Add each word score:')
    running = model.intercept_[0]
    for idx, val in zip(features.col, features.data):
        score = coefs[idx] * val
        running += score
        print(f'   "{names[idx]}" adds {score:+.4f}  -->  running total = {running:.4f}')
    print(f'Step 3 - FINAL SCORE: {running:.4f}')
    if running > 0:
        print(f'Step 4 - {running:.4f} is ABOVE 0 --> SPAM')
    else:
        print(f'Step 4 - {running:.4f} is BELOW 0 --> HAM (Safe)')
    print()

print('='*60)
print('  HOW THE SPAM/HAM CUTOFF WORKS')
print('='*60)
print()
print(f'The model starts every email at: {model.intercept_[0]:.4f}')
print(f'Then adds/subtracts based on each word.')
print(f'CUTOFF = 0 (zero)')
print(f'  Above 0 = SPAM')
print(f'  Below 0 = HAM')
print()
print('-'*60)
print('EXAMPLE 1: A clear HAM email')
print('-'*60)
show_decision('Hey can we meet tomorrow for dinner thanks')

print('-'*60)
print('EXAMPLE 2: A clear SPAM email')
print('-'*60)
show_decision('Congratulations you won free prize claim your reward now')

print('-'*60)
print('EXAMPLE 3: A work email (HAM)')
print('-'*60)
show_decision('Please find attached the report for the meeting')
