import os
import pickle
import numpy as np
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not found. Explanations will be disabled.")


from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import nltk
from nltk.corpus import stopwords
import string

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__, template_folder='app/templates', static_folder='app/static')

# Load model and vectorizer
with open('app/models/spam_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('app/models/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Configure Hugging Face API
# Hugging Face API key is loaded from environment variables via dotenv
from huggingface_hub import InferenceClient
hf_client = InferenceClient(api_key=os.getenv("HUGGINGFACE_API_KEY"))



def preprocess_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    stop_words = set(stopwords.words('english'))
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

@app.route('/')
def home():
    return render_template('index.html')

def get_nlp_summary(text):
    text_lower = text.lower()
    text_no_punct = "".join([char for char in text_lower if char not in string.punctuation])
    words = text_no_punct.split()
    stop_words = set(stopwords.words('english'))
    removed_stopwords = [word for word in words if word in stop_words]
    return {
        "original_length": len(text),
        "tokens_count": len(words),
        "stopwords_removed": len(removed_stopwords)
    }

def _get_spam_word_reason(word):
    """Return a dynamic, simple reason why a word looks spammy — unique per word."""
    w = word.lower()

    # Money / financial lure words
    if w in ('free', 'bonus', 'prize', 'reward', 'cash', 'money', 'dollar',
             'dollars', 'earn', 'income', 'profit', 'million', 'billion',
             'jackpot', 'lottery', 'win', 'winner', 'won', 'payout'):
        return "promises money or rewards — a classic trick used in scam emails"
    if w in ('offer', 'deal', 'discount', 'sale', 'cheap', 'bargain',
             'lowest', 'price', 'cost', 'save', 'saving', 'affordable'):
        return "pushes a deal or discount to tempt you into clicking"
    if w in ('buy', 'order', 'purchase', 'shop', 'store', 'checkout'):
        return "tries to get you to buy something — common in unwanted promotions"

    # Urgency / pressure words
    if w in ('urgent', 'immediately', 'hurry', 'rush', 'fast', 'quick',
             'limited', 'expire', 'expires', 'expiring', 'deadline',
             'asap', 'instant', 'now', 'today', 'act', 'action',
             'dont', 'miss', 'last', 'final', 'ending'):
        return "creates pressure to act fast — a common trick in scam emails"

    # Phishing / deception words
    if w in ('click', 'link', 'url', 'visit', 'website', 'http', 'www',
             'href', 'redirect', 'browse', 'download', 'install'):
        return "tries to get you to click a link — often used in phishing attacks"
    if w in ('verify', 'confirm', 'validate', 'update', 'secure',
             'security', 'protect', 'protection', 'authentication',
             'password', 'login', 'signin', 'credential', 'credentials',
             'account', 'suspend', 'suspended', 'locked', 'unauthorized'):
        return "pretends there is a security issue to steal your information"
    if w in ('bank', 'paypal', 'visa', 'mastercard', 'credit', 'debit',
             'card', 'transaction', 'billing', 'payment', 'pay', 'invoice'):
        return "references banking or payments — often used to trick people into sharing financial details"

    # Spam marketing words
    if w in ('subscribe', 'unsubscribe', 'newsletter', 'promotion',
             'promotional', 'advertise', 'advertisement', 'marketing',
             'campaign', 'bulk', 'mass', 'list', 'opt', 'optin'):
        return "is typical of mass marketing and bulk promotional emails"
    if w in ('guarantee', 'guaranteed', 'promise', 'risk', 'riskfree',
             'obligation', 'refund', 'satisfaction', 'certified'):
        return "makes unrealistic promises — a red flag in suspicious emails"
    if w in ('congratulations', 'congrats', 'selected', 'chosen',
             'exclusive', 'special', 'vip', 'member', 'membership'):
        return "uses flattery or fake exclusivity to lure you in"

    # Suspicious content words
    if w in ('pill', 'pills', 'medication', 'pharmacy', 'drug', 'drugs',
             'prescription', 'viagra', 'weight', 'diet', 'supplement',
             'health', 'cure', 'treatment', 'doctor', 'medical'):
        return "is commonly found in health-related spam and fake pharmacy emails"
    if w in ('call', 'contact', 'reply', 'respond', 'send', 'forward',
             'txt', 'text', 'sms', 'mobile', 'phone', 'number'):
        return "asks you to respond or share contact info — a typical spam tactic"
    if w in ('information', 'info', 'personal', 'details', 'data',
             'name', 'address', 'social', 'ssn'):
        return "tries to collect your personal information — a warning sign"
    if w in ('access', 'unlock', 'open', 'enable', 'activate',
             'registration', 'register', 'signup', 'sign'):
        return "pushes you to sign up or unlock something — often a trick"
    if w in ('claim', 'collect', 'redeem', 'receive', 'apply',
             'request', 'submit', 'fill', 'form'):
        return "wants you to claim or submit something — commonly seen in scam emails"
    if w in ('customer', 'service', 'support', 'team', 'representative',
             'agent', 'department', 'helpdesk', 'admin'):
        return "impersonates a customer service team — a common phishing strategy"

    # Generic fallback — still dynamic using the word itself
    return f"our system learned this word appears more often in spam than in safe emails"


def _get_ham_word_reason(word):
    """Return a dynamic, simple reason why a word looks safe — unique per word."""
    w = word.lower()

    # Greetings / polite words
    if w in ('thanks', 'thank', 'thankyou', 'appreciate', 'grateful',
             'gratitude', 'cheers', 'regards', 'sincerely', 'kindly',
             'please', 'welcome', 'hi', 'hello', 'hey', 'dear', 'greetings'):
        return "is a polite word used in friendly, everyday conversation"
    if w in ('good', 'great', 'nice', 'wonderful', 'awesome', 'excellent',
             'amazing', 'fantastic', 'brilliant', 'lovely', 'fine', 'well'):
        return "is a positive word found in normal, genuine conversations"

    # Work / professional words
    if w in ('meeting', 'schedule', 'agenda', 'project', 'report',
             'review', 'update', 'status', 'progress', 'task', 'plan',
             'planning', 'deadline', 'milestone', 'deliverable'):
        return "is a work-related word commonly found in professional emails"
    if w in ('team', 'colleague', 'manager', 'boss', 'office',
             'department', 'company', 'organization', 'workplace',
             'coworker', 'staff', 'employee'):
        return "refers to workplace communication — a sign of a genuine email"
    if w in ('discuss', 'discussion', 'talk', 'chat', 'conversation',
             'call', 'mention', 'share', 'idea', 'thought', 'opinion',
             'feedback', 'suggestion', 'input', 'question'):
        return "is about having a discussion — typical in real conversations"

    # Personal / everyday words
    if w in ('home', 'house', 'family', 'friend', 'friends', 'kids',
             'children', 'mom', 'dad', 'brother', 'sister', 'wife',
             'husband', 'parents', 'baby', 'love'):
        return "is a personal word used in real-life conversations with people you know"
    if w in ('time', 'day', 'week', 'month', 'year', 'today',
             'tomorrow', 'yesterday', 'morning', 'evening', 'night',
             'afternoon', 'weekend', 'date', 'soon', 'later'):
        return "is a time reference — very common in normal everyday emails"
    if w in ('know', 'think', 'feel', 'hope', 'wish', 'want', 'need',
             'like', 'love', 'enjoy', 'remember', 'understand', 'believe',
             'sure', 'guess', 'maybe', 'probably', 'actually'):
        return "is a common everyday word people use in personal messages"
    if w in ('go', 'going', 'come', 'coming', 'get', 'got', 'make',
             'made', 'take', 'look', 'see', 'try', 'give', 'help',
             'work', 'working', 'done', 'start', 'back', 'keep'):
        return "is a simple action word found in normal daily communication"
    if w in ('food', 'eat', 'dinner', 'lunch', 'breakfast', 'restaurant',
             'cook', 'recipe', 'coffee', 'drink', 'movie', 'book',
             'music', 'game', 'play', 'trip', 'travel', 'vacation'):
        return "is about daily activities — a sign of genuine personal communication"
    if w in ('school', 'class', 'teacher', 'student', 'study',
             'learn', 'course', 'education', 'university', 'college',
             'homework', 'exam', 'test', 'grade', 'lecture'):
        return "is related to education — commonly found in legitimate emails"
    if w in ('said', 'told', 'asked', 'answer', 'replied', 'wrote',
             'sent', 'received', 'read', 'heard', 'saw', 'found'):
        return "is a communication word used in everyday storytelling and updates"

    # Generic fallback — still dynamic using the word itself
    return "our system learned this word appears more often in safe emails than in spam"


def generate_explanation(result, feature_contributions):
    """Generate a dynamic, simple explanation — easy for anyone to understand."""

    # Separate word contributions
    word_features = []
    for c in feature_contributions:
        if c['word'] != 'Base Normalcy (Intercept)':
            word_features.append(c)

    spam_words = [(c['word'], c['contribution']) for c in word_features if c['contribution'] > 0]
    ham_words = [(c['word'], abs(c['contribution'])) for c in word_features if c['contribution'] < 0]

    lines = []

    if result == "Spam":
        # ── SPAM VERDICT ──
        lines.append("<strong>\U0001f6a8 This email is SPAM.</strong>")
        lines.append("<br><br>Our system read through this email and spotted words that "
                      "are commonly found in scam, phishing, or junk emails. "
                      "Here's what raised the alarm:")

        # Show ONLY spam words — no ham words for spam emails
        if spam_words:
            lines.append("<br><br><strong>🔍 Words that flagged this email:</strong>")
            show_count = min(len(spam_words), 5)
            for i, (word, score) in enumerate(spam_words[:show_count]):
                reason = _get_spam_word_reason(word)
                lines.append(
                    f"<br>{i + 1}. \u2018<em>{word}</em>\u2019 — {reason}"
                )
            if len(spam_words) > 5:
                lines.append(f"<br><em>...and {len(spam_words) - 5} more suspicious word(s).</em>")

        # Dynamic conclusion based on actual top words
        top_words = [w for w, _ in spam_words[:3]]
        if top_words:
            mention = _join_words(top_words)
            # Build a dynamic closing paragraph
            if any(w.lower() in ('click', 'link', 'url', 'visit', 'website', 'verify',
                                  'confirm', 'password', 'login', 'account', 'suspend',
                                  'locked', 'bank', 'paypal', 'credential')
                   for w in top_words):
                context = ("This looks like a phishing attempt trying to steal your "
                           "personal information or login details.")
            elif any(w.lower() in ('free', 'win', 'winner', 'prize', 'cash', 'money',
                                    'lottery', 'bonus', 'reward', 'million', 'jackpot')
                     for w in top_words):
                context = ("This looks like a fake prize or money scam designed to "
                           "trick you into sharing personal details.")
            elif any(w.lower() in ('offer', 'deal', 'discount', 'buy', 'sale', 'cheap',
                                    'order', 'purchase', 'shop', 'subscribe', 'promotion')
                     for w in top_words):
                context = ("This looks like an unsolicited promotional or marketing email "
                           "that you didn't sign up for.")
            elif any(w.lower() in ('urgent', 'immediately', 'hurry', 'limited', 'expire',
                                    'act', 'now', 'fast', 'quick', 'asap')
                     for w in top_words):
                context = ("This email uses high-pressure language to rush you into "
                           "making a decision — a common manipulation tactic.")
            else:
                context = ("The combination of these words closely matches patterns "
                           "our system has seen in thousands of known spam emails.")

            lines.append(
                f"<br><br><strong>💡 Bottom Line:</strong> The words {mention} "
                f"are the biggest red flags here. {context} "
                f"Our system has classified this as <strong>SPAM</strong>."
            )
        else:
            lines.append(
                "<br><br><strong>💡 Bottom Line:</strong> No single word was extremely "
                "suspicious on its own, but the overall combination of words in this email "
                "closely matches patterns found in known spam messages. "
                "Our system has classified this as <strong>SPAM</strong>."
            )

    else:
        # ── HAM VERDICT ──
        lines.append("<strong>\u2705 This email is Safe.</strong>")
        lines.append("<br><br>Our system read through this email and found that its "
                      "language looks normal and trustworthy — like something a real "
                      "person would send. Here's what confirmed it as safe:")

        # Show ONLY ham words — no spam words for ham emails
        if ham_words:
            lines.append("<br><br><strong>✅ Words that confirm this email is safe:</strong>")
            show_count = min(len(ham_words), 5)
            for i, (word, score) in enumerate(ham_words[:show_count]):
                reason = _get_ham_word_reason(word)
                lines.append(
                    f"<br>{i + 1}. \u2018<em>{word}</em>\u2019 — {reason}"
                )
            if len(ham_words) > 5:
                lines.append(f"<br><em>...and {len(ham_words) - 5} more safe word(s).</em>")

        # Dynamic conclusion based on actual top words
        top_words = [w for w, _ in ham_words[:3]]
        if top_words:
            mention = _join_words(top_words)
            # Build a dynamic closing paragraph
            if any(w.lower() in ('meeting', 'schedule', 'project', 'report', 'team',
                                  'office', 'manager', 'work', 'colleague', 'company',
                                  'department', 'review', 'update', 'status', 'plan')
                   for w in top_words):
                context = ("This reads like a normal work or professional email — "
                           "nothing suspicious about it.")
            elif any(w.lower() in ('home', 'family', 'friend', 'love', 'kids',
                                    'dinner', 'movie', 'trip', 'vacation', 'weekend',
                                    'birthday', 'party', 'fun')
                     for w in top_words):
                context = ("This reads like a personal message between friends or family — "
                           "completely normal and safe.")
            elif any(w.lower() in ('thanks', 'thank', 'please', 'appreciate', 'regards',
                                    'sincerely', 'hello', 'hi', 'dear', 'welcome')
                     for w in top_words):
                context = ("The polite, conversational tone of this email is consistent "
                           "with genuine human communication.")
            else:
                context = ("The language and tone match what our system has seen in "
                           "thousands of legitimate, everyday emails.")

            lines.append(
                f"<br><br><strong>💡 Bottom Line:</strong> Words like {mention} "
                f"are signs of a genuine email. {context} "
                f"Our system has classified this as <strong>Safe (Ham)</strong>."
            )
        else:
            lines.append(
                "<br><br><strong>💡 Bottom Line:</strong> The overall language and tone of this "
                "email match normal, trustworthy communication. No suspicious patterns were "
                "found. Our system has classified this as <strong>Safe (Ham)</strong>."
            )

    return "".join(lines)


def _join_words(words):
    """Format a list of words into a readable English phrase."""
    styled = [f"\u2018<em>{w}</em>\u2019" for w in words]
    if len(styled) == 1:
        return styled[0]
    return ", ".join(styled[:-1]) + " and " + styled[-1]

@app.route('/generate_mail', methods=['POST'])
def generate_mail():
    data = request.json
    if not data:
        return jsonify({'error': 'Invalid JSON format'}), 400
        
    prompt_type = data.get('type')
    topic = data.get('topic')
    
    if not prompt_type or not topic:
         return jsonify({'error': 'Missing type or topic'}), 400
    
    prompt = f"Write a {prompt_type} email about {topic}."
    
    try:
        messages = [{"role": "user", "content": prompt}]
        response = hf_client.chat_completion(
            messages=messages, model="Qwen/Qwen2.5-72B-Instruct", max_tokens=500
        )
        return jsonify({'email': response.choices[0].message.content})
    except Exception as e:
        print(f"Hugging Face API Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/detect_mail', methods=['POST'])
def detect_mail():
    data = request.json
    if not data:
        return jsonify({'error': 'Invalid JSON format'}), 400
        
    email_text = data.get('email_text', '')
    if not email_text.strip():
        return jsonify({'error': 'Email text is required'}), 400
        
    processed_text = preprocess_text(email_text)
    nlp_summary = get_nlp_summary(email_text)
    vectorized_text = vectorizer.transform([processed_text])
    
    prediction = model.predict(vectorized_text)[0]
    result = "Spam" if prediction == 1 else "Ham"
    
    # SHAP Explanation
    feature_contributions = []
    try:
        feature_names = vectorizer.get_feature_names_out()
        coefs = model.coef_[0]
        intercept = model.intercept_[0]
        
        # Add the baseline intercept
        feature_contributions.append({
            'word': 'Base Normalcy (Intercept)',
            'contribution': float(intercept)
        })
        
        # Get non-zero features in the input
        input_features = vectorized_text.tocoo()
        
        for idx, value in zip(input_features.col, input_features.data):
            feature_contributions.append({
                'word': feature_names[idx],
                'contribution': float(coefs[idx] * value)
            })
            
        feature_contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)
    except Exception as e:
        print(f"Error computing contributions: {e}")

    # Generate dynamic, human-like explanation
    explanation = generate_explanation(result, feature_contributions)

    return jsonify({
        'result': result,
        'nlp_summary': nlp_summary,
        'contributions': feature_contributions[:20],  # Top 20 features
        'explanation': explanation
    })



if __name__ == '__main__':
    app.run(debug=True)