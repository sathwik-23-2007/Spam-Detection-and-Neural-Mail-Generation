
import os
import pickle
import string
from pathlib import Path
from html import escape

import nltk
from nltk.corpus import stopwords
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, send_from_directory
from huggingface_hub import InferenceClient

# Optional SHAP import. The explanations below use model coefficients,
# not actual SHAP values.
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


# ==================================================
# CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# HTML and CSS are in the same directory as app.py.
# Disable Flask's default static route and define a safe one below.
app = Flask(
    __name__,
    template_folder=str(BASE_DIR),
    static_folder=None,
)


# ==================================================
# LOAD MODEL AND VECTORIZER
# ==================================================

MODEL_PATH = BASE_DIR / "spam_model.pkl"
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"

with MODEL_PATH.open("rb") as file:
    model = pickle.load(file)

with VECTORIZER_PATH.open("rb") as file:
    vectorizer = pickle.load(file)


# ==================================================
# HUGGING FACE API
# ==================================================

api_key = os.getenv("HUGGINGFACE_API_KEY")

hf_client = InferenceClient(api_key=api_key) if api_key else None


# ==================================================
# STATIC FILES
# ==================================================

@app.route("/static/<path:filename>", endpoint="static")
def serve_static(filename):
    allowed_extensions = {
        ".css", ".png", ".jpg", ".jpeg",
        ".gif", ".svg", ".webp", ".ico",
        ".woff", ".woff2", ".ttf"
    }

    requested_path = Path(filename)

    # Only serve root-level public assets, never Python/model/env files.
    if (
        requested_path.name != filename
        or requested_path.suffix.lower() not in allowed_extensions
        or filename.startswith(".")
    ):
        return jsonify({"error": "Not found"}), 404

    file_path = BASE_DIR / filename

    if not file_path.is_file():
        return jsonify({"error": "Not found"}), 404

    return send_from_directory(str(BASE_DIR), filename)


# ==================================================
# NLTK STOPWORDS
# ==================================================

NLTK_DATA_DIR = "/tmp/nltk_data"

if NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.append(NLTK_DATA_DIR)

try:
    stopwords.words("english")
except LookupError:
    try:
        os.makedirs(NLTK_DATA_DIR, exist_ok=True)
        nltk.download(
            "stopwords",
            download_dir=NLTK_DATA_DIR,
            quiet=True
        )
    except Exception:
        app.logger.exception("Could not download NLTK stopwords")


def get_stop_words():
    try:
        return set(stopwords.words("english"))
    except LookupError:
        app.logger.warning(
            "NLTK stopwords unavailable; proceeding without stopword removal"
        )
        return set()


# ==================================================
# TEXT PREPROCESSING
# ==================================================

def preprocess_text(text):
    text = text.lower()

    text = "".join(
        char for char in text
        if char not in string.punctuation
    )

    stop_words = get_stop_words()

    return " ".join(
        word for word in text.split()
        if word not in stop_words
    )


def get_nlp_summary(text):
    text_lower = text.lower()

    text_no_punct = "".join(
        char for char in text_lower
        if char not in string.punctuation
    )

    words = text_no_punct.split()
    stop_words = get_stop_words()

    removed_count = sum(
        1 for word in words if word in stop_words
    )

    return {
        "original_length": len(text),
        "tokens_count": len(words),
        "stopwords_removed": removed_count
    }


# ==================================================
# WORD EXPLANATION HELPERS
# ==================================================

SPAM_WORD_REASONS = {
    "free": "promises money or rewards",
    "bonus": "promises money or rewards",
    "prize": "promises money or rewards",
    "reward": "promises money or rewards",
    "cash": "promises money or rewards",
    "money": "promises money or rewards",
    "earn": "promises money or rewards",
    "income": "promises money or rewards",
    "profit": "promises money or rewards",
    "million": "promises money or rewards",
    "jackpot": "promises money or rewards",
    "lottery": "promises money or rewards",
    "win": "promises money or rewards",
    "winner": "promises money or rewards",
    "payout": "promises money or rewards",

    "offer": "promotes a deal or discount",
    "deal": "promotes a deal or discount",
    "discount": "promotes a deal or discount",
    "sale": "promotes a deal or discount",
    "cheap": "promotes a deal or discount",
    "buy": "promotes a purchase",
    "order": "promotes a purchase",
    "purchase": "promotes a purchase",
    "shop": "promotes a purchase",

    "urgent": "uses urgency or pressure to encourage quick action",
    "immediately": "uses urgency or pressure to encourage quick action",
    "hurry": "uses urgency or pressure to encourage quick action",
    "limited": "uses urgency or pressure to encourage quick action",
    "expire": "uses urgency or pressure to encourage quick action",
    "deadline": "uses urgency or pressure to encourage quick action",
    "now": "uses urgency or pressure to encourage quick action",
    "today": "uses urgency or pressure to encourage quick action",

    "click": "refers to clicking a link or opening content",
    "link": "refers to clicking a link or opening content",
    "url": "refers to clicking a link or opening content",
    "website": "refers to clicking a link or opening content",
    "download": "refers to clicking a link or opening content",

    "verify": "may relate to account or security requests",
    "confirm": "may relate to account or security requests",
    "password": "may relate to account or security requests",
    "login": "may relate to account or security requests",
    "account": "may relate to account or security requests",
    "suspended": "may relate to account or security requests",
    "locked": "may relate to account or security requests",

    "bank": "refers to banking or payment details",
    "paypal": "refers to banking or payment details",
    "credit": "refers to banking or payment details",
    "debit": "refers to banking or payment details",
    "payment": "refers to banking or payment details",
    "invoice": "refers to banking or payment details",

    "subscribe": "is associated with bulk marketing or promotions",
    "newsletter": "is associated with bulk marketing or promotions",
    "promotion": "is associated with bulk marketing or promotions",
    "marketing": "is associated with bulk marketing or promotions",

    "guarantee": "may be used in promotional promises",
    "refund": "may be used in promotional promises",
    "congratulations": "uses prize or exclusivity language",
    "selected": "uses prize or exclusivity language",
    "exclusive": "uses prize or exclusivity language",

    "personal": "refers to personal information",
    "information": "refers to personal information",
    "address": "refers to personal information",
    "claim": "asks the reader to claim something",
    "redeem": "asks the reader to claim something",
    "submit": "asks the reader to submit something",
    "support": "refers to customer service or support",
}


HAM_WORD_REASONS = {
    "thanks": "is often used in polite, everyday conversation",
    "thank": "is often used in polite, everyday conversation",
    "please": "is often used in polite, everyday conversation",
    "regards": "is often used in polite, everyday conversation",
    "hello": "is often used in polite, everyday conversation",
    "hi": "is often used in polite, everyday conversation",
    "welcome": "is often used in polite, everyday conversation",

    "good": "is common in ordinary conversational language",
    "great": "is common in ordinary conversational language",
    "meeting": "is often used in work-related communication",
    "schedule": "is often used in work-related communication",
    "project": "is often used in work-related communication",
    "report": "is often used in work-related communication",
    "review": "is often used in work-related communication",
    "progress": "is often used in work-related communication",
    "task": "is often used in work-related communication",
    "team": "refers to workplace communication",
    "manager": "refers to workplace communication",
    "office": "refers to workplace communication",
    "company": "refers to workplace communication",

    "discuss": "is used in ordinary discussions and exchanges",
    "conversation": "is used in ordinary discussions and exchanges",
    "feedback": "is used in ordinary discussions and exchanges",
    "question": "is used in ordinary discussions and exchanges",

    "home": "is often used in personal communication",
    "family": "is often used in personal communication",
    "friend": "is often used in personal communication",
    "friends": "is often used in personal communication",
    "parents": "is often used in personal communication",

    "time": "is a common time reference",
    "day": "is a common time reference",
    "week": "is a common time reference",
    "tomorrow": "is a common time reference",
    "morning": "is a common time reference",
    "weekend": "is a common time reference",

    "think": "is common in everyday messages",
    "hope": "is common in everyday messages",
    "want": "is common in everyday messages",
    "need": "is common in everyday messages",
    "understand": "is common in everyday messages",

    "help": "is a common action word",
    "work": "is a common action word",
    "start": "is a common action word",
    "school": "is related to education or learning",
    "student": "is related to education or learning",
    "study": "is related to education or learning",
    "course": "is related to education or learning",
    "college": "is related to education or learning",
    "exam": "is related to education or learning",
}


def _get_spam_word_reason(word):
    return SPAM_WORD_REASONS.get(
        word.lower(),
        "the trained model assigns this word a spam-associated weight"
    )


def _get_ham_word_reason(word):
    return HAM_WORD_REASONS.get(
        word.lower(),
        "the trained model assigns this word a ham-associated weight"
    )


def _join_words(words):
    styled = [
        f"‘<em>{escape(str(word))}</em>’"
        for word in words
    ]

    if len(styled) == 1:
        return styled[0]

    return ", ".join(styled[:-1]) + " and " + styled[-1]


def generate_explanation(result, feature_contributions):
    word_features = [
        item for item in feature_contributions
        if item["word"] != "Base Normalcy (Intercept)"
    ]

    spam_words = [
        (item["word"], item["contribution"])
        for item in word_features
        if item["contribution"] > 0
    ]

    ham_words = [
        (item["word"], abs(item["contribution"]))
        for item in word_features
        if item["contribution"] < 0
    ]

    lines = []

    if result == "Spam":
        lines.append(
            "<strong>🚨 This email is classified as SPAM.</strong>"
        )
        lines.append(
            "<br><br>The model found words that push its prediction "
            "toward spam. These signals are not proof that an email "
            "is malicious:"
        )

        selected_words = spam_words[:5]

        if selected_words:
            lines.append(
                "<br><br><strong>🔍 Words influencing the prediction:</strong>"
            )

            for index, (word, _score) in enumerate(
                selected_words, start=1
            ):
                lines.append(
                    f"<br>{index}. ‘<em>{escape(str(word))}</em>’ — "
                    f"{_get_spam_word_reason(word)}"
                )

            if len(spam_words) > 5:
                lines.append(
                    f"<br><em>...and {len(spam_words) - 5} "
                    "more contributing word(s).</em>"
                )

            top_words = [
                word for word, _score in selected_words[:3]
            ]

            lines.append(
                "<br><br><strong>💡 Bottom line:</strong> "
                f"The words {_join_words(top_words)} contributed "
                "toward the spam prediction. Consider the full "
                "message and sender before deciding."
            )
        else:
            lines.append(
                "<br><br><strong>💡 Bottom line:</strong> The model "
                "classified this message as spam based on its overall "
                "learned pattern."
            )

    else:
        lines.append(
            "<strong>✅ This email is classified as HAM (not spam).</strong>"
        )
        lines.append(
            "<br><br>The model found words that pushed its prediction "
            "away from spam. This is not a guarantee that the email "
            "is safe:"
        )

        selected_words = ham_words[:5]

        if selected_words:
            lines.append(
                "<br><br><strong>✅ Words influencing the prediction:</strong>"
            )

            for index, (word, _score) in enumerate(
                selected_words, start=1
            ):
                lines.append(
                    f"<br>{index}. ‘<em>{escape(str(word))}</em>’ — "
                    f"{_get_ham_word_reason(word)}"
                )

            if len(ham_words) > 5:
                lines.append(
                    f"<br><em>...and {len(ham_words) - 5} "
                    "more contributing word(s).</em>"
                )

            top_words = [
                word for word, _score in selected_words[:3]
            ]

            lines.append(
                "<br><br><strong>💡 Bottom line:</strong> "
                f"The words {_join_words(top_words)} contributed "
                "toward the non-spam prediction. Be cautious with "
                "unexpected links and requests for sensitive information."
            )
        else:
            lines.append(
                "<br><br><strong>💡 Bottom line:</strong> The model "
                "classified this message as ham based on its overall "
                "learned pattern."
            )

    return "".join(lines)


# ==================================================
# HOME PAGE
# ==================================================

@app.route("/")
def home():
    return render_template("index.html")


# ==================================================
# EMAIL GENERATION
# ==================================================

@app.route("/generate_mail", methods=["POST"])
def generate_mail():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON format"}), 400

    prompt_type = data.get("type")
    topic = data.get("topic")

    if not isinstance(prompt_type, str) or not prompt_type.strip():
        return jsonify({"error": "Missing type or topic"}), 400

    if not isinstance(topic, str) or not topic.strip():
        return jsonify({"error": "Missing type or topic"}), 400

    if hf_client is None:
        return jsonify({
            "error": "Hugging Face API key is not configured on the server."
        }), 500

    prompt = (
        f"Write a {prompt_type.strip()} email about {topic.strip()}. "
        "Return only the email draft."
    )

    try:
        response = hf_client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model="Qwen/Qwen2.5-72B-Instruct",
            max_tokens=500,
        )

        return jsonify({
            "email": response.choices[0].message.content or ""
        })

    except Exception:
        app.logger.exception("Hugging Face API error")

        return jsonify({
            "error": (
                "Email generation failed. Verify the Hugging Face API key, "
                "model availability, and Vercel logs."
            )
        }), 502


# ==================================================
# SPAM DETECTION
# ==================================================

@app.route("/detect_mail", methods=["POST"])
def detect_mail():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON format"}), 400

    email_text = data.get("email_text", "")

    if not isinstance(email_text, str) or not email_text.strip():
        return jsonify({"error": "Email text is required"}), 400

    try:
        processed_text = preprocess_text(email_text)
        nlp_summary = get_nlp_summary(email_text)

        vectorized_text = vectorizer.transform([processed_text])
        prediction = model.predict(vectorized_text)[0]

        # This assumes the model uses 1 = Spam and 0 = Ham.
        # Confirm these labels match the model's training labels.
        result = "Spam" if prediction == 1 else "Ham"

        feature_contributions = []

        # Coefficient-based contributions for Logistic Regression.
        # These are not SHAP values.
        try:
            feature_names = vectorizer.get_feature_names_out()
            coefficients = model.coef_[0]
            intercept = float(model.intercept_[0])

            feature_contributions.append({
                "word": "Base Normalcy (Intercept)",
                "contribution": intercept
            })

            input_features = vectorized_text.tocoo()

            for index, value in zip(
                input_features.col,
                input_features.data
            ):
                feature_contributions.append({
                    "word": str(feature_names[index]),
                    "contribution": float(
                        coefficients[index] * value
                    )
                })

            feature_contributions.sort(
                key=lambda item: abs(item["contribution"]),
                reverse=True
            )

        except Exception:
            app.logger.exception(
                "Error computing feature contributions"
            )

        explanation = generate_explanation(
            result,
            feature_contributions
        )

        return jsonify({
            "result": result,
            "nlp_summary": nlp_summary,
            "contributions": feature_contributions[:20],
            "explanation": explanation
        })

    except Exception:
        app.logger.exception("Email detection failed")

        return jsonify({
            "error": "Email detection failed. Check the Vercel function logs."
        }), 500


# ==================================================
# LOCAL DEVELOPMENT
# ==================================================

if __name__ == "__main__":
    app.run(debug=True)
