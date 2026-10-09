
import os
import pickle
import string
from pathlib import Path
from html import escape

import nltk
from nltk.corpus import stopwords
from dotenv import load_dotenv
from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_from_directory,
)
from huggingface_hub import InferenceClient


# ==================================================
# CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

app = Flask(
    __name__,
    template_folder=str(BASE_DIR),
    static_folder=None,
)

MODEL_PATH = BASE_DIR / "spam_model.pkl"
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"


# ==================================================
# LOAD SPAM MODEL AND VECTORIZER
# ==================================================

try:
    with MODEL_PATH.open("rb") as file:
        model = pickle.load(file)

    with VECTORIZER_PATH.open("rb") as file:
        vectorizer = pickle.load(file)

except Exception:
    app.logger.exception("Could not load spam model or vectorizer")
    raise


# ==================================================
# HUGGING FACE CONFIGURATION
# ==================================================

api_key = os.getenv("HUGGINGFACE_API_KEY")

# Can be overridden in Vercel environment variables.
HF_MODEL = os.getenv(
    "HF_MODEL",
    "Qwen/Qwen2.5-72B-Instruct",
)

hf_client = (
    InferenceClient(
        provider="auto",
        api_key=api_key,
    )
    if api_key
    else None
)


# ==================================================
# STATIC FILES
# ==================================================

@app.route("/static/<path:filename>", endpoint="static")
def serve_static(filename):
    allowed_extensions = {
        ".css", ".png", ".jpg", ".jpeg",
        ".gif", ".svg", ".webp", ".ico",
        ".woff", ".woff2", ".ttf",
    }

    requested_path = Path(filename)

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


def get_stop_words():
    try:
        return set(stopwords.words("english"))

    except LookupError:
        try:
            os.makedirs(NLTK_DATA_DIR, exist_ok=True)

            downloaded = nltk.download(
                "stopwords",
                download_dir=NLTK_DATA_DIR,
                quiet=True,
                raise_on_error=True,
            )

            if downloaded:
                return set(stopwords.words("english"))

        except Exception:
            app.logger.exception(
                "Could not load or download NLTK stopwords"
            )

        # Keep the application usable if stopwords are unavailable.
        return set()


# ==================================================
# TEXT PREPROCESSING
# ==================================================

def preprocess_text(text):
    text = text.lower()

    text = "".join(
        char
        for char in text
        if char not in string.punctuation
    )

    stop_words = get_stop_words()

    return " ".join(
        word
        for word in text.split()
        if word not in stop_words
    )


def get_nlp_summary(text):
    text_lower = text.lower()

    text_no_punct = "".join(
        char
        for char in text_lower
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
        "stopwords_removed": removed_count,
    }


# ==================================================
# WORD EXPLANATION HELPERS
# These are coefficient-based explanations, not SHAP.
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

    "urgent": "uses urgency or pressure",
    "immediately": "uses urgency or pressure",
    "hurry": "uses urgency or pressure",
    "limited": "uses urgency or pressure",
    "expire": "uses urgency or pressure",
    "deadline": "uses urgency or pressure",
    "now": "uses urgency or pressure",
    "today": "uses urgency or pressure",

    "click": "refers to clicking a link",
    "link": "refers to clicking a link",
    "url": "refers to clicking a link",
    "website": "refers to clicking a link",
    "download": "refers to downloading content",

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

    "subscribe": "is associated with marketing",
    "newsletter": "is associated with marketing",
    "promotion": "is associated with marketing",
    "marketing": "is associated with marketing",

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
    "thanks": "is often used in polite conversation",
    "thank": "is often used in polite conversation",
    "please": "is often used in polite conversation",
    "regards": "is often used in polite conversation",
    "hello": "is often used in everyday conversation",
    "hi": "is often used in everyday conversation",
    "welcome": "is often used in everyday conversation",

    "good": "is common in ordinary conversation",
    "great": "is common in ordinary conversation",
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

    "discuss": "is used in ordinary discussions",
    "conversation": "is used in ordinary discussions",
    "feedback": "is used in ordinary discussions",
    "question": "is used in ordinary discussions",

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
    "school": "is related to education",
    "student": "is related to education",
    "study": "is related to education",
    "course": "is related to education",
    "college": "is related to education",
    "exam": "is related to education",
}


def _get_spam_word_reason(word):
    return SPAM_WORD_REASONS.get(
        word.lower(),
        "the trained model assigns this word a spam-associated weight",
    )


def _get_ham_word_reason(word):
    return HAM_WORD_REASONS.get(
        word.lower(),
        "the trained model assigns this word a ham-associated weight",
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
        item
        for item in feature_contributions
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
                "<br><br><strong>🔍 Words influencing the prediction:"
                "</strong>"
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
            "<strong>✅ This email is classified as HAM (not spam)."
            "</strong>"
        )
        lines.append(
            "<br><br>The model found words that pushed its prediction "
            "away from spam. This is not a guarantee that the email "
            "is safe:"
        )

        selected_words = ham_words[:5]

        if selected_words:
            lines.append(
                "<br><br><strong>✅ Words influencing the prediction:"
                "</strong>"
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
# EMAIL GENERATION - HUGGING FACE
# ==================================================

@app.route("/generate_mail", methods=["POST"])
def generate_mail():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON format"}), 400

    prompt_type = data.get("type")
    topic = data.get("topic")

    if not isinstance(prompt_type, str) or not prompt_type.strip():
        return jsonify({"error": "Email type is required"}), 400

    if not isinstance(topic, str) or not topic.strip():
        return jsonify({"error": "Email topic is required"}), 400

    if hf_client is None:
        app.logger.error(
            "HUGGINGFACE_API_KEY is missing from the server environment"
        )
        return jsonify({
            "error": "Hugging Face API key is not configured on the server."
        }), 500

    prompt = (
        f"Write a {prompt_type.strip()} email about {topic.strip()}. "
        "Return only the email draft. Include a suitable subject line "
        "when appropriate. Do not invent specific facts or commitments."
    )

    try:
        response = hf_client.chat_completion(
            model=HF_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional email-writing assistant. "
                        "Write clear, natural, context-appropriate emails."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            max_tokens=500,
            temperature=0.7,
        )

        email = response.choices[0].message.content

        if not isinstance(email, str) or not email.strip():
            app.logger.error(
                "Hugging Face returned an empty email response"
            )
            return jsonify({
                "error": "The AI provider returned an empty response."
            }), 502

        return jsonify({"email": email.strip()}), 200

  
    except Exception as exc:
        app.logger.exception(
            "HF_GENERATION_FAILURE type=%s error=%s",
            type(exc).__name__,
            str(exc)[:1000],
        )

        return jsonify({
            "error": "Email generation failed.",
            "error_type": type(exc).__name__,
            "details": str(exc)[:500],
        }), 502, 502


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

        # Assumes the trained model uses 1 = Spam and 0 = Ham.
        # Verify this against your model's actual training labels.
        result = "Spam" if prediction == 1 else "Ham"

        feature_contributions = []

        # Logistic Regression coefficient-based explanations.
        # These are NOT actual SHAP values.
        try:
            feature_names = vectorizer.get_feature_names_out()
            coefficients = model.coef_[0]
            intercept = float(model.intercept_[0])

            feature_contributions.append({
                "word": "Base Normalcy (Intercept)",
                "contribution": intercept,
            })

            input_features = vectorized_text.tocoo()

            for index, value in zip(
                input_features.col,
                input_features.data,
            ):
                feature_contributions.append({
                    "word": str(feature_names[index]),
                    "contribution": float(
                        coefficients[index] * value
                    ),
                })

            feature_contributions.sort(
                key=lambda item: abs(item["contribution"]),
                reverse=True,
            )

        except Exception:
            app.logger.exception(
                "Error computing feature contributions"
            )

        explanation = generate_explanation(
            result,
            feature_contributions,
        )

        return jsonify({
            "result": result,
            "nlp_summary": nlp_summary,
            "contributions": feature_contributions[:20],
            "explanation": explanation,
        }), 200

    except Exception:
        app.logger.exception("Email detection failed")

        return jsonify({
            "error": (
                "Email detection failed. Check the server logs."
            )
        }), 500


# ==================================================
# LOCAL DEVELOPMENT
# ==================================================


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
        debug=False,
    )
