
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

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not found. Explanations will be disabled.")


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

# All files are in the same directory as app.py.
app = Flask(
    __name__,
    template_folder=str(BASE_DIR),
    static_folder=None
)


# --------------------------------------------------
# LOAD TRAINED MODEL AND VECTORIZER
# --------------------------------------------------

MODEL_PATH = BASE_DIR / "spam_model.pkl"
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"

with MODEL_PATH.open("rb") as f:
    model = pickle.load(f)

with VECTORIZER_PATH.open("rb") as f:
    vectorizer = pickle.load(f)


# --------------------------------------------------
# HUGGING FACE CONFIGURATION
# --------------------------------------------------

api_key = os.getenv("HUGGINGFACE_API_KEY")

hf_client = (
    InferenceClient(api_key=api_key)
    if api_key
    else None
)


# --------------------------------------------------
# NLTK STOPWORDS
# --------------------------------------------------

try:
    stopwords.words("english")
except LookupError:
    nltk_dir = "/tmp/nltk_data"
    os.makedirs(nltk_dir, exist_ok=True)
    nltk.data.path.append(nltk_dir)

    try:
        nltk.download(
            "stopwords",
            download_dir=nltk_dir,
            quiet=True
        )
    except Exception:
        app.logger.exception(
            "Could not download NLTK stopwords"
        )


def get_stop_words():
    try:
        return set(stopwords.words("english"))
    except LookupError:
        app.logger.warning(
            "NLTK stopwords unavailable; proceeding without stopword removal"
        )
        return set()


# --------------------------------------------------
# TEXT PREPROCESSING
# --------------------------------------------------

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

    removed_stopwords = [
        word for word in words
        if word in stop_words
    ]

    return {
        "original_length": len(text),
        "tokens_count": len(words),
        "stopwords_removed": len(removed_stopwords)
    }


# --------------------------------------------------
# WORD-LEVEL EXPLANATION HELPERS
# --------------------------------------------------

def _get_spam_word_reason(word):
    w = word.lower()

    if w in (
        "free", "bonus", "prize", "reward", "cash",
        "money", "dollar", "dollars", "earn", "income",
        "profit", "million", "billion", "jackpot",
        "lottery", "win", "winner", "won", "payout"
    ):
        return "promises money or rewards"

    if w in (
        "offer", "deal", "discount", "sale", "cheap",
        "bargain", "lowest", "price", "cost", "save",
        "saving", "affordable"
    ):
        return "promotes a deal or discount"

    if w in (
        "buy", "order", "purchase", "shop", "store",
        "checkout"
    ):
        return "promotes a purchase"

    if w in (
        "urgent", "immediately", "hurry", "rush", "fast",
        "quick", "limited", "expire", "expires",
        "expiring", "deadline", "asap", "instant",
        "now", "today", "act", "action", "dont", "miss",
        "last", "final", "ending"
    ):
        return "uses urgency or pressure to encourage quick action"

    if w in (
        "click", "link", "url", "visit", "website",
        "http", "www", "href", "redirect", "browse",
        "download", "install"
    ):
        return "refers to clicking a link or opening content"

    if w in (
        "verify", "confirm", "validate", "update",
        "secure", "security", "protect", "protection",
        "authentication", "password", "login", "signin",
        "credential", "credentials", "account", "suspend",
        "suspended", "locked", "unauthorized"
    ):
        return "may appear in messages requesting account or security actions"

    if w in (
        "bank", "paypal", "visa", "mastercard", "credit",
        "debit", "card", "transaction", "billing",
        "payment", "pay", "invoice"
    ):
        return "refers to banking or payment details"

    if w in (
        "subscribe", "unsubscribe", "newsletter",
        "promotion", "promotional", "advertise",
        "advertisement", "marketing", "campaign", "bulk",
        "mass", "list", "opt", "optin"
    ):
        return "is associated with bulk marketing or promotions"

    if w in (
        "guarantee", "guaranteed", "promise", "risk",
        "riskfree", "obligation", "refund", "satisfaction",
        "certified"
    ):
        return "may be used in promotional promises"

    if w in (
        "congratulations", "congrats", "selected",
        "chosen", "exclusive", "special", "vip",
        "member", "membership"
    ):
        return "uses prize, selection, or exclusivity language"

    if w in (
        "pill", "pills", "medication", "pharmacy", "drug",
        "drugs", "prescription", "viagra", "weight", "diet",
        "supplement", "health", "cure", "treatment",
        "doctor", "medical"
    ):
        return "appears in health-related promotional content"

    if w in (
        "call", "contact", "reply", "respond", "send",
        "forward", "txt", "text", "sms", "mobile",
        "phone", "number"
    ):
        return "requests a response or contact"

    if w in (
        "information", "info", "personal", "details",
        "data", "name", "address", "social", "ssn"
    ):
        return "refers to personal information"

    if w in (
        "access", "unlock", "open", "enable", "activate",
        "registration", "register", "signup", "sign"
    ):
        return "refers to activating or signing up for something"

    if w in (
        "claim", "collect", "redeem", "receive", "apply",
        "request", "submit", "fill", "form"
    ):
        return "asks the reader to claim or submit something"

    if w in (
        "customer", "service", "support", "team",
        "representative", "agent", "department",
        "helpdesk", "admin"
    ):
        return "refers to customer service or support"

    return "the trained model assigns this word a spam-associated weight"


def _get_ham_word_reason(word):
    w = word.lower()

    if w in (
        "thanks", "thank", "thankyou", "appreciate",
        "grateful", "gratitude", "cheers", "regards",
        "sincerely", "kindly", "please", "welcome",
        "hi", "hello", "hey", "dear", "greetings"
    ):
        return "is often used in polite, everyday conversation"

    if w in (
        "good", "great", "nice", "wonderful", "awesome",
        "excellent", "amazing", "fantastic", "brilliant",
        "lovely", "fine", "well"
    ):
        return "is common in ordinary conversational language"

    if w in (
        "meeting", "schedule", "agenda", "project",
        "report", "review", "update", "status", "progress",
        "task", "plan", "planning", "deadline", "milestone",
        "deliverable"
    ):
        return "is often used in work-related communication"

    if w in (
        "team", "colleague", "manager", "boss", "office",
        "department", "company", "organization",
        "workplace", "coworker", "staff", "employee"
    ):
        return "refers to workplace communication"

    if w in (
        "discuss", "discussion", "talk", "chat",
        "conversation", "call", "mention", "share", "idea",
        "thought", "opinion", "feedback", "suggestion",
        "input", "question"
    ):
        return "is used in ordinary discussions and exchanges"

    if w in (
        "home", "house", "family", "friend", "friends",
        "kids", "children", "mom", "dad", "brother",
        "sister", "wife", "husband", "parents", "baby", "love"
    ):
        return "is often used in personal communication"

    if w in (
        "time", "day", "week", "month", "year", "today",
        "tomorrow", "yesterday", "morning", "evening",
        "night", "afternoon", "weekend", "date", "soon", "later"
    ):
        return "is a common time reference"

    if w in (
        "know", "think", "feel", "hope", "wish", "want",
        "need", "like", "love", "enjoy", "remember",
        "understand", "believe", "sure", "guess", "maybe",
        "probably", "actually"
    ):
        return "is common in everyday messages"

    if w in (
        "go", "going", "come", "coming", "get", "got",
        "make", "made", "take", "look", "see", "try",
        "give", "help", "work", "working", "done",
        "start", "back", "keep"
    ):
        return "is a common action word"

    if w in (
        "food", "eat", "dinner", "lunch", "breakfast",
        "restaurant", "cook", "recipe", "coffee", "drink",
        "movie", "book", "music", "game", "play", "trip",
        "travel", "vacation"
    ):
        return "is often used when discussing everyday activities"

    if w in (
        "school", "class", "teacher", "student", "study",
        "learn", "course", "education", "university",
        "college", "homework", "exam", "test", "grade", "lecture"
    ):
        return "is related to education or learning"

    if w in (
        "said", "told", "asked", "answer", "replied",
        "wrote", "sent", "received", "read", "heard",
        "saw", "found"
    ):
        return "is used in ordinary communication or storytelling"

    return "the trained model assigns this word a ham-associated weight"


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


# --------------------------------------------------
# ROOT-LEVEL STATIC FILES
# --------------------------------------------------

@app.route("/<path:filename>")
def root_assets(filename):
    allowed_extensions = {
        ".css", ".js", ".png", ".jpg", ".jpeg",
        ".gif", ".svg", ".webp", ".ico",
        ".woff", ".woff2", ".ttf"
    }

    requested_path = Path(filename)

    # This no-folders setup serves assets from the repository root only.
    if (
        requested_path.name != filename
        or requested_path.suffix.lower() not in allowed_extensions
    ):
        return jsonify({"error": "Not found"}), 404

    file_path = BASE_DIR / filename

    if not file_path.is_file():
        return jsonify({"error": "Not found"}), 404

    return send_from_directory(str(BASE_DIR), filename)


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# --------------------------------------------------
# EMAIL GENERATION
# --------------------------------------------------

@app.route("/generate_mail", methods=["POST"])
def generate_mail():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({
            "error": "Invalid JSON format"
        }), 400

    prompt_type = data.get("type")
    topic = data.get("topic")

    if not prompt_type or not topic:
        return jsonify({
            "error": "Missing type or topic"
        }), 400

    if hf_client is None:
        return jsonify({
            "error": "Hugging Face API key is not configured on the server."
        }), 500

    prompt = f"Write a {prompt_type} email about {topic}."

    try:
        response = hf_client.chat_completion(
            messages=[
                {"role": "user", "content": prompt}
            ],
            model="Qwen/Qwen2.5-72B-Instruct",
            max_tokens=500
        )

        return jsonify({
            "email": response.choices[0].message.content
        })

    except Exception:
        app.logger.exception("Hugging Face API error")

        return jsonify({
            "error": (
                "Email generation failed. Check the Vercel logs "
                "and verify your Hugging Face API key and model access."
            )
        }), 502


# --------------------------------------------------
# SPAM DETECTION
# --------------------------------------------------

@app.route("/detect_mail", methods=["POST"])
def detect_mail():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({
            "error": "Invalid JSON format"
        }), 400

    email_text = data.get("email_text", "")

    if not isinstance(email_text, str) or not email_text.strip():
        return jsonify({
            "error": "Email text is required"
        }), 400

    try:
        processed_text = preprocess_text(email_text)
        nlp_summary = get_nlp_summary(email_text)

        vectorized_text = vectorizer.transform([
            processed_text
        ])

        prediction = model.predict(vectorized_text)[0]

        # Preserves your original label mapping:
        # 1 = Spam, 0 = Ham.
        result = "Spam" if prediction == 1 else "Ham"

        feature_contributions = []

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
            "error": (
                "Email detection failed. Check the Vercel function logs."
            )
        }), 500


# --------------------------------------------------
# LOCAL DEVELOPMENT
# --------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
