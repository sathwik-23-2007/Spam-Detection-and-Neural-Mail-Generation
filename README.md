# 📧 Inbox AI — Detect & Compose

An intelligent email application that combines **Spam/Ham email classification, Explainable AI using SHAP, and AI-powered email generation** into a single Flask-based platform.

The system does not simply tell the user whether an email is **Spam or Ham**. It also explains **why the machine learning model made that prediction**, making the detection process more transparent and understandable.

---

## 🚀 Project Overview

Email users commonly face two problems:

1. Identifying unwanted or suspicious emails.
2. Writing professional email responses quickly.

**Inbox AI — Detect & Compose** addresses both problems in one application.

The project contains two major intelligent modules:

### 🔐 Spam Detection

A machine learning pipeline analyzes an email and predicts whether it is:

* **Spam**
* **Ham**

The project uses **Logistic Regression** for classification.

### 🧠 Explainable AI with SHAP

The major distinguishing feature of this project is **Explainable AI (XAI)**.

Instead of providing only:

> `Prediction: Spam`

the system also explains the factors that contributed to that prediction using **SHAP (SHapley Additive exPlanations)**.

This helps the user understand **why the email was classified as Spam or Ham**.

### ✍️ AI Email Generation

The application also provides an AI-powered email generation feature using the **Hugging Face API**, allowing users to generate context-aware and professional email content.

---

## ⭐ What Makes This Project Different?

Most basic spam detection projects follow:

```text
Email
  ↓
Machine Learning Model
  ↓
Spam / Ham
```

This project goes one step further:

```text
                 ┌──────────────────┐
                 │   Email Input    │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │  Preprocessing   │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │ Logistic         │
                 │ Regression       │
                 └────────┬─────────┘
                          ↓
                  ┌───────────────┐
                  │  Spam / Ham   │
                  └───────┬───────┘
                          ↓
                 ┌──────────────────┐
                 │ SHAP Explanation │
                 └────────┬─────────┘
                          ↓
              Why Spam? / Why Ham?
```

The user therefore gets both:

**Prediction + Explanation**

rather than a black-box prediction.

---

# ✨ Key Features

## 🔴 1. Spam/Ham Email Detection

The application uses a supervised machine learning model to classify emails as:

* **Spam**
* **Ham**

The classification pipeline includes data preprocessing and machine learning using Python and Scikit-learn.

---

## 🧠 2. Explainable AI using SHAP

SHAP is one of the core features of this project.

The system uses SHAP to provide an explanation behind the model's prediction.

For example:

```text
Prediction: SPAM

Important contributing features:
• suspicious wording
• promotional language
• unusual email patterns
• other learned text features
```

For a Ham email, the explanation can show the features that contributed toward the Ham prediction.

### Why SHAP?

Traditional machine learning systems can behave like a black box:

```text
Email → Model → Spam
```

SHAP makes the process more transparent:

```text
Email
  ↓
Model
  ↓
Spam
  ↓
SHAP
  ↓
Why the model predicted Spam
```

This makes the project more useful for **trust, transparency, debugging, and understanding model behavior**.

---

## ✍️ 3. AI-Powered Email Generation

The application integrates the **Hugging Face API** to generate email content based on the user's requirements.

The generated content is intended to be:

* Context-aware
* Grammatically appropriate
* Professional
* Editable by the user

Basic workflow:

```text
User Prompt
     ↓
Flask Backend
     ↓
Hugging Face API
     ↓
Generated Email
     ↓
User Reviews / Edits
```

---

## 🔐 4. User Authentication

The application includes user authentication functionality so users can access their email-related features securely.

The planned application architecture includes:

* Registration
* Login
* Logout
* User profile
* Email-related functionality

---

## 🎨 5. Responsive Web Interface

The project provides a web-based interface designed to make the spam detection and email generation features accessible through a single application.

Frontend technologies include:

* HTML
* CSS
* Bootstrap

Backend:

* Flask
* Python

---

# 🗂️ Dataset

The spam detection model was trained using a dataset containing:

* **6,570 rows**
* **2 columns**

The dataset is used to train the machine learning model to distinguish between **Spam and Ham** emails.

> Note: The exact dataset column names are intentionally not assumed here.

---

# 🤖 Machine Learning Pipeline

The spam detection component follows a typical machine learning workflow:

```text
Raw Email Dataset
       ↓
Data Loading
       ↓
Data Preprocessing
       ↓
Feature Preparation
       ↓
Train Machine Learning Model
       ↓
Logistic Regression
       ↓
Spam / Ham Prediction
       ↓
SHAP Explanation
```

### Model Used

**Logistic Regression**

Logistic Regression is used as the classification algorithm for predicting whether an email belongs to the Spam or Ham class.

---

# 🔍 Explainable AI Pipeline

The most important additional layer is the SHAP explanation.

```text
                    Email
                      ↓
              Text Preprocessing
                      ↓
              Feature Extraction
                      ↓
             Logistic Regression
                      ↓
              Spam / Ham Output
                      ↓
                    SHAP
                      ↓
          Feature Contribution Analysis
                      ↓
          Human-Understandable Explanation
```

SHAP helps identify which model features contributed toward the prediction and in which direction.

This makes it possible to investigate the model's decision rather than simply accepting the prediction.

---

# 🏗️ System Architecture

```text
                    ┌─────────────────┐
                    │      User       │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │  Web Interface  │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Flask Backend   │
                    └───────┬─┬───────┘
                            │ │
              ┌─────────────┘ └─────────────┐
              ↓                             ↓
    ┌───────────────────┐        ┌──────────────────┐
    │ Spam Detection    │        │ Email Generation │
    │                   │        │                  │
    │ Logistic          │        │ Hugging Face API │
    │ Regression        │        └────────┬─────────┘
    └─────────┬─────────┘                 ↓
              ↓                    Generated Email
        Spam / Ham
              ↓
        ┌───────────┐
        │   SHAP    │
        │ Explanation│
        └─────┬─────┘
              ↓
       Why Spam / Ham?
```

---

# 🛠️ Technology Stack

### Programming Language

* Python

### Backend

* Flask
* Flask-SQLAlchemy
* SQLAlchemy

### Machine Learning

* Scikit-learn
* Logistic Regression

### Explainable AI

* SHAP

### Natural Language Processing

* NLTK
* Transformers

### Generative AI

* Hugging Face API

### Database

* SQLite

### Frontend

* HTML
* CSS
* Bootstrap 5
* Jinja2

### Development Environment

* Visual Studio Code
* Windows

---

# 📁 Project Structure

A representative structure for the application is:

```text
Inbox-AI/
│
├── app.py
│
├── model/
│   ├── spam_model
│   └── preprocessing
│
├── templates/
│   ├── login.html
│   ├── register.html
│   ├── dashboard.html
│   ├── spam_detection.html
│   └── compose.html
│
├── static/
│   ├── css/
│   ├── js/
│   └── images/
│
├── dataset/
│   └── spam_dataset
│
├── requirements.txt
│
└── README.md
```

> The exact filenames and folder structure may differ depending on the final project implementation.

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone <your-repository-url>
```

```bash
cd Inbox-AI
```

## 2. Create a Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Configure API Credentials

Add the required Hugging Face API credentials using your preferred environment-variable configuration.

Do not commit API keys directly to GitHub.

## 5. Run the Application

```bash
python app.py
```

Open the local Flask application in your browser.

---

# 🧪 How the Spam Detection Works

### Step 1 — Email Input

The user enters or provides an email.

### Step 2 — Preprocessing

The email is processed so that it can be passed to the machine learning pipeline.

### Step 3 — Prediction

The processed email is passed to the Logistic Regression classifier.

### Step 4 — Classification

The model produces one of two classes:

```text
SPAM
```

or

```text
HAM
```

### Step 5 — Explanation

SHAP analyzes the model prediction and provides feature-level explanations.

Therefore, the final output is not just:

```text
Spam
```

but conceptually:

```text
Prediction: Spam

Explanation:
These features contributed to the prediction.
```

---

# 📊 Why Explainability Matters

A machine learning model can achieve good predictive performance while still being difficult for users to understand.

For an email security system, simply displaying:

```text
❌ SPAM
```

does not tell the user why the system reached that conclusion.

With SHAP:

```text
❌ SPAM

Why?
↓
Feature contributions
↓
Model explanation
```

This improves the transparency of the system and can also help developers investigate incorrect predictions.

---

# 🔄 Complete Application Workflow

```text
                 USER
                  │
        ┌─────────┴─────────┐
        ↓                   ↓
  Spam Detection       Email Generation
        │                   │
        ↓                   ↓
 Preprocessing        User Prompt
        │                   │
        ↓                   ↓
 Logistic Regression  Hugging Face API
        │                   │
        ↓                   ↓
   Spam / Ham          Generated Email
        │                   │
        ↓                   ↓
       SHAP             Edit / Review
        │
        ↓
 Why Spam / Ham?
```

---

# 🎯 Project Objectives

The main objectives of Inbox AI are:

* Detect unwanted emails using machine learning.
* Classify emails into **Spam or Ham**.
* Provide explanations for predictions using **SHAP**.
* Reduce the black-box nature of machine learning predictions.
* Generate professional emails using generative AI.
* Provide authentication and user management.
* Provide a responsive and accessible web interface.
* Combine traditional machine learning and generative AI in one application.

---

# 💡 Core Innovation

The core idea of the project can be represented as:

```text
                 TRADITIONAL
               SPAM DETECTION
                     │
                     ↓
              Spam / Ham
                     │
                     +
                     │
                     ↓
             EXPLAINABLE AI
                  (SHAP)
                     │
                     ↓
              WHY THE MODEL
               DECIDED THIS
                     │
                     +
                     │
                     ↓
              GENERATIVE AI
                     │
                     ↓
             EMAIL COMPOSITION
```

Instead of building only another spam classifier, the project combines:

**Machine Learning + Explainable AI + Generative AI + Web Application**

---

# 🔮 Future Enhancements

Potential future improvements include:

* Integration with Gmail and Outlook.
* Real-time inbox scanning.
* More advanced NLP-based spam detection.
* Improved SHAP visualization.
* User-specific email preferences.
* Better email generation controls.
* Advanced phishing detection.
* Model performance monitoring.
* Larger and more diverse email datasets.
* Deployment as a cloud-based application.

---

# 📚 Project Domain

**Domain:** Web Development + Machine Learning + Natural Language Processing + Generative AI + Explainable AI