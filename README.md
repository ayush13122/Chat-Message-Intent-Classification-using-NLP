# 💬 Chat Message Intent Classification using NLP

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Scikit--learn-1.3%2B-orange?logo=scikit-learn&logoColor=white" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/NLP-NLTK-green?logo=natural-language-processing" alt="NLP">
  <img src="https://img.shields.io/badge/Model-Logistic%20Regression-blueviolet" alt="Model">
  <img src="https://img.shields.io/badge/Accuracy-72.22%25-success" alt="Accuracy">
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" alt="License">
</p>

<p align="center">
  A complete NLP-based Machine Learning project that classifies customer/chat support messages
  into <strong>Support</strong>, <strong>Complaint</strong>, or <strong>Query</strong> using
  TF-IDF vectorization and Logistic Regression.
</p>

---

## 📌 Project Overview

In real-world customer support systems, automatically categorizing incoming messages helps
route them to the right teams, prioritize urgent complaints, and improve response times.

This project builds an end-to-end **text classification pipeline** that:
- Preprocesses raw chat messages using NLP techniques
- Converts text to numerical features using **TF-IDF**
- Trains a **Logistic Regression** classifier
- Evaluates with accuracy, confusion matrix, and classification report
- Provides a **live prediction system** for new messages
- Saves model artifacts for deployment

**Intent Categories:**

| Label      | Description                              | Example                                      |
|------------|------------------------------------------|----------------------------------------------|
| 🛠️ Support  | User needs help or assistance            | "I need help resetting my password"          |
| 😠 Complaint | User is expressing dissatisfaction       | "This service is absolutely terrible!"       |
| ❓ Query    | User is asking for information           | "What are the available subscription plans?" |

---

## 🧠 ML Workflow

```
Raw CSV Dataset
      │
      ▼
 Text Preprocessing (lowercase → punctuation removal → stopwords → tokenize)
      │
      ▼
 TF-IDF Vectorization (unigrams + bigrams, top 5000 features)
      │
      ▼
 Train-Test Split (80% train / 20% test, stratified)
      │
      ▼
 Logistic Regression Training
      │
      ▼
 Evaluation (Accuracy · Confusion Matrix · Classification Report)
      │
      ▼
 Save Model (.pkl) + Vectorizer (.pkl)
      │
      ▼
 Prediction System (CLI / importable module)
```

---

## 📊 Algorithms & Techniques

| Component           | Choice                          | Why                                                   |
|---------------------|----------------------------------|-------------------------------------------------------|
| Text Preprocessing  | NLTK                             | Industry-standard NLP library, robust tokenization    |
| Vectorization       | TF-IDF (1-gram + 2-gram)        | Captures word importance + phrase context             |
| Classifier          | Logistic Regression              | Fast, interpretable, strong on text classification    |
| Evaluation          | Accuracy + Confusion Matrix      | Full picture of per-class and overall performance     |
| Persistence         | Joblib                           | Efficient binary serialization for sklearn objects    |

---

## 📈 Results

| Metric             | Score   |
|--------------------|---------|
| Test Accuracy      | 72.22%  |
| Support F1-score   | 0.83    |
| Complaint F1-score | 0.67    |
| Query F1-score     | 0.67    |
| Macro Avg F1       | 0.72    |

**Classification Report:**
```
              precision    recall  f1-score   support

   complaint       0.67      0.67      0.67         6
       query       0.67      0.67      0.67         6
     support       0.83      0.83      0.83         6

    accuracy                           0.72        18
   macro avg       0.72      0.72      0.72        18
weighted avg       0.72      0.72      0.72        18
```

---

## 🖼️ Screenshots

### Confusion Matrix
> Located at `outputs/confusion_matrix.png`

![Confusion Matrix](outputs/confusion_matrix.png)

### Sample Predictions
```
  ┌─────────────────────────────────────────────────┐
  │  Message    : I need help resetting my account password
  │  Prediction : 🛠️  Support
  │  Confidence : 68.2%
  │  Probabilities:
  │    🛠️  Support        :  68.2% █████████████
  │    ❓ Query            :  16.4% ███
  │    😠 Complaint        :  15.4% ███
  └─────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────┐
  │  Message    : This is absolutely ridiculous, worst service!
  │  Prediction : 😠 Complaint
  │  Confidence : 50.7%
  │  Probabilities:
  │    😠 Complaint        :  50.7% ██████████
  │    ❓ Query            :  27.0% █████
  │    🛠️  Support        :  22.3% ████
  └─────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────┐
  │  Message    : What payment methods do you accept?
  │  Prediction : ❓ Query
  │  Confidence : 45.1%
  │  Probabilities:
  │    ❓ Query            :  45.1% █████████
  │    😠 Complaint        :  28.0% █████
  │    🛠️  Support        :  26.9% █████
  └─────────────────────────────────────────────────┘
```

---

## 📁 Folder Structure

```
chat-message-intent-classification/
│
├── data/
│   └── intents_dataset.csv         ← 90 labelled chat messages (balanced)
│
├── src/
│   ├── preprocess.py               ← NLP preprocessing functions
│   ├── train_model.py              ← Full training pipeline
│   └── predict.py                  ← Prediction module (CLI + importable)
│
├── models/
│   ├── intent_model.pkl            ← Trained Logistic Regression model
│   └── vectorizer.pkl              ← Fitted TF-IDF vectorizer
│
├── notebooks/
│   └── intent_classification.ipynb ← Full walkthrough Jupyter notebook
│
├── outputs/
│   └── confusion_matrix.png        ← Saved evaluation plot
│
├── requirements.txt                ← Python dependencies
├── README.md                       ← This file
└── .gitignore                      ← Git ignore rules
```

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/chat-message-intent-classification.git
cd chat-message-intent-classification
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download NLTK resources
```python
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab')"
```

---

## 🚀 How to Run

### Step 1 — Train the Model
```bash
python src/train_model.py
```
This will:
- Load and preprocess the dataset
- Vectorize text with TF-IDF
- Train the Logistic Regression model
- Print accuracy and classification report
- Save `confusion_matrix.png` to `outputs/`
- Save `intent_model.pkl` and `vectorizer.pkl` to `models/`

---

### Step 2 — Predict on New Messages

**Interactive mode** (type messages one by one):
```bash
python src/predict.py
```

**Single message prediction:**
```bash
python src/predict.py --message "I need help with my account"
```

**Demo mode** (8 preset messages):
```bash
python src/predict.py --demo
```

---

### Step 3 — Run the Jupyter Notebook
```bash
jupyter notebook notebooks/intent_classification.ipynb
```

---

## 🔮 Sample Predictions

| Message                                        | Predicted Label | Confidence |
|------------------------------------------------|-----------------|------------|
| "I need help resetting my password"            | 🛠️ Support       | 68.2%      |
| "This is absolutely terrible service!"         | 😠 Complaint     | 50.7%      |
| "What payment methods do you accept?"          | ❓ Query         | 45.1%      |
| "Please assist me with onboarding"             | 🛠️ Support       | 62.3%      |
| "I've been waiting 3 weeks, no response"       | 😠 Complaint     | 39.1%      |
| "How much does the premium plan cost?"         | ❓ Query         | 36.4%      |
| "My app keeps crashing, I'm very frustrated"   | 😠 Complaint     | 46.1%      |
| "Can you help me set up 2FA?"                  | 🛠️ Support       | 63.1%      |

---

## 🔭 Future Scope

- [ ] **Expand dataset** — 500+ messages with real customer support data
- [ ] **Deep learning models** — LSTM / BERT / DistilBERT for higher accuracy
- [ ] **More intent classes** — Feedback, Cancellation, Billing, etc.
- [ ] **Flask / FastAPI web app** — REST API for real-time predictions
- [ ] **Streamlit dashboard** — Interactive UI for demo and testing
- [ ] **Multi-label classification** — A message can have multiple intents
- [ ] **Confidence thresholds** — Flag low-confidence predictions for human review
- [ ] **Language support** — Hindi, Spanish, French multilingual models
- [ ] **Active learning** — Improve model with user-corrected predictions

---

## 🛠️ Tech Stack

| Library       | Version  | Purpose                    |
|---------------|----------|----------------------------|
| Python        | 3.8+     | Core language              |
| scikit-learn  | 1.3+     | ML pipeline & evaluation   |
| pandas        | 2.0+     | Data handling              |
| numpy         | 1.24+    | Numerical operations       |
| nltk          | 3.8+     | NLP preprocessing          |
| matplotlib    | 3.7+     | Plotting                   |
| seaborn       | 0.12+    | Styled visualizations      |
| joblib        | 1.3+     | Model persistence          |

---

## 👤 Author

**Your Name**  
📧 your.email@example.com  
🔗 [LinkedIn](https://linkedin.com/in/yourprofile)  
🐙 [GitHub](https://github.com/yourusername)

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use it for your college
projects, portfolio, or internship applications.

---

<p align="center">
  Made with ❤️ for learning NLP and Machine Learning
</p>
