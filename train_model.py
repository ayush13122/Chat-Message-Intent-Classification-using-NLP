# =============================================================================
# train_model.py
# Model Training Script for Chat Message Intent Classification
# =============================================================================
# Workflow:
#   1. Load dataset
#   2. Preprocess text
#   3. TF-IDF vectorization
#   4. Train-test split
#   5. Train Logistic Regression
#   6. Evaluate (accuracy, confusion matrix, classification report)
#   7. Save model and vectorizer
# =============================================================================

import os
import sys
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings('ignore')

# Add src/ to path so we can import preprocess.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import preprocess_series

# =============================================================================
# PATH CONFIGURATION
# =============================================================================
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE_DIR, 'data', 'intents_dataset.csv')
MODEL_PATH  = os.path.join(BASE_DIR, 'models', 'intent_model.pkl')
TFIDF_PATH  = os.path.join(BASE_DIR, 'models', 'vectorizer.pkl')
OUTPUT_PATH = os.path.join(BASE_DIR, 'outputs', 'confusion_matrix.png')

# Create output directories if they don't exist
os.makedirs(os.path.join(BASE_DIR, 'models'),  exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'outputs'), exist_ok=True)


# =============================================================================
# STEP 1: LOAD DATASET
# =============================================================================
def load_data(path: str) -> pd.DataFrame:
    """Load CSV dataset and display basic info."""
    print("\n" + "=" * 60)
    print("  STEP 1: Loading Dataset")
    print("=" * 60)

    df = pd.read_csv(path)
    print(f"  Dataset loaded successfully from: {path}")
    print(f"  Total samples   : {len(df)}")
    print(f"  Columns         : {list(df.columns)}")
    print(f"\n  Label Distribution:")
    print(df['label'].value_counts().to_string())
    return df


# =============================================================================
# STEP 2: PREPROCESS TEXT
# =============================================================================
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply NLP preprocessing to all messages."""
    print("\n" + "=" * 60)
    print("  STEP 2: Preprocessing Text")
    print("=" * 60)

    df = df.copy()
    df['cleaned_message'] = preprocess_series(df['message'])

    # Show a before/after sample
    print("\n  Sample before vs after preprocessing:")
    print(f"  {'ORIGINAL':<50} {'CLEANED'}")
    print(f"  {'-'*50} {'-'*40}")
    for _, row in df.sample(3, random_state=42).iterrows():
        orig    = row['message'][:48]
        cleaned = row['cleaned_message'][:38]
        print(f"  {orig:<50} {cleaned}")

    print(f"\n  Preprocessing complete. {len(df)} messages processed.")
    return df


# =============================================================================
# STEP 3: TF-IDF VECTORIZATION
# =============================================================================
def vectorize_text(X_train, X_test):
    """
    Convert text into numerical TF-IDF features.

    TF-IDF (Term Frequency–Inverse Document Frequency) measures how
    important a word is to a document relative to the entire corpus.
    """
    print("\n" + "=" * 60)
    print("  STEP 3: TF-IDF Vectorization")
    print("=" * 60)

    tfidf = TfidfVectorizer(
        max_features=5000,   # use top 5000 terms by frequency
        ngram_range=(1, 2),  # use unigrams and bigrams
        sublinear_tf=True    # apply log normalization to TF
    )

    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf  = tfidf.transform(X_test)

    print(f"  Vocabulary size         : {len(tfidf.vocabulary_)}")
    print(f"  Training matrix shape   : {X_train_tfidf.shape}")
    print(f"  Testing matrix shape    : {X_test_tfidf.shape}")

    return tfidf, X_train_tfidf, X_test_tfidf


# =============================================================================
# STEP 4: TRAIN-TEST SPLIT
# =============================================================================
def split_data(df: pd.DataFrame):
    """Split dataset into training and testing sets (80/20 split)."""
    print("\n" + "=" * 60)
    print("  STEP 4: Train-Test Split")
    print("=" * 60)

    X = df['cleaned_message']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,      # 20% data for testing
        random_state=42,    # reproducibility
        stratify=y          # maintain label balance in both sets
    )

    print(f"  Total samples    : {len(df)}")
    print(f"  Training samples : {len(X_train)} ({len(X_train)/len(df)*100:.0f}%)")
    print(f"  Testing samples  : {len(X_test)}  ({len(X_test)/len(df)*100:.0f}%)")

    return X_train, X_test, y_train, y_test


# =============================================================================
# STEP 5: TRAIN LOGISTIC REGRESSION MODEL
# =============================================================================
def train_model(X_train_tfidf, y_train) -> LogisticRegression:
    """
    Train Logistic Regression classifier.

    Logistic Regression is a linear model ideal for text classification.
    It learns weights for each TF-IDF feature and predicts class probabilities.
    """
    print("\n" + "=" * 60)
    print("  STEP 5: Training Logistic Regression Model")
    print("=" * 60)

    model = LogisticRegression(
        max_iter=1000,    # ensure convergence
        solver='lbfgs',   # efficient solver for multiclass
        C=1.0,            # regularization strength (default)
        random_state=42
    )

    model.fit(X_train_tfidf, y_train)
    print("  Model training complete!")
    print(f"  Algorithm  : Logistic Regression")
    print(f"  Solver     : lbfgs")
    print(f"  Max iter   : 1000")
    print(f"  Classes    : {list(model.classes_)}")

    return model


# =============================================================================
# STEP 6: EVALUATE MODEL
# =============================================================================
def evaluate_model(model, tfidf, X_test_tfidf, y_test, X_test):
    """Evaluate model and print accuracy, classification report, confusion matrix."""
    print("\n" + "=" * 60)
    print("  STEP 6: Model Evaluation")
    print("=" * 60)

    y_pred = model.predict(X_test_tfidf)

    # --- Accuracy ---
    acc = accuracy_score(y_test, y_pred)
    print(f"\n  ✅ Test Accuracy : {acc * 100:.2f}%\n")

    # --- Classification Report ---
    print("  Classification Report:")
    print("  " + "-" * 50)
    report = classification_report(y_test, y_pred, target_names=model.classes_)
    for line in report.split('\n'):
        print("  " + line)

    # --- Confusion Matrix ---
    print("\n  Generating confusion matrix plot...")
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    _plot_confusion_matrix(cm, model.classes_)

    # --- Sample Predictions ---
    print("\n  Sample Predictions on Test Set:")
    print(f"  {'MESSAGE':<48} {'ACTUAL':<12} {'PREDICTED'}")
    print(f"  {'-'*48} {'-'*12} {'-'*10}")
    sample_df = pd.DataFrame({'message': X_test, 'actual': y_test, 'predicted': y_pred})
    for _, row in sample_df.sample(min(6, len(sample_df)), random_state=7).iterrows():
        msg  = str(row['message'])[:46]
        tick = "✓" if row['actual'] == row['predicted'] else "✗"
        print(f"  {msg:<48} {row['actual']:<12} {row['predicted']} {tick}")

    return acc


def _plot_confusion_matrix(cm, class_names):
    """Generate and save a styled confusion matrix heatmap."""
    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        linecolor='gray',
        annot_kws={"size": 14, "weight": "bold"}
    )

    plt.title('Confusion Matrix — Intent Classification', fontsize=15, fontweight='bold', pad=15)
    plt.ylabel('Actual Label',    fontsize=12, labelpad=10)
    plt.xlabel('Predicted Label', fontsize=12, labelpad=10)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11, rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Confusion matrix saved to: {OUTPUT_PATH}")


# =============================================================================
# STEP 7: SAVE MODEL AND VECTORIZER
# =============================================================================
def save_artifacts(model, tfidf):
    """Persist trained model and TF-IDF vectorizer to disk using joblib."""
    print("\n" + "=" * 60)
    print("  STEP 7: Saving Model & Vectorizer")
    print("=" * 60)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(tfidf, TFIDF_PATH)

    print(f"  Model saved     → {MODEL_PATH}")
    print(f"  Vectorizer saved→ {TFIDF_PATH}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def main():
    print("\n" + "=" * 60)
    print("  Chat Message Intent Classification")
    print("  ML Training Pipeline")
    print("=" * 60)

    # Load → Preprocess → Split → Vectorize → Train → Evaluate → Save
    df                              = load_data(DATA_PATH)
    df                              = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    tfidf, X_train_tfidf, X_test_tfidf = vectorize_text(X_train, X_test)
    model                           = train_model(X_train_tfidf, y_train)
    acc                             = evaluate_model(model, tfidf, X_test_tfidf, y_test, X_test)
    save_artifacts(model, tfidf)

    print("\n" + "=" * 60)
    print(f"  Pipeline complete! Final Accuracy: {acc*100:.2f}%")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
