# =============================================================================
# predict.py
# Prediction Module for Chat Message Intent Classification
# =============================================================================
# Loads the saved model + vectorizer and predicts intent for new messages.
# Can be run interactively or imported as a module.
# =============================================================================

import os
import sys
import joblib

# Add src/ to path so we can import preprocess.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import preprocess_text

# =============================================================================
# PATH CONFIGURATION
# =============================================================================
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'intent_model.pkl')
TFIDF_PATH = os.path.join(BASE_DIR, 'models', 'vectorizer.pkl')

# Label → emoji mapping for friendly display
LABEL_EMOJIS = {
    'support':   '🛠️  Support',
    'complaint': '😠 Complaint',
    'query':     '❓ Query',
}


def load_artifacts():
    """
    Load the saved model and TF-IDF vectorizer from disk.

    Returns:
        tuple: (model, tfidf_vectorizer)

    Raises:
        FileNotFoundError: If model/vectorizer files don't exist yet.
    """
    if not os.path.exists(MODEL_PATH) or not os.path.exists(TFIDF_PATH):
        raise FileNotFoundError(
            "\n  [ERROR] Model files not found. "
            "Please run `python src/train_model.py` first to train the model.\n"
        )

    model = joblib.load(MODEL_PATH)
    tfidf = joblib.load(TFIDF_PATH)
    return model, tfidf


def predict_intent(message: str, model=None, tfidf=None) -> dict:
    """
    Predict the intent label for a single chat message.

    Args:
        message (str): Raw user message.
        model: Pre-loaded Logistic Regression model (optional).
        tfidf: Pre-loaded TF-IDF vectorizer (optional).

    Returns:
        dict: {
            'message'    : original message,
            'cleaned'    : preprocessed message,
            'label'      : predicted label string,
            'confidence' : confidence score (0–100),
            'probabilities': {'support': x, 'complaint': y, 'query': z}
        }
    """
    # Load model artifacts if not provided
    if model is None or tfidf is None:
        model, tfidf = load_artifacts()

    # Preprocess the input message
    cleaned = preprocess_text(message)

    # Vectorize using the trained TF-IDF vectorizer
    vector = tfidf.transform([cleaned])

    # Predict class label and probability scores
    label       = model.predict(vector)[0]
    proba_array = model.predict_proba(vector)[0]

    # Build probability dict {class_name: score}
    proba_dict = {
        cls: round(float(prob) * 100, 2)
        for cls, prob in zip(model.classes_, proba_array)
    }

    confidence = proba_dict[label]

    return {
        'message'      : message,
        'cleaned'      : cleaned,
        'label'        : label,
        'confidence'   : confidence,
        'probabilities': proba_dict,
    }


def print_prediction(result: dict):
    """Pretty-print a prediction result to the console."""
    label_display = LABEL_EMOJIS.get(result['label'], result['label'])

    print("\n  ┌─────────────────────────────────────────────────┐")
    print(f"  │  Message    : {result['message'][:47]}")
    print(f"  │  Cleaned    : {result['cleaned'][:47]}")
    print(f"  │  Prediction : {label_display}")
    print(f"  │  Confidence : {result['confidence']:.1f}%")
    print(f"  │  Probabilities:")
    for cls, prob in sorted(result['probabilities'].items(), key=lambda x: -x[1]):
        bar   = '█' * int(prob / 5)
        emoji = LABEL_EMOJIS.get(cls, cls)
        print(f"  │    {emoji:<18} : {prob:5.1f}% {bar}")
    print("  └─────────────────────────────────────────────────┘")


# =============================================================================
# BATCH PREDICTION DEMO
# =============================================================================
def run_demo():
    """Run predictions on a set of predefined demo messages."""
    demo_messages = [
        "I need help resetting my account password",
        "This is absolutely ridiculous, your service is the worst!",
        "What payment methods do you accept?",
        "Please assist me with the onboarding process",
        "I have been waiting three weeks and no one is responding to me",
        "How much does the premium plan cost per month?",
        "My app keeps crashing and I am very frustrated",
        "Can you help me set up two-factor authentication?",
    ]

    print("\n" + "=" * 55)
    print("  INTENT CLASSIFICATION — DEMO PREDICTIONS")
    print("=" * 55)

    model, tfidf = load_artifacts()

    for msg in demo_messages:
        result = predict_intent(msg, model, tfidf)
        print_prediction(result)


# =============================================================================
# INTERACTIVE PREDICTION MODE
# =============================================================================
def interactive_mode():
    """Let user type messages and see predictions in real time."""
    print("\n" + "=" * 55)
    print("  INTENT PREDICTION — INTERACTIVE MODE")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 55)

    model, tfidf = load_artifacts()
    print("  Model loaded. Ready for predictions!\n")

    while True:
        try:
            message = input("  Enter message: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Exiting prediction mode. Goodbye!")
            break

        if not message:
            print("  [!] Please enter a non-empty message.\n")
            continue

        if message.lower() in ('quit', 'exit', 'q'):
            print("  Exiting prediction mode. Goodbye!")
            break

        result = predict_intent(message, model, tfidf)
        print_prediction(result)
        print()


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Predict intent of chat messages.'
    )
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run demo predictions on preset messages'
    )
    parser.add_argument(
        '--message', '-m',
        type=str,
        help='Predict intent for a single message (wrap in quotes)'
    )
    args = parser.parse_args()

    if args.demo:
        run_demo()
    elif args.message:
        model, tfidf = load_artifacts()
        result       = predict_intent(args.message, model, tfidf)
        print_prediction(result)
    else:
        interactive_mode()
