# =============================================================================
# preprocess.py
# Text Preprocessing Module for Chat Message Intent Classification
# =============================================================================
# This module handles all NLP text preprocessing steps:
#   - Lowercasing
#   - Punctuation removal
#   - Stopword removal
#   - Tokenization
# =============================================================================

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK resources (runs only if not already downloaded)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)


def preprocess_text(text: str) -> str:
    """
    Apply full NLP preprocessing pipeline to a raw text string.

    Steps:
        1. Lowercase conversion
        2. Punctuation removal
        3. Tokenization
        4. Stopword removal

    Args:
        text (str): Raw input message string.

    Returns:
        str: Cleaned and preprocessed text.
    """

    # Step 1: Convert to lowercase
    # e.g., "Hello World!" → "hello world!"
    text = text.lower()

    # Step 2: Remove punctuation
    # Removes all characters in string.punctuation: !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Step 3: Remove extra whitespace and strip leading/trailing spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Step 4: Tokenize the text (split sentence into individual words)
    tokens = word_tokenize(text)

    # Step 5: Remove stopwords (common words like "the", "is", "and", etc.)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # Step 6: Rejoin tokens into a single cleaned string
    cleaned_text = ' '.join(filtered_tokens)

    return cleaned_text


def preprocess_series(text_series):
    """
    Apply preprocessing to an entire Pandas Series of text messages.

    Args:
        text_series (pd.Series): Series of raw text messages.

    Returns:
        pd.Series: Series of cleaned text messages.
    """
    return text_series.apply(preprocess_text)


# ---------------------------------------------------------------
# Quick demo — run this file directly to test preprocessing
# ---------------------------------------------------------------
if __name__ == "__main__":
    sample_messages = [
        "I need help resetting my PASSWORD!",
        "This product is COMPLETELY broken and unusable!!!",
        "What are the pricing plans available for new users?",
    ]

    print("=" * 55)
    print("  Text Preprocessing Demo")
    print("=" * 55)

    for msg in sample_messages:
        cleaned = preprocess_text(msg)
        print(f"\nOriginal : {msg}")
        print(f"Cleaned  : {cleaned}")

    print("\n" + "=" * 55)
