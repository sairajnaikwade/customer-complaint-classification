"""
Natural Language Preprocessing Pipeline
College NLP PBL Project - Academic Year 2026-27
Sanjivani College of Engineering, Kopargaon

Performs text normalization, tokenization, stop-word removal, and lemmatization.
"""

import re
import string
from typing import List, Dict, Any, Union

# Common English contractions dictionary for normalization
CONTRACTIONS_MAP = {
    "wasn't": "was not",
    "wasnt": "was not",
    "isn't": "is not",
    "isnt": "is not",
    "aren't": "are not",
    "arent": "are not",
    "haven't": "have not",
    "havent": "have not",
    "hasn't": "has not",
    "hasnt": "has not",
    "hadn't": "had not",
    "hadnt": "had not",
    "won't": "will not",
    "wont": "will not",
    "don't": "do not",
    "dont": "do not",
    "doesn't": "does not",
    "doesnt": "does not",
    "didn't": "did not",
    "didnt": "did not",
    "can't": "cannot",
    "cant": "cannot",
    "couldn't": "could not",
    "couldnt": "could not",
    "shouldn't": "should not",
    "shouldnt": "should not",
    "wouldn't": "would not",
    "wouldnt": "would not",
    "i'm": "i am",
    "im": "i am",
    "i've": "i have",
    "ive": "i have",
    "i'll": "i will",
    "ill": "i will",
    "i'd": "i would",
    "they're": "they are",
    "we're": "we are",
    "you're": "you are",
    "it's": "it is",
    "that's": "that is"
}

# Standard English Stop Words (curated for complaint classification)
# Preserves semantic keywords while discarding non-informative functional words
DEFAULT_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down",
    "during", "each", "few", "for", "from", "further", "had", "has", "have", "having",
    "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i",
    "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more", "most",
    "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once", "only",
    "or", "other", "our", "ours", "ourselves", "out", "over", "own", "s", "same",
    "she", "should", "so", "some", "such", "than", "that", "the", "their", "theirs",
    "them", "themselves", "then", "there", "these", "they", "this", "those", "through",
    "to", "too", "under", "until", "up", "very", "was", "we", "were", "what", "when",
    "where", "which", "while", "who", "whom", "why", "with", "would", "you", "your",
    "yours", "yourself", "yourselves", "please", "kindly", "hello", "hi", "sir", "madam",
    "dear", "also", "already", "today", "yesterday", "since", "still", "even"
}

# Initialize Lemmatizer with graceful fallback
try:
    from nltk.stem import WordNetLemmatizer
    import nltk
    lemmatizer = WordNetLemmatizer()
    # Test if WordNet dataset is accessible
    lemmatizer.lemmatize("crashing", pos="v")
    _NLTK_LEMMATIZER_AVAILABLE = True
except Exception:
    _NLTK_LEMMATIZER_AVAILABLE = False


def _rule_based_lemmatize(word: str) -> str:
    """
    Lightweight rule-based morphological lemmatization fallback.
    Handles standard verb suffixes (-ing, -ed, -es, -s, -ies, -tion).
    """
    word = word.lower()
    if len(word) <= 3:
        return word
    
    # Common irregular verbs & terms in customer domain
    irregulars = {
        "deducted": "deduct",
        "charged": "charge",
        "crashed": "crash",
        "crashing": "crash",
        "arrived": "arrive",
        "arriving": "arrive",
        "cancelled": "cancel",
        "canceling": "cancel",
        "received": "receive",
        "receiving": "receive",
        "delivered": "deliver",
        "delivering": "deliver",
        "transferred": "transfer",
        "transferring": "transfer",
        "processed": "process",
        "processing": "process",
        "verified": "verify",
        "verifying": "verify",
        "responded": "respond",
        "responding": "respond",
        "escalated": "escalate",
        "escalating": "escalate",
        "disputed": "dispute",
        "disputing": "dispute",
        "logged": "log",
        "logging": "log",
        "refunded": "refund",
        "refunding": "refund",
        "paid": "pay",
        "paying": "pay",
        "debited": "debit",
        "debiting": "debit",
        "credited": "credit",
        "crediting": "credit",
        "billed": "bill",
        "billing": "bill",
        "ordered": "order",
        "ordering": "order"
    }
    
    if word in irregulars:
        return irregulars[word]
        
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("ing") and len(word) > 4:
        # e.g., working -> work, passing -> pass
        if word[-4] == word[-5] and word[-4] not in "lsz":
            return word[:-4]
        return word[:-3]
    if word.endswith("ed") and len(word) > 4:
        if word[-3] == word[-4] and word[-3] not in "lsz":
            return word[:-3]
        return word[:-2]
    if word.endswith("es") and len(word) > 3:
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
        
    return word


def lemmatize_word(word: str) -> str:
    """
    Lemmatizes a single token to its root dictionary form.
    """
    if _NLTK_LEMMATIZER_AVAILABLE:
        try:
            # Try verb first, then noun
            lemma = lemmatizer.lemmatize(word, pos="v")
            if lemma == word:
                lemma = lemmatizer.lemmatize(word, pos="n")
            return lemma
        except Exception:
            return _rule_based_lemmatize(word)
    return _rule_based_lemmatize(word)


def expand_contractions(text: str) -> str:
    """
    Expands common conversational English contractions.
    e.g. "wasn't" -> "was not"
    """
    words = text.split()
    expanded_words = [CONTRACTIONS_MAP.get(w.lower(), w) for w in words]
    return " ".join(expanded_words)


def clean_text(text: Union[str, float, None]) -> str:
    """
    Primary NLP Preprocessing Pipeline:
    1. Lowercasing
    2. Contraction expansion
    3. Noise & punctuation removal
    4. Tokenization
    5. Stop-word removal
    6. Lemmatization
    7. Whitespace normalization
    """
    if text is None or not isinstance(text, str):
        return ""
    
    # 1. Lowercase
    text = text.lower().strip()
    if not text:
        return ""
    
    # 2. Expand Contractions
    text = expand_contractions(text)
    
    # 3. Remove URLs, emails, special symbols, digits
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    
    # 4. Tokenization
    tokens = text.split()
    
    # 5. Stop-words removal & 6. Lemmatization
    cleaned_tokens = []
    for token in tokens:
        token = token.strip()
        if len(token) > 1 and token not in DEFAULT_STOPWORDS:
            lemma = lemmatize_word(token)
            if lemma and len(lemma) > 1:
                cleaned_tokens.append(lemma)
                
    # 7. Recombine to clean string
    return " ".join(cleaned_tokens)


def get_pipeline_steps_breakdown(text: str) -> Dict[str, Any]:
    """
    Detailed diagnostic function for the college project UI/viva.
    Returns the intermediate state at every stage of the NLP pipeline.
    """
    if text is None or not isinstance(text, str):
        text = ""
        
    raw_input = text
    lower_text = raw_input.lower().strip()
    expanded_text = expand_contractions(lower_text)
    
    # Regex sanitized
    no_noise = re.sub(r"https?://\S+|www\.\S+", " ", expanded_text)
    no_noise = re.sub(r"\S+@\S+", " ", no_noise)
    no_noise = re.sub(r"\d+", " ", no_noise)
    no_punct = re.sub(r"[^\w\s]", " ", no_noise)
    normalized = " ".join(no_punct.split())
    
    tokens = normalized.split()
    tokens_no_stops = [t for t in tokens if len(t) > 1 and t not in DEFAULT_STOPWORDS]
    lemmatized_tokens = [lemmatize_word(t) for t in tokens_no_stops]
    final_output = " ".join([t for t in lemmatized_tokens if len(t) > 1])
    
    return {
        "raw_text": raw_input,
        "normalized_text": normalized,
        "tokens": tokens,
        "tokens_without_stopwords": tokens_no_stops,
        "lemmatized_tokens": lemmatized_tokens,
        "final_cleaned_text": final_output
    }


if __name__ == "__main__":
    sample = "My PAYMENT was deducted, but the order wasn't confirmed!"
    print("Sample Input:", sample)
    print("Cleaned Output:", clean_text(sample))
    print("\nDetailed Steps:")
    import pprint
    pprint.pprint(get_pipeline_steps_breakdown(sample))
