"""
Feature Extraction Pipeline using TF-IDF
College NLP PBL Project - Academic Year 2026-27
Sanjivani College of Engineering, Kopargaon

Converts preprocessed text into numerical TF-IDF feature vectors.
"""

import os
import joblib
import numpy as np
from typing import Tuple, List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")


def create_vectorizer(
    max_features: int = 3000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 1,
    sublinear_tf: bool = True
) -> TfidfVectorizer:
    """
    Initializes a configured TF-IDF Vectorizer.
    - ngram_range=(1, 2): Extracts unigrams and meaningful bigrams (e.g. 'payment fail', 'card charge')
    - sublinear_tf=True: Applies sublinear scaling 1 + log(tf) to dampen excessive frequency impact
    - min_df=2: Discards rare noise terms occurring only once across the corpus
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=sublinear_tf,
        strip_accents="unicode",
        norm="l2"
    )


def fit_and_save_vectorizer(
    corpus: List[str],
    save_path: str = VECTORIZER_PATH
) -> Tuple[TfidfVectorizer, np.ndarray]:
    """
    Fits TF-IDF vectorizer on the training text corpus and persists to disk.
    """
    vectorizer = create_vectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(vectorizer, save_path)
    print(f"TF-IDF Vectorizer fitted with {len(vectorizer.get_feature_names_out())} features.")
    print(f"Saved vectorizer to: {save_path}")
    
    return vectorizer, tfidf_matrix


def load_vectorizer(load_path: str = VECTORIZER_PATH) -> TfidfVectorizer:
    """
    Loads persisted TF-IDF vectorizer from disk.
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"TF-IDF vectorizer not found at {load_path}. Please train the model first.")
    return joblib.load(load_path)


def transform_text(text: str, vectorizer: Optional[TfidfVectorizer] = None) -> np.ndarray:
    """
    Transforms a single cleaned text string into TF-IDF vector representation.
    """
    if vectorizer is None:
        vectorizer = load_vectorizer()
    return vectorizer.transform([text])


def get_top_tfidf_features(text: str, vectorizer: Optional[TfidfVectorizer] = None, top_n: int = 5) -> List[dict]:
    """
    Extracts the highest-weighted TF-IDF terms for a given input text.
    Helpful for student viva and frontend visual feedback.
    """
    if vectorizer is None:
        vectorizer = load_vectorizer()
        
    vec = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()
    
    # Non-zero coordinates
    coo = vec.tocoo()
    tuples = zip(coo.col, coo.data)
    sorted_items = sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)[:top_n]
    
    return [
        {"term": feature_names[idx], "tfidf_score": round(float(score), 4)}
        for idx, score in sorted_items
    ]


if __name__ == "__main__":
    sample_corpus = [
        "payment deduct order confirm",
        "card charge twice invoice billing",
        "application crash login error",
        "package arrive delay tracking",
        "unable access account reset password",
        "customer support response rude ticket"
    ]
    vec, mat = fit_and_save_vectorizer(sample_corpus, "models/test_tfidf.pkl")
    print("Test Transformation Shape:", mat.shape)
    print("Top features for sample 1:", get_top_tfidf_features(sample_corpus[0], vec))
    if os.path.exists("models/test_tfidf.pkl"):
        os.remove("models/test_tfidf.pkl")
