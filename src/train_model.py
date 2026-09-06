"""
Model Training Pipeline
College NLP PBL Project - Academic Year 2026-27
Sanjivani College of Engineering, Kopargaon

Trains three supervised ML classifiers:
  1. Multinomial Naive Bayes
  2. Logistic Regression
  3. Linear SVM (LinearSVC)

Uses stratified train/test split (80/20) and persists the best model.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# Add project root to sys.path so src can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import clean_text
from src.feature_extraction import fit_and_save_vectorizer, create_vectorizer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATASET_PATH       = os.path.join("dataset", "complaints.csv")
MODEL_PATH         = os.path.join("models", "complaint_model.pkl")
VECTORIZER_PATH    = os.path.join("models", "tfidf_vectorizer.pkl")
LABEL_ENCODER_PATH = os.path.join("models", "label_encoder.pkl")
METRICS_PATH       = os.path.join("models", "model_metrics.json")


def load_dataset(path: str = DATASET_PATH) -> pd.DataFrame:
    """Loads and validates the complaints CSV dataset."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")
    df = pd.read_csv(path)
    if "complaint_text" not in df.columns or "category" not in df.columns:
        raise ValueError("Dataset must contain 'complaint_text' and 'category' columns.")
    df.dropna(subset=["complaint_text", "category"], inplace=True)
    df = df[df["complaint_text"].str.strip().astype(bool)]  # remove empty texts
    df.drop_duplicates(subset=["complaint_text"], inplace=True)
    print(f"Loaded {len(df)} records across {df['category'].nunique()} categories.")
    return df


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Applies the NLP preprocessing pipeline to every complaint text."""
    print("Applying NLP preprocessing pipeline …")
    df = df.copy()
    df["clean_text"] = df["complaint_text"].apply(clean_text)
    # Drop any rows where cleaning produces empty string
    df = df[df["clean_text"].str.strip().astype(bool)]
    print(f"After preprocessing: {len(df)} valid records.")
    return df


def get_classifiers() -> dict:
    """
    Returns the three classifiers to compare.
    LinearSVC is wrapped in CalibratedClassifierCV so we can call
    predict_proba() for confidence scores.
    """
    return {
        "Naive Bayes": MultinomialNB(alpha=1.0),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            random_state=42
        ),
        "SVM": CalibratedClassifierCV(
            LinearSVC(max_iter=2000, C=1.0, random_state=42)
        )
    }


def train_and_evaluate(
    X_train, X_test,
    y_train, y_test,
    classifiers: dict,
    label_encoder: LabelEncoder
) -> dict:
    """
    Trains each classifier on (X_train, y_train) and evaluates on (X_test, y_test).
    Returns a dict with per-model metrics.
    """
    results = {}
    classes = label_encoder.classes_.tolist()

    for name, clf in classifiers.items():
        print(f"\n  >> Training: {name}")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        cm = confusion_matrix(y_test, y_pred, labels=label_encoder.transform(classes))
        report = classification_report(
            y_test, y_pred,
            target_names=classes,
            zero_division=0,
            output_dict=True
        )

        results[name] = {
            "accuracy":          round(float(acc),  4),
            "precision":         round(float(prec), 4),
            "recall":            round(float(rec),  4),
            "f1_score":          round(float(f1),   4),
            "confusion_matrix":  cm.tolist(),
            "classification_report": report,
            "classes":           classes
        }

        print(f"     Accuracy : {acc:.4f}  |  Precision : {prec:.4f}  |  Recall : {rec:.4f}  |  F1 : {f1:.4f}")

    return results


def select_best_model(results: dict, classifiers: dict) -> tuple:
    """Selects the best classifier by weighted F1-score."""
    best_name = max(results, key=lambda k: results[k]["f1_score"])
    print(f"\nBest model: {best_name}  (F1-score = {results[best_name]['f1_score']:.4f})")
    return best_name, classifiers[best_name]


def run_training():
    """End-to-end training pipeline orchestrator with strict train/test separation."""
    os.makedirs("models", exist_ok=True)

    # ── 1. Load & preprocess ──────────────────────────────────────────────
    df = load_dataset()
    df = preprocess_dataset(df)

    # ── 2. Encode labels ──────────────────────────────────────────────────
    le = LabelEncoder()
    y  = le.fit_transform(df["category"])
    joblib.dump(le, LABEL_ENCODER_PATH)
    print(f"Categories: {list(le.classes_)}")

    # ── 3. Stratified train/test split (80 / 20) on clean text ─────────────
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["clean_text"].tolist(), y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\nTrain samples: {len(X_train_text)}  |  Test samples: {len(X_test_text)}")

    # ── 4. TF-IDF vectorisation — fitted ONLY on training data ────────────
    print("\nFitting TF-IDF vectorizer ONLY on training set …")
    vectorizer = create_vectorizer()
    X_train = vectorizer.fit_transform(X_train_text)
    X_test  = vectorizer.transform(X_test_text)

    # Save the vectorizer fitted ONLY on training data
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"TF-IDF Vectorizer fitted with {len(vectorizer.get_feature_names_out())} vocabulary features.")
    print(f"Saved training-fitted vectorizer to: {VECTORIZER_PATH}")

    # ── 5. Train & evaluate classifiers on training data only ─────────────
    print("\n=== Training Classifiers ===")
    classifiers = get_classifiers()
    results = train_and_evaluate(X_train, X_test, y_train, y_test, classifiers, le)

    # ── 6. Select best model ──────────────────────────────────────────────
    best_name, best_clf = select_best_model(results, classifiers)

    # ── 7. Persist best model ─────────────────────────────────────────────
    joblib.dump(best_clf, MODEL_PATH)
    print(f"Saved best model ({best_name}) to: {MODEL_PATH}")

    # ── 8. Save metrics JSON ──────────────────────────────────────────────
    metrics_payload = {
        "best_model":       best_name,
        "selection_reason": "Highest weighted F1-score on held-out 20% test set",
        "averaging":        "weighted",
        "test_size":        0.20,
        "train_samples":    int(X_train.shape[0]),
        "test_samples":     int(X_test.shape[0]),
        "models":           results
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"Saved evaluation metrics to: {METRICS_PATH}")

    # ── 9. Print summary table ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"{'Model':<25} {'Accuracy':>10} {'Precision':>11} {'Recall':>8} {'F1':>8}")
    print("-" * 70)
    for model_name, m in results.items():
        marker = " [BEST]" if model_name == best_name else ""
        print(
            f"{model_name:<25} {m['accuracy']:>10.4f} {m['precision']:>11.4f}"
            f" {m['recall']:>8.4f} {m['f1_score']:>8.4f}{marker}"
        )
    print("=" * 70)
    print("\nTraining complete!")
    return metrics_payload


if __name__ == "__main__":
    run_training()

