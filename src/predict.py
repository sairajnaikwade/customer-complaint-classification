"""
Prediction Module
College NLP PBL Project - Academic Year 2026-27
Sanjivani College of Engineering, Kopargaon

Loads the saved model, vectorizer, and label encoder to classify
an incoming complaint text. Also handles department routing.
"""

import os
import sys
import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import clean_text, get_pipeline_steps_breakdown
from src.feature_extraction import load_vectorizer, get_top_tfidf_features

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_PATH         = os.path.join("models", "complaint_model.pkl")
VECTORIZER_PATH    = os.path.join("models", "tfidf_vectorizer.pkl")
LABEL_ENCODER_PATH = os.path.join("models", "label_encoder.pkl")

# ---------------------------------------------------------------------------
# Department routing map (rule-based layer; separate from ML prediction)
# ---------------------------------------------------------------------------
DEPARTMENT_ROUTING = {
    "Payment Issue":   {
        "department":    "Finance / Payments Team",
        "description":   "This complaint is related to payment problems such as failed transactions, double charges, or refund issues.",
        "icon":          "💳",
        "color":         "#3b82f6"
    },
    "Billing Issue":   {
        "department":    "Billing Team",
        "description":   "This complaint concerns billing discrepancies, incorrect invoices, or unauthorized charges.",
        "icon":          "🧾",
        "color":         "#f59e0b"
    },
    "Technical Issue": {
        "department":    "Technical Support",
        "description":   "This complaint relates to application crashes, errors, or system malfunctions.",
        "icon":          "🔧",
        "color":         "#ef4444"
    },
    "Delivery Issue":  {
        "department":    "Logistics / Delivery Team",
        "description":   "This complaint is about delayed, missing, or damaged deliveries.",
        "icon":          "🚚",
        "color":         "#10b981"
    },
    "Account Issue":   {
        "department":    "Account Support",
        "description":   "This complaint is related to login failures, password resets, or account access issues.",
        "icon":          "👤",
        "color":         "#8b5cf6"
    },
    "Service Issue":   {
        "department":    "Customer Support",
        "description":   "This complaint concerns poor customer service, unresponsive agents, or service quality.",
        "icon":          "🎧",
        "color":         "#ec4899"
    }
}

# Module-level cache for loaded artefacts (avoids repeated disk reads)
_model      = None
_vectorizer = None
_le         = None


def _load_artefacts():
    """Lazy-loads model, vectorizer, and label-encoder from disk."""
    global _model, _vectorizer, _le

    for path, label in [
        (MODEL_PATH,         "model"),
        (VECTORIZER_PATH,    "vectorizer"),
        (LABEL_ENCODER_PATH, "label encoder"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required {label} not found at '{path}'. "
                "Please run 'python src/train_model.py' first."
            )

    if _model is None:
        _model = joblib.load(MODEL_PATH)
    if _vectorizer is None:
        _vectorizer = joblib.load(VECTORIZER_PATH)
    if _le is None:
        _le = joblib.load(LABEL_ENCODER_PATH)


def route_department(category: str) -> dict:
    """Returns department routing info for a given predicted category."""
    return DEPARTMENT_ROUTING.get(category, {
        "department":  "General Support",
        "description": "Your complaint has been logged and will be reviewed.",
        "icon":        "📋",
        "color":       "#6b7280"
    })


def predict(complaint_text: str) -> dict:
    """
    Full prediction pipeline:
      1. Validate input
      2. NLP preprocessing
      3. TF-IDF transformation
      4. ML model prediction + confidence score
      5. Department routing
      6. Return structured result dict

    Returns a dict with keys:
        raw_complaint, clean_text, category, confidence,
        department_info, top_features, pipeline_steps
    """
    # ── Input Validation ────────────────────────────────────────────────
    if not complaint_text or not isinstance(complaint_text, str):
        return {"error": "Complaint text is required."}

    complaint_text = complaint_text.strip()

    if len(complaint_text) < 5:
        return {"error": "Complaint text is too short. Please provide more detail."}

    if len(complaint_text) > 2000:
        return {"error": "Complaint text is too long. Please limit to 2000 characters."}

    # ── Load artefacts (cached) ──────────────────────────────────────────
    try:
        _load_artefacts()
    except FileNotFoundError as e:
        return {"error": str(e)}

    # ── Preprocessing ────────────────────────────────────────────────────
    pipeline_steps = get_pipeline_steps_breakdown(complaint_text)
    cleaned = pipeline_steps["final_cleaned_text"]

    if not cleaned.strip():
        return {"error": "Complaint contains only stop-words or punctuation. Please rephrase."}

    # ── TF-IDF Transformation ────────────────────────────────────────────
    X = _vectorizer.transform([cleaned])

    # ── Prediction + Confidence ──────────────────────────────────────────
    predicted_label   = _model.predict(X)[0]
    predicted_category = _le.inverse_transform([predicted_label])[0]

    # Confidence via predict_proba (all three models support this)
    try:
        proba       = _model.predict_proba(X)[0]
        confidence  = float(np.max(proba)) * 100
        all_proba   = {
            _le.inverse_transform([i])[0]: round(float(p) * 100, 2)
            for i, p in enumerate(proba)
        }
    except Exception:
        confidence = None
        all_proba  = {}

    # ── Department Routing ───────────────────────────────────────────────
    dept_info = route_department(predicted_category)

    # ── Top contributing TF-IDF features ────────────────────────────────
    top_features = get_top_tfidf_features(cleaned, _vectorizer, top_n=5)

    return {
        "raw_complaint":    complaint_text,
        "clean_text":       cleaned,
        "category":         predicted_category,
        "confidence":       round(confidence, 2) if confidence is not None else None,
        "all_probabilities": all_proba,
        "department_info":  dept_info,
        "top_features":     top_features,
        "pipeline_steps":   pipeline_steps
    }


if __name__ == "__main__":
    test_complaints = [
        "My payment was deducted but the order was not confirmed.",
        "The mobile application crashes whenever I try to login.",
        "My package has not arrived even after the expected delivery date.",
        "I forgot my password and cannot access my account.",
        "I was charged twice for the same order.",
        "I contacted customer support but nobody has responded."
    ]

    for text in test_complaints:
        result = predict(text)
        if "error" in result:
            print(f"ERROR: {result['error']}")
        else:
            conf_str = f"{result['confidence']:.1f}%" if result['confidence'] else "N/A"
            print(f"Input   : {text[:60]}...")
            print(f"Category: {result['category']}  |  Confidence: {conf_str}")
            print(f"Dept    : {result['department_info']['department']}")
            print()
