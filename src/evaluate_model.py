"""
Model Evaluation Module
College NLP PBL Project - Academic Year 2026-27
Sanjivani College of Engineering, Kopargaon

Loads saved metrics and provides helper functions for dashboard and
model-comparison page rendering.
"""

import os
import json

METRICS_PATH = os.path.join("models", "model_metrics.json")


def load_metrics() -> dict:
    """Loads model evaluation metrics from the persisted JSON file."""
    if not os.path.exists(METRICS_PATH):
        raise FileNotFoundError(
            "model_metrics.json not found. Run 'python src/train_model.py' first."
        )
    with open(METRICS_PATH, "r") as f:
        return json.load(f)


def get_comparison_table(metrics: dict = None) -> list:
    """
    Returns a list of dicts suitable for rendering the model comparison table.
    Each dict: {model, accuracy, precision, recall, f1_score, is_best}
    """
    if metrics is None:
        metrics = load_metrics()

    rows = []
    best = metrics.get("best_model", "")
    for model_name, m in metrics["models"].items():
        raw_acc  = m.get("accuracy", 0)
        acc_pct  = raw_acc * 100 if raw_acc <= 1.0 else raw_acc
        
        raw_prec = m.get("precision", 0)
        prec_pct = raw_prec * 100 if raw_prec <= 1.0 else raw_prec
        
        raw_rec  = m.get("recall", 0)
        rec_pct  = raw_rec * 100 if raw_rec <= 1.0 else raw_rec
        
        raw_f1   = m.get("f1_score", 0)
        f1_pct   = raw_f1 * 100 if raw_f1 <= 1.0 else raw_f1

        rows.append({
            "model":     model_name,
            "accuracy":  round(float(acc_pct), 2),
            "precision": round(float(prec_pct), 2),
            "recall":    round(float(rec_pct), 2),
            "f1_score":  round(float(f1_pct), 2),
            "is_best":   model_name == best
        })
    return rows


def get_confusion_matrix_data(model_name: str = None, metrics: dict = None) -> dict:
    """
    Returns confusion matrix and class labels for the specified model.
    Defaults to the best model if model_name is None.
    """
    if metrics is None:
        metrics = load_metrics()

    if model_name is None:
        model_name = metrics.get("best_model")

    model_data = metrics["models"].get(model_name)
    if model_data is None:
        raise ValueError(f"Model '{model_name}' not found in metrics.")

    return {
        "model":            model_name,
        "classes":          model_data["classes"],
        "confusion_matrix": model_data["confusion_matrix"]
    }


def get_best_model_info(metrics: dict = None) -> dict:
    """Returns summary info about the best-performing model."""
    if metrics is None:
        metrics = load_metrics()

    best_name = metrics.get("best_model", "Unknown")
    best_data = metrics["models"].get(best_name, {})
    
    raw_acc  = best_data.get("accuracy", 0)
    acc_pct  = raw_acc * 100 if raw_acc <= 1.0 else raw_acc
    
    raw_prec = best_data.get("precision", 0)
    prec_pct = raw_prec * 100 if raw_prec <= 1.0 else raw_prec
    
    raw_rec  = best_data.get("recall", 0)
    rec_pct  = raw_rec * 100 if raw_rec <= 1.0 else raw_rec
    
    raw_f1   = best_data.get("f1_score", 0)
    f1_pct   = raw_f1 * 100 if raw_f1 <= 1.0 else raw_f1

    return {
        "model":     best_name,
        "accuracy":  round(float(acc_pct), 2),
        "precision": round(float(prec_pct), 2),
        "recall":    round(float(rec_pct), 2),
        "f1_score":  round(float(f1_pct), 2),
        "reason":    metrics.get("selection_reason", "")
    }


def get_per_class_metrics(model_name: str = None, metrics: dict = None) -> list:
    """
    Returns per-class precision, recall, f1 for the specified model.
    Useful for the About / Model-Comparison page breakdown.
    """
    if metrics is None:
        metrics = load_metrics()
    if model_name is None:
        model_name = metrics["best_model"]

    report = metrics["models"][model_name].get("classification_report", {})
    rows = []
    for label, vals in report.items():
        if isinstance(vals, dict) and "precision" in vals:
            rows.append({
                "category":  label,
                "precision": round(vals["precision"] * 100, 2),
                "recall":    round(vals["recall"]    * 100, 2),
                "f1_score":  round(vals["f1-score"]  * 100, 2),
                "support":   int(vals["support"])
            })
    return rows


if __name__ == "__main__":
    try:
        m = load_metrics()
        print("Best model:", m["best_model"])
        print("\nComparison table:")
        for row in get_comparison_table(m):
            print(row)
        print("\nConfusion matrix (best model):")
        cm_data = get_confusion_matrix_data(metrics=m)
        print("Classes:", cm_data["classes"])
        import pprint
        pprint.pprint(cm_data["confusion_matrix"])
    except FileNotFoundError as e:
        print(e)
