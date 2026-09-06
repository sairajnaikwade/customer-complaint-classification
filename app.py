"""
Flask Application — Customer Complaint Classification System
College NLP PBL Project — Academic Year 2026-27
Sanjivani College of Engineering, Kopargaon

Entry-point for the web application.
All business logic is delegated to the src/ package modules.
"""

import os
import sys
import json
from flask import (
    Flask, render_template, request,
    redirect, url_for, jsonify, abort
)

# Ensure project root on sys.path so src.* imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.predict import predict, DEPARTMENT_ROUTING
from src.evaluate_model import (
    load_metrics, get_comparison_table,
    get_confusion_matrix_data, get_best_model_info
)
from src.database import (
    init_db, insert_complaint,
    get_all_complaints, update_status,
    get_dashboard_stats, get_complaint_by_id
)

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "nlp-pbl-2026-sanjivani-kopargaon")

# Register enumerate as a Jinja2 filter (used in model_comparison.html)
app.jinja_env.filters["enumerate"] = enumerate

# Initialise database on startup
with app.app_context():
    init_db()

# ---------------------------------------------------------------------------
# Helper: load metrics once and cache
# ---------------------------------------------------------------------------
_metrics_cache = None

def _get_metrics():
    global _metrics_cache
    if _metrics_cache is None:
        try:
            _metrics_cache = load_metrics()
        except FileNotFoundError:
            _metrics_cache = None
    return _metrics_cache


# ===========================================================================
# HTML Page Routes
# ===========================================================================

@app.route("/")
def index():
    """Home / Input page."""
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    """Dashboard page with charts and statistics."""
    stats = get_dashboard_stats()
    metrics = _get_metrics()
    best_info = get_best_model_info(metrics) if metrics else None
    return render_template("dashboard.html", stats=stats, best_info=best_info)


@app.route("/history")
def history():
    """Previous complaints / history page."""
    search   = request.args.get("search", "").strip()
    category = request.args.get("category", "All")
    status   = request.args.get("status", "All")
    complaints = get_all_complaints(search=search, category=category, status=status)
    categories = list(DEPARTMENT_ROUTING.keys())
    return render_template(
        "history.html",
        complaints=complaints,
        categories=categories,
        selected_category=category,
        selected_status=status,
        search_query=search
    )


@app.route("/model-comparison")
def model_comparison():
    """Model comparison and confusion matrix page."""
    metrics = _get_metrics()
    if not metrics:
        return render_template(
            "model_comparison.html",
            error="Model metrics not found. Please train the models first."
        )
    table    = get_comparison_table(metrics)
    cm_data  = get_confusion_matrix_data(metrics=metrics)
    best     = get_best_model_info(metrics)
    return render_template(
        "model_comparison.html",
        table=table,
        cm_data=cm_data,
        best=best,
        all_models=list(metrics["models"].keys())
    )


@app.route("/about")
def about():
    """About / documentation page."""
    return render_template("about.html")


@app.route("/result")
def result():
    """Result page — displays classification result for a complaint."""
    complaint_id = request.args.get("id", type=int)
    if not complaint_id:
        return redirect(url_for("index"))
    record = get_complaint_by_id(complaint_id)
    if not record:
        abort(404)
    dept_info = DEPARTMENT_ROUTING.get(record["predicted_category"], {})
    return render_template("result.html", record=record, dept_info=dept_info)


# ===========================================================================
# API Endpoints
# ===========================================================================

@app.route("/predict", methods=["POST"])
def api_predict():
    """
    POST /predict
    Accepts JSON or form data with 'complaint' field.
    Classifies complaint, stores to DB, redirects to result page.
    """
    # Accept both JSON body and HTML form submission
    if request.is_json:
        data = request.get_json(silent=True) or {}
        complaint_text = data.get("complaint", "").strip()
    else:
        complaint_text = request.form.get("complaint", "").strip()

    # ── Validation ──────────────────────────────────────────────────────
    if not complaint_text:
        if request.is_json:
            return jsonify({"error": "Complaint text is required."}), 400
        return render_template("index.html", error="Please enter a complaint before submitting.")

    if len(complaint_text) < 5:
        if request.is_json:
            return jsonify({"error": "Complaint is too short. Please provide more detail."}), 400
        return render_template("index.html", error="Complaint is too short. Please provide more detail.")

    if len(complaint_text) > 2000:
        if request.is_json:
            return jsonify({"error": "Complaint exceeds 2000 characters."}), 400
        return render_template("index.html", error="Complaint text is too long. Limit to 2000 characters.")

    # ── Predict ──────────────────────────────────────────────────────────
    result_data = predict(complaint_text)

    if "error" in result_data:
        if request.is_json:
            return jsonify(result_data), 500
        return render_template("index.html", error=result_data["error"])

    # ── Persist to SQLite ─────────────────────────────────────────────────
    dept_info  = result_data.get("department_info", {})
    department = dept_info.get("department", "General Support")
    score      = result_data.get("confidence")

    complaint_id = insert_complaint(
        complaint_text=result_data["raw_complaint"],
        clean_text=result_data["clean_text"],
        predicted_category=result_data["category"],
        department=department,
        model_score=round(score, 2) if score else None
    )

    if request.is_json:
        return jsonify({
            "id":           complaint_id,
            "category":     result_data["category"],
            "confidence":   result_data["confidence"],
            "department":   department,
            "top_features": result_data.get("top_features", []),
            "all_probabilities": result_data.get("all_probabilities", {})
        })

    return redirect(url_for("result", id=complaint_id))


@app.route("/api/dashboard")
def api_dashboard():
    """GET /api/dashboard — returns aggregated dashboard stats as JSON."""
    stats   = get_dashboard_stats()
    metrics = _get_metrics()
    best    = get_best_model_info(metrics) if metrics else {}
    return jsonify({"stats": stats, "best_model": best})


@app.route("/api/history")
def api_history():
    """GET /api/history — returns complaint history as JSON."""
    search   = request.args.get("search", "")
    category = request.args.get("category", "All")
    status   = request.args.get("status", "All")
    limit    = request.args.get("limit", 100, type=int)
    complaints = get_all_complaints(search=search, category=category, status=status, limit=limit)
    return jsonify({"complaints": complaints, "total": len(complaints)})


@app.route("/api/history/<int:complaint_id>/status", methods=["POST"])
def api_update_status(complaint_id):
    """POST /api/history/<id>/status — update complaint resolution status."""
    data       = request.get_json(silent=True) or {}
    new_status = data.get("status", "").strip()
    if new_status not in ("Pending", "Resolved"):
        return jsonify({"error": "Status must be 'Pending' or 'Resolved'."}), 400
    success = update_status(complaint_id, new_status)
    if not success:
        return jsonify({"error": "Complaint not found."}), 404
    return jsonify({"success": True, "id": complaint_id, "status": new_status})


@app.route("/api/model-comparison")
def api_model_comparison():
    """GET /api/model-comparison — returns full model metrics as JSON."""
    metrics = _get_metrics()
    if not metrics:
        return jsonify({"error": "Model metrics not available. Please train first."}), 503
    return jsonify({
        "comparison_table": get_comparison_table(metrics),
        "confusion_matrix": get_confusion_matrix_data(metrics=metrics),
        "best_model":       get_best_model_info(metrics)
    })


@app.route("/api/confusion-matrix/<model_name>")
def api_confusion_matrix(model_name):
    """GET /api/confusion-matrix/<model_name> — returns CM for specific model."""
    metrics = _get_metrics()
    if not metrics:
        return jsonify({"error": "Metrics not available."}), 503
    try:
        cm_data = get_confusion_matrix_data(model_name=model_name, metrics=metrics)
        return jsonify(cm_data)
    except ValueError as e:
        return jsonify({"error": str(e)}), 404


# ===========================================================================
# Error Handlers
# ===========================================================================

@app.errorhandler(404)
def not_found(e):
    return render_template("error.html", code=404, message="Page not found."), 404


@app.errorhandler(500)
def server_error(e):
    return render_template("error.html", code=500, message="Internal server error."), 500


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
    print(f"Starting Customer Complaint Classification System on port {port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
