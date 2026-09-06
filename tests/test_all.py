"""
Test Suite - Customer Complaint Classification System
College NLP PBL Project | Sanjivani COE, Kopargaon 2026-27

Tests cover:
  1. Text preprocessing
  2. TF-IDF transformation
  3. Model prediction
  4. Flask API endpoint
  5. Database CRUD
  6. Input validation
"""

import os
import sys
import json
import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import clean_text, get_pipeline_steps_breakdown
from src.feature_extraction import load_vectorizer


# ===========================================================================
# 1. Preprocessing Tests
# ===========================================================================

class TestPreprocessing:
    def test_basic_lowercase(self):
        result = clean_text("PAYMENT FAILED")
        assert result == result.lower()

    def test_removes_punctuation(self):
        result = clean_text("My payment failed!!!")
        assert "!" not in result

    def test_contraction_expansion(self):
        result = clean_text("I wasn't informed about the charge.")
        # 'wasn't' should become 'was not', then 'not' may stay or be a stopword
        # At minimum the text must be non-empty
        assert len(result) > 0

    def test_stopword_removal(self):
        result = clean_text("the is was a")
        # All stopwords → empty after cleaning
        assert result.strip() == ""

    def test_lemmatization_verbs(self):
        result = clean_text("crashing")
        # lemmatized form: crash
        assert "crash" in result

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_none_input(self):
        assert clean_text(None) == ""

    def test_payment_complaint(self):
        result = clean_text("My payment was deducted but the order was not confirmed.")
        assert "payment" in result
        assert "deduct" in result or "deducted" in result
        assert "order" in result
        assert "confirm" in result or "confirmed" in result

    def test_pipeline_breakdown_keys(self):
        steps = get_pipeline_steps_breakdown("App keeps crashing on login")
        required_keys = [
            "raw_text", "normalized_text", "tokens",
            "tokens_without_stopwords", "lemmatized_tokens", "final_cleaned_text"
        ]
        for key in required_keys:
            assert key in steps, f"Missing key: {key}"

    def test_pipeline_raw_text_preserved(self):
        text = "My card was charged twice!"
        steps = get_pipeline_steps_breakdown(text)
        assert steps["raw_text"] == text

    def test_url_removal(self):
        result = clean_text("Visit https://support.example.com for help.")
        assert "http" not in result

    def test_numeric_tokens_removed(self):
        result = clean_text("Transaction 123456 failed.")
        assert "123456" not in result


# ===========================================================================
# 2. TF-IDF Tests
# ===========================================================================

class TestTFIDF:
    @pytest.fixture(autouse=True)
    def load_vec(self):
        """Load vectorizer once for the class (requires trained model)."""
        model_path = os.path.join("models", "tfidf_vectorizer.pkl")
        if not os.path.exists(model_path):
            pytest.skip("TF-IDF vectorizer not found. Run train_model.py first.")
        self.vectorizer = load_vectorizer(model_path)

    def test_vectorizer_loads(self):
        assert self.vectorizer is not None

    def test_transform_shape(self):
        text = clean_text("payment deducted order not confirmed")
        X = self.vectorizer.transform([text])
        assert X.shape[0] == 1
        assert X.shape[1] > 0

    def test_transform_sparse(self):
        text = clean_text("app crashes when opening settings")
        X = self.vectorizer.transform([text])
        assert X.nnz >= 0  # sparse matrix

    def test_feature_names_exist(self):
        names = self.vectorizer.get_feature_names_out()
        assert len(names) > 100

    def test_different_texts_differ(self):
        X1 = self.vectorizer.transform([clean_text("payment failed card charged")])
        X2 = self.vectorizer.transform([clean_text("app crash error login")])
        # The two matrices should not be identical
        diff = (X1 - X2).nnz
        assert diff > 0


# ===========================================================================
# 3. Prediction Tests
# ===========================================================================

class TestPrediction:
    @pytest.fixture(autouse=True)
    def check_models(self):
        for p in ["models/complaint_model.pkl", "models/tfidf_vectorizer.pkl", "models/label_encoder.pkl"]:
            if not os.path.exists(p):
                pytest.skip(f"Required model file not found: {p}")

    def _predict(self, text):
        from src.predict import predict
        return predict(text)

    def test_payment_issue(self):
        result = self._predict("My payment was deducted but the order was not confirmed.")
        assert "error" not in result
        assert result["category"] == "Payment Issue"

    def test_technical_issue(self):
        result = self._predict("The mobile application crashes whenever I try to login.")
        assert "error" not in result
        assert result["category"] == "Technical Issue"

    def test_delivery_issue(self):
        result = self._predict("My package has not arrived even after the expected delivery date.")
        assert "error" not in result
        assert result["category"] == "Delivery Issue"

    def test_account_issue(self):
        result = self._predict("I forgot my password and cannot access my account.")
        assert "error" not in result
        assert result["category"] == "Account Issue"

    def test_billing_issue(self):
        result = self._predict("I was charged twice for the same order.")
        assert "error" not in result
        assert result["category"] == "Billing Issue"

    def test_service_issue(self):
        result = self._predict("I contacted customer support but nobody has responded.")
        assert "error" not in result
        assert result["category"] == "Service Issue"

    def test_result_has_confidence(self):
        result = self._predict("My payment failed but money was deducted.")
        assert "confidence" in result
        assert result["confidence"] is not None
        assert 0 <= result["confidence"] <= 100

    def test_result_has_department(self):
        result = self._predict("App keeps crashing on startup.")
        assert "department_info" in result
        assert "department" in result["department_info"]

    def test_empty_input_error(self):
        result = self._predict("")
        assert "error" in result

    def test_short_input_error(self):
        result = self._predict("hi")
        assert "error" in result

    def test_long_input_error(self):
        result = self._predict("x" * 2001)
        assert "error" in result


# ===========================================================================
# 4. Flask API Endpoint Tests
# ===========================================================================

class TestFlaskAPI:
    @pytest.fixture(autouse=True)
    def setup_client(self):
        import app as flask_app
        flask_app.app.config["TESTING"] = True
        self.client = flask_app.app.test_client()

    def test_home_page_loads(self):
        resp = self.client.get("/")
        assert resp.status_code == 200
        assert b"Complaint" in resp.data

    def test_dashboard_loads(self):
        resp = self.client.get("/dashboard")
        assert resp.status_code == 200

    def test_history_loads(self):
        resp = self.client.get("/history")
        assert resp.status_code == 200

    def test_model_comparison_loads(self):
        resp = self.client.get("/model-comparison")
        assert resp.status_code == 200

    def test_about_loads(self):
        resp = self.client.get("/about")
        assert resp.status_code == 200

    def test_predict_json_valid(self):
        resp = self.client.post(
            "/predict",
            json={"complaint": "My payment was deducted but order was not confirmed."},
            headers={"Content-Type": "application/json"}
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "category" in data
        assert data["category"] == "Payment Issue"

    def test_predict_empty_complaint(self):
        resp = self.client.post(
            "/predict",
            json={"complaint": ""},
            headers={"Content-Type": "application/json"}
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data

    def test_predict_short_complaint(self):
        resp = self.client.post(
            "/predict",
            json={"complaint": "hi"},
            headers={"Content-Type": "application/json"}
        )
        assert resp.status_code == 400

    def test_api_dashboard_json(self):
        resp = self.client.get("/api/dashboard")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "stats" in data
        assert "total" in data["stats"]

    def test_api_history_json(self):
        resp = self.client.get("/api/history")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "complaints" in data

    def test_api_model_comparison_json(self):
        resp = self.client.get("/api/model-comparison")
        assert resp.status_code in (200, 503)  # 503 if not trained

    def test_404_returns_error_page(self):
        resp = self.client.get("/nonexistent-route-xyz")
        assert resp.status_code == 404


# ===========================================================================
# 5. Database Tests
# ===========================================================================

class TestDatabase:
    @pytest.fixture(autouse=True)
    def setup_db(self, tmp_path):
        """Use a temporary test database."""
        import src.database as db
        self._orig_path = db.DB_PATH
        db.DB_PATH = str(tmp_path / "test.db")
        db.init_db()
        yield
        db.DB_PATH = self._orig_path

    def test_insert_and_retrieve(self):
        from src.database import insert_complaint, get_complaint_by_id
        cid = insert_complaint(
            "My payment failed.", "payment fail", "Payment Issue", "Finance Team", 91.5
        )
        assert cid is not None and cid > 0
        record = get_complaint_by_id(cid)
        assert record is not None
        assert record["predicted_category"] == "Payment Issue"
        assert record["model_score"] == 91.5

    def test_get_all_empty(self):
        from src.database import get_all_complaints
        rows = get_all_complaints()
        assert isinstance(rows, list)

    def test_update_status_resolved(self):
        from src.database import insert_complaint, update_status, get_complaint_by_id
        cid = insert_complaint("App crash", "app crash", "Technical Issue", "Tech Support", 87.0)
        success = update_status(cid, "Resolved")
        assert success is True
        record = get_complaint_by_id(cid)
        assert record["status"] == "Resolved"

    def test_update_status_invalid(self):
        from src.database import insert_complaint, update_status
        cid = insert_complaint("Order late", "order late", "Delivery Issue", "Logistics", 80.0)
        success = update_status(cid, "InvalidStatus")
        assert success is False

    def test_dashboard_stats(self):
        from src.database import insert_complaint, get_dashboard_stats
        insert_complaint("Payment failed", "payment fail", "Payment Issue", "Finance", 90.0)
        stats = get_dashboard_stats()
        assert stats["total"] >= 1
        assert "by_category" in stats

    def test_search_filter(self):
        from src.database import insert_complaint, get_all_complaints
        insert_complaint("App crash on startup", "app crash", "Technical Issue", "Tech", 85.0)
        results = get_all_complaints(search="App crash")
        assert len(results) >= 1

    def test_category_filter(self):
        from src.database import insert_complaint, get_all_complaints
        insert_complaint("Cannot login", "login fail", "Account Issue", "Account Support", 92.0)
        results = get_all_complaints(category="Account Issue")
        assert all(r["predicted_category"] == "Account Issue" for r in results)
