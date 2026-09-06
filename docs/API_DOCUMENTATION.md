# 🔌 REST API Documentation

The Customer Complaint Classification System exposes a clean JSON REST API for enterprise integration.

---

## 1. Classify Complaint

Classifies raw complaint text into a functional department category, extracts top keywords, confidence score, and returns probability distribution.

- **Endpoint:** `/predict`
- **Method:** `POST`
- **Headers:** `Content-Type: application/json` (or `application/x-www-form-urlencoded`)

### Request Body (JSON)
```json
{
  "complaint": "I noticed an unauthorized charge of $850 on my credit card statement yesterday. Please freeze my card and initiate a fraud investigation immediately!"
}
```

### Response (`200 OK`)
```json
{
  "id": 1,
  "category": "Credit Card / Prepaid Card",
  "confidence": 0.98,
  "department": "Credit Card Dispute Department",
  "top_features": ["unauthorized charge", "statement", "freeze card", "investigation"],
  "all_probabilities": {
    "Bank Account Services": 0.01,
    "Credit Card / Prepaid Card": 0.98,
    "Credit Reporting": 0.00,
    "Mortgages & Loans": 0.00,
    "Theft / Fraud Reporting": 0.01
  }
}
```

### Error Responses
- **`400 Bad Request`**: When `complaint` field is empty or missing.
```json
{
  "error": "Complaint text is required."
}
```

---

## 2. Real-Time Dashboard Statistics

Retrieves aggregated metrics, total counts, category distributions, recent complaints, and best model metadata.

- **Endpoint:** `/api/dashboard`
- **Method:** `GET`

### Response (`200 OK`)
```json
{
  "stats": {
    "total_complaints": 24,
    "pending_count": 18,
    "resolved_count": 6,
    "category_counts": {
      "Credit Card / Prepaid Card": 8,
      "Bank Account Services": 5,
      "Mortgages & Loans": 4,
      "Theft / Fraud Reporting": 4,
      "Credit Reporting": 3
    }
  },
  "best_model": {
    "name": "Linear SVM",
    "accuracy": 0.98,
    "macro_f1": 0.98
  }
}
```

---

## 3. History & Audit Log API

Retrieves past complaints with pagination, category filter, urgency filter, and sentiment filter.

- **Endpoint:** `/api/history`
- **Method:** `GET`
- **Query Parameters:**
  - `page` (optional, default: 1): Page number
  - `limit` (optional, default: 20): Items per page
  - `category` (optional): Filter by category
  - `urgency` (optional): Filter by urgency (`High`, `Medium`, `Low`)
  - `sentiment` (optional): Filter by sentiment (`Negative`, `Neutral`, `Positive`)
  - `search` (optional): Full-text keyword search

---

## 4. Export CSV Endpoint

Downloads the complete audit database as a standardized CSV file.

- **Endpoint:** `/export-csv`
- **Method:** `GET`
- **Response Header:** `Content-Disposition: attachment; filename=complaint_records.csv`
