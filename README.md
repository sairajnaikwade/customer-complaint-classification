# 🏦 Customer Complaint Classification System
### Academic B.Tech Project-Based Learning (NLP PBL) — Academic Year 2026–27
**Sanjivani College of Engineering, Kopargaon**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Flask 3.1+](https://img.shields.io/badge/Flask-3.1%2B-green.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6%2B-orange.svg)](https://scikit-learn.org/)
[![NLTK](https://img.shields.io/badge/NLTK-3.9%2B-yellow.svg)](https://www.nltk.org/)
[![Tests Passing](https://img.shields.io/badge/Tests-47%2F47%20Passed-brightgreen.svg)](tests/)

---

## 📌 1. Project Overview

The **Customer Complaint Classification System** is an end-to-end Natural Language Processing (NLP) and Machine Learning (ML) solution designed to automatically categorize unstructured customer grievances into functional categories and route them to corresponding departments.

The system features:
- **Clean Academic Light-Theme Web Interface** with a dark navy sidebar across all pages.
- **Automated Text Classification & Department Routing** using Linear SVM.
- **Calibrated Prediction Confidence Scores**.
- **Interactive Analytics Dashboard** displaying complaint distribution across all 6 categories with Chart.js.
- **Audit & History Log** with real-time search, status filtering, and resolution management.
- **Model Comparison Suite** with radar charts and a full 6×6 confusion matrix.

---

## 🗂️ 2. Complaint Categories & Department Routing

The system supports **6 distinct complaint categories**:

| # | Category | Description | Routed Department |
|---|:---|:---|:---|
| 1 | 💳 **Payment Issue** | Failed transactions, double charges, payment gateway timeouts | Finance / Payments Team |
| 2 | 🧾 **Billing Issue** | Invoice discrepancies, unexpected subscription fees, overbilling | Billing Department |
| 3 | 🔧 **Technical Issue** | Application crashes, server 500 errors, bugs, API failures | Technical Support / IT |
| 4 | 📦 **Delivery Issue** | Delayed shipments, damaged parcels, tracking failures | Logistics / Delivery Team |
| 5 | 👤 **Account Issue** | Login failures, 2FA verification issues, locked profiles | Account & Security Team |
| 6 | 🎧 **Service Issue** | Rude staff behavior, unhelpful representatives, delayed response | Customer Support / Escalations |

---

## 🔬 3. NLP Preprocessing Pipeline

Every incoming complaint undergoes a rigorous 6-step NLP cleaning and normalization pipeline:

1. **Lowercase Conversion**: Standardizes text case.
2. **Contraction Expansion**: Normalizes abbreviations (e.g., `wasn't` → `was not`, `can't` → `cannot`).
3. **Noise & Punctuation Removal**: Removes URLs, email addresses, non-alphabetic symbols, and standalone digits.
4. **Tokenization**: Segments text into individual lexical tokens using NLTK.
5. **Stop-Word Removal**: Filters high-frequency, non-discriminative English stop-words.
6. **WordNet Lemmatization**: Converts inflected word forms to canonical dictionary lemmas (e.g., `crashing` → `crash`, `deducted` → `deduct`).

---

## 📊 4. TF-IDF Feature Extraction & Data Leakage Prevention

- **TF-IDF Configuration**:
  - `ngram_range = (1, 2)` (Unigrams and Bigrams)
  - `max_features = 5000` (Top 5,000 most informative n-gram features)
  - `sublinear_tf = True` (Applies logarithmic sublinear term-frequency scaling: $1 + \log(\text{tf})$)
  - `norm = 'l2'` (Cosine normalization)

- **Methodological Rigor (Zero Data Leakage)**:
  - Dataset is partitioned using **80/20 Stratified Train-Test Split** (`test_size=0.20`, `random_state=42`, `stratify=y`).
  - The `TfidfVectorizer` is **fitted strictly on the training set (`X_train`) only**.
  - The test set (`X_test`) is transformed using the pre-fitted vectorizer (`vectorizer.transform(X_test)`), ensuring no test distribution information leaks into feature extraction or model training.

---

## 🏆 5. Machine Learning Models & Evaluation Benchmark

All models were evaluated on the held-out 20% stratified test set:

| Machine Learning Model | Test Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) | Status |
|:---|:---:|:---:|:---:|:---:|:---:|
| 🥇 **Linear Support Vector Machine (LinearSVC)** | **98.37%** | **98.51%** | **98.37%** | **98.38%** | 🏆 **Best Model (Production)** |
| 🥈 **Logistic Regression** | 97.55% | 97.72% | 97.55% | 97.56% | Benchmark |
| 🥉 **Multinomial Naive Bayes** | 96.73% | 96.87% | 96.73% | 96.74% | Benchmark |

*Linear SVM with probability calibration via `CalibratedClassifierCV` is deployed as the active production model.*

---

## 📁 6. Repository Structure

```
customer-complaint-classification/
├── app.py                     # Flask application entry point & routes
├── requirements.txt           # Python dependencies
├── .gitignore                 # Excludes caches, venvs, and sqlite databases
├── README.md                  # Comprehensive project documentation
├── dataset/
│   ├── generate_dataset.py    # Balanced dataset generator
│   ├── complaints.csv         # Labeled complaints dataset (2,450 samples)
│   └── README.md              # Dataset schema documentation
├── src/
│   ├── __init__.py            # Package initialization
│   ├── preprocessing.py       # Full NLP cleaning, tokenization & lemmatization
│   ├── feature_extraction.py  # TF-IDF vectorization & feature pipeline
│   ├── train_model.py         # Model training & serialization
│   ├── evaluate_model.py      # Classification reports, confusion matrices & metrics
│   ├── predict.py             # Production inference engine & department routing
│   └── database.py            # SQLite schema, CRUD operations & analytics aggregations
├── models/
│   ├── complaint_model.pkl    # Serialized production classifier (Calibrated LinearSVC)
│   ├── tfidf_vectorizer.pkl   # Serialized TF-IDF vectorizer
│   └── model_metrics.json     # Benchmarking results and evaluation reports
├── templates/
│   ├── base.html              # Base layout with dark navy sidebar & topbar
│   ├── index.html             # Complaint input & classification interface
│   ├── result.html            # Classification result, confidence & department routing
│   ├── dashboard.html         # Interactive analytics with Chart.js (all 6 categories)
│   ├── history.html           # Complaint history table with filters & status toggle
│   ├── model_comparison.html  # Model benchmark table, radar chart & 6x6 confusion matrix
│   ├── about.html             # Academic project documentation & system workflow
│   └── error.html             # Error handling page (404/500)
├── static/
│   ├── css/
│   │   └── style.css          # Unified light-theme academic styling
│   └── js/
│       ├── main.js            # General UI helpers
│       ├── dashboard.js       # Dashboard counter animation
│       ├── history.js         # Status update AJAX handler
│       └── model_comparison.js# Dynamic confusion matrix fetcher
├── docs/
│   ├── ARCHITECTURE.md        # Architectural design document
│   └── API_DOCUMENTATION.md   # REST API endpoint specifications
└── tests/
    └── test_all.py            # 47 comprehensive unit & integration tests
```

---

## ⚙️ 7. Installation & Setup

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- Git

### Step 1: Clone Repository & Set Up Virtual Environment
```bash
git clone <repository_url>
cd customer-complaint-classification
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Step 2: Install Dependencies & Download NLTK Corpora
```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### Step 3: Run the Test Suite
```bash
python -m pytest tests/test_all.py -v
```
*Expected result: 47 passed.*

### Step 4: (Optional) Retrain the Machine Learning Models
```bash
python src/train_model.py
```

### Step 5: Launch the Web Application
```bash
python app.py
```
Open your browser and navigate to: **`http://127.0.0.1:5000`**

---

## 📡 8. REST API Endpoints

| Method | Endpoint | Description |
|:---|:---|:---|
| `POST` | `/predict` | Classify complaint text (accepts form data or JSON `{"complaint": "..."}`) |
| `GET` | `/api/dashboard` | JSON dashboard statistics and best model summary |
| `GET` | `/api/history` | Query complaint records with optional search, category, and status filters |
| `POST` | `/api/history/<id>/status` | Update complaint status (`Pending` / `Resolved`) |
| `GET` | `/api/model-comparison` | Full benchmark comparison table, confusion matrix, and model info |
| `GET` | `/api/confusion-matrix/<model>` | Confusion matrix for a specific model |

---

## 🛠️ 9. Technology Stack

- **Core Language**: Python 3.13
- **Web Framework**: Flask 3.1
- **Machine Learning**: scikit-learn 1.6+ (LinearSVC, LogisticRegression, MultinomialNB, CalibratedClassifierCV)
- **NLP Library**: NLTK 3.9+ (Tokenization, Stopwords, WordNet Lemmatizer)
- **Data Handling**: Pandas & NumPy
- **Database**: SQLite3
- **Visualization**: Chart.js 4.4
- **Testing**: pytest 9.1+

---

## 🎓 Academic Attribution
- **Project**: Customer Complaint Classification System
- **Subject**: Natural Language Processing (NLP) — Project-Based Learning (PBL)
- **Institution**: Sanjivani College of Engineering, Kopargaon
- **Academic Year**: 2026–27

