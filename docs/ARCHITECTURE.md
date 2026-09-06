# 🏛️ System Architecture & Data Flow

## 1. High-Level Architectural Diagram

```
+-------------------------------------------------------------------------------+
|                             CLIENT / USER INTERFACE                           |
|  - Modern Dark-Mode Glassmorphic UI (HTML5, Vanilla CSS3, Vanilla JS)        |
|  - Interactive Dashboards & Metrics (Chart.js 4.4)                            |
+---------------------------------------+---------------------------------------+
                                        | (HTTP / REST JSON)
                                        v
+-------------------------------------------------------------------------------+
|                            FLASK APPLICATION SERVER                           |
|  - App Routing & REST API (`app.py`)                                          |
|  - Jinja2 Template Engine Rendering                                          |
|  - Input Validation & Error Handling (HTTP 400/404/500)                       |
+-------------------+---------------------------------------+-------------------+
                    |                                       |
                    v                                       v
+---------------------------------------+   +-----------------------------------+
|            NLP & ML INFERENCE         |   |         DATABASE LAYER            |
|  - Text Preprocessing (`preprocessing`)  |   |  - SQLite Database Engine         |
|  - TF-IDF Vectorizer (`feature_extr.`) |   |  - Thread-Safe Connections        |
|  - Multi-Class Model (`predict.py`)   |   |  - CRUD & Aggregation Queries     |
|  - VADER Sentiment & Urgency Rules    |   |  - Audit Log Tracking             |
+---------------------------------------+   +-----------------------------------+
```

## 2. Component Specifications

### 2.1 Preprocessing Module (`src/preprocessing.py`)
- **Step 1: Lowercase Conversion & Stripping:** Converts text to lowercase and strips extraneous spaces.
- **Step 2: Contraction Expansion:** Replaces contractions with full equivalents.
- **Step 3: Noise Cleaning:** Removes URLs (`http\S+`), email addresses (`\S+@\S+`), HTML tags (`<.*?>`), and special characters.
- **Step 4: Tokenization:** Uses NLTK's `word_tokenize` with fallback whitespace splitting.
- **Step 5: Stopwords Removal:** Removes standard English stopwords while keeping negation terms when appropriate.
- **Step 6: POS Tagging & Lemmatization:** Uses NLTK `pos_tag` and maps treebank tags to WordNet constants (`NOUN`, `VERB`, `ADJ`, `ADV`) for accurate root-word lemmatization.

### 2.2 Feature Extraction Module (`src/feature_extraction.py`)
- **Vectorizer:** `TfidfVectorizer` from `scikit-learn`
- **N-gram Range:** Unigrams & Bigrams $(1, 2)$
- **Vocabulary Size:** Max 5,000 features
- **Scaling:** Sublinear TF scaling enabled ($1 + \log(\text{tf})$)

### 2.3 Machine Learning Pipeline (`src/train_model.py` & `src/evaluate_model.py`)
- Benchmarks 4 candidate architectures using stratified 80/20 train-test splits:
  1. `LinearSVC(C=1.0, max_iter=2000)` (or calibrated `CalibratedClassifierCV` / `SGDClassifier(loss='log_loss')`)
  2. `LogisticRegression(C=1.0, max_iter=1000)`
  3. `MultinomialNB(alpha=0.1)`
  4. `RandomForestClassifier(n_estimators=100)`
- Evaluates metrics: Accuracy, Precision, Recall, Macro F1, Weighted F1, Confusion Matrix.
- Serializes top model artifact to `models/complaint_model.pkl`.

### 2.4 Database Schema (`database/complaints.db`)
Table: `complaints`
```sql
CREATE TABLE IF NOT EXISTS complaints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    complaint_text TEXT NOT NULL,
    cleaned_text TEXT NOT NULL,
    predicted_category TEXT NOT NULL,
    confidence REAL NOT NULL,
    sentiment TEXT NOT NULL,
    sentiment_score REAL NOT NULL,
    urgency TEXT NOT NULL,
    keywords TEXT,
    department TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```
Indexes are placed on `predicted_category`, `urgency`, `sentiment`, and `created_at` for rapid filtering and dashboard aggregations.
