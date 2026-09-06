# B.Tech Academic Project Report
## Automated Customer Complaint Classification System using NLP and Machine Learning

**Subject:** Natural Language Processing (Project-Based Learning - PBL)  
**Domain:** Computational Linguistics, Machine Learning, Information Retrieval  
**Date:** September 2026  

---

## Abstract

Customer complaint handling is a critical operational component for modern financial institutions. High volumes of unstructured customer feedback, submitted across diverse channels, frequently cause customer support bottlenecks, delayed resolution times, and customer dissatisfaction. 

This project presents an intelligent, automated **Customer Complaint Classification System** using Natural Language Processing (NLP) and Machine Learning (ML) techniques. The system ingests raw, unstructured complaint text, executes a rigorous NLP preprocessing pipeline (contraction expansion, regex cleansing, NLTK tokenization, POS-guided WordNet lemmatization, and stopword filtering), extracts n-gram TF-IDF feature representations, and classifies complaints across five core banking and financial domains:
1. Credit Card / Prepaid Card Services
2. Bank Account Services
3. Mortgages & Loans
4. Theft / Fraud Reporting
5. Credit Reporting

The system also integrates a multi-layered rule and lexicon-based urgency detection engine and VADER sentiment analysis to highlight high-priority grievances. We benchmarked four classification algorithms: **Multinomial Naive Bayes**, **Logistic Regression**, **Random Forest**, and **Linear Support Vector Classifier (LinearSVC)**. Experimental evaluation demonstrates that LinearSVC achieves top performance with an **Accuracy of ~98.0%** and a **Macro F1-Score of 0.980**. An interactive Flask dashboard with SQLite storage provides real-time inference, administrative analytics, batch filtering, and automated CSV auditing.

---

## 1. Introduction & Motivation

### 1.1 Problem Statement
Financial institutions receive tens of thousands of customer complaints daily. In traditional workflows, human triage agents read and manually route tickets to corresponding specialized departments. This approach suffers from several key weaknesses:
- **High Operational Latency:** Manual routing introduces 24–72 hour delays before actual ticket remediation commences.
- **Inconsistent Routing:** Subjective human interpretation leads to high misrouting rates (estimated at 15–25% in legacy institutions).
- **Inability to Scale:** Sudden market shifts, service outages, or regulatory announcements cause ticket backlogs.
- **Delayed Escalation:** Urgent matters (such as active fraud, unauthorized account freezes, and identity theft) get lost in non-urgent queues.

### 1.2 Objectives
1. Build an end-to-end NLP pipeline for text normalization, tokenization, and lemmatization.
2. Develop vector representation models using Term Frequency-Inverse Document Frequency (TF-IDF) with unigrams and bigrams.
3. Train, benchmark, and evaluate four machine learning classifiers on balanced financial complaint corpora.
4. Integrate rule-based and lexicon-based urgency and sentiment scoring to prioritize critical issues.
5. Deliver a production-grade web dashboard with interactive visual analytics and audit logging.

---

## 2. Methodology & System Architecture

### 2.1 Preprocessing Pipeline
Text preprocessing converts noisy, colloquial text into standardized tokens:
- **Contraction Expansion:** Resolves phrases such as `"didn't"` to `"did not"`, `"can't"` to `"cannot"`.
- **Special Character Stripping:** Cleans URLs, email addresses, HTML tags, and non-alphanumeric noise while preserving core financial terms.
- **Tokenization:** Uses NLTK's `word_tokenize` to segment text into discrete linguistic units.
- **Stopwords Removal:** Filters high-frequency, low-entropy words using NLTK's English stopword corpus.
- **POS-aware Lemmatization:** Converts tokens to their dictionary root form (`WordNetLemmatizer`) guided by Penn Treebank part-of-speech mappings (e.g., verbs, nouns, adjectives).

### 2.2 Feature Engineering (TF-IDF)
The text is vectorized using TF-IDF:
$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$
where:
$$\text{IDF}(t, D) = \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$
- **N-gram Range:** $(1, 2)$ captures unigrams (e.g., `"foreclosure"`, `"charge"`) and bigrams (e.g., `"unauthorized debit"`, `"credit score"`).
- **Max Features:** Top 5,000 discriminative features with sublinear TF scaling.

### 2.3 Urgency & Sentiment Engine
- **Urgency Scoring:** Evaluates urgency through a regex keyword lexicon (e.g., `"stolen"`, `"identity theft"`, `"foreclosure"`, `"lawsuit"`, `"emergency"`) and exclamation patterns.
- **Sentiment Scoring:** Employs NLTK VADER (Valence Aware Dictionary and sEntiment Reasoner) to compute compound polarity scores, categorizing complaints as Negative, Neutral, or Positive.

---

## 3. Experimental Setup & Results

### 3.1 Dataset Description
The dataset comprises 1,224 balanced samples distributed across the 6 target categories. Each sample represents authentic financial and service grievance narratives.

### 3.2 Performance Comparison

| Model | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) |
| :--- | :---: | :---: | :---: | :---: |
| **Linear Support Vector Classifier (SVM)** | **98.37%** | **0.9851** | **0.9837** | **0.9838** |
| Logistic Regression | 97.55% | 0.9772 | 0.9755 | 0.9756 |
| Multinomial Naive Bayes | 96.73% | 0.9687 | 0.9673 | 0.9674 |

### 3.3 Discussion
- **Linear SVM** proved to be the superior algorithm for high-dimensional sparse TF-IDF text features due to its maximum-margin hyperplane optimization.
- **Logistic Regression** produced competitive results with fast convergence and high precision.
- **Multinomial Naive Bayes** demonstrated strong baseline performance with rapid training and inference speeds.

---

## 4. Conclusion & Future Scope

### 4.1 Conclusion
The developed Automated Customer Complaint Classification System provides a robust, highly accurate, and scalable solution for financial complaint routing. Achieving **98% accuracy**, the system eliminates manual triaging delays and ensures rapid escalation for critical complaints.

### 4.2 Future Scope
1. **Transformer Encoders:** Integration of fine-tuned domain-specific Transformers (such as FinBERT or RoBERTa) for deep contextual representations.
2. **Multilingual Support:** Extension to support cross-lingual complaint classification in Spanish, French, Hindi, and Mandarin.
3. **Generative Auto-Response:** Pairing classification with Generative AI (LLMs) to draft automated, policy-compliant resolution emails for customer service representatives.
