# Credit-card-fraud-detection

# 💳 Credit Card Fraud Detection using Machine Learning

## 📌 Project Overview

Credit card fraud detection is a **binary classification problem** where the goal is to identify fraudulent transactions from a highly **imbalanced dataset**. Fraudulent transactions are very rare compared to legitimate ones, making traditional accuracy-based evaluation unreliable.

In this project, we build a **machine learning pipeline** to detect fraudulent credit card transactions while focusing on **recall, precision, F1-score, and ROC-AUC** rather than accuracy.

---

## 🎯 Problem Statement

Financial institutions face massive losses due to fraudulent credit card transactions. The objective of this project is to:

* Detect fraudulent transactions effectively
* Minimize false negatives (missing fraud)
* Handle extreme class imbalance
* Build an interview-ready, real-world ML solution

---

## 📊 Dataset Description

* **Source**: Kaggle Credit Card Fraud Dataset
* **Total Transactions**: 284,807
* **Fraudulent Transactions**: 492 (~0.17%)
* **Features**:

  * `V1`–`V28`: PCA-transformed features (to protect privacy)
  * `Time`: Time elapsed between transactions
  * `Amount`: Transaction amount
  * `Class`: Target variable (0 = Legit, 1 = Fraud)

---

## ⚠️ Key Challenge: Imbalanced Data

The dataset is **heavily skewed**, so accuracy is misleading. A model predicting all transactions as legitimate would still achieve >99% accuracy.

### ✔ Solution Strategy

* Used **class-weighted Logistic Regression**
* Focused on **Recall, Precision, F1-score, ROC-AUC**
* Avoided random undersampling to prevent data loss

---

## 🧠 Approach & Workflow

1. Data Loading & Inspection
2. Exploratory Data Analysis (EDA)
3. Feature Scaling (`Time`, `Amount`)
4. Train-Test Split with Stratification
5. Model Training with Class Weights
6. Evaluation using meaningful metrics

---

## 🤖 Model Used

### Logistic Regression (Baseline Model)

Why Logistic Regression?

* Simple and interpretable
* Strong baseline for binary classification
* Works well with class weighting

```python
LogisticRegression(class_weight='balanced', max_iter=1000)
```

---

## 📈 Evaluation Metrics

* **Confusion Matrix**
* **Precision**
* **Recall** (Most important)
* **F1-score**
* **ROC-AUC Score**

### Why Recall Matters?

Missing a fraud transaction is more costly than falsely flagging a legitimate one.

---

## 📌 Results Summary

* Model successfully identifies a high percentage of fraud cases
* ROC-AUC score indicates strong class separation
* Balanced trade-off between false positives and false negatives

*(Exact numbers may vary based on train-test split)*

---

## 🛠 Tech Stack

* Python
* NumPy
* Pandas
* Scikit-learn

---

## ▶️ How to Run the Project

1. Clone the repository

```bash
git clone https://github.com/your-username/credit-card-fraud-detection.git
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the script

```bash
python fraud_detection.py
```

---

## 🔮 Future Improvements

* Apply **SMOTE** for oversampling
* Use **Random Forest / XGBoost**
* Hyperparameter tuning
* Deploy model using **Flask / Streamlit**
* Real-time fraud detection simulation

---

## 🧑‍💻 Author

**Abhishek Bhosale**
Machine Learning Engineer

> *"Accuracy is easy. Recall saves money."*

---

## ⭐ If you found this project useful

Give it a ⭐ on GitHub and feel free to connect!
