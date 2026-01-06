# =========================
# 1. IMPORT LIBRARIES
# =========================
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score
)

# =========================
# 2. LOAD DATASET
# =========================
data = pd.read_csv(
    "C:\\Users\\bhosa\\MACHINE_LEARNING\\Credit_card_fraud_detection\\creditcard.csv"
)

print(data.head())
print(data.info())

# =========================
# 3. CHECK CLASS IMBALANCE
# =========================
print("\nClass distribution:")
print(data['Class'].value_counts())

# =========================
# 4. FEATURE & TARGET
# =========================
X = data.drop(columns='Class', axis=1)
Y = data['Class']

# =========================
# 5. FEATURE SCALING
# =========================
scaler = StandardScaler()
X[['Time', 'Amount']] = scaler.fit_transform(X[['Time', 'Amount']])

# =========================
# 6. TRAIN-TEST SPLIT
# =========================
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    stratify=Y,
    random_state=42
)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)

# =========================
# 7. MODEL TRAINING
# =========================
model = LogisticRegression(
    class_weight='balanced',   # VERY IMPORTANT
    max_iter=1000
)

model.fit(X_train, Y_train)

# =========================
# 8. PREDICTION
# =========================
Y_pred = model.predict(X_test)
Y_prob = model.predict_proba(X_test)[:, 1]

# =========================
# 9. EVALUATION
# =========================
print("\nCONFUSION MATRIX:")
print(confusion_matrix(Y_test, Y_pred))

print("\nCLASSIFICATION REPORT:")
print(classification_report(Y_test, Y_pred))

print("ROC-AUC SCORE:", roc_auc_score(Y_test, Y_prob))
