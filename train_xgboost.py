import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, classification_report

# Load splits
X_train = pd.read_parquet("data/splits/X_train.parquet")
y_train = pd.read_parquet("data/splits/y_train.parquet")["label"]

X_val = pd.read_parquet("data/splits/X_val.parquet")
y_val = pd.read_parquet("data/splits/y_val.parquet")["label"]

X_test = pd.read_parquet("data/splits/X_test.parquet")
y_test = pd.read_parquet("data/splits/y_test.parquet")["label"]

model = XGBClassifier(
    max_depth=4,
    learning_rate=0.05,
    n_estimators=300,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=True
)

val_probs = model.predict_proba(X_val)[:, 1]
val_preds = (val_probs >= 0.5).astype(int)

precision_1 = precision_score(y_val, val_preds, pos_label=1)
recall_1 = recall_score(y_val, val_preds, pos_label=1)

print("\nVALIDATION METRICS (class 1)")
print("Precision:", round(precision_1, 4))
print("Recall:   ", round(recall_1, 4))

print("\nValidation prediction confidence (probability bins):")

bins = [0.0, 0.55, 0.6, 0.65, 0.7, 0.8, 1.0]
hist = pd.cut(val_probs, bins=bins).value_counts().sort_index()

print(hist)

test_probs = model.predict_proba(X_test)[:, 1]
test_preds = (test_probs >= 0.5).astype(int)

test_precision = precision_score(y_test, test_preds, pos_label=1)
test_recall = recall_score(y_test, test_preds, pos_label=1)

print("\nTEST METRICS (class 1)")
print("Precision:", round(test_precision, 4))
print("Recall:   ", round(test_recall, 4))

print("\nClassification report (TEST):")
print(classification_report(y_test, test_preds))

import joblib
joblib.dump(model, "data/models/xgb_tp_sl_model.pkl")

print("Model saved")
