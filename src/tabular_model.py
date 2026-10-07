"""
STAGE 6 — Tabular ML pipeline (Wisconsin Diagnostic Breast Cancer).
Proper train/test split BEFORE fitting anything; scaler fit on train only;
evaluation strictly on the held-out test set.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve,
)
from xgboost import XGBClassifier


def load_and_clean(path="data/data.csv"):
    df = pd.read_csv(path)
    for col in ["Unnamed: 32", "id"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
    df = df.dropna(subset=["diagnosis"])
    return df


def split_and_scale(df, test_size=0.2, seed=42):
    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"].astype(int)

    # split FIRST
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    # scaler fit on TRAIN ONLY, then applied to test -> no leakage
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X.columns, index=X_test.index
    )
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_xgb(X_train, y_train, seed=42):
    model = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        base_score=0.5, random_state=seed, eval_metric="logloss",
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    return metrics, cm, (fpr, tpr), y_pred, y_prob


def save_model(model, scaler, path_model="models/tabular_xgb.joblib", path_scaler="models/tabular_scaler.joblib"):
    joblib.dump(model, path_model)
    joblib.dump(scaler, path_scaler)
