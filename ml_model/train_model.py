from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

from utils.data_processing import (
    CATEGORICAL_COLS,
    FEATURE_COLS,
    NUMERICAL_COLS,
    TARGET_COL,
    process_heart_disease_data,
)

MODEL_PATH = Path("ml_model/decision_tree_model.pkl")
METRICS_PATH = Path("reports/ml_metrics.json")
SCALER_PATH = Path("data/scaler.pkl")
PREPROCESS_ARTIFACTS_PATH = Path("data/preprocessing_artifacts.pkl")
CLEANED_PATH = Path("data/cleaned_data.csv")


def evaluate_model(
    model: DecisionTreeClassifier,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict:
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    return {
        "train_accuracy": float(accuracy_score(y_train, y_train_pred)),
        "train_precision": float(precision_score(y_train, y_train_pred, zero_division=0)),
        "train_recall": float(recall_score(y_train, y_train_pred, zero_division=0)),
        "train_f1": float(f1_score(y_train, y_train_pred, zero_division=0)),
        "test_accuracy": float(accuracy_score(y_test, y_test_pred)),
        "test_precision": float(precision_score(y_test, y_test_pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, y_test_pred, zero_division=0)),
        "test_f1": float(f1_score(y_test, y_test_pred, zero_division=0)),
        "overfitting_gap_f1": float(
            f1_score(y_train, y_train_pred, zero_division=0)
            - f1_score(y_test, y_test_pred, zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(y_train.tolist() + y_test.tolist(), list(y_train_pred) + list(y_test_pred)).tolist(),
        "classification_report": classification_report(
            y_test,
            y_test_pred,
            zero_division=0,
            output_dict=True,
        ),
    }


def save_preprocessing_artifacts(feature_columns: list[str]) -> None:
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Scaler not found: {SCALER_PATH}")

    scaler = joblib.load(SCALER_PATH)
    artifacts = {
        "target_col": TARGET_COL,
        "feature_cols": FEATURE_COLS,
        "numeric_cols": NUMERICAL_COLS,
        "categorical_cols": CATEGORICAL_COLS,
        "encoded_columns": feature_columns,
        "selected_features_by_correlation": feature_columns,
        "scaler": scaler,
    }
    PREPROCESS_ARTIFACTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts, PREPROCESS_ARTIFACTS_PATH)


def run_training() -> dict:
    # Match notebook flow: train on cleaned/encoded dataset.
    cleaned_df = process_heart_disease_data(
        input_path="data/raw_data.csv",
        raw_path="data/raw_data.csv",
        output_path=str(CLEANED_PATH),
        scaler_path=str(SCALER_PATH),
        corr_threshold=0.05,
    )

    if TARGET_COL not in cleaned_df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' is missing from cleaned data.")

    X = cleaned_df.drop(columns=[TARGET_COL]).copy()
    y = cleaned_df[TARGET_COL].astype(int).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
    f1_folds = cross_val_score(model, X, y, cv=5, scoring="f1")
    cv_f1_mean = float(f1_folds.mean())

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREPROCESS_ARTIFACTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CLEANED_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    save_preprocessing_artifacts(feature_columns=list(X.columns))
    cleaned_df.to_csv(CLEANED_PATH, index=False)

    payload = {
        "model": "DecisionTreeClassifier",
        "best_params": {"random_state": 42},
        "selection_metric": "notebook_style_default_tree",
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "rows_used": int(len(cleaned_df)),
        "cross_validation": {
            "cv": 5,
            "metric": "f1",
            "f1_folds": [float(s) for s in f1_folds],
            "f1_mean": cv_f1_mean,
        },
        "metrics": metrics,
    }

    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("Decision Tree training complete (notebook-aligned flow).")
    print(f"Train Accuracy: {metrics['train_accuracy']:.4f}")
    print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Train F1: {metrics['train_f1']:.4f}")
    print(f"Test F1: {metrics['test_f1']:.4f}")
    print(f"Cross-validation F1 mean (5-fold): {cv_f1_mean:.4f}")
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved scaler to: {SCALER_PATH}")
    print(f"Saved preprocessing artifacts to: {PREPROCESS_ARTIFACTS_PATH}")
    print(f"Saved cleaned data to: {CLEANED_PATH}")
    print(f"Saved metrics to: {METRICS_PATH}")

    return payload


if __name__ == "__main__":
    run_training()
