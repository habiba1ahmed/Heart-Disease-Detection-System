from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from rule_based_system.rules import assess_patient
from utils.data_processing import FEATURE_COLS, TARGET_COL, handle_missing_values, load_dataset

DEFAULT_RAW_DATA_PATH = Path("data/raw_data.csv")
DEFAULT_PREDICTIONS_PATH = Path("data/expert_system_predictions.csv")
DEFAULT_METRICS_PATH = Path("reports/expert_metrics.json")


def _risk_level_to_prediction(risk_level: str) -> int:
    risk = str(risk_level).strip().lower()
    if risk == "high":
        return 1
    if risk == "low":
        return 0
    # Moderate/Normal/Unknown -> conservative disease-positive mapping.
    return 1


def _evaluate_rows(df: pd.DataFrame) -> tuple[list[int], list[str], list[int]]:
    predictions: list[int] = []
    risk_levels: list[str] = []
    actuals: list[int] = []

    for _, row in df.iterrows():
        patient_data = {col: row[col] for col in FEATURE_COLS}
        result = assess_patient(patient_data)
        risk_level = str(result.get("risk_level", "Moderate"))
        prediction = _risk_level_to_prediction(risk_level)

        predictions.append(prediction)
        risk_levels.append(risk_level)
        actuals.append(int(row[TARGET_COL]))

    return predictions, risk_levels, actuals


def run_expert_evaluation(
    data_path: Path = DEFAULT_RAW_DATA_PATH,
    metrics_path: Path = DEFAULT_METRICS_PATH,
    predictions_path: Path = DEFAULT_PREDICTIONS_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Evaluate expert rules on the same split policy used by ML training.
    Returns metrics payload and saves metrics/prediction artifacts.
    """
    df = load_dataset(str(data_path))
    df = handle_missing_values(df)

    required_cols = FEATURE_COLS + [TARGET_COL]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Dataset is missing required columns: {missing_str}")

    # Keep comparison fair: same split style used in ml_model/train_model.py.
    _, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    predictions, risk_levels, actuals = _evaluate_rows(test_df)

    accuracy = float(accuracy_score(actuals, predictions))
    precision = float(precision_score(actuals, predictions, zero_division=0))
    recall = float(recall_score(actuals, predictions, zero_division=0))
    f1 = float(f1_score(actuals, predictions, zero_division=0))

    payload = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "test_size": int(len(test_df)),
        "split": {"test_size": test_size, "random_state": random_state},
    }

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    out_df = test_df.copy()
    out_df["expert_risk_level"] = risk_levels
    out_df["expert_system_prediction"] = predictions
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(predictions_path, index=False)

    print("Expert System evaluation complete.")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1: {f1:.4f}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved predictions to: {predictions_path}")

    return payload


def evaluate_expert_system(data_path):
    """
    Backward-compatible wrapper returning (predictions, accuracy_percent).
    """
    payload = run_expert_evaluation(data_path=Path(data_path))
    preds_df = pd.read_csv(DEFAULT_PREDICTIONS_PATH)
    predictions = preds_df["expert_system_prediction"].astype(int).tolist()
    accuracy_percent = payload["accuracy"] * 100.0
    return predictions, accuracy_percent


if __name__ == "__main__":
    run_expert_evaluation()
