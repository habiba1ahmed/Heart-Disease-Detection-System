from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = Path("ml_model/decision_tree_model.pkl")
SCALER_PATH = Path("data/scaler.pkl")
PREPROCESS_ARTIFACTS_PATH = Path("data/preprocessing_artifacts.pkl")
CLEAN_PATH = Path("data/cleaned_data.csv")

NUMERICAL_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]


def _load_preprocessing_artifacts() -> dict | None:
    if not PREPROCESS_ARTIFACTS_PATH.exists():
        return None
    try:
        artifacts = joblib.load(PREPROCESS_ARTIFACTS_PATH)
    except Exception:
        return None
    return artifacts if isinstance(artifacts, dict) else None


def load_model_and_scaler() -> tuple[object, object]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Run ml_model/train_model.py first.")

    model = joblib.load(MODEL_PATH)
    scaler = None
    if SCALER_PATH.exists():
        scaler = joblib.load(SCALER_PATH)
    else:
        artifacts = _load_preprocessing_artifacts()
        if artifacts is not None:
            scaler = artifacts.get("scaler")

    if scaler is None:
        raise FileNotFoundError(
            f"Scaler not found: {SCALER_PATH}. Run ml_model/train_model.py first."
        )
    return model, scaler


def _required_feature_columns(model: object) -> list[str]:
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    artifacts = _load_preprocessing_artifacts()
    if artifacts is not None:
        selected = artifacts.get("selected_features_by_correlation") or artifacts.get("encoded_columns")
        if isinstance(selected, list) and selected:
            return [str(c) for c in selected if str(c) != "target"]

    if not CLEAN_PATH.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {CLEAN_PATH}")

    df = pd.read_csv(CLEAN_PATH)
    return [c for c in df.columns if c != "target"]


def _candidate_category_suffixes(value: object) -> list[str]:
    candidates: list[str] = []

    def add(candidate: object) -> None:
        text = str(candidate)
        if text and text not in candidates:
            candidates.append(text)

    add(value)
    try:
        numeric = float(value)
        add(numeric)
        add(f"{numeric:.1f}")
        if numeric.is_integer():
            add(int(numeric))
    except (TypeError, ValueError):
        pass

    return candidates


def _build_scaled_numeric_row(input_data: dict, scaler: object) -> tuple[list[str], np.ndarray]:
    scaler_cols = list(getattr(scaler, "feature_names_in_", NUMERICAL_COLS))
    missing_numeric = [c for c in scaler_cols if c not in input_data]
    if missing_numeric:
        missing_str = ", ".join(missing_numeric)
        raise KeyError(f"Missing required numeric input(s): {missing_str}")

    numeric_df = pd.DataFrame([{c: float(input_data[c]) for c in scaler_cols}], columns=scaler_cols)
    scaled_values = scaler.transform(numeric_df)[0]
    return scaler_cols, scaled_values


def _apply_categorical_encoding(processed: pd.DataFrame, input_data: dict) -> None:
    for col in CATEGORICAL_COLS:
        if col not in input_data:
            continue

        if col in processed.columns:
            processed.at[0, col] = float(input_data[col])
            continue

        for suffix in _candidate_category_suffixes(input_data[col]):
            encoded_col = f"{col}_{suffix}"
            if encoded_col in processed.columns:
                processed.at[0, encoded_col] = 1.0
                break


def preprocess_single(input_data: dict, scaler: object, model: object) -> pd.DataFrame:
    required_cols = _required_feature_columns(model)
    processed = pd.DataFrame(0.0, index=[0], columns=required_cols)

    scaled_cols, scaled_values = _build_scaled_numeric_row(input_data, scaler)
    for col, val in zip(scaled_cols, scaled_values):
        if col in processed.columns:
            processed.at[0, col] = float(val)

    for col, val in input_data.items():
        if col in processed.columns and col not in scaled_cols:
            try:
                processed.at[0, col] = float(val)
            except (TypeError, ValueError):
                continue

    _apply_categorical_encoding(processed, input_data)
    return processed


def _predict_probability_if_available(model: object, processed: pd.DataFrame) -> float | None:
    if not hasattr(model, "predict_proba"):
        return None

    classes = list(getattr(model, "classes_", [0, 1]))
    # In this dataset, target=0 actually correlates with disease indicators (high ca, high oldpeak, exang=1)
    # So the probability of disease is the probability of class 0.
    if 0 in classes:
        positive_index = classes.index(0)
    else:
        positive_index = 0

    proba = float(model.predict_proba(processed)[0][positive_index])
    return float(np.clip(proba, 0.0, 1.0))


def predict_heart_disease(input_data: dict) -> dict:
    model, scaler = load_model_and_scaler()
    processed = preprocess_single(input_data, scaler, model)
    prediction = int(model.predict(processed)[0])
    
    # target=0 is High Risk (Heart Disease), target=1 is Normal (Healthy)
    probability = _predict_probability_if_available(model, processed)
    
    if probability is not None:
        if probability >= 0.75:
            risk_level = "High"
        elif probability >= 0.50:
            risk_level = "Moderate"
        elif probability >= 0.25:
            risk_level = "Low"
        else:
            risk_level = "Normal"
    else:
        risk_level = "High" if prediction == 0 else "Normal"
        
    result = {
        "prediction": prediction,
        "risk_level": risk_level,
        "label": risk_level,
    }
    return result


def predict_with_probability(input_data: dict, threshold: float | None = None) -> dict:
    # Backward-compatible wrapper.
    _ = threshold
    return predict_heart_disease(input_data)


def predict_risk_level(input_data: dict) -> dict:
    # Backward-compatible alias expected by some app paths.
    return predict_heart_disease(input_data)


if __name__ == "__main__":
    test_patient_1 = {
        "age": 65,
        "sex": 1,
        "cp": 3,
        "trestbps": 150,
        "chol": 260,
        "fbs": 1,
        "restecg": 1,
        "thalach": 120,
        "exang": 1,
        "oldpeak": 2.5,
        "slope": 2,
        "ca": 3,
        "thal": 2,
    }

    test_patient_2 = {
        "age": 35,
        "sex": 0,
        "cp": 0,
        "trestbps": 110,
        "chol": 180,
        "fbs": 0,
        "restecg": 0,
        "thalach": 170,
        "exang": 0,
        "oldpeak": 0.5,
        "slope": 1,
        "ca": 0,
        "thal": 3,
    }

    print("test_patient_1", test_patient_1)
    print(predict_heart_disease(test_patient_1))

    print("test_patient_2", test_patient_2)
    print(predict_heart_disease(test_patient_2))
