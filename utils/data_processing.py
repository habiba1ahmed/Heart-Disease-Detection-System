import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler



NUMERICAL_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
TARGET_COL = "target"

FEATURE_COLS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]


def load_dataset(input_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    df = df.drop_duplicates()
    print(f"  Original shape : {df.shape}")
    print(f"  Columns        : {list(df.columns)}")
    return df


def save_raw_backup(df: pd.DataFrame, raw_path: str, source_path: str | None = None) -> None:
    if source_path and os.path.abspath(source_path) == os.path.abspath(raw_path):
        print(f"  Raw backup skipped (source already at: {raw_path})")
        return

    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    df.to_csv(raw_path, index=False)
    print(f"  Raw backup saved to: {raw_path}")


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    print("\nHandling missing values")

    out = df.copy()

    for col in NUMERICAL_COLS:
        if col in out.columns and out[col].isnull().sum() > 0:
            mean_val = out[col].mean()
            out[col] = out[col].fillna(mean_val)
            print(f"  Filled '{col}' with mean ({mean_val:.2f})")

    for col in CATEGORICAL_COLS:
        if col in out.columns and out[col].isnull().sum() > 0:
            mode_val = out[col].mode()[0]
            out[col] = out[col].fillna(mode_val)
            print(f"  Filled '{col}' with mode ({mode_val})")

    if TARGET_COL in out.columns:
        out[TARGET_COL] = out[TARGET_COL].astype(int)

    return out


def normalize_features(df: pd.DataFrame, scaler_path: str) -> pd.DataFrame:
    print("\nNormalizing numerical features")

    out = df.copy()
    existing_num_cols = [col for col in NUMERICAL_COLS if col in out.columns]

    scaler = MinMaxScaler()
    out[existing_num_cols] = scaler.fit_transform(out[existing_num_cols])

    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)

    print(f"  Scaled columns : {existing_num_cols}")
    print(f"  Scaler saved to: {scaler_path}")

    return out


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    print("\nEncoding categorical variables")

    out = df.copy()
    existing_cat_cols = [col for col in CATEGORICAL_COLS if col in out.columns]
    out = pd.get_dummies(out, columns=existing_cat_cols, drop_first=True, dtype=int)

    bool_cols = out.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        out[bool_cols] = out[bool_cols].astype(int)

    if TARGET_COL in out.columns:
        out[TARGET_COL] = out[TARGET_COL].astype(int)

    print(f"  Encoded columns : {existing_cat_cols}")
    print(f"  New shape       : {out.shape}")

    return out


def select_features(df: pd.DataFrame, corr_threshold: float = 0.05) -> pd.DataFrame:
    print("\nPerforming feature selection (correlation)")

    corr = df.corr(numeric_only=True)[TARGET_COL].abs().sort_values(ascending=False)

    print("\n  Top 10 features correlated with target:")
    print(corr.head(11).to_string())

    important_features = corr[corr > corr_threshold].index.tolist()
    out = df[important_features].copy()

    bool_cols = out.select_dtypes(include=["bool", "boolean"]).columns.tolist()
    if bool_cols:
        out[bool_cols] = out[bool_cols].astype(int)

    if TARGET_COL in out.columns:
        out[TARGET_COL] = out[TARGET_COL].astype(int)

    print(f"\n  Kept {len(important_features)} features (threshold = {corr_threshold})")

    return out


def save_cleaned_data(df: pd.DataFrame, output_path: str) -> None:
    out = df.copy()
    bool_cols = out.select_dtypes(include=["bool", "boolean"]).columns.tolist()
    if bool_cols:
        out[bool_cols] = out[bool_cols].astype(int)

    if TARGET_COL in out.columns:
        out[TARGET_COL] = out[TARGET_COL].astype(int)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"\n  Cleaned dataset saved to: {output_path}")
    print(f"  Final shape: {out.shape}")


def process_heart_disease_data(
    input_path: str = "data/raw_data.csv",
    raw_path: str = "data/raw_data.csv",
    output_path: str = "data/cleaned_data.csv",
    scaler_path: str = "data/scaler.pkl",
    corr_threshold: float = 0.05,
) -> pd.DataFrame:
    df = load_dataset(input_path)
    save_raw_backup(df, raw_path, source_path=input_path)
    df = handle_missing_values(df)
    df = normalize_features(df, scaler_path)
    df = encode_categorical(df)
    df = select_features(df, corr_threshold)
    save_cleaned_data(df, output_path)

    return df


if __name__ == "__main__":
    process_heart_disease_data()
