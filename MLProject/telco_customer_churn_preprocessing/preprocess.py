from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
PREPROCESSING_DIR = ROOT / "telco_customer_churn_preprocessing"
RAW_DATA_PATH = PREPROCESSING_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
PROCESSED_DIR = PREPROCESSING_DIR / "processed"
ARTIFACT_DIR = PREPROCESSING_DIR / "artifacts"

TARGET_COLUMN = "Churn"
DROP_COLUMNS = ["customerID"]
RANDOM_STATE = 42
TEST_SIZE = 0.2


def _validate_input(data: pd.DataFrame) -> None:
    required_columns = {TARGET_COLUMN, "TotalCharges"}
    missing_columns = sorted(required_columns.difference(data.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    target_values = set(data[TARGET_COLUMN].dropna().unique())
    expected_values = {"Yes", "No"}
    unknown_values = sorted(target_values.difference(expected_values))
    if unknown_values:
        raise ValueError(
            f"Column {TARGET_COLUMN!r} contains unsupported values: {unknown_values}"
        )


def preprocess() -> dict[str, object]:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            "Raw Telco Customer Churn dataset was not found at "
            f"{RAW_DATA_PATH}. Place the Kaggle CSV at this path before running."
        )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(RAW_DATA_PATH)
    _validate_input(data)

    data = data.copy()
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    y = data[TARGET_COLUMN].map({"No": 0, "Yes": 1}).astype(int)
    X = data.drop(columns=[TARGET_COLUMN, *DROP_COLUMNS], errors="ignore")

    numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = X.select_dtypes(exclude=["number"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ]
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out().tolist()

    pd.DataFrame(X_train_processed, columns=feature_names).to_csv(
        PROCESSED_DIR / "X_train.csv", index=False
    )
    pd.DataFrame(X_test_processed, columns=feature_names).to_csv(
        PROCESSED_DIR / "X_test.csv", index=False
    )
    y_train.rename(TARGET_COLUMN).to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    y_test.rename(TARGET_COLUMN).to_csv(PROCESSED_DIR / "y_test.csv", index=False)

    joblib.dump(preprocessor, ARTIFACT_DIR / "preprocessor.pkl")

    (PROCESSED_DIR / "feature_names.json").write_text(
        json.dumps(feature_names, indent=2), encoding="utf-8"
    )

    summary = {
        "raw_data_path": str(RAW_DATA_PATH.relative_to(ROOT)),
        "rows": int(data.shape[0]),
        "columns": int(data.shape[1]),
        "target_column": TARGET_COLUMN,
        "target_mapping": {"No": 0, "Yes": 1},
        "dropped_columns": DROP_COLUMNS,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "missing_values_after_totalcharges_conversion": int(data["TotalCharges"].isna().sum()),
        "output_files": [
            "X_train.csv",
            "X_test.csv",
            "y_train.csv",
            "y_test.csv",
            "feature_names.json",
            "preprocessing_summary.json",
        ],
        "artifact_files": ["preprocessor.pkl"],
    }
    (PROCESSED_DIR / "preprocessing_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("Preprocessing completed.")
    print(f"Processed data saved to: {PROCESSED_DIR}")
    print(f"Preprocessor artifact saved to: {ARTIFACT_DIR / 'preprocessor.pkl'}")
    return summary


if __name__ == "__main__":
    preprocess()
