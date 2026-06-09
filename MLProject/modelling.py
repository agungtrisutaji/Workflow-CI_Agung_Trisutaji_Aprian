from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT
PREPROCESSING_DIR = MODEL_DIR / "telco_customer_churn_preprocessing"
PROCESSED_DIR = PREPROCESSING_DIR / "processed"
ARTIFACT_DIR = MODEL_DIR / "artifacts"
PREPROCESS_SCRIPT = PREPROCESSING_DIR / "preprocess.py"
EXPERIMENT_NAME = "telco-customer-churn-agungtrisutaji"
RANDOM_STATE = 42


def configure_mlflow() -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    dagshub_username = os.getenv("DAGSHUB_USERNAME")
    dagshub_token = os.getenv("DAGSHUB_TOKEN")

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())

    if dagshub_username and dagshub_token:
        os.environ.setdefault("MLFLOW_TRACKING_USERNAME", dagshub_username)
        os.environ.setdefault("MLFLOW_TRACKING_PASSWORD", dagshub_token)

    if not os.getenv("MLFLOW_RUN_ID"):
        mlflow.set_experiment(EXPERIMENT_NAME)


def ensure_processed_data() -> None:
    required_files = [
        PROCESSED_DIR / "X_train.csv",
        PROCESSED_DIR / "X_test.csv",
        PROCESSED_DIR / "y_train.csv",
        PROCESSED_DIR / "y_test.csv",
    ]
    if all(path.exists() for path in required_files):
        return

    subprocess.run([sys.executable, str(PREPROCESS_SCRIPT)], check=True)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    ensure_processed_data()
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv")["Churn"]
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv")["Churn"]
    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def log_confusion_matrix(model_name: str, model, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    y_pred = model.predict(X_test)
    display = ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=["No Churn", "Churn"],
        cmap="Blues",
        colorbar=False,
    )
    display.ax_.set_title(f"Confusion Matrix - {model_name}")
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "confusion_matrix.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        mlflow.log_artifact(str(output_path), artifact_path="evaluation")


def train() -> dict[str, object]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    configure_mlflow()
    mlflow.sklearn.autolog(log_models=False)
    X_train, X_test, y_train, y_test = load_data()

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    best_result: dict[str, object] | None = None

    for model_name, model in models.items():
        with mlflow.start_run(
            run_name=f"baseline_{model_name}",
            nested=bool(os.getenv("MLFLOW_RUN_ID")),
        ):
            model.fit(X_train, y_train)
            metrics = evaluate_model(model, X_test, y_test)
            y_pred = model.predict(X_test)

            mlflow.log_param("model_name", model_name)
            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)

            report = classification_report(y_test, y_pred, target_names=["No Churn", "Churn"])
            with tempfile.TemporaryDirectory() as temp_dir:
                report_path = Path(temp_dir) / "classification_report.txt"
                report_path.write_text(report, encoding="utf-8")
                mlflow.log_artifact(str(report_path), artifact_path="evaluation")

            log_confusion_matrix(model_name, model, X_test, y_test)

            input_example = X_test.head(5)
            signature = infer_signature(input_example, model.predict(input_example))
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                input_example=input_example,
                signature=signature,
            )
            run_id = mlflow.active_run().info.run_id

            result = {
                "model_name": model_name,
                "metrics": metrics,
                "params": model.get_params(),
                "run_id": run_id,
                "model_uri": f"runs:/{run_id}/model",
                "model_artifact_uri": f"runs:/{run_id}/model",
                "model": model,
            }
            if best_result is None or metrics["f1_score"] > best_result["metrics"]["f1_score"]:
                best_result = result

    if best_result is None:
        raise RuntimeError("No model was trained.")

    best_model = best_result.pop("model")
    joblib.dump(best_model, ARTIFACT_DIR / "best_model.pkl")

    model_info = {
        "experiment_name": EXPERIMENT_NAME,
        "selection_metric": "f1_score",
        "best_metric": "f1_score",
        "best_metric_value": best_result["metrics"]["f1_score"],
        "best_model_name": best_result["model_name"],
        "best_run_id": best_result["run_id"],
        "model_uri": best_result["model_uri"],
        "model_artifact_uri": best_result["model_artifact_uri"],
        "metrics": best_result["metrics"],
        "artifact": str((ARTIFACT_DIR / "best_model.pkl").relative_to(ROOT)),
    }
    (ARTIFACT_DIR / "model_info.json").write_text(
        json.dumps(model_info, indent=2), encoding="utf-8"
    )

    print("Baseline modelling completed.")
    print(f"Best model: {model_info['best_model_name']}")
    print(f"Best f1_score: {model_info['metrics']['f1_score']:.4f}")
    print(f"Best model saved to: {ARTIFACT_DIR / 'best_model.pkl'}")
    return model_info


if __name__ == "__main__":
    train()
