from __future__ import annotations

import json
import os
import subprocess
import sys
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
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV


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


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def save_and_log_evaluation_artifacts(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    metrics: dict[str, float],
) -> None:
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, target_names=["No Churn", "Churn"])
    report_path = ARTIFACT_DIR / "tuning_classification_report.txt"
    report_path.write_text(report, encoding="utf-8")
    mlflow.log_artifact(str(report_path), artifact_path="tuning_evaluation")

    metric_info_path = ARTIFACT_DIR / "tuning_metric_info.json"
    metric_info_path.write_text(
        json.dumps(
            {
                "selection_metric": "f1_score",
                "metrics": metrics,
                "notes": "Metrics calculated on the held-out test set.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    mlflow.log_artifact(str(metric_info_path), artifact_path="tuning_evaluation")

    display = ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=["No Churn", "Churn"],
        cmap="Blues",
        colorbar=False,
    )
    display.ax_.set_title("Confusion Matrix - Tuned RandomForestClassifier")
    confusion_matrix_path = ARTIFACT_DIR / "tuning_confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(confusion_matrix_path, dpi=150)
    plt.close()
    mlflow.log_artifact(str(confusion_matrix_path), artifact_path="tuning_evaluation")

    if hasattr(model, "feature_importances_"):
        feature_importance = (
            pd.DataFrame(
                {
                    "feature": X_test.columns,
                    "importance": model.feature_importances_,
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        feature_importance_path = ARTIFACT_DIR / "tuning_feature_importance.csv"
        feature_importance.to_csv(feature_importance_path, index=False)
        mlflow.log_artifact(str(feature_importance_path), artifact_path="tuning_evaluation")


def tune() -> dict[str, object]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    configure_mlflow()
    X_train, X_test, y_train, y_test = load_data()

    estimator = RandomForestClassifier(
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    param_grid = {
        "n_estimators": [80, 120],
        "max_depth": [8, 12],
        "min_samples_leaf": [1, 2],
    }
    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring="f1",
        cv=3,
        n_jobs=-1,
        verbose=0,
    )

    with mlflow.start_run(
        run_name="tuning_random_forest",
        nested=bool(os.getenv("MLFLOW_RUN_ID")),
    ):
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        metrics = evaluate(best_model, X_test, y_test)

        mlflow.log_param("model_name", "RandomForestClassifier")
        mlflow.log_params(search.best_params_)
        mlflow.log_metric("best_cv_f1_score", search.best_score_)
        mlflow.log_metrics(metrics)
        save_and_log_evaluation_artifacts(best_model, X_test, y_test, metrics)

        input_example = X_test.head(5)
        signature = infer_signature(input_example, best_model.predict(input_example))
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            input_example=input_example,
            signature=signature,
        )

        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/model"
        model_artifact_uri = model_uri

    joblib.dump(best_model, ARTIFACT_DIR / "tuned_model.pkl")
    results = {
        "experiment_name": EXPERIMENT_NAME,
        "model_name": "RandomForestClassifier",
        "selection_metric": "f1_score",
        "best_cv_f1_score": float(search.best_score_),
        "best_params": search.best_params_,
        "test_metrics": metrics,
        "run_id": run_id,
        "model_uri": model_uri,
        "model_artifact_uri": model_artifact_uri,
        "artifact": str((ARTIFACT_DIR / "tuned_model.pkl").relative_to(ROOT)),
        "local_evaluation_artifacts": [
            "tuning_classification_report.txt",
            "tuning_confusion_matrix.png",
            "tuning_metric_info.json",
            "tuning_feature_importance.csv",
        ],
    }
    (ARTIFACT_DIR / "tuning_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    print("Hyperparameter tuning completed.")
    print(f"Best params: {results['best_params']}")
    print(f"Test f1_score: {results['test_metrics']['f1_score']:.4f}")
    print(f"Tuned model saved to: {ARTIFACT_DIR / 'tuned_model.pkl'}")
    return results


if __name__ == "__main__":
    tune()
