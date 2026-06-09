# Telco Customer Churn Preprocessing

This folder stores the raw Kaggle dataset and the reproducible preprocessing script.

Expected raw CSV:

```text
Membangun_model/telco_customer_churn_preprocessing/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

Run from the repository root after activating `.venv`:

```bash
python Membangun_model/telco_customer_churn_preprocessing/preprocess.py
```

Outputs:

```text
processed/X_train.csv
processed/X_test.csv
processed/y_train.csv
processed/y_test.csv
processed/feature_names.json
processed/preprocessing_summary.json
artifacts/preprocessor.pkl
```

The preprocessor fits only on training data, converts `TotalCharges` to numeric, maps `Churn` from `No`/`Yes` to `0`/`1`, imputes missing values, scales numeric features, and one-hot encodes categorical features.

