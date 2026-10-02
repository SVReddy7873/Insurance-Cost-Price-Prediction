
from __future__ import annotations

import argparse
import json
import logging
import math
import re
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger("insurance-trainer")

TARGET = "charges"
DEFAULT_DATA = Path("data/insurance.csv")
DEFAULT_MODEL = Path("models/insurance_cost_pipeline.joblib")
DEFAULT_METRICS = Path("models/metrics.json")

ALIASES = {
    "age": "age", "sex": "sex", "gender": "sex",
    "bmi": "bmi", "body_mass_index": "bmi",
    "children": "children", "child": "children",
    "smoker": "smoker", "smoking": "smoker",
    "region": "region", "area": "region",
    "charges": "charges", "charge": "charges",
    "insurance_charges": "charges", "medical_cost": "charges",
}

def normalize_name(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", str(name).strip().lower()).strip("_")
    return ALIASES.get(s, s)

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [normalize_name(c) for c in out.columns]
    # Remove accidental duplicate columns while preserving the first.
    out = out.loc[:, ~out.columns.duplicated()]
    return out

def coerce_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out

def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    required = {"age", "bmi", "children", "smoker", "region", TARGET}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(sorted(missing)) +
            ". Expected columns similar to: age, sex, bmi, children, smoker, region, charges."
        )

    df = coerce_numeric(df, ["age", "bmi", "children", TARGET])

    # Invalid values become NaN and are handled by the pipeline's imputers.
    # Rows without a valid target cannot be used for supervised training.
    df = df.dropna(subset=[TARGET]).copy()
    if df.empty:
        raise ValueError("No valid target values were found in the dataset.")

    # Keep physically plausible ranges; this protects the model from obvious corrupt rows.
    df.loc[(df["age"] < 0) | (df["age"] > 120), "age"] = np.nan
    df.loc[(df["bmi"] <= 0) | (df["bmi"] > 100), "bmi"] = np.nan
    df.loc[(df["children"] < 0) | (df["children"] > 30), "children"] = np.nan

    for col in ["sex", "smoker", "region"]:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip().str.lower()

    if len(df) < 20:
        raise ValueError("At least 20 usable rows are recommended for training.")

    return df

def make_pipeline(random_state: int = 42) -> Pipeline:
    numeric = ["age", "bmi", "children"]
    categorical = ["sex", "smoker", "region"]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical),
    ], remainder="drop")

    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=1,
        max_features=0.8,
        random_state=random_state,
        n_jobs=-1,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

def train(data_path: Path, model_path: Path, metrics_path: Path, test_size: float, seed: int) -> dict:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    raw = pd.read_csv(data_path)
    df = validate_dataset(clean_columns(raw))

    features = ["age", "bmi", "children", "sex", "smoker", "region"]
    X = df[features].copy()
    y = df[TARGET].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    pipeline = make_pipeline(seed)
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    metrics = {
        "rows_total": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2),
        "features": features,
        "target": TARGET,
        "random_state": seed,
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    LOGGER.info("Model saved to %s", model_path)
    LOGGER.info("MAE=%.2f | RMSE=%.2f | R²=%.4f", mae, rmse, r2)
    return metrics

def main() -> None:
    parser = argparse.ArgumentParser(description="Train a robust health insurance cost prediction model.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0.05 <= args.test_size <= 0.40:
        raise ValueError("--test-size must be between 0.05 and 0.40.")

    train(args.data, args.model, args.metrics, args.test_size, args.seed)

if __name__ == "__main__":
    main()
