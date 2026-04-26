"""Train baseline and CP1 regression experiments for Stepik demand prediction."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TARGET_LOG = "target_log_learners_count"
TARGET_RAW = "learners_count"
SPLIT_COLUMN = "split"

DROP_COLUMNS = {
    TARGET_RAW,
    TARGET_LOG,
    SPLIT_COLUMN,
    "course_id",
    "section_id",
    "unit_id",
    "lesson_id",
    "step_id",
    "title",
    "summary",
    "description",
    "lesson_title",
    "_source",
    "_parsed_at_utc",
}

TEXT_ID_COLUMNS = {"title", "summary", "description", "lesson_title"}


@dataclass(frozen=True)
class DatasetSplit:
    """Container for train/validation/test arrays."""

    x_train: pd.DataFrame
    y_train: pd.Series
    x_val: pd.DataFrame
    y_val: pd.Series
    x_test: pd.DataFrame
    y_test: pd.Series


def load_dataset(path: Path) -> pd.DataFrame:
    """Load processed CSV and validate required columns."""
    df = pd.read_csv(path)
    required = {TARGET_LOG, TARGET_RAW, SPLIT_COLUMN}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if not {"train", "val", "test"}.issubset(set(df[SPLIT_COLUMN])):
        raise ValueError("Dataset must contain train, val and test splits")
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return model feature columns, excluding targets, ids and raw long text."""
    return [
        col
        for col in df.columns
        if col not in DROP_COLUMNS and col not in TEXT_ID_COLUMNS and not col.startswith("Unnamed")
    ]


def split_dataset(df: pd.DataFrame) -> DatasetSplit:
    """Split processed dataframe according to saved split labels."""
    features = feature_columns(df)
    train = df[df[SPLIT_COLUMN] == "train"].copy()
    val = df[df[SPLIT_COLUMN] == "val"].copy()
    test = df[df[SPLIT_COLUMN] == "test"].copy()
    return DatasetSplit(
        x_train=train[features],
        y_train=train[TARGET_LOG],
        x_val=val[features],
        y_val=val[TARGET_LOG],
        x_test=test[features],
        y_test=test[TARGET_LOG],
    )


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    """Build preprocessing transformer for numeric and categorical features."""
    numeric_features = x.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [col for col in x.columns if col not in numeric_features]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ],
        remainder="drop",
    )


def make_models() -> dict[str, Any]:
    """Create CP1 baseline and several out-of-the-box models."""
    return {
        "dummy_median": DummyRegressor(strategy="median"),
        "ridge_alpha_1": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "ridge_alpha_10": Ridge(alpha=10.0, random_state=RANDOM_STATE),
        "random_forest": RandomForestRegressor(
            n_estimators=250,
            max_depth=14,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=250,
            max_depth=16,
            min_samples_leaf=4,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=250,
            learning_rate=0.06,
            max_leaf_nodes=31,
            l2_regularization=0.1,
            random_state=RANDOM_STATE,
        ),
    }


def evaluate(model: Pipeline, x: pd.DataFrame, y_log: pd.Series) -> dict[str, float]:
    """Calculate metrics in log and original learner-count scales."""
    pred_log = model.predict(x)
    pred_log = np.maximum(pred_log, 0)
    true_raw = np.expm1(y_log)
    pred_raw = np.expm1(pred_log)
    return {
        "rmse_log": float(np.sqrt(mean_squared_error(y_log, pred_log))),
        "mae_log": float(mean_absolute_error(y_log, pred_log)),
        "r2_log": float(r2_score(y_log, pred_log)),
        "rmse_learners": float(np.sqrt(mean_squared_error(true_raw, pred_raw))),
        "mae_learners": float(mean_absolute_error(true_raw, pred_raw)),
    }


def train_experiments(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Train all CP1 models, save best model and metrics table."""
    data = split_dataset(df)
    preprocessor = build_preprocessor(data.x_train)
    models = make_models()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    best_name: str | None = None
    best_pipeline: Pipeline | None = None
    best_val_rmse = np.inf

    for name, estimator in models.items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
        pipeline.fit(data.x_train, data.y_train)
        val_metrics = evaluate(pipeline, data.x_val, data.y_val)
        rows.append({"model": name, "split": "val", **val_metrics})

        if val_metrics["rmse_log"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse_log"]
            best_name = name
            best_pipeline = pipeline

    if best_pipeline is None or best_name is None:
        raise RuntimeError("No model was trained")

    test_metrics = evaluate(best_pipeline, data.x_test, data.y_test)
    rows.append({"model": f"best:{best_name}", "split": "test", **test_metrics})

    metrics = pd.DataFrame(rows).sort_values(["split", "rmse_log"]).reset_index(drop=True)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    joblib.dump(best_pipeline, output_dir / "best_model.joblib")

    metadata = {
        "target": TARGET_LOG,
        "raw_target": TARGET_RAW,
        "best_model": best_name,
        "selection_metric": "min validation RMSE on log1p(learners_count)",
        "features": feature_columns(df),
        "random_state": RANDOM_STATE,
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CP1 models for Stepik course-demand regression")
    parser.add_argument("--input", type=Path, default=Path("data/processed/stepik_course_steps_processed.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    args = parser.parse_args()

    df = load_dataset(args.input)
    metrics = train_experiments(df, args.output_dir)
    print(f"Saved metrics to {args.output_dir / 'metrics.csv'}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
