"""Train baseline, tuned models and interpretation artifacts for Stepik demand prediction."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TARGET_LOG = "target_log_learners_count"
TARGET_RAW = "learners_count"
SPLIT_COLUMN = "split"
GROUP_COLUMN = "course_id"

DROP_COLUMNS = {
    TARGET_RAW,
    TARGET_LOG,
    SPLIT_COLUMN,
    GROUP_COLUMN,
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


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clip numeric feature outliers using train-only quantile borders.

    The transformer is fitted only on the training fold inside the sklearn
    pipeline, so validation/test values do not leak into clipping thresholds.
    """

    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99) -> None:
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, x: pd.DataFrame | np.ndarray, y: pd.Series | None = None) -> "QuantileClipper":
        frame = pd.DataFrame(x)
        self.lower_bounds_ = frame.quantile(self.lower_quantile, numeric_only=True)
        self.upper_bounds_ = frame.quantile(self.upper_quantile, numeric_only=True)
        return self

    def transform(self, x: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        frame = pd.DataFrame(x).copy()
        return frame.clip(lower=self.lower_bounds_, upper=self.upper_bounds_, axis=1)


@dataclass(frozen=True)
class DatasetSplit:
    """Container for train/validation/test arrays."""

    x_train: pd.DataFrame
    y_train: pd.Series
    groups_train: pd.Series
    x_val: pd.DataFrame
    y_val: pd.Series
    x_test: pd.DataFrame
    y_test: pd.Series


def load_dataset(path: Path) -> pd.DataFrame:
    """Load processed CSV and validate required columns."""
    df = pd.read_csv(path)
    required = {TARGET_LOG, TARGET_RAW, SPLIT_COLUMN, GROUP_COLUMN}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if not {"train", "val", "test"}.issubset(set(df[SPLIT_COLUMN])):
        raise ValueError("Dataset must contain train, val and test splits")
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return model feature columns, excluding targets, ids and raw long text."""
    return [
        column
        for column in df.columns
        if column not in DROP_COLUMNS
        and column not in TEXT_ID_COLUMNS
        and not column.startswith("Unnamed")
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
        groups_train=train[GROUP_COLUMN].astype(str),
        x_val=val[features],
        y_val=val[TARGET_LOG],
        x_test=test[features],
        y_test=test[TARGET_LOG],
    )


def build_preprocessor(x: pd.DataFrame, *, clip_outliers: bool = True) -> ColumnTransformer:
    """Build preprocessing transformer for numeric and categorical features."""
    numeric_features = x.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [column for column in x.columns if column not in numeric_features]

    numeric_steps: list[tuple[str, Any]] = []
    if clip_outliers:
        numeric_steps.append(("clipper", QuantileClipper(lower_quantile=0.01, upper_quantile=0.99)))
    numeric_steps.extend(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    numeric_pipe = Pipeline(steps=numeric_steps)
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


def make_base_models() -> dict[str, Any]:
    """Create baseline and several non-tuned models."""
    return {
        "dummy_median": DummyRegressor(strategy="median"),
        "ridge_alpha_1": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "ridge_alpha_10": Ridge(alpha=10.0, random_state=RANDOM_STATE),
        "random_forest_default": RandomForestRegressor(
            n_estimators=250,
            max_depth=14,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "extra_trees_default": ExtraTreesRegressor(
            n_estimators=250,
            max_depth=16,
            min_samples_leaf=4,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "hist_gradient_boosting_default": HistGradientBoostingRegressor(
            max_iter=250,
            learning_rate=0.06,
            max_leaf_nodes=31,
            l2_regularization=0.1,
            random_state=RANDOM_STATE,
        ),
    }


def make_searches(preprocessor: ColumnTransformer, n_iter: int) -> dict[str, RandomizedSearchCV]:
    """Create RandomizedSearchCV objects for the strongest model families."""
    cv = GroupKFold(n_splits=3)

    hist_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", HistGradientBoostingRegressor(random_state=RANDOM_STATE)),
        ]
    )
    extra_trees_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", ExtraTreesRegressor(n_jobs=-1, random_state=RANDOM_STATE)),
        ]
    )

    return {
        "hist_gradient_boosting_random_search": RandomizedSearchCV(
            estimator=hist_pipeline,
            param_distributions={
                "model__learning_rate": [0.02, 0.04, 0.06, 0.08, 0.1],
                "model__max_iter": [150, 250, 350, 500],
                "model__max_leaf_nodes": [15, 31, 63],
                "model__min_samples_leaf": [10, 20, 40, 80],
                "model__l2_regularization": [0.0, 0.01, 0.1, 1.0],
            },
            n_iter=n_iter,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            refit=True,
            return_train_score=True,
        ),
        "extra_trees_random_search": RandomizedSearchCV(
            estimator=extra_trees_pipeline,
            param_distributions={
                "model__n_estimators": [200, 400, 700],
                "model__max_depth": [8, 12, 16, 24, None],
                "model__min_samples_leaf": [1, 2, 4, 8],
                "model__max_features": ["sqrt", "log2", 0.5, 0.8, 1.0],
            },
            n_iter=n_iter,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            refit=True,
            return_train_score=True,
        ),
    }


def evaluate(model: Pipeline, x: pd.DataFrame, y_log: pd.Series) -> dict[str, float]:
    """Calculate metrics in log and original learner-count scales."""
    pred_log = np.maximum(model.predict(x), 0)
    true_raw = np.expm1(y_log)
    pred_raw = np.expm1(pred_log)
    return {
        "rmse_log": float(np.sqrt(mean_squared_error(y_log, pred_log))),
        "mae_log": float(mean_absolute_error(y_log, pred_log)),
        "r2_log": float(r2_score(y_log, pred_log)),
        "rmse_learners": float(np.sqrt(mean_squared_error(true_raw, pred_raw))),
        "mae_learners": float(mean_absolute_error(true_raw, pred_raw)),
    }


def make_outlier_report(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Save IQR/quantile outlier diagnostics for numeric columns."""
    numeric = df[feature_columns(df) + [TARGET_RAW, TARGET_LOG]].select_dtypes(include=["number", "bool"])
    rows: list[dict[str, Any]] = []
    for column in numeric.columns:
        series = pd.to_numeric(numeric[column], errors="coerce").dropna()
        if series.empty:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_iqr = q1 - 1.5 * iqr
        upper_iqr = q3 + 1.5 * iqr
        rows.append(
            {
                "feature": column,
                "n_missing": int(numeric[column].isna().sum()),
                "p01": float(series.quantile(0.01)),
                "p25": float(q1),
                "median": float(series.median()),
                "p75": float(q3),
                "p99": float(series.quantile(0.99)),
                "min": float(series.min()),
                "max": float(series.max()),
                "iqr_outliers": int(((series < lower_iqr) | (series > upper_iqr)).sum()),
                "iqr_outlier_share": float(((series < lower_iqr) | (series > upper_iqr)).mean()),
            }
        )
    report = pd.DataFrame(rows).sort_values("iqr_outlier_share", ascending=False)
    report.to_csv(output_dir / "outlier_report.csv", index=False)
    return report


def save_permutation_importance(model: Pipeline, data: DatasetSplit, output_dir: Path) -> pd.DataFrame:
    """Save model-agnostic feature importance on validation data."""
    result = permutation_importance(
        model,
        data.x_val,
        data.y_val,
        n_repeats=5,
        random_state=RANDOM_STATE,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    importance = pd.DataFrame(
        {
            "feature": data.x_val.columns,
            "importance_mean_rmse_increase": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean_rmse_increase", ascending=False)
    importance.to_csv(output_dir / "permutation_importance.csv", index=False)
    return importance


def _append_metrics_row(
    rows: list[dict[str, Any]],
    model_name: str,
    split: str,
    metrics: dict[str, float],
    params: dict[str, Any] | None = None,
) -> None:
    rows.append(
        {
            "model": model_name,
            "split": split,
            **metrics,
            "best_params": json.dumps(params or {}, ensure_ascii=False, sort_keys=True),
        }
    )


def train_experiments(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    run_search: bool = True,
    search_iter: int = 12,
) -> pd.DataFrame:
    """Train models, optional hyperparameter searches, and save artifacts."""
    data = split_dataset(df)
    output_dir.mkdir(parents=True, exist_ok=True)
    make_outlier_report(df, output_dir)

    rows: list[dict[str, Any]] = []
    best_name: str | None = None
    best_pipeline: Pipeline | None = None
    best_val_rmse = np.inf

    preprocessor = build_preprocessor(data.x_train, clip_outliers=True)
    for name, estimator in make_base_models().items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
        pipeline.fit(data.x_train, data.y_train)
        val_metrics = evaluate(pipeline, data.x_val, data.y_val)
        _append_metrics_row(rows, name, "val", val_metrics)
        if val_metrics["rmse_log"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse_log"]
            best_name = name
            best_pipeline = pipeline

    if run_search:
        for name, search in make_searches(preprocessor, n_iter=search_iter).items():
            search.fit(data.x_train, data.y_train, groups=data.groups_train)
            pipeline = search.best_estimator_
            val_metrics = evaluate(pipeline, data.x_val, data.y_val)
            _append_metrics_row(rows, name, "val", val_metrics, search.best_params_)
            pd.DataFrame(search.cv_results_).to_csv(output_dir / f"{name}_cv_results.csv", index=False)
            if val_metrics["rmse_log"] < best_val_rmse:
                best_val_rmse = val_metrics["rmse_log"]
                best_name = name
                best_pipeline = pipeline

    if best_pipeline is None or best_name is None:
        raise RuntimeError("No model was trained")

    test_metrics = evaluate(best_pipeline, data.x_test, data.y_test)
    _append_metrics_row(rows, f"best:{best_name}", "test", test_metrics)

    metrics = pd.DataFrame(rows).sort_values(["split", "rmse_log"]).reset_index(drop=True)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    joblib.dump(best_pipeline, output_dir / "best_model.joblib")
    importance = save_permutation_importance(best_pipeline, data, output_dir)

    metadata = {
        "target": TARGET_LOG,
        "raw_target": TARGET_RAW,
        "best_model": best_name,
        "selection_metric": "min validation RMSE on log1p(learners_count)",
        "secondary_metrics": ["mae_log", "r2_log", "rmse_learners", "mae_learners"],
        "features": feature_columns(df),
        "top_permutation_features": importance.head(10).to_dict(orient="records"),
        "random_state": RANDOM_STATE,
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "outlier_processing": "numeric feature clipping by 1st/99th train quantiles inside sklearn pipeline",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CP2 models for Stepik course-demand regression")
    parser.add_argument("--input", type=Path, default=Path("data/processed/stepik_course_steps_processed.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--no-search", action="store_true", help="Disable RandomizedSearchCV for a fast smoke run")
    parser.add_argument("--search-iter", type=int, default=12, help="Number of sampled hyperparameter sets per tuned model")
    args = parser.parse_args()

    df = load_dataset(args.input)
    metrics = train_experiments(
        df,
        args.output_dir,
        run_search=not args.no_search,
        search_iter=args.search_iter,
    )
    print(f"Saved metrics to {args.output_dir / 'metrics.csv'}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
