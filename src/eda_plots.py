"""Create CP2 EDA plots: outliers and feature-vs-target dependencies."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TARGET_RAW = "learners_count"
TARGET_LOG = "target_log_learners_count"
SPLIT_COLUMN = "split"


NUMERIC_FEATURES_FOR_PLOTS = [
    "rating",
    "reviews_count",
    "lessons_count",
    "sections_count",
    "lesson_steps_count",
    "title_len",
    "description_word_count",
]
CATEGORICAL_FEATURES_FOR_PLOTS = ["language", "step_type", "is_paid", "has_certificate"]


def _save_current(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_target_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(df[TARGET_RAW].clip(lower=0), bins=60)
    plt.xlabel("learners_count")
    plt.ylabel("Количество строк")
    plt.title("Распределение таргета в исходной шкале")
    _save_current(output_dir / "target_distribution_raw.png")

    plt.figure(figsize=(8, 5))
    plt.hist(df[TARGET_LOG], bins=60)
    plt.xlabel("log1p(learners_count)")
    plt.ylabel("Количество строк")
    plt.title("Распределение таргета после log1p")
    _save_current(output_dir / "target_distribution_log.png")


def plot_numeric_dependency(df: pd.DataFrame, feature: str, output_dir: Path) -> None:
    if feature not in df.columns:
        return
    sample = df[[feature, TARGET_LOG]].dropna().sample(
        n=min(5000, df[[feature, TARGET_LOG]].dropna().shape[0]),
        random_state=42,
    )
    if sample.empty:
        return

    x = sample[feature]
    if x.min() >= 0 and x.max() > 100:
        x = np.log1p(x)
        xlabel = f"log1p({feature})"
    else:
        xlabel = feature

    plt.figure(figsize=(8, 5))
    plt.scatter(x, sample[TARGET_LOG], alpha=0.25, s=10)
    plt.xlabel(xlabel)
    plt.ylabel("log1p(learners_count)")
    plt.title(f"Зависимость таргета от {feature}")
    _save_current(output_dir / f"feature_vs_target_{feature}.png")


def plot_categorical_dependency(df: pd.DataFrame, feature: str, output_dir: Path) -> None:
    if feature not in df.columns:
        return
    work = df[[feature, TARGET_LOG]].dropna().copy()
    if work.empty:
        return
    work[feature] = work[feature].astype(str).fillna("unknown")
    top_values = work[feature].value_counts().head(12).index
    work = work[work[feature].isin(top_values)]
    means = work.groupby(feature)[TARGET_LOG].mean().sort_values(ascending=False)

    plt.figure(figsize=(9, 5))
    plt.bar(means.index.astype(str), means.values)
    plt.xlabel(feature)
    plt.ylabel("Средний log1p(learners_count)")
    plt.title(f"Средний таргет по категориям: {feature}")
    plt.xticks(rotation=35, ha="right")
    _save_current(output_dir / f"feature_vs_target_{feature}.png")


def make_correlation_table(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    numeric = df.select_dtypes(include=["number", "bool"]).copy()
    if TARGET_LOG not in numeric.columns:
        return pd.DataFrame()
    corr = (
        numeric.corr(numeric_only=True)[TARGET_LOG]
        .drop(index=[TARGET_LOG], errors="ignore")
        .sort_values(key=lambda values: values.abs(), ascending=False)
        .rename("corr_with_target_log")
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    corr.to_csv(output_dir / "feature_target_correlations.csv", index=False)
    return corr


def make_outlier_table(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    numeric = df.select_dtypes(include=["number", "bool"])
    rows: list[dict[str, float | int | str]] = []
    for column in numeric.columns:
        series = pd.to_numeric(numeric[column], errors="coerce").dropna()
        if series.empty:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (series < lower) | (series > upper)
        rows.append(
            {
                "feature": column,
                "min": float(series.min()),
                "p01": float(series.quantile(0.01)),
                "median": float(series.median()),
                "p99": float(series.quantile(0.99)),
                "max": float(series.max()),
                "iqr_outliers": int(mask.sum()),
                "iqr_outlier_share": float(mask.mean()),
            }
        )
    outliers = pd.DataFrame(rows).sort_values("iqr_outlier_share", ascending=False)
    outliers.to_csv(output_dir / "eda_outlier_summary.csv", index=False)
    return outliers


def create_plots(input_path: Path, output_dir: Path) -> None:
    df = pd.read_csv(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_target_distribution(df, output_dir)
    for feature in NUMERIC_FEATURES_FOR_PLOTS:
        plot_numeric_dependency(df, feature, output_dir)
    for feature in CATEGORICAL_FEATURES_FOR_PLOTS:
        plot_categorical_dependency(df, feature, output_dir)
    make_correlation_table(df, output_dir)
    make_outlier_table(df, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create EDA plots for CP2 report")
    parser.add_argument("--input", type=Path, default=Path("data/processed/stepik_course_steps_processed.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("report/images"))
    args = parser.parse_args()
    create_plots(args.input, args.output_dir)
    print(f"Saved EDA plots and tables to {args.output_dir}")


if __name__ == "__main__":
    main()
