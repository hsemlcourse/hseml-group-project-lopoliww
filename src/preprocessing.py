"""Preprocess raw Stepik course cards for regression experiments."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

RANDOM_STATE = 42
TARGET = "learners_count"
GROUP_COLUMN = "course_id"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file with one JSON object per line."""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _first_existing(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    """Return the first available column or an all-NA series."""
    for column in candidates:
        if column in df.columns:
            return df[column]
    return pd.Series([pd.NA] * len(df), index=df.index)


def parse_price(value: Any) -> float:
    """Extract numeric price from API field or text representation."""
    if value is None or value is pd.NA:
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        for key in ("amount", "price", "value"):
            if key in value:
                return parse_price(value[key])
        return np.nan
    text = str(value).replace(",", ".")
    match = re.search(r"\d+(?:\.\d+)?", text.replace(" ", ""))
    return float(match.group(0)) if match else np.nan


def count_list_like(value: Any) -> int:
    """Count elements in list-like API fields."""
    if isinstance(value, list):
        return len(value)
    if isinstance(value, str) and value:
        return 1
    if pd.isna(value):
        return 0
    return 1


def make_features(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Flatten raw course JSON and create ML-ready features."""
    if not records:
        raise ValueError("No records found in raw JSONL")

    df = pd.json_normalize(records, sep="__")

    # The parser supports two real granularities:
    # 1) old course-level rows from /api/courses;
    # 2) preferred CP1 step-level rows enriched with course_* fields.
    if "course_id" in df.columns:
        df["course_id"] = pd.to_numeric(df["course_id"], errors="coerce")
    else:
        df["course_id"] = pd.to_numeric(_first_existing(df, ["id"]), errors="coerce")

    if "step_id" in df.columns:
        df["step_id"] = pd.to_numeric(df["step_id"], errors="coerce")
        df = df.drop_duplicates(subset=["course_id", "section_id", "unit_id", "lesson_id", "step_id"]).reset_index(drop=True)
    else:
        df = df.drop_duplicates(subset=["course_id"]).reset_index(drop=True)

    # Target. For step-level rows target is inherited from the parent course.
    target_source = _first_existing(df, ["course_learners_count", TARGET])
    df[TARGET] = pd.to_numeric(target_source, errors="coerce")
    df = df[df[TARGET].notna() & (df[TARGET] >= 0)].copy()
    df["target_log_learners_count"] = np.log1p(df[TARGET])

    # Stable identifiers and text features.
    df["title"] = _first_existing(df, ["course_title", "title"]).fillna("").astype(str)
    df["summary"] = _first_existing(df, ["course_summary", "summary", "short_description"]).fillna("").astype(str)
    df["description"] = _first_existing(df, ["course_description", "description", "intro"]).fillna("").astype(str)
    df["language"] = _first_existing(df, ["course_language", "language"]).fillna("unknown").astype(str)
    df["lesson_title"] = _first_existing(df, ["lesson_title"]).fillna("").astype(str)
    df["step_type"] = _first_existing(df, ["step_type"]).fillna("unknown").astype(str)

    df["title_len"] = df["title"].str.len()
    df["summary_len"] = df["summary"].str.len()
    df["description_len"] = df["description"].str.len()
    df["lesson_title_len"] = df["lesson_title"].str.len()
    df["title_word_count"] = df["title"].str.split().str.len().fillna(0)
    df["description_word_count"] = df["description"].str.split().str.len().fillna(0)
    df["lesson_title_word_count"] = df["lesson_title"].str.split().str.len().fillna(0)

    # Numeric API fields. Missing fields are kept as NaN and later imputed by models.
    numeric_candidates = {
        "reviews_count": ["course_review_count", "review_summary__count", "reviews_count"],
        "rating": ["course_review_average", "review_summary__average", "rating"],
        "lessons_count": ["course_lessons_count_api", "lessons_count"],
        "sections_count": ["course_sections_count_api", "sections_count"],
        "time_to_complete": ["course_time_to_complete", "time_to_complete"],
        "submissions_count": ["submissions_count"],
        "videos_duration": ["videos_duration"],
    }
    for new_column, candidates in numeric_candidates.items():
        df[new_column] = pd.to_numeric(_first_existing(df, candidates), errors="coerce")

    # Boolean/categorical fields that often exist in Stepik course cards.
    boolean_candidates = {
        "is_paid": ["course_is_paid", "is_paid", "paid"],
        "is_public": ["course_is_public", "is_public", "public"],
        "is_archived": ["course_is_archived", "is_archived", "archived"],
        "has_certificate": ["course_certificate", "certificate", "certificate_link", "certificate_regular"],
        "is_featured": ["is_featured", "featured"],
    }
    for new_column, candidates in boolean_candidates.items():
        source = _first_existing(df, candidates)
        df[new_column] = source.fillna(False).astype(bool).astype(int)

    # Price/payment proxy. API fields may differ by course type and region.
    raw_price = _first_existing(df, ["course_price", "price", "display_price", "currency_code", "discount_price"])
    df["price_numeric"] = raw_price.map(parse_price)
    df["has_price_info"] = df["price_numeric"].notna().astype(int)

    # List-like complexity/reputation features.
    for source_col, new_col in [
        ("authors", "authors_count"),
        ("instructors", "instructors_count"),
        ("requirements", "requirements_count"),
        ("tags", "tags_count"),
        ("certificate_distinction_thresholds", "certificate_thresholds_count"),
    ]:
        if source_col in df.columns:
            df[new_col] = df[source_col].map(count_list_like)
        else:
            df[new_col] = 0

    # Dates.
    date_source = _first_existing(df, ["course_create_date", "create_date", "created", "course_begin_date", "begin_date", "last_deadline"])
    parsed_date = pd.to_datetime(date_source, errors="coerce", utc=True)
    df["created_year"] = parsed_date.dt.year
    df["created_month"] = parsed_date.dt.month

    # Keep useful raw/source columns plus engineered features.
    selected = [
        "course_id",
        "section_id",
        "unit_id",
        "lesson_id",
        "step_id",
        "title",
        "summary",
        "description",
        "language",
        "lesson_title",
        "step_type",
        TARGET,
        "target_log_learners_count",
        "reviews_count",
        "rating",
        "lessons_count",
        "sections_count",
        "time_to_complete",
        "submissions_count",
        "videos_duration",
        "section_position",
        "section_units_count",
        "unit_position_in_section",
        "unit_progress_weight",
        "lesson_steps_count",
        "step_position_in_lesson",
        "step_cost",
        "step_has_video",
        "is_paid",
        "is_public",
        "is_archived",
        "has_certificate",
        "is_featured",
        "price_numeric",
        "has_price_info",
        "authors_count",
        "instructors_count",
        "requirements_count",
        "tags_count",
        "certificate_thresholds_count",
        "title_len",
        "summary_len",
        "description_len",
        "lesson_title_len",
        "title_word_count",
        "description_word_count",
        "lesson_title_word_count",
        "created_year",
        "created_month",
        "_source",
        "_parsed_page",
        "_parsed_at_utc",
    ]
    existing = [column for column in selected if column in df.columns]
    result = df[existing].copy()

    # Remove impossible duplicates and rows without stable ID.
    result = result[result["course_id"].notna()].reset_index(drop=True)
    return result


def add_splits(df: pd.DataFrame) -> pd.DataFrame:
    """Add train/val/test split with fixed seed and course-level grouping.

    For the preferred step-level dataset, many rows belong to the same course and
    inherit the same target. A regular row-wise split would leak course-level
    information across train/validation/test. GroupShuffleSplit keeps every
    course_id in exactly one split.
    """
    result = df.copy()
    groups = result[GROUP_COLUMN].astype(str)

    if groups.nunique() < 3:
        raise ValueError("Need at least 3 unique course_id groups for train/val/test split")

    first_split = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=RANDOM_STATE)
    train_pos, temp_pos = next(first_split.split(result, groups=groups))

    temp = result.iloc[temp_pos]
    temp_groups = temp[GROUP_COLUMN].astype(str)
    second_split = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=RANDOM_STATE)
    val_local_pos, test_local_pos = next(second_split.split(temp, groups=temp_groups))

    train_idx = result.index[train_pos]
    val_idx = temp.index[val_local_pos]
    test_idx = temp.index[test_local_pos]

    result["split"] = "train"
    result.loc[val_idx, "split"] = "val"
    result.loc[test_idx, "split"] = "test"

    overlap = (
        set(result.loc[result["split"] == "train", GROUP_COLUMN])
        & set(result.loc[result["split"] == "val", GROUP_COLUMN])
        | set(result.loc[result["split"] == "train", GROUP_COLUMN])
        & set(result.loc[result["split"] == "test", GROUP_COLUMN])
        | set(result.loc[result["split"] == "val", GROUP_COLUMN])
        & set(result.loc[result["split"] == "test", GROUP_COLUMN])
    )
    if overlap:
        raise RuntimeError("Course-level leakage detected in splits")

    return result

def preprocess(input_path: Path, output_path: Path) -> pd.DataFrame:
    """Run the full preprocessing pipeline."""
    records = read_jsonl(input_path)
    df = make_features(records)
    df = add_splits(df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Stepik course dataset")
    parser.add_argument("--input", type=Path, default=Path("data/raw/stepik_course_steps.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/stepik_course_steps_processed.csv"))
    args = parser.parse_args()

    df = preprocess(args.input, args.output)
    print(f"Saved processed dataset: {args.output}")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(df["split"].value_counts().to_string())


if __name__ == "__main__":
    main()
