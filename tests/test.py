import json
from pathlib import Path

import pandas as pd

from src.modeling import feature_columns, split_dataset
from src.preprocessing import make_features, parse_price, preprocess


def test_parse_price_extracts_number():
    assert parse_price("1 990 ₽") == 1990.0
    assert parse_price({"amount": "2500"}) == 2500.0


def test_make_features_supports_course_level_rows():
    records = [
        {
            "id": 1,
            "title": "Python course",
            "summary": "Intro",
            "description": "Learn Python",
            "language": "ru",
            "learners_count": 100,
            "review_summary": {"average": 4.8, "count": 10},
            "authors": [11, 12],
            "_source": "stepik",
            "_parsed_page": 1,
            "_parsed_at_utc": "2026-04-25T00:00:00Z",
        },
        {
            "id": 2,
            "title": "SQL course",
            "summary": "DB",
            "description": "Learn SQL",
            "language": "ru",
            "learners_count": 10,
            "review_summary": {"average": 4.1, "count": 3},
            "authors": [13],
            "_source": "stepik",
            "_parsed_page": 1,
            "_parsed_at_utc": "2026-04-25T00:00:00Z",
        },
    ]
    df = make_features(records)

    assert "target_log_learners_count" in df.columns
    assert "title_len" in df.columns
    assert "authors_count" in df.columns
    assert df.shape[0] == 2


def test_make_features_supports_step_level_rows():
    records = [
        {
            "record_type": "course_step",
            "course_id": 1,
            "course_title": "Python course",
            "course_summary": "Intro",
            "course_description": "Learn Python",
            "course_language": "ru",
            "course_learners_count": 100,
            "course_review_average": 4.8,
            "course_review_count": 10,
            "section_id": 10,
            "unit_id": 100,
            "lesson_id": 1000,
            "step_id": 10000,
            "lesson_title": "Variables",
            "step_type": "text",
            "lesson_steps_count": 5,
            "step_position_in_lesson": 1,
            "_source": "stepik",
            "_parsed_page": 1,
            "_parsed_at_utc": "2026-04-25T00:00:00Z",
        }
    ]
    df = make_features(records)

    assert df.loc[0, "learners_count"] == 100
    assert df.loc[0, "course_id"] == 1
    assert "lesson_title_len" in df.columns
    assert "step_type" in df.columns


def test_preprocess_writes_csv_and_keeps_course_groups_separate(tmp_path: Path):
    raw = tmp_path / "raw.jsonl"
    records = []
    for course_id in range(30):
        for step_no in range(3):
            records.append(
                {
                    "record_type": "course_step",
                    "course_id": course_id,
                    "course_title": f"Course {course_id}",
                    "course_summary": "Short",
                    "course_description": "Long description",
                    "course_language": "ru",
                    "course_learners_count": course_id + 1,
                    "section_id": course_id * 10,
                    "unit_id": course_id * 100,
                    "lesson_id": course_id * 1000,
                    "step_id": course_id * 10000 + step_no,
                    "lesson_title": "Lesson",
                    "step_type": "text",
                    "_source": "stepik",
                    "_parsed_page": 1,
                    "_parsed_at_utc": "2026-04-25T00:00:00Z",
                }
            )
    raw.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    output = tmp_path / "processed.csv"
    df = preprocess(raw, output)

    assert output.exists()
    assert set(df["split"]) == {"train", "val", "test"}

    train_courses = set(df.loc[df["split"] == "train", "course_id"])
    val_courses = set(df.loc[df["split"] == "val", "course_id"])
    test_courses = set(df.loc[df["split"] == "test", "course_id"])
    assert train_courses.isdisjoint(val_courses)
    assert train_courses.isdisjoint(test_courses)
    assert val_courses.isdisjoint(test_courses)


def test_modeling_feature_columns_exclude_ids_and_target():
    df = pd.DataFrame(
        {
            "course_id": [1, 2, 3],
            "step_id": [10, 20, 30],
            "learners_count": [10, 20, 30],
            "target_log_learners_count": [1.0, 2.0, 3.0],
            "split": ["train", "val", "test"],
            "language": ["ru", "en", "ru"],
            "title_len": [5, 6, 7],
        }
    )
    cols = feature_columns(df)
    assert "course_id" not in cols
    assert "step_id" not in cols
    assert "target_log_learners_count" not in cols
    assert "language" in cols
    assert "title_len" in cols


def test_split_dataset_uses_saved_split_labels():
    df = pd.DataFrame(
        {
            "course_id": [1, 2, 3],
            "learners_count": [10, 20, 30],
            "target_log_learners_count": [1.0, 2.0, 3.0],
            "split": ["train", "val", "test"],
            "language": ["ru", "en", "ru"],
            "title_len": [5, 6, 7],
        }
    )
    data = split_dataset(df)
    assert data.x_train.shape[0] == 1
    assert data.x_val.shape[0] == 1
    assert data.x_test.shape[0] == 1
