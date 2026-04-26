"""Parser for real online education data from the public Stepik API.

The main CP1 dataset is built at the level of course content items: one row is a
course step enriched with course, section, unit and lesson metadata. This usually
produces 10k+ real rows, while keeping the target connected to online-education
traffic: course learners count.

Only public GET endpoints are used. The parser respects pagination, retries
temporary errors and adds a delay between requests.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

API_ROOT = "https://stepik.org/api"
DEFAULT_HEADERS = {
    "User-Agent": (
        "hse-ml-cp1-online-edu-parser/1.1 "
        "(educational project; contact: add-your-email@example.com)"
    ),
    "Accept": "application/json",
}
ID_CHUNK_SIZE = 40


def build_session() -> requests.Session:
    """Create a requests session with retries for temporary server errors."""
    session = requests.Session()
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(DEFAULT_HEADERS)
    return session


def _api_get(
    session: requests.Session,
    endpoint: str,
    params: dict[str, Any] | list[tuple[str, Any]],
    timeout: float = 30.0,
) -> dict[str, Any]:
    """GET a Stepik API endpoint and return JSON payload."""
    url = f"{API_ROOT}/{endpoint.strip('/')}"
    response = session.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _chunked(values: list[int], size: int = ID_CHUNK_SIZE) -> Iterable[list[int]]:
    """Yield chunks preserving order."""
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _unique_ints(values: Iterable[Any]) -> list[int]:
    """Convert values to unique ints preserving order."""
    result: list[int] = []
    seen: set[int] = set()
    for value in values:
        if value is None:
            continue
        try:
            int_value = int(value)
        except (TypeError, ValueError):
            continue
        if int_value not in seen:
            seen.add(int_value)
            result.append(int_value)
    return result


def _id_params(ids: list[int]) -> list[tuple[str, int]]:
    """Build Stepik-compatible ids[]= query params."""
    return [("ids[]", object_id) for object_id in ids]


def _fetch_by_ids(
    session: requests.Session,
    endpoint: str,
    object_key: str,
    ids: list[int],
    sleep: float,
) -> dict[int, dict[str, Any]]:
    """Fetch Stepik objects by ids in batches."""
    if not ids:
        return {}

    result: dict[int, dict[str, Any]] = {}
    for chunk in tqdm(list(_chunked(ids)), desc=f"Fetching {endpoint}"):
        payload = _api_get(session, endpoint, params=_id_params(chunk))
        objects = payload.get(object_key, [])
        for obj in objects:
            object_id = obj.get("id")
            if object_id is not None:
                result[int(object_id)] = obj
        time.sleep(sleep)
    return result


def fetch_courses_page(
    session: requests.Session,
    page: int,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Fetch a single courses page from Stepik API."""
    payload = _api_get(session, "courses", params={"page": page}, timeout=timeout)
    if "courses" not in payload:
        raise ValueError(f"Unexpected Stepik response on page {page}: no 'courses' key")
    return payload


def collect_courses(
    session: requests.Session,
    pages: int,
    sleep: float,
    stop_on_last_page: bool = True,
) -> list[dict[str, Any]]:
    """Collect unique Stepik course cards from paginated API."""
    courses: list[dict[str, Any]] = []
    seen_ids: set[int] = set()

    for page in tqdm(range(1, pages + 1), desc="Parsing Stepik courses"):
        payload = fetch_courses_page(session=session, page=page)
        for course in payload.get("courses", []):
            course_id = course.get("id")
            if course_id in seen_ids:
                continue
            if course_id is not None:
                seen_ids.add(int(course_id))
            course["_source"] = "stepik"
            course["_parsed_page"] = page
            course["_parsed_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            courses.append(course)

        meta = payload.get("meta", {})
        if stop_on_last_page and not meta.get("has_next", True):
            logging.info("API reports no next page after page %s", page)
            break
        time.sleep(sleep)

    return courses


def save_jsonl(records: Iterable[dict[str, Any]], output: Path) -> int:
    """Save records as JSON Lines and return number of written rows."""
    output.parent.mkdir(parents=True, exist_ok=True)
    saved = 0
    with output.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            saved += 1
    return saved


def _review_count(course: dict[str, Any]) -> Any:
    review_summary = course.get("review_summary")
    if isinstance(review_summary, dict):
        return review_summary.get("count")
    return course.get("reviews_count")


def _review_average(course: dict[str, Any]) -> Any:
    review_summary = course.get("review_summary")
    if isinstance(review_summary, dict):
        return review_summary.get("average")
    return course.get("rating")


def _course_public_features(course: dict[str, Any]) -> dict[str, Any]:
    """Keep useful course-level fields with stable names."""
    return {
        "course_id": course.get("id"),
        "course_title": course.get("title"),
        "course_summary": course.get("summary") or course.get("short_description"),
        "course_description": course.get("description") or course.get("intro"),
        "course_language": course.get("language"),
        "course_learners_count": course.get("learners_count"),
        "course_review_count": _review_count(course),
        "course_review_average": _review_average(course),
        "course_lessons_count_api": course.get("lessons_count"),
        "course_sections_count_api": course.get("sections_count"),
        "course_time_to_complete": course.get("time_to_complete"),
        "course_is_paid": course.get("is_paid") or course.get("paid"),
        "course_is_public": course.get("is_public") or course.get("public"),
        "course_is_archived": course.get("is_archived") or course.get("archived"),
        "course_certificate": course.get("certificate")
        or course.get("certificate_regular")
        or course.get("certificate_link"),
        "course_price": course.get("price") or course.get("display_price") or course.get("discount_price"),
        "course_authors_count": len(course.get("authors") or []),
        "course_tags_count": len(course.get("tags") or []),
        "course_create_date": course.get("create_date") or course.get("created"),
        "course_begin_date": course.get("begin_date"),
        "_source": "stepik",
        "_parsed_page": course.get("_parsed_page"),
        "_parsed_at_utc": course.get("_parsed_at_utc"),
    }


def expand_courses_to_step_rows(
    session: requests.Session,
    courses: list[dict[str, Any]],
    sleep: float,
) -> list[dict[str, Any]]:
    """Build one row per course step using public Stepik hierarchy endpoints."""
    course_by_id = {int(course["id"]): course for course in courses if course.get("id") is not None}

    section_ids = _unique_ints(
        section_id
        for course in courses
        for section_id in (course.get("sections") or [])
    )
    sections = _fetch_by_ids(session, "sections", "sections", section_ids, sleep=sleep)

    unit_ids = _unique_ints(
        unit_id
        for section in sections.values()
        for unit_id in (section.get("units") or [])
    )
    units = _fetch_by_ids(session, "units", "units", unit_ids, sleep=sleep)

    lesson_ids = _unique_ints(unit.get("lesson") for unit in units.values())
    lessons = _fetch_by_ids(session, "lessons", "lessons", lesson_ids, sleep=sleep)

    step_ids = _unique_ints(
        step_id
        for lesson in lessons.values()
        for step_id in (lesson.get("steps") or [])
    )
    steps = _fetch_by_ids(session, "steps", "steps", step_ids, sleep=sleep)

    rows: list[dict[str, Any]] = []
    for section in tqdm(sections.values(), desc="Building step-level rows"):
        course_id = section.get("course")
        if course_id is None or int(course_id) not in course_by_id:
            continue
        course_features = _course_public_features(course_by_id[int(course_id)])

        for unit_position, unit_id in enumerate(section.get("units") or [], start=1):
            unit = units.get(int(unit_id))
            if unit is None:
                continue
            lesson_id = unit.get("lesson")
            lesson = lessons.get(int(lesson_id)) if lesson_id is not None else None
            if lesson is None:
                continue

            lesson_step_ids = lesson.get("steps") or []
            for step_position, step_id in enumerate(lesson_step_ids, start=1):
                step = steps.get(int(step_id), {}) if step_id is not None else {}
                block = step.get("block") if isinstance(step.get("block"), dict) else {}
                rows.append(
                    {
                        "record_type": "course_step",
                        **course_features,
                        "section_id": section.get("id"),
                        "section_position": section.get("position"),
                        "section_units_count": len(section.get("units") or []),
                        "unit_id": unit.get("id"),
                        "unit_position_in_section": unit_position,
                        "unit_progress_weight": unit.get("progress_weight"),
                        "lesson_id": lesson.get("id"),
                        "lesson_title": lesson.get("title"),
                        "lesson_steps_count": len(lesson_step_ids),
                        "step_id": step_id,
                        "step_position_in_lesson": step_position,
                        "step_type": block.get("name") or step.get("type"),
                        "step_cost": step.get("cost"),
                        "step_has_video": int(bool(block.get("video"))) if block else 0,
                    }
                )
    return rows


def parse_courses(
    pages: int,
    output: Path,
    sleep: float = 0.25,
    stop_on_last_page: bool = True,
    granularity: str = "steps",
) -> int:
    """Parse Stepik data and save raw JSONL.

    Args:
        pages: Maximum number of course pages to request.
        output: Path to the output JSONL file.
        sleep: Delay between requests in seconds.
        stop_on_last_page: Stop when API meta.has_next is false.
        granularity: "courses" saves one row per course, "steps" saves one row per step.

    Returns:
        Number of saved records.
    """
    session = build_session()
    courses = collect_courses(
        session=session,
        pages=pages,
        sleep=sleep,
        stop_on_last_page=stop_on_last_page,
    )
    logging.info("Collected %s unique course cards", len(courses))

    if granularity == "courses":
        records = courses
    elif granularity == "steps":
        records = expand_courses_to_step_rows(session=session, courses=courses, sleep=sleep)
        logging.info("Expanded courses to %s step-level records", len(records))
    else:
        raise ValueError("granularity must be either 'courses' or 'steps'")

    return save_jsonl(records, output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse real online-education data from Stepik API")
    parser.add_argument("--pages", type=int, default=600, help="Maximum number of course pages to parse")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/stepik_course_steps.jsonl"),
        help="Path to raw JSONL output",
    )
    parser.add_argument("--sleep", type=float, default=0.25, help="Delay between requests")
    parser.add_argument(
        "--granularity",
        choices=("steps", "courses"),
        default="steps",
        help="Dataset row type: steps gives 10k+ rows; courses keeps old course-level mode",
    )
    parser.add_argument(
        "--no-stop-on-last-page",
        action="store_true",
        help="Do not stop when API meta.has_next is false",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    saved = parse_courses(
        pages=args.pages,
        output=args.output,
        sleep=args.sleep,
        stop_on_last_page=not args.no_stop_on_last_page,
        granularity=args.granularity,
    )
    logging.info("Saved %s raw records to %s", saved, args.output)


if __name__ == "__main__":
    main()
