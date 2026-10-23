"""Pluggable question-bank registry.

Adding a new exam / question set is one loader function plus one `SOURCES`
entry.  Every loader returns `Item`s with `exam`, `subject`, `chapter` (when the
source tags it) and `source` populated.

Registered today
----------------
grafite : JEE Main + AIEEE, 2002-2024, chapter/topic tagged, with explanations
neet    : NEET 2025 + 2026 (Physics/Chemistry/Biology), with solutions
pwjee   : JEE Main 2025 Maths (January + April shifts)
ckjee   : JEE Main 2025 Maths (shift-wise)

Deliberately excluded: datasets that are cut-off tables, article corpora, or
figure-dependent questions (a text model cannot answer "see the figure").
"""

from __future__ import annotations

import csv
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .data import (
    GRAFITE_URL,
    LETTERS,
    SUBJECT_ALIASES,
    Item,
    clean_text,
    download,
    extract_year,
    load_csv,
    parse_grafite_row,
    quality_rationale,
    stable_hash,
)

DATASETS_SERVER = "https://datasets-server.huggingface.co"
NEET_DATASET = "geekyrakshit/indian-exam-questions"
PW_BASE = "https://huggingface.co/datasets/PhysicsWallahAI/JEE-Main-2025-Math/resolve/main/"
PW_FILES = [("main2025-jan.jsonl", "jee-main-2025-jan"), ("main2025-apr.jsonl", "jee-main-2025-apr")]
CK_URL = "https://huggingface.co/datasets/CK0607/2025-Jee-Mains-Question/resolve/main/math_with_uuid.csv"


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _get_json(url: str, timeout: int = 180, attempts: int = 6) -> dict:
    """HF's datasets-server 429s aggressively; back off instead of failing."""
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "fkjee/0.1"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code not in (429, 500, 502, 503, 504):
                raise
            time.sleep(min(30, 2 ** attempt) + 1)
        except (urllib.error.URLError, TimeoutError) as exc:
            last = exc
            time.sleep(min(15, 2 ** attempt))
    assert last is not None
    raise last


def _cached_json(path: Path, fetch: Callable[[], object]):
    if path.exists() and path.stat().st_size > 0:
        return json.loads(path.read_text(encoding="utf-8"))
    data = fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return data


def _maybe_list(value) -> list:
    if isinstance(value, str):
        v = value.strip()
        if not v:
            return []
        try:
            value = json.loads(v)
        except json.JSONDecodeError:
            return []
    return value if isinstance(value, list) else []


def _subject(value: str | None) -> str | None:
    return SUBJECT_ALIASES.get((value or "").strip().lower())


def split_numbered_options(question: str) -> tuple[str, list[str]] | None:
    """Many Indian exam dumps embed options as '(1) .. (2) .. (3) .. (4) ..'."""
    m = re.search(r"\(1\)", question)
    if not m:
        return None
    stem = question[: m.start()].strip()
    parts = [p for p in re.split(r"(?=\([1-4]\))", question[m.start():]) if p.strip()]
    if len(parts) != 4:
        return None
    opts = []
    for part in parts:
        mm = re.match(r"\([1-4]\)\s*", part)
        if not mm:
            return None
        opts.append(part[mm.end():].strip())
    return (stem, opts) if all(opts) else None


# --------------------------------------------------------------------------- #
# grafite: JEE Main / AIEEE
# --------------------------------------------------------------------------- #
def load_grafite(raw_dir: Path, csv_path: str | None = None) -> list[Item]:
    path = Path(csv_path) if csv_path else download(GRAFITE_URL, Path(raw_dir) / "grafite.csv")
    return [it for row in load_csv(path) if (it := parse_grafite_row(row)) is not None]


# --------------------------------------------------------------------------- #
# NEET (all configs discovered dynamically, so new codes appear automatically)
# --------------------------------------------------------------------------- #
def hf_configs(dataset: str) -> dict:
    return _get_json(f"{DATASETS_SERVER}/info?dataset={urllib.parse.quote(dataset)}").get(
        "dataset_info", {}
    )


def hf_rows(dataset: str, config: str, split: str, total: int, page: int = 100) -> list[dict]:
    rows: list[dict] = []
    while len(rows) < total:
        n = min(page, total - len(rows))
        url = (
            f"{DATASETS_SERVER}/rows?dataset={urllib.parse.quote(dataset)}"
            f"&config={urllib.parse.quote(config)}&split={urllib.parse.quote(split)}"
            f"&offset={len(rows)}&length={n}"
        )
        batch = [r["row"] for r in _get_json(url).get("rows", [])]
        if not batch:
            break
        rows.extend(batch)
        time.sleep(0.3)  # be polite to the shared API
    return rows


def _neet_item(row: dict, config: str, year: int | None) -> Item | None:
    if (row.get("answer_status") or "normal").strip().lower() != "normal":
        return None  # dropped / bonus questions have no valid key

    raw_q = row.get("question") or ""
    if re.search(r"image_\d+", raw_q):
        return None  # figure-dependent: not answerable from text alone

    question = clean_text(raw_q)
    options = [clean_text(str(o)) for o in _maybe_list(row.get("options"))]
    correct = _maybe_list(row.get("correct_option"))
    if len(question) < 8 or len(options) != 4 or any(not o for o in options):
        return None
    if len(correct) != 1:
        return None
    try:
        idx = int(correct[0])
    except (TypeError, ValueError):
        return None
    if not 0 <= idx < 4:
        return None

    subject = _subject(row.get("subject"))
    if subject is None:
        return None

    solution = clean_text(re.sub(r"image_\d+", " ", str(row.get("solution") or "")))
    return Item(
        id=f"{config}-{row.get('question_number')}",
        subject=subject,
        question=question,
        options=options,
        gold=LETTERS[idx],
        kind="mcq",
        year=year,
        paper=config,
        rationale=quality_rationale(solution, options[idx], min_words=2),
        source="neet",
        exam="neet",
    )


def load_neet(raw_dir: Path, csv_path: str | None = None) -> list[Item]:
    items: list[Item] = []
    raw_dir = Path(raw_dir)
    for config, meta in hf_configs(NEET_DATASET).items():
        if not re.search(r"neet", config, re.I):
            continue
        year = extract_year(config)
        for split, split_meta in (meta.get("splits") or {}).items():
            count = split_meta.get("num_examples", 0) if isinstance(split_meta, dict) else split_meta
            rows = _cached_json(
                raw_dir / f"neet-{config}.json",
                lambda c=config, s=split, n=int(count): hf_rows(NEET_DATASET, c, s, n),
            )
            for row in rows:
                item = _neet_item(row, config, year)
                if item is not None:
                    items.append(item)
    return items


# --------------------------------------------------------------------------- #
# JEE Main 2025 Maths, clean JSONL
# --------------------------------------------------------------------------- #
def load_pwjee(raw_dir: Path, csv_path: str | None = None) -> list[Item]:
    items: list[Item] = []
    for filename, paper in PW_FILES:
        path = download(PW_BASE + filename, Path(raw_dir) / filename)
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            question = clean_text(row.get("question"))
            options = [clean_text(str(o)) for o in _maybe_list(row.get("options"))]
            correct = _maybe_list(row.get("correct_options"))
            if len(question) < 8:
                continue

            if not options:
                # numeric-answer question (question_type 0): keep it, different kind
                answer = clean_text(row.get("answer"))
                if not re.fullmatch(r"-?\d+(?:\.\d+)?", answer):
                    continue
                items.append(
                    Item(
                        id=f"pw-{stable_hash(question):x}",
                        subject="math",
                        question=question,
                        options=[],
                        gold=answer,
                        kind="integer",
                        year=extract_year(paper),
                        paper=paper,
                        source="pwjee",
                        exam="jee",
                    )
                )
                continue

            if len(options) != 4 or any(not o for o in options) or len(correct) != 1:
                continue
            try:
                idx = int(correct[0])
            except (TypeError, ValueError):
                continue
            if not 0 <= idx < 4:
                continue
            items.append(
                Item(
                    id=f"pw-{stable_hash(question):x}",
                    subject="math",
                    question=question,
                    options=options,
                    gold=LETTERS[idx],
                    kind="mcq",
                    year=extract_year(paper),
                    paper=paper,
                    source="pwjee",
                    exam="jee",
                )
            )
    return items


# --------------------------------------------------------------------------- #
# JEE Main 2025 Maths, options embedded in the question text
# --------------------------------------------------------------------------- #
def load_ckjee(raw_dir: Path, csv_path: str | None = None) -> list[Item]:
    path = download(CK_URL, Path(raw_dir) / "ckjee.csv")
    items: list[Item] = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            paper = (row.get("Shift Name") or "jee-main-2025").strip()
            subject = _subject(row.get("Subject")) or "math"
            parsed = split_numbered_options(clean_text(row.get("Question Text")))

            if parsed is None:
                # fill-in-the-blank / numeric-answer question
                answer = str(row.get("Correct Option") or "").strip()
                stem = clean_text(row.get("Question Text")).replace("________", "").strip()
                if not re.fullmatch(r"-?\d+(?:\.\d+)?", answer) or len(stem) < 8:
                    continue
                items.append(
                    Item(
                        id=f"ck-{row.get('unique_id') or stable_hash(stem):}",
                        subject=subject,
                        question=stem,
                        options=[],
                        gold=answer,
                        kind="integer",
                        year=extract_year(paper),
                        paper=paper,
                        source="ckjee",
                        exam="jee",
                    )
                )
                continue

            stem, options = parsed
            try:
                idx = int(str(row.get("Correct Option")).strip()) - 1
            except (TypeError, ValueError):
                continue
            if not 0 <= idx < 4:
                continue
            items.append(
                Item(
                    id=f"ck-{row.get('unique_id') or stable_hash(stem):}",
                    subject=subject,
                    question=stem,
                    options=options,
                    gold=LETTERS[idx],
                    kind="mcq",
                    year=extract_year(paper),
                    paper=paper,
                    source="ckjee",
                    exam="jee",
                )
            )
    return items


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Source:
    name: str
    exam: str
    description: str
    loader: Callable[..., list[Item]]


SOURCES: dict[str, Source] = {
    "grafite": Source("grafite", "jee", "JEE Main/AIEEE 2002-2024 (chapter tagged)", load_grafite),
    "neet": Source("neet", "neet", "NEET 2025-2026 Physics/Chemistry/Biology", load_neet),
    "pwjee": Source("pwjee", "jee", "JEE Main 2025 Maths, Jan + Apr", load_pwjee),
    "ckjee": Source("ckjee", "jee", "JEE Main 2025 Maths, shift-wise", load_ckjee),
}


def load_source(name: str, raw_dir: Path, csv_path: str | None = None) -> list[Item]:
    return SOURCES[name].loader(Path(raw_dir), csv_path=csv_path)
