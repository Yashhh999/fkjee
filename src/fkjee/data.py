"""Dataset preparation for fkjee.

Sources
-------
grafite  : ruh-ai/grafite-jee-mains-qna-no-img
           11.4k real JEE Main / AIEEE questions (2002-2024) with options,
           answer keys and explanations.  Our training + in-domain test set.
jeebench : daman1209arora/jeebench
           515 JEE Advanced questions.  Out-of-domain evaluation only.

Splitting is done by *paper*, never by row, so no shift of the same paper can
leak across train/test.  All papers from `--holdout-year` onwards are held out.

This module is stdlib-only so data prep runs on a CPU box before torch is
installed.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import html
import json
import re
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

GRAFITE_URL = (
    "https://huggingface.co/datasets/ruh-ai/grafite-jee-mains-qna-no-img"
    "/resolve/main/filtered_dataset.csv"
)
JEEBENCH_URL = (
    "https://huggingface.co/datasets/daman1209arora/jeebench/resolve/main/test.json"
)

LETTERS = "ABCD"
DEFAULT_HOLDOUT = {"jee": 2024, "neet": 2026}
SUBJECT_ALIASES = {
    "maths": "math",
    "math": "math",
    "mathematics": "math",
    "phy": "physics",
    "physics": "physics",
    "chem": "chemistry",
    "chemistry": "chemistry",
    "bio": "biology",
    "biology": "biology",
    "botany": "biology",
    "zoology": "biology",
}


# --------------------------------------------------------------------------- #
# text normalisation
# --------------------------------------------------------------------------- #
_IMG = re.compile(r"<img[^>]*>", re.I)
_BR = re.compile(r"<br\s*/?>", re.I)
_BLOCK_END = re.compile(r"</\s*(?:p|div|li|tr)\s*>", re.I)
_SUB = re.compile(r"<\s*sub\s*>(.*?)<\s*/\s*sub\s*>", re.I | re.S)
_SUP = re.compile(r"<\s*sup\s*>(.*?)<\s*/\s*sup\s*>", re.I | re.S)
_JUNK_TAG = re.compile(
    r"<\s*/?\s*(?:b|i|strong|em|u|span|center|table|tbody|thead|td|th|tr|"
    r"hline|small|font|div|p|a|ul|ol|li)\b[^>]*>",
    re.I,
)
_ANY_TAG = re.compile(r"<[^>]+>")


def clean_text(value: str | None) -> str:
    """Strip the exam-portal HTML while keeping the physics/maths readable."""
    if not value:
        return ""
    s = value.replace("\r", "")
    s = _IMG.sub(" ", s)
    s = _BR.sub("\n", s)
    s = _BLOCK_END.sub("\n", s)
    s = _SUB.sub(r"\1", s)          # H<sub>2</sub>SO<sub>4</sub> -> H2SO4
    s = _SUP.sub(r"^\1", s)         # 10<sup>-34</sup> -> 10^-34
    s = _JUNK_TAG.sub("", s)
    s = _ANY_TAG.sub("", s)
    s = html.unescape(s).replace("\xa0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n\s*\n\s*\n+", "\n\n", s)
    return s.strip()


def norm_key(text: str | None) -> str:
    """Aggressive normalisation used for dedupe / string answer matching."""
    s = clean_text(text).lower()
    s = re.sub(r"\\(?:text|mathrm|mathbf|left|right)\b", "", s)
    s = re.sub(r"[\s$\\{}()\[\],.;:'\"`~!?^*_|=+/\\-]+", "", s)
    return s


def stable_hash(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)


def extract_year(paper_id: str) -> int | None:
    m = re.search(r"(?:19|20)\d{2}", paper_id or "")
    return int(m.group(0)) if m else None


# --------------------------------------------------------------------------- #
# item model
# --------------------------------------------------------------------------- #
@dataclass
class Item:
    id: str
    subject: str            # math | physics | chemistry | biology
    question: str
    options: list[str]      # 4 option texts for MCQ, [] for integer
    gold: str               # "A".."D" for MCQ, numeric string for integer
    kind: str               # mcq | integer
    year: int | None = None
    paper: str = ""
    rationale: str | None = None
    source: str = "grafite"
    exam: str = "jee"       # jee | neet | ...
    chapter: str = ""       # syllabus module, when the source tags it

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Item":
        return cls(**d)

    @property
    def gold_index(self) -> int | None:
        return LETTERS.index(self.gold) if self.kind == "mcq" and self.gold in LETTERS else None


def load_jsonl(path: str | Path) -> list[Item]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(Item.from_dict(json.loads(line)))
    return items


def write_jsonl(path: str | Path, items: list[Item]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it.to_dict(), ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# robuster parsing helpers
# --------------------------------------------------------------------------- #
def _as_list(value) -> list:
    """grafite stores JSON arrays inside CSV strings; be forgiving."""
    v = (value or "").strip()
    if not v:
        return []
    try:
        out = json.loads(v)
    except (json.JSONDecodeError, TypeError):
        try:
            out = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            return []
    return out if isinstance(out, list) else []


def parse_options(value) -> list[str]:
    opts = []
    for entry in _as_list(value)[:4]:
        if isinstance(entry, dict):
            entry = entry.get("content", "")
        opts.append(clean_text(str(entry)))
    return opts


def parse_option_letters(value) -> list[str]:
    out = []
    for entry in _as_list(value):
        if isinstance(entry, dict):
            entry = entry.get("identifier", "")
        s = str(entry).strip().upper()
        if len(s) == 1 and s in LETTERS:
            out.append(s)
    return out


def quality_rationale(
    explanation: str | None, gold_text: str, min_words: int = 5
) -> str | None:
    """Keep a rationale only if it is real prose, not a restated answer or an
    image-only explanation.  We never invent reasoning the source lacks.

    `min_words=2` is for equation-heavy sources (NEET solutions) where a
    derivation is mostly LaTeX and only a couple of English words.
    """
    text = clean_text(explanation)
    if not text:
        return None
    words = re.findall(r"[A-Za-z]{2,}", text)
    if len(text) < 40 or len(words) < min_words:
        return None
    if norm_key(text) == norm_key(gold_text):
        return None
    return text


# --------------------------------------------------------------------------- #
# source adapters
# --------------------------------------------------------------------------- #
def parse_grafite_row(row: dict) -> Item | None:
    subject = SUBJECT_ALIASES.get((row.get("subject") or "").strip().lower())
    if subject is None:
        return None

    question = clean_text(row.get("question"))
    if len(question) < 8:
        return None

    kind = (row.get("question_type") or "").strip().lower()
    paper = (row.get("paper_id") or "").strip()
    rid = (row.get("question_id") or "").strip() or f"h{stable_hash(question):x}"

    if kind == "mcq":
        options = parse_options(row.get("options"))
        if len(options) != 4 or any(not o for o in options):
            return None
        letters = parse_option_letters(row.get("correct_option"))
        if len(letters) != 1:
            return None
        gold = letters[0]
        gold_text = options[LETTERS.index(gold)]
    elif kind == "integer":
        answer = clean_text(row.get("answer"))
        if not answer:
            return None
        options, gold, gold_text = [], answer, answer
    else:
        return None

    return Item(
        id=rid,
        subject=subject,
        question=question,
        options=options,
        gold=gold,
        kind=kind,
        year=extract_year(paper),
        paper=paper,
        rationale=quality_rationale(row.get("explanation"), gold_text),
        exam="jee",
        chapter=clean_text(row.get("chapter")),
    )


def split_inline_options(question: str) -> tuple[str, list[str]] | None:
    """jeebench embeds options in the prompt as '(A) .. (B) .. (C) .. (D) ..'."""
    m = re.search(r"\(A\)", question)
    if not m:
        return None
    stem = question[: m.start()].strip()
    parts = re.split(r"(?=\([ABCD]\))", question[m.start():])
    parts = [p for p in parts if p.strip()]
    if len(parts) != 4:
        return None
    opts = []
    for p in parts:
        mm = re.match(r"\([ABCD]\)\s*", p)
        if not mm:
            return None
        opts.append(p[mm.end():].strip())
    if any(not o for o in opts):
        return None
    return stem, opts


def parse_jeebench(data: list[dict]) -> list[Item]:
    items = []
    for i, row in enumerate(data):
        if (row.get("type") or "").strip() != "MCQ":
            continue  # MCQ(multiple) / Numeric / Integer need other parsers
        gold = (row.get("gold") or "").strip().upper()
        if len(gold) != 1 or gold not in LETTERS:
            continue
        parsed = split_inline_options(clean_text(row.get("question")))
        if parsed is None:
            continue
        stem, options = parsed
        items.append(
            Item(
                id=f"jeebench-{i}",
                subject=SUBJECT_ALIASES.get((row.get("subject") or "").lower(), "unknown"),
                question=stem,
                options=options,
                gold=gold,
                kind="mcq",
                year=extract_year(row.get("description") or ""),
                paper=(row.get("description") or "").strip(),
                source="jeebench",
            )
        )
    return items


# --------------------------------------------------------------------------- #
# dedupe / split
# --------------------------------------------------------------------------- #
def dedupe(items: list[Item]) -> list[Item]:
    """Same question can appear in several shifts. Keep one, preferring a
    version that carries an explanation."""
    kept: dict[tuple[str, str], Item] = {}
    for it in items:
        key = (it.subject, norm_key(it.question))
        prev = kept.get(key)
        if prev is None or (prev.rationale is None and it.rationale is not None):
            kept[key] = it
    return list(kept.values())


def split_items(
    items: list[Item],
    holdout: dict[str, int] | int | None = None,
    dev_mod: int = 10,
) -> tuple[list[Item], list[Item], list[Item]]:
    """Paper-level split.

    `holdout` is either one cutoff year for every exam or a per-exam map such as
    {"jee": 2024, "neet": 2026}.  Every paper at or after its exam's cutoff is
    test.  Everything else is split train/dev by a stable hash of paper_id, so
    all questions from one paper land on the same side.
    """
    if holdout is None:
        holdout = DEFAULT_HOLDOUT

    def is_held_out(it: Item) -> bool:
        cutoff = holdout if isinstance(holdout, int) else holdout.get(it.exam)
        return cutoff is not None and it.year is not None and it.year >= cutoff

    test, rest = [], []
    for it in items:
        (test if is_held_out(it) else rest).append(it)

    dev, train = [], []
    for it in rest:
        bucket = stable_hash(it.paper or it.id) % dev_mod
        (dev if bucket == 0 else train).append(it)
    return train, dev, test


# --------------------------------------------------------------------------- #
# io
# --------------------------------------------------------------------------- #
def download(url: str, dest: str | Path, timeout: int = 180) -> Path:
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "fkjee/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(dest, "wb") as f:
        f.write(resp.read())
    return dest


def load_csv(path: str | Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def summarise(name: str, items: list[Item]) -> dict:
    by_subject: dict[str, int] = {}
    by_kind: dict[str, int] = {}
    by_exam: dict[str, int] = {}
    by_chapter: dict[str, int] = {}
    for it in items:
        by_subject[it.subject] = by_subject.get(it.subject, 0) + 1
        by_kind[it.kind] = by_kind.get(it.kind, 0) + 1
        by_exam[it.exam] = by_exam.get(it.exam, 0) + 1
        by_chapter[it.chapter or "(untagged)"] = by_chapter.get(it.chapter or "(untagged)", 0) + 1
    with_cot = sum(1 for it in items if it.rationale)
    return {
        "split": name,
        "n": len(items),
        "by_subject": by_subject,
        "by_kind": by_kind,
        "by_exam": by_exam,
        "modules": dict(sorted(by_chapter.items(), key=lambda kv: -kv[1])),
        "n_modules": len([k for k in by_chapter if k != "(untagged)"]),
        "untagged": by_chapter.get("(untagged)", 0),
        "with_rationale": with_cot,
        "rationale_rate": round(with_cot / len(items), 3) if items else 0.0,
        "papers": len({it.paper for it in items}),
        "years": sorted({it.year for it in items if it.year}),
    }


def check_leakage(train: list[Item], test: list[Item]) -> int:
    """Exact-duplicate questions must never straddle the split."""
    seen = {(it.subject, norm_key(it.question)) for it in train}
    return sum(1 for it in test if (it.subject, norm_key(it.question)) in seen)


def write_coverage(path: str | Path, train: list[Item], test: list[Item]) -> None:
    """Per-syllabus-module train/test counts -- the 'did we cover everything' table."""
    rows: dict[tuple[str, str], dict[str, int]] = {}
    for split, items in (("train", train), ("test", test)):
        for it in items:
            key = (it.subject, it.chapter or "(untagged)")
            row = rows.setdefault(key, {"train": 0, "test": 0})
            row[split] += 1

    lines = [
        "# Module coverage",
        "",
        "`(untagged)` rows come from sources without chapter metadata (NEET, JEE 2025).",
        "",
        "| subject | module | train | test |",
        "|---|---|---:|---:|",
    ]
    ordered = sorted(rows.items(), key=lambda kv: (kv[0][0], -(kv[1]["train"] + kv[1]["test"])))
    for (subject, chapter), c in ordered:
        lines.append(f"| {subject} | {chapter} | {c['train']} | {c['test']} |")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare(args: argparse.Namespace) -> dict:
    from .sources import SOURCES, load_source

    data_dir = Path(args.data_dir)
    raw_dir = data_dir / "raw"
    wanted = [s.strip() for s in args.sources.split(",")] if args.sources else list(SOURCES)

    items: list[Item] = []
    per_source: dict[str, int] = {}
    for name in wanted:
        if name not in SOURCES:
            print(f"[warn] unknown source {name!r}, skipping")
            continue
        try:
            got = load_source(name, raw_dir, csv_path=args.csv)
        except Exception as exc:  # a dead dataset must not kill the whole build
            print(f"[warn] source {name!r} failed: {exc}")
            continue
        per_source[name] = len(got)
        items.extend(got)
        print(f"  loaded {name:<10} {len(got):>6}")

    if args.subjects:
        allowed = {s.strip() for s in args.subjects.split(",")}
        items = [it for it in items if it.subject in allowed]

    items = dedupe(items)

    holdout: dict[str, int] | int
    if args.holdout_year is not None:
        holdout = args.holdout_year
    else:
        holdout = {"jee": args.holdout_jee, "neet": args.holdout_neet}

    train, dev, test = split_items(items, holdout, args.dev_mod)
    write_jsonl(data_dir / "train.jsonl", train)
    write_jsonl(data_dir / "dev.jsonl", dev)
    write_jsonl(data_dir / "test_holdout.jsonl", test)
    write_coverage(data_dir / "coverage.md", train, test)
    leakage = check_leakage(train, test)
    if leakage:
        print(f"[WARN] {leakage} test questions also appear in train")

    stats = {
        "sources": wanted,
        "per_source": per_source,
        "after_dedupe": len(items),
        "leakage": leakage,
        "holdout": holdout,
        "subjects": sorted({it.subject for it in items}),
        "splits": [
            summarise("train", train),
            summarise("dev", dev),
            summarise("test_holdout", test),
        ],
    }

    if not args.skip_jeebench:
        try:
            bench_path = download(JEEBENCH_URL, raw_dir / "jeebench.json")
            bench = parse_jeebench(json.loads(Path(bench_path).read_text(encoding="utf-8")))
            write_jsonl(data_dir / "jeebench_mcq.jsonl", bench)
            stats["splits"].append(summarise("jeebench_mcq", bench))
        except Exception as exc:  # network failure should not kill data prep
            print(f"[warn] jeebench skipped: {exc}")

    (data_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"\nafter dedupe    {len(items)} items across {len(per_source)} sources")
    for s in stats["splits"]:
        print(
            f"  {s['split']:<14} n={s['n']:<6} papers={s['papers']:<4} "
            f"modules={s['n_modules']:<3} rationale={s['rationale_rate']:<5} "
            f"{s['by_subject']} {s['by_exam']}"
        )
    return stats


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="prepare fkjee datasets")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="download, clean, dedupe and split")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--sources", default=None, help="comma list; default = every registered source")
    p.add_argument("--subjects", default=None, help="comma list, e.g. math,physics,chemistry")
    p.add_argument("--holdout-year", type=int, default=None,
                   help="single cutoff year applied to all exams")
    p.add_argument("--holdout-jee", type=int, default=DEFAULT_HOLDOUT["jee"])
    p.add_argument("--holdout-neet", type=int, default=DEFAULT_HOLDOUT["neet"])
    p.add_argument("--dev-mod", type=int, default=10)
    p.add_argument("--csv", default=None, help="use a local grafite CSV instead of downloading")
    p.add_argument("--skip-jeebench", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "prepare":
        prepare(args)


if __name__ == "__main__":
    main()
