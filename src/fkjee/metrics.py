"""Scoring.  Pure stdlib so it can be unit-tested without torch."""

from __future__ import annotations

from collections import Counter

from .data import LETTERS, norm_key
from .parsing import as_number


def is_correct(pred: str | None, gold: str | None) -> bool:
    if pred is None or gold is None:
        return False
    p, g = str(pred).strip(), str(gold).strip()
    if len(p) == 1 and len(g) == 1 and p in LETTERS and g in LETTERS:
        return p == g
    a, b = as_number(p), as_number(g)
    if a is not None and b is not None:
        return abs(a - b) <= max(1e-6, 1e-3 * abs(b))
    return norm_key(p) == norm_key(g)


def _accuracy(records: list[dict]) -> float | None:
    if not records:
        return None
    return round(sum(is_correct(r.get("pred"), r.get("gold")) for r in records) / len(records), 4)


def _slice(records: list[dict], key: str, limit: int | None = None) -> dict:
    groups: dict[str, list[dict]] = {}
    for r in records:
        groups.setdefault(str(r.get(key)), []).append(r)
    ordered = sorted(groups.items(), key=lambda kv: -len(kv[1]))
    if limit:
        ordered = ordered[:limit]
    return {k: {"n": len(v), "accuracy": _accuracy(v)} for k, v in ordered}


def summarize(records: list[dict], name: str = "eval") -> dict:
    n = len(records)
    gold_letters = [r["gold"] for r in records if str(r.get("gold", "")).strip() in set(LETTERS)]
    valid = [r for r in records if r.get("pred") is not None]
    mcq = [r for r in records if len(str(r.get("gold", ""))) == 1 and r["gold"] in LETTERS]

    majority = 0.0
    if mcq:
        majority = round(max(Counter(r["gold"] for r in mcq).values()) / len(mcq), 4)

    return {
        "name": name,
        "n": n,
        "n_correct": sum(is_correct(r.get("pred"), r.get("gold")) for r in records),
        "accuracy": _accuracy(records) or 0.0,
        "chance_4way": 0.25 if mcq else None,
        "majority_baseline": majority if mcq else None,
        "parsed_rate": round(len(valid) / n, 4) if n else 0.0,
        "by_subject": _slice(records, "subject"),
        "by_kind": _slice(records, "kind"),
        "by_exam": _slice(records, "exam"),
        "by_chapter": _slice(records, "chapter", limit=25),
        "by_year": _slice(records, "year"),
        "pred_distribution": dict(Counter(str(r.get("pred")) for r in records).most_common(8)),
        "gold_distribution": dict(Counter(gold_letters).most_common()),
    }


def one_line(name: str, s: dict) -> str:
    return (
        f"{name:<28} n={s['n']:<5} acc={s['accuracy']:.3f} "
        f"parsed={s['parsed_rate']:.2f} "
        f"math={_get(s, 'by_subject', 'math')} "
        f"phy={_get(s, 'by_subject', 'physics')} "
        f"chem={_get(s, 'by_subject', 'chemistry')}"
    )


def _get(s: dict, section: str, key: str) -> str:
    v = s.get(section, {}).get(key)
    return "-" if not v else f"{v['accuracy']:.3f}"
