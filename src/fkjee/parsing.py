"""Answer extraction from model generations.  Pure stdlib."""

from __future__ import annotations

import re

BOXED = re.compile(r"\\boxed\s*")
ANSWERISH = re.compile(
    r"(?:final\s+answer|correct\s+(?:option|answer|choice)|answer|option|choice)"
    r"\s*(?:is|:|=)?\s*[\(\[]?\s*([A-Da-d]|\-?\d+(?:\.\d+)?)\s*[\)\]]?",
    re.I,
)
FRAC = re.compile(r"\\d?frac\s*\{\s*(-?\d+(?:\.\d+)?)\s*\}\s*\{\s*(-?\d+(?:\.\d+)?)\s*\}")
FLOATY = re.compile(r"^-?\d+(?:\.\d+)?(?:e[+-]?\d+)?$", re.I)
TRAIL_PUNCT = " \t\n.,;:!?$"


def _strip(s: str) -> str:
    return s.strip().strip("$").strip().strip(TRAIL_PUNCT).strip()


def as_number(s: str) -> float | None:
    """Parse the numeric forms these answers actually take."""
    if s is None:
        return None
    t = _strip(str(s).replace(",", ""))
    if not t:
        return None
    m = FRAC.fullmatch(t)
    if m:
        try:
            den = float(m.group(2))
            return float(m.group(1)) / den if den else None
        except ValueError:
            return None
    t = t.replace("\\times", "x").replace("×", "x")
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)\s*x\s*10\s*\^?\s*\{?\s*(-?\d+)\s*\}?", t)
    if m:
        return float(m.group(1)) * (10 ** int(m.group(2)))
    t = t.rstrip("%")
    if FLOATY.fullmatch(t):
        try:
            return float(t)
        except ValueError:
            return None
    return None


def normalise_letter(s: str) -> str | None:
    t = _strip(s).upper()
    if not t:
        return None
    if len(t) == 1 and t in "ABCD":
        return t
    m = re.fullmatch(r"\(?([ABCD])\)?", t)
    if m:
        return m.group(1)
    m = re.fullmatch(r"(?:OPTION|CHOICE|ANS(?:WER)?)\s*[:\-]?\s*\(?([ABCD])\)?", t)
    if m:
        return m.group(1)
    return None


def canon(value: str) -> str | None:
    """Canonicalise one candidate: single letter, or a clean numeric string."""
    letter = normalise_letter(value)
    if letter:
        return letter
    num = as_number(value)
    if num is not None:
        return _num_key(num)
    t = _strip(value)
    return t or None


def _num_key(x: float) -> str:
    if x == int(x) and abs(x) < 1e15:
        return str(int(x))
    return f"{x:.10g}"


def _boxed_contents(text: str) -> list[str]:
    """Extract \\boxed{...} bodies with balanced braces, so \\frac{1}{2} works."""
    out: list[str] = []
    for m in BOXED.finditer(text):
        start = m.end()
        if start >= len(text) or text[start] != "{":
            continue
        depth = 0
        for j in range(start, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    out.append(text[start + 1 : j])
                    break
    return out


def parse_answer(text: str) -> tuple[str | None, str]:
    """Return (answer, method).  Prefers the last \\boxed{...}, then answer-ish
    phrases, then a trailing standalone letter."""
    if not text:
        return None, "none"

    boxed = [c for c in _boxed_contents(text) if c.strip()]
    for cand in reversed(boxed):
        val = canon(cand)
        if val:
            return val, "boxed"

    matches = list(ANSWERISH.finditer(text))
    for m in reversed(matches):
        val = canon(m.group(1))
        if val:
            return val, "pattern"

    tail = text.strip().splitlines()[-1] if text.strip() else ""
    letter = normalise_letter(tail)
    if letter:
        return letter, "tail"
    return None, "none"
