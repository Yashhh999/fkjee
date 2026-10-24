"""Lexical MCQ ranker -- the smallest possible "predict" model.

No neural network, no tokenizer, no training loop.  Each option gets a handful
of hand-built features (lexical overlap with the question, length, numeric
agreement, "none of the above" flags) and a softmax over the four options is
trained by gradient descent.  That is 8 parameters: 7 feature weights + bias.

It exists for two reasons:

1. It is the floor for the "smallest model that can answer JEE" question.  If a
   33M cross-encoder cannot beat 7 numbers, the encoder is not doing anything.
2. It is a diagnostic.  A lexical model clearing 25% means the task has
   learnable text signal and the neural collapse is an optimisation bug; a
   lexical model *at* 25% means the options are not separable from surface
   form and the task genuinely needs subject knowledge.

Usage:
    python -m fkjee.lexical --data-dir data --out results/lexical.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

from .data import LETTERS, Item, load_jsonl

WORD = re.compile(r"[A-Za-z]{2,}|\d+(?:\.\d+)?")
NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
STOP = {
    "the", "of", "and", "is", "are", "to", "in", "a", "an", "for", "with", "on",
    "by", "as", "that", "which", "be", "it", "at", "from", "or", "if", "then",
    "following", "correct", "question", "option", "options", "value", "equal",
}
FLAGS = ("all of the above", "none of the above", "both", "cannot be determined")


def tokens(text: str) -> list[str]:
    return [w.lower() for w in WORD.findall(text) if w.lower() not in STOP]


class Tfidf:
    def __init__(self, docs: list[list[str]]):
        df: dict[str, int] = {}
        for d in docs:
            for w in set(d):
                df[w] = df.get(w, 0) + 1
        n = max(1, len(docs))
        self.idf = {w: math.log((1 + n) / (1 + c)) + 1.0 for w, c in df.items()}
        self.default = math.log(1 + n) + 1.0

    def vec(self, toks: list[str]) -> dict[str, float]:
        tf: dict[str, float] = {}
        for w in toks:
            tf[w] = tf.get(w, 0.0) + 1.0
        return {w: (1 + math.log(c)) * self.idf.get(w, self.default) for w, c in tf.items()}


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(v * b.get(w, 0.0) for w, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def features(item: Item, option: str, tfidf: Tfidf) -> list[float]:
    q = tokens(item.question)
    o = tokens(option)
    qv, ov = tfidf.vec(q), tfidf.vec(o)
    qset, oset = set(q), set(o)

    overlap = len(qset & oset) / max(1, len(qset))
    qnums = set(NUMBER.findall(item.question))
    onums = set(NUMBER.findall(option))
    num_agree = len(qnums & onums) / max(1, len(qnums)) if qnums else 0.0
    low = option.lower()
    return [
        cosine(qv, ov),
        overlap,
        math.log1p(len(o)),
        num_agree,
        float(any(f in low for f in FLAGS)),
        float(len(o) == 0),
        len(onums) / max(1, len(qnums)) if qnums else 0.0,
    ]


def build(items: list[Item], tfidf: Tfidf) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for it in items:
        if it.kind != "mcq" or it.gold_index is None:
            continue
        X.append([features(it, o, tfidf) for o in it.options])
        y.append(it.gold_index)
    return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.int64)


def fit(X: np.ndarray, y: np.ndarray, steps: int = 2000, lr: float = 0.5) -> np.ndarray:
    """Softmax over the 4 options with a shared weight vector (7 params)."""
    n, k, d = X.shape
    w = np.zeros(d + 1)  # +1 bias
    Xb = np.concatenate([X, np.ones((n, k, 1))], axis=2)
    for _ in range(steps):
        logits = Xb @ w
        logits -= logits.max(1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(1, keepdims=True)
        grad = np.zeros_like(w)
        for c in range(k):
            coef = p[:, c] - (y == c)
            grad += Xb[:, c, :].T @ coef
        w -= lr * grad / n
    return w


def score(X: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n, k, _ = X.shape
    Xb = np.concatenate([X, np.ones((n, k, 1))], axis=2)
    logits = Xb @ w
    logits -= logits.max(1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(1, keepdims=True)
    return p.argmax(1), p.max(1)


def report(name: str, pred: np.ndarray, conf: np.ndarray, y: np.ndarray,
           items: list[Item]) -> dict:
    ok = pred == y
    by_subject: dict[str, list[int]] = {}
    for i, it in enumerate(items):
        by_subject.setdefault(it.subject, []).append(int(ok[i]))
    return {
        "suite": name,
        "n": int(len(y)),
        "accuracy": round(float(ok.mean()), 4),
        "mean_confidence": round(float(conf.mean()), 4),
        "by_subject": {
            s: {"n": len(v), "accuracy": round(float(np.mean(v)), 4)}
            for s, v in sorted(by_subject.items())
        },
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="8-parameter lexical MCQ ranker")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="results/lexical.json")
    args = ap.parse_args(argv)

    data = Path(args.data_dir)
    train_items = [i for i in load_jsonl(data / "train.jsonl") if i.kind == "mcq"]
    test_items = [i for i in load_jsonl(data / "test_holdout.jsonl") if i.kind == "mcq"]

    tfidf = Tfidf([tokens(i.question) for i in train_items])
    Xtr, ytr = build(train_items, tfidf)
    w = fit(Xtr, ytr)

    out = {"model": "lexical-softmax", "params": int(len(w))}
    for name, items in (("train", train_items), ("holdout", test_items)):
        X, y = build(items, tfidf)
        pred, conf = score(X, w)
        kept = [i for i in items if i.kind == "mcq" and i.gold_index is not None]
        res = report(name, pred, conf, y, kept)
        out[name] = res
        print(f"{name:<8} n={res['n']:<5} acc={res['accuracy']:.4f}  {res['by_subject']}")

    print("\nweights (bias last):", np.round(w, 4).tolist())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
