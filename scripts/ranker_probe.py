#!/usr/bin/env python3
"""Frozen-feature probe: does the encoder contain the answer signal at all?

Trains a *linear* head on cached encoder features.  This separates two very
different failures that look identical in a training log:

* **optimisation failure** — a linear head on frozen features beats ln 4, so the
  signal is there and the fine-tuning recipe/LR is at fault;
* **capacity failure** — even the linear probe sits at ln 4, so the 4.4M/22.7M
  representation genuinely cannot rank JEE options, and no amount of tuning on
  that encoder will fix it.

It runs in a couple of minutes on a GPU (a few minutes per model on CPU), which
is much cheaper than discovering the same thing after a 45-minute fine-tune.

    python scripts/ranker_probe.py                       # all three encoders
    python scripts/ranker_probe.py --models l6           # just MiniLM-L6
    python scripts/ranker_probe.py --train-n 4000 --dev-n 600

Output is one line per (encoder, query_mode, pooling) plus a verdict.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fkjee.classifier import ListwiseCollator, MCQDataset  # noqa: E402
from fkjee.data import load_jsonl  # noqa: E402

ENCODERS = {
    "tinybert": "cross-encoder/ms-marco-TinyBERT-L-2-v2",
    "l6": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "l12": "cross-encoder/ms-marco-MiniLM-L-12-v2",
}
LN4 = math.log(4)


def pick(name: str) -> str:
    return ENCODERS.get(name, name)


@torch.no_grad()
def encode(model, tokenizer, items, query_mode: str, max_len: int,
           batch_size: int, device: str):
    """Return (cls, doc_mean, labels), each shaped (n_questions, 4, hidden)."""
    ds = MCQDataset(items, tokenizer, max_len=max_len, query_mode=query_mode)
    coll = ListwiseCollator(tokenizer.pad_token_id)
    cls_f, doc_f, labels = [], [], []
    t0 = time.time()
    for i in range(0, len(ds), batch_size):
        batch = coll([ds[j] for j in range(i, min(i + batch_size, len(ds)))])
        labels.append(batch.pop("labels"))
        batch = {k: v.to(device) for k, v in batch.items()}
        hidden = model(**batch, output_hidden_states=True).hidden_states[-1].float()
        hid = hidden.shape[-1]
        cls_f.append(hidden[:, 0].reshape(-1, 4, hid).cpu())
        if "token_type_ids" in batch:
            mask = (batch["token_type_ids"] == 1).unsqueeze(-1).float()
        else:  # no segment ids: mean over everything after the first token
            mask = batch["attention_mask"].unsqueeze(-1).float()
            mask[:, 0] = 0.0
        doc_f.append(((hidden * mask).sum(1) / mask.sum(1).clamp(min=1))
                     .reshape(-1, 4, hid).cpu())
    secs = time.time() - t0
    print(f"    encoded {len(ds)} questions in {secs:.0f}s", flush=True)
    return torch.cat(cls_f), torch.cat(doc_f), torch.cat(labels)


def fit_linear(f_tr, y_tr, f_dv, y_dv, device: str, steps: int = 600,
               lr: float = 1e-2, batch: int = 256) -> dict:
    w = torch.zeros(f_tr.shape[-1], requires_grad=True, device=device)
    b = torch.zeros(1, requires_grad=True, device=device)
    opt = torch.optim.AdamW([w, b], lr=lr, weight_decay=1e-4)
    f_tr, y_tr = f_tr.to(device), y_tr.to(device)
    for _ in range(steps):
        idx = torch.randint(0, f_tr.shape[0], (batch,), device=device)
        loss = F.cross_entropy(f_tr[idx] @ w + b, y_tr[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        tr = f_tr @ w + b
        dv = f_dv.to(device) @ w + b
        return {
            "train_ce": float(F.cross_entropy(tr, y_tr)),
            "train_acc": float((tr.argmax(-1) == y_tr).float().mean()),
            "dev_ce": float(F.cross_entropy(dv, y_dv.to(device))),
            "dev_acc": float((dv.argmax(-1) == y_dv.to(device)).float().mean()),
        }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="*", default=list(ENCODERS),
                    help=f"keys {list(ENCODERS)} or full HF model ids")
    ap.add_argument("--modes", nargs="*", default=["full", "question"])
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--train-n", type=int, default=2000)
    ap.add_argument("--dev-n", type=int, default=400)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--out", default="results/ranker_probe.json")
    args = ap.parse_args(argv)

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = Path(args.data_dir)
    train_items = [i for i in load_jsonl(data / "train.jsonl") if i.kind == "mcq"]
    dev_items = [i for i in load_jsonl(data / "dev.jsonl") if i.kind == "mcq"]
    train_items, dev_items = train_items[: args.train_n], dev_items[: args.dev_n]
    print(f"device={device}  train={len(train_items)}  dev={len(dev_items)}  "
          f"max_len={args.max_len}  ln4={LN4:.4f}", flush=True)

    import json
    rows = []
    for key in args.models:
        model_id = pick(key)
        print(f"\n=== {model_id}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.sep_token
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id, num_labels=1
        ).to(device).eval()

        for mode in args.modes:
            print(f"  query_mode={mode}", flush=True)
            c_tr, d_tr, y_tr = encode(model, tokenizer, train_items, mode,
                                      args.max_len, args.batch_size, device)
            c_dv, d_dv, y_dv = encode(model, tokenizer, dev_items, mode,
                                      args.max_len, args.batch_size, device)
            se = math.sqrt(0.25 * 0.75 / max(1, len(dev_items)))
            for pooling, a, b in (("cls", c_tr, c_dv), ("doc", d_tr, d_dv)):
                m = fit_linear(a, y_tr, b, y_dv, device, steps=args.steps)
                # A linear head on 800 questions can memorise, so "train CE < ln4"
                # alone is NOT evidence of usable signal.  Generalisation needs a
                # dev improvement beyond one standard error, in both CE and acc.
                if m["train_ce"] >= LN4 - 0.02:
                    verdict = "flat"          # cannot even fit the train split
                elif (m["dev_ce"] < LN4 - 0.02
                      and m["dev_acc"] > 0.25 + se):
                    verdict = "SIGNAL"        # fits train AND beats chance on dev
                elif m["dev_ce"] < LN4:
                    verdict = "weak"          # dev CE improves, accuracy is within noise
                else:
                    verdict = "memorises"     # fits train, at/below chance on dev
                print(f"    {pooling:3s} train CE={m['train_ce']:.4f} "
                      f"acc={m['train_acc']:.3f} | dev CE={m['dev_ce']:.4f} "
                      f"acc={m['dev_acc']:.3f}  [{verdict}]", flush=True)
                rows.append({"model": model_id, "query_mode": mode,
                             "pooling": pooling, **m, "verdict": verdict,
                             "dev_se": se})

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"ln4": LN4, "train_n": len(train_items),
                               "dev_n": len(dev_items), "rows": rows}, indent=2),
                   encoding="utf-8")

    signal = [r for r in rows if r["verdict"] == "SIGNAL"]
    memorised = [r for r in rows if r["verdict"] == "memorises"]
    weak = [r for r in rows if r["verdict"] == "weak"]
    print("\n" + "=" * 68)
    if signal:
        best = max(signal, key=lambda r: r["dev_acc"])
        print(f"VERDICT: signal exists. Best dev acc {best['dev_acc']:.3f} "
              f"({best['model'].split('/')[-1]}, {best['query_mode']}, {best['pooling']}).")
        print("         Fine-tuning can work -> check the LR/head recipe.")
    else:
        print("VERDICT: no generalising signal in any encoder tested.")
        if weak:
            best = max(weak, key=lambda r: r["dev_acc"])
            print(f"         Best row is only weak: dev acc {best['dev_acc']:.3f} "
                  f"(chance 0.25 +- {best['dev_se']:.3f} SE) with dev CE at chance.")
        if memorised:
            print("         Some rows fit the train questions (CE < ln4) but sit at or")
            print("         below chance on dev -- that is memorisation, not usable signal.")
        print("         These encoders are below the JEE threshold; stop tuning them.")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
