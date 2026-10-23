"""Listwise option ranker -- the "predict instead of solve" track.

A tiny cross-encoder scores each of the 4 options given the question and the
full option set, then a softmax over those 4 scores is trained with
cross-entropy.  This is a 4-way classification per question, not generation.

    query = question + all four options (letters shown)
    doc   = the candidate option's text (letter NOT shown)

Leaving the letter out of the doc is deliberate: it removes the model's easiest
shortcut and forces it to match content.

Ablations (`--ablate`) exist because a 4-way task with a 25% floor is trivially
cheatable:
    question : blank the question, keep options  -> tests option-shape priors
    options  : blank option text, keep question  -> tests whether content matters
    shuffle  : permute options per question      -> tests letter/position bias
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from .config import load_config
from .data import LETTERS, Item, load_jsonl, stable_hash
from .hub import (
    checkpoint_callback,
    enabled as hub_enabled,
    ensure_repo,
    upload_dir,
    upload_files,
)
from .metrics import one_line, summarize
from .runtime import warmup_kwargs

QUERY_HEADER = "Question and options:\n"


def build_query(item: Item, options: list[str], question: str | None = None) -> str:
    q = item.question if question is None else question
    opts = "\n".join(f"({L}) {o.strip()}" for L, o in zip(LETTERS, options))
    return f"{QUERY_HEADER}{q.strip()}\n\nOptions:\n{opts}"


class MCQDataset(Dataset):
    """Pre-tokenises 4 (query, option) pairs per question."""

    def __init__(self, items: list[Item], tokenizer, max_len: int = 384, ablate: str = "none"):
        self.items = items
        self.encodings: list[list[dict]] = []
        self.labels: list[int] = []
        self.gold_letters: list[str] = []

        for item in self.items:
            if item.kind != "mcq" or item.gold_index is None:
                continue
            options = list(item.options)
            gold_idx = item.gold_index

            if ablate == "shuffle":
                perm = list(range(len(options)))
                random.Random(stable_hash(item.id)).shuffle(perm)
                options = [item.options[i] for i in perm]
                gold_idx = perm.index(gold_idx)

            question = "" if ablate == "question" else item.question
            query = build_query(item, options, question)

            encs = []
            for opt in options:
                doc = "option" if ablate == "options" else opt
                enc = tokenizer(
                    query,
                    doc,
                    truncation=True,
                    max_length=max_len,
                    padding=False,
                    return_token_type_ids=True,
                )
                encs.append({k: list(v) for k, v in enc.items()})
            self.encodings.append(encs)
            self.labels.append(gold_idx)
            self.gold_letters.append(LETTERS[gold_idx])

        self.questions = [it for it in self.items if it.kind == "mcq" and it.gold_index is not None]

    def __len__(self) -> int:
        return len(self.encodings)

    def __getitem__(self, i: int) -> dict:
        return {"encs": self.encodings[i], "label": self.labels[i]}


class ListwiseCollator:
    def __init__(self, pad_token_id: int, n_options: int = 4):
        self.pad = pad_token_id
        self.n_options = n_options

    def __call__(self, features: list[dict]) -> dict:
        flat = [enc for f in features for enc in f["encs"]]
        max_len = max(len(e["input_ids"]) for e in flat)
        has_types = bool(flat) and "token_type_ids" in flat[0]

        input_ids, attn = [], []
        for e in flat:
            pad = max_len - len(e["input_ids"])
            input_ids.append(e["input_ids"] + [self.pad] * pad)
            attn.append(e["attention_mask"] + [0] * pad)

        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor([f["label"] for f in features], dtype=torch.long),
        }
        if has_types:
            type_ids = []
            for e in flat:
                pad = max_len - len(e["token_type_ids"])
                type_ids.append(e["token_type_ids"] + [0] * pad)
            batch["token_type_ids"] = torch.tensor(type_ids, dtype=torch.long)
        return batch


class ListwiseTrainer(Trainer):
    def __init__(self, *args, n_options: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_options = n_options

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits.reshape(-1, self.n_options)
        loss = F.cross_entropy(logits, labels)
        return (loss, outputs) if return_outputs else loss


@torch.no_grad()
def predict(model, loader, device, n_options: int = 4) -> tuple[list[int], list[float]]:
    model.eval()
    preds, conf = [], []
    for batch in loader:
        labels = batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits.reshape(-1, n_options)
        probs = logits.softmax(-1)
        preds.extend(logits.argmax(-1).cpu().tolist())
        conf.extend(probs.max(-1).values.cpu().tolist())
    return preds, conf


def evaluate(model, tokenizer, items: list[Item], *, max_len: int, batch_size: int,
             ablate: str = "none", device: str = "cuda") -> list[dict]:
    ds = MCQDataset(items, tokenizer, max_len=max_len, ablate=ablate)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=ListwiseCollator(tokenizer.pad_token_id),
        num_workers=2,
    )
    preds, conf = predict(model, loader, device)
    records = []
    for item, gold, pred, c in zip(ds.questions, ds.gold_letters, preds, conf):
        records.append(
            {
                "id": item.id,
                "subject": item.subject,
                "exam": item.exam,
                "chapter": item.chapter,
                "kind": item.kind,
                "year": item.year,
                "gold": gold,
                "pred": LETTERS[pred],
                "confidence": round(float(c), 4),
                "ablate": ablate,
            }
        )
    return records


def train(args: argparse.Namespace) -> Path:
    cfg = load_config(
        args.config,
        {
            "epochs": args.epochs,
            "limit": args.limit,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "max_len": args.max_len,
        },
    )
    name = cfg.get("name") or Path(args.config).stem
    data_dir = Path(cfg.get("data_dir", "data"))
    out_dir = Path(cfg.get("out_dir", "runs")) / name
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token

    train_items = load_jsonl(data_dir / "train.jsonl")
    if cfg.get("limit"):
        train_items = train_items[: int(cfg["limit"])]
    ds = MCQDataset(train_items, tokenizer, max_len=cfg.get("max_len", 384))

    model = AutoModelForSequenceClassification.from_pretrained(cfg["model_id"], num_labels=1)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model={cfg['model_id']} total_params={total/1e6:.2f}M trainable={trainable/1e6:.2f}M")
    print(f"train questions={len(ds)}")

    targs = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=float(cfg.get("epochs", 3)),
        per_device_train_batch_size=int(cfg.get("batch_size", 8)),
        gradient_accumulation_steps=int(cfg.get("grad_accum", 1)),
        learning_rate=float(cfg.get("lr", 2e-5)),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
        **warmup_kwargs(
            float(cfg.get("warmup_ratio", 0.06)),
            math.ceil(len(ds) / max(1, int(cfg.get("batch_size", 8)) * int(cfg.get("grad_accum", 1))))
            * max(1, int(float(cfg.get("epochs", 3)))),
        ),
        lr_scheduler_type="cosine",
        logging_steps=int(cfg.get("logging_steps", 50)),
        save_strategy="steps",
        save_steps=int(cfg.get("save_steps", 500)),
        save_total_limit=int(cfg.get("save_total_limit", 2)),
        eval_strategy="no",
        report_to="none",
        seed=int(cfg.get("seed", 42)),
        remove_unused_columns=False,
        dataloader_num_workers=2,
        fp16=device == "cuda" and cfg.get("fp16", True),
    )

    callbacks = []
    if hub_enabled(cfg):
        ensure_repo(cfg)  # fail early, not at the first checkpoint
        if cfg.get("push_checkpoints", True):
            callbacks.append(checkpoint_callback(cfg, name))

    trainer = ListwiseTrainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=ListwiseCollator(tokenizer.pad_token_id),
        n_options=4,
        callbacks=callbacks,
    )
    trainer.train(resume_from_checkpoint=True if args.resume else None)

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    meta = {
        "track": "ranker",
        "config": cfg,
        "total_params": total,
        "trainable_params": trainable,
        "train_questions": len(ds),
        "train_loss": next(
            (h["loss"] for h in reversed(trainer.state.log_history) if "loss" in h), None
        ),
    }
    (out_dir / "run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"saved {out_dir}")

    if hub_enabled(cfg):
        upload_dir(cfg, out_dir, f"{name}/adapter", ignore=["checkpoints/*", "checkpoints"])
        upload_files(cfg, {f"{name}/run.json": out_dir / "run.json"})
    return out_dir


def load_ranker(path: str | Path, device: str):
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=1).to(device)
    return model, tokenizer


def run_eval(model_dir: str, data_dir: str = "data", batch_size: int = 16,
             max_len: int = 384, limit: int | None = None, ablations: list[str] | None = None,
             out: str | None = None) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_ranker(model_dir, device)
    ablations = ablations or ["none"]
    suites = {
        "holdout": Path(data_dir) / "test_holdout.jsonl",
        "jeebench": Path(data_dir) / "jeebench_mcq.jsonl",
    }
    results: dict = {"model_dir": model_dir, "runs": {}}
    for suite, path in suites.items():
        if not path.exists():
            continue
        items = load_jsonl(path)
        if limit:
            items = items[:limit]
        for ablate in ablations:
            tag = f"{suite}/{ablate}"
            records = evaluate(
                model, tokenizer, items,
                max_len=max_len, batch_size=batch_size, ablate=ablate, device=device,
            )
            summary = summarize(records, name=tag)
            results["runs"][tag] = summary
            print(one_line(tag, summary))
            if out:
                outp = Path(out)
                outp.parent.mkdir(parents=True, exist_ok=True)
                outp.with_suffix(".preds.jsonl").write_text(
                    "\n".join(json.dumps(r) for r in records), encoding="utf-8"
                )
    if out:
        Path(out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="tiny listwise MCQ ranker")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--config", required=True)
    t.add_argument("--epochs", type=float, default=None)
    t.add_argument("--lr", type=float, default=None)
    t.add_argument("--batch-size", type=int, default=None)
    t.add_argument("--max-len", type=int, default=None)
    t.add_argument("--limit", type=int, default=None)
    t.add_argument("--resume", action="store_true")

    e = sub.add_parser("eval")
    e.add_argument("--model-dir", required=True)
    e.add_argument("--data-dir", default="data")
    e.add_argument("--batch-size", type=int, default=16)
    e.add_argument("--max-len", type=int, default=384)
    e.add_argument("--limit", type=int, default=None)
    e.add_argument("--ablate", nargs="+", default=["none"])
    e.add_argument("--out", default=None)

    args = ap.parse_args(argv)
    if args.cmd == "train":
        train(args)
    else:
        run_eval(
            args.model_dir, args.data_dir, args.batch_size, args.max_len,
            args.limit, args.ablate, args.out,
        )


if __name__ == "__main__":
    main()
