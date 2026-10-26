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
    solo     : drop the option set from the query, score question vs one option
               (the classic cross-encoder form)  -> tests reliance on contrast
    shuffle  : permute options per question      -> tests letter/position bias

There is deliberately no "blank the option text" ablation: with all four
candidate texts replaced by the same placeholder the four scores are identical,
so the softmax is uniform and the accuracy is 0.25 *by construction*.  It looks
like a clean result and measures nothing.
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
QUERY_MODES = ("full", "question")


def build_query(item: Item, options: list[str], question: str | None = None,
                include_options: bool = True) -> str:
    q = item.question if question is None else question
    if not include_options:
        # Classic cross-encoder form: judge each option against the question
        # alone.  Putting all four options in the query makes every candidate
        # equally "present", which is a real (and avoidable) confound.
        return f"{QUERY_HEADER}{q.strip()}"
    opts = "\n".join(f"({L}) {o.strip()}" for L, o in zip(LETTERS, options))
    return f"{QUERY_HEADER}{q.strip()}\n\nOptions:\n{opts}"


def encode_pair(tokenizer, query: str, doc: str, max_len: int) -> dict:
    """Tokenise (query, doc) with a budget that can never eat the option.

    Plain `truncation=True` cuts the *end* of the pair, so a long JEE question
    silently deletes the option text and the model ends up scoring four empty
    strings.  We instead reserve space for the doc and give the query the rest.
    """
    if tokenizer.cls_token_id is None or tokenizer.sep_token_id is None:
        enc = tokenizer(query, doc, truncation=True, max_length=max_len,
                        return_token_type_ids=True)
        return {k: list(v) for k, v in enc.items()}

    special = tokenizer.num_special_tokens_to_add(pair=True)
    doc_ids = tokenizer(doc, add_special_tokens=False)["input_ids"]
    doc_ids = doc_ids[: max(8, max_len // 2)]
    q_budget = max(1, max_len - special - len(doc_ids))
    q_ids = tokenizer(query, add_special_tokens=False)["input_ids"][:q_budget]

    ids = [tokenizer.cls_token_id] + q_ids + [tokenizer.sep_token_id] + doc_ids + [
        tokenizer.sep_token_id
    ]
    types = [0] * (len(q_ids) + 2) + [1] * (len(doc_ids) + 1)
    return {"input_ids": ids, "token_type_ids": types, "attention_mask": [1] * len(ids)}


class MCQDataset(Dataset):
    """Pre-tokenises 4 (query, option) pairs per question."""

    def __init__(self, items: list[Item], tokenizer, max_len: int = 384, ablate: str = "none",
                 query_mode: str = "full"):
        self.items = items
        self.encodings: list[list[dict]] = []
        self.labels: list[int] = []
        self.gold_letters: list[str] = []
        if query_mode not in QUERY_MODES:
            raise ValueError(f"query_mode must be one of {QUERY_MODES}, got {query_mode!r}")

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
            # `solo` scores each option against the question alone instead of
            # against the whole option set (the classic cross-encoder form).
            show_options = query_mode == "full" and ablate != "solo"
            query = build_query(item, options, question, include_options=show_options)

            encs = []
            for opt in options:
                encs.append(encode_pair(tokenizer, query, opt, max_len))
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
    """Trainer with honest 4-way CE logging.

    `Trainer` accumulates the *sum* of the micro-batch losses and divides by the
    number of optimizer steps, so with `gradient_accumulation_steps=k` the logged
    "loss" is k x the real CE (verified: 2.773 logged vs 1.386 true at k=2, i.e.
    ln 16 instead of ln 4).  That makes a stuck model look like a 16-way problem
    and hides whether anything is learning.  We therefore log the true mean CE
    computed in `compute_loss` and keep the raw value as `loss_raw`.
    """

    def __init__(self, *args, n_options: int = 4, head_lr: float | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_options = n_options
        self.head_lr = head_lr
        self._ce_sum = 0.0
        self._ce_n = 0

    def create_optimizer(self):
        """Optional separate LR for a *fresh* classification head.

        Only useful when the head is randomly initialised (e.g. a plain BERT
        backbone loaded with a new `classifier`).  The `cross-encoder/ms-marco-*`
        models keep their pretrained scoring head, so they must use one LR for
        everything -- giving a pretrained head a 50x step destroys it.
        """
        if self.optimizer is not None:
            return self.optimizer
        head_lr = self.head_lr or self.args.learning_rate
        body = [p for n, p in self.model.named_parameters()
                if "classifier" not in n and p.requires_grad]
        head = [p for n, p in self.model.named_parameters()
                if "classifier" in n and p.requires_grad]
        groups = [{"params": body, "lr": self.args.learning_rate}] if body else []
        groups.append({"params": head, "lr": head_lr})
        self.optimizer = torch.optim.AdamW(
            groups, lr=self.args.learning_rate, weight_decay=self.args.weight_decay
        )
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits.reshape(-1, self.n_options)
        loss = F.cross_entropy(logits, labels)
        if self.model.training:
            self._ce_sum += float(loss.detach())
            self._ce_n += 1
        return (loss, outputs) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        if logs.get("loss") is not None and self._ce_n:
            logs = {**logs, "loss": round(self._ce_sum / self._ce_n, 4),
                    "loss_raw": logs["loss"]}
            self._ce_sum, self._ce_n = 0.0, 0
        return super().log(logs, *args, **kwargs)

    @staticmethod
    def compute_metrics(eval_pred) -> dict:
        """Accuracy over 4-way logits (eval_loss is already the true CE)."""
        import numpy as np

        logits, labels = eval_pred
        logits = np.asarray(logits)
        gold = np.asarray(labels).reshape(-1)
        if logits.ndim == 1 or logits.shape[-1] != 4:
            logits = logits.reshape(len(gold), 4)
        pred = logits.argmax(-1)
        n = min(len(pred), len(gold))
        return {"accuracy": float((pred[:n] == gold[:n]).mean())}

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Return one logit row per *question*, not per option.

        `Trainer` gathers predictions through `accelerator.gather_for_metrics`,
        which truncates the last batch to the number of remaining *dataset
        items* (questions).  With logits shaped (batch*4, 1) that silently drops
        three quarters of the last batch -- observed as 71 predictions for 23
        questions.  Reshaping to (batch, 4) here makes the truncation correct and
        keeps `compute_metrics` trivial.
        """
        loss, logits, labels = super().prediction_step(
            model, inputs, prediction_loss_only, ignore_keys
        )
        if logits is not None:
            logits = logits.reshape(-1, self.n_options)
        return loss, logits, labels


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
             ablate: str = "none", device: str = "cuda",
             query_mode: str = "full") -> list[dict]:
    ds = MCQDataset(items, tokenizer, max_len=max_len, ablate=ablate, query_mode=query_mode)
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
    query_mode = cfg.get("query_mode", "full")
    ds = MCQDataset(train_items, tokenizer, max_len=cfg.get("max_len", 384),
                    query_mode=query_mode)

    dev_items = load_jsonl(data_dir / "dev.jsonl")
    dev_limit = int(cfg.get("eval_limit", 500))
    eval_ds = MCQDataset(dev_items[:dev_limit], tokenizer, max_len=cfg.get("max_len", 384),
                         query_mode=query_mode)

    model = AutoModelForSequenceClassification.from_pretrained(cfg["model_id"], num_labels=1)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model={cfg['model_id']} total_params={total/1e6:.2f}M trainable={trainable/1e6:.2f}M")
    print(f"train questions={len(ds)} | dev questions={len(eval_ds)} (chance 0.25)")

    eval_steps = int(cfg.get("eval_steps", 100))
    save_steps = int(cfg.get("save_steps", 500))
    # Saving the last step is how a model that peaked at dev 0.328 ends up
    # shipped at 0.25: the run overfits after the peak.  Keep the best dev
    # checkpoint instead.  transformers requires save_steps % eval_steps == 0
    # and that every eval step has a saved checkpoint to reload.
    select_best = bool(cfg.get("load_best_model_at_end", True)) and save_steps % eval_steps == 0
    if not select_best:
        print(f"[warn] save_steps={save_steps} is not a multiple of eval_steps={eval_steps}; "
              "the last checkpoint will be saved, not the best one")

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
        save_steps=save_steps,
        save_total_limit=int(cfg.get("save_total_limit", 2)),
        eval_strategy="steps",
        eval_steps=eval_steps,
        load_best_model_at_end=select_best,
        metric_for_best_model="accuracy" if select_best else None,
        greater_is_better=True if select_best else None,
        per_device_eval_batch_size=int(cfg.get("eval_batch_size", 16)),
        logging_first_step=True,
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
        eval_dataset=eval_ds,
        data_collator=ListwiseCollator(tokenizer.pad_token_id),
        compute_metrics=ListwiseTrainer.compute_metrics,
        n_options=4,
        head_lr=float(cfg["head_lr"]) if cfg.get("head_lr") else None,
        callbacks=callbacks,
    )
    trainer.train(resume_from_checkpoint=True if args.resume else None)

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    meta = {
        "track": "ranker",
        "config": cfg,
        "selected": "best-dev" if select_best else "last-step",
        "total_params": total,
        "trainable_params": trainable,
        "train_questions": len(ds),
        "train_loss": next(
            (h["loss"] for h in reversed(trainer.state.log_history) if "loss" in h), None
        ),
        "evals": [
            {"step": h.get("step"), "eval_loss": h.get("eval_loss"),
             "eval_accuracy": h.get("eval_accuracy")}
            for h in trainer.state.log_history if "eval_accuracy" in h
        ],
    }
    evals = meta["evals"]
    if evals:
        best = max(evals, key=lambda e: e["eval_accuracy"])
        meta["best_eval"] = best
        print(
            f"train CE (last logged)={meta['train_loss']} | "
            f"best dev acc={best['eval_accuracy']:.4f} @ step {best['step']} "
            f"(chance 0.25)"
        )
        if best["eval_accuracy"] < 0.28 and len(evals) >= 3:
            print(
                "[collapse] dev accuracy never left chance over "
                f"{len(evals)} evals -- this config is dead, do not add epochs."
            )
    (out_dir / "run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"saved {out_dir}")

    if hub_enabled(cfg):
        upload_dir(cfg, out_dir, f"{name}/adapter", ignore=["checkpoints/*", "checkpoints"])
        upload_files(cfg, {f"{name}/run.json": out_dir / "run.json"})
    return out_dir


def load_ranker(path: str | Path, device: str):
    """Load a saved ranker plus the input format it was trained with.

    `query_mode` is part of the model's interface: evaluating a question-only
    model with option-augmented queries silently scores ~chance.
    """
    path = Path(path)
    query_mode = "full"
    run_json = path / "run.json"
    if run_json.exists():
        query_mode = json.loads(run_json.read_text(encoding="utf-8")).get(
            "config", {}
        ).get("query_mode", "full")
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=1).to(device)
    return model, tokenizer, query_mode


def run_eval(model_dir: str, data_dir: str = "data", batch_size: int = 16,
             max_len: int = 384, limit: int | None = None, ablations: list[str] | None = None,
             out: str | None = None) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, query_mode = load_ranker(model_dir, device)
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
                query_mode=query_mode,
            )
            summary = summarize(records, name=tag)
            results["runs"][tag] = summary
            print(one_line(tag, summary))
            if out:
                outp = Path(out)
                outp.parent.mkdir(parents=True, exist_ok=True)
                # One file per ablation -- a single fixed name meant every
                # ablation overwrote the previous one, so only "shuffle" survived.
                suffix = ".preds.jsonl" if ablate == "none" else f".{ablate}.preds.jsonl"
                outp.with_suffix(suffix).write_text(
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
