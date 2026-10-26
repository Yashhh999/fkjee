"""Greedy-decode evaluation for the generative track."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import LETTERS, Item, load_jsonl
from .metrics import one_line, summarize
from .parsing import parse_answer
from .prompts import build_messages
from .train import _dtype_kwarg, pick_dtype


def load_model(model_id: str, adapter: str | None, dtype: torch.dtype):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_id, **_dtype_kwarg(dtype))
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate(model, tokenizer, items: list[Item], *, batch_size: int, max_new_tokens: int,
             max_len: int, device: str) -> list[str]:
    outputs: list[str] = []
    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                build_messages(it), tokenize=False, add_generation_prompt=True
            )
            for it in chunk
        ]
        enc = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True,
            max_length=max_len, add_special_tokens=False,
        ).to(device)
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        new_tokens = gen[:, enc["input_ids"].shape[1] :]
        outputs.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
        done = min(start + batch_size, len(items))
        print(f"\r  generated {done}/{len(items)}", end="", flush=True)
    print()
    return outputs


def score(generations: list[str], items: list[Item], tag: str) -> tuple[dict, list[dict]]:
    records = []
    for text, item in zip(generations, items):
        pred, method = parse_answer(text)
        records.append(
            {
                "id": item.id,
                "subject": item.subject,
                "exam": item.exam,
                "chapter": item.chapter,
                "kind": item.kind,
                "year": item.year,
                "gold": item.gold,
                "pred": pred,
                "method": method,
                "raw": text.strip()[-600:],
            }
        )
    return summarize(records, name=tag), records


def take_evenly(items: list[Item], n: int | None) -> list[Item]:
    """Deterministic evenly-spaced sample.

    Generator eval is generation-bound, so we score a slice.  Taking the first
    N rows would score whichever paper happens to be first, so spread the slice
    across the whole suite instead (same ordering, no RNG to explain).
    """
    if not n or len(items) <= n:
        return items
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def evaluate(args: argparse.Namespace) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[warn] no CUDA device: generation will be extremely slow")

    model_id = args.model
    adapter = args.adapter
    if args.run:
        meta = json.loads((Path(args.run) / "run.json").read_text(encoding="utf-8"))
        model_id = meta["config"]["model_id"]
        adapter = str(Path(args.run))
        print(f"run={args.run} base={model_id} adapter={adapter}")

    dtype = pick_dtype()
    model, tokenizer = load_model(model_id, adapter, dtype)
    model.to(device)

    results: dict = {"model": model_id, "adapter": adapter, "files": {}}
    for path in args.files:
        p = Path(path)
        if not p.exists():
            print(f"[skip] {p} missing")
            continue
        items = load_jsonl(p)
        if args.kind:
            items = [it for it in items if it.kind == args.kind]
        if args.limit:
            items = items[: args.limit]
        items = take_evenly(items, args.sample)
        if not items:
            continue

        print(f"{p.name}: {len(items)} items")
        gens = generate(
            model, tokenizer, items,
            batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
            max_len=args.max_len, device=device,
        )
        summary, records = score(gens, items, p.stem)
        results["files"][p.stem] = summary
        print(one_line(p.stem, summary))

        out = Path(args.out) if args.out else Path("results") / f"{p.stem}-gen.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        preds = out.with_suffix(".preds.jsonl")
        preds.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")

        if args.show_errors:
            bad = [r for r in records if r["pred"] != r["gold"]][: args.show_errors]
            for r in bad:
                print(f"\n--- {r['id']} gold={r['gold']} pred={r['pred']} ({r['method']})")
                print(r["raw"][-300:])
    return results


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="evaluate a generative fkjee model")
    ap.add_argument("--run", default=None, help="adapter dir containing run.json")
    ap.add_argument("--model", default=None, help="base model id (if not using --run)")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--files", nargs="+", default=["data/test_holdout.jsonl"])
    ap.add_argument("--kind", default=None, choices=["mcq", "integer"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sample", type=int, default=None,
                    help="evenly-spaced subsample of each suite (generation is slow)")
    ap.add_argument("--show-errors", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if not args.run and not args.model:
        ap.error("pass --run or --model")
    evaluate(args)


if __name__ == "__main__":
    main()
