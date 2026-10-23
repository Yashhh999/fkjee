"""LoRA / QLoRA supervised fine-tuning for the generative track."""

from __future__ import annotations

import argparse
import inspect
import json
import math
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from .config import load_config
from .data import Item, load_jsonl
from .hub import (
    checkpoint_callback,
    enabled as hub_enabled,
    ensure_repo,
    upload_dir,
    upload_files,
)
from .prompts import build_messages, target_text
from .runtime import warmup_kwargs

DEFAULT_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def pick_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def _dtype_kwarg(dtype: torch.dtype) -> dict:
    """transformers renamed torch_dtype -> dtype (v5 makes the old name warn)."""
    import transformers

    try:
        major = int(transformers.__version__.split(".")[0])
    except ValueError:
        major = 0
    if major >= 5:
        return {"dtype": dtype}
    params = inspect.signature(AutoModelForCausalLM.from_pretrained).parameters
    return {"dtype" if "dtype" in params else "torch_dtype": dtype}


class SFTDataset(torch.utils.data.Dataset):
    """Prompt tokens are masked out of the loss; only the completion is learned."""

    def __init__(self, items: list[Item], tokenizer, max_len: int):
        self.rows: list[tuple[list[int], list[int]]] = []
        self.skipped = 0
        eos = tokenizer.eos_token_id

        for item in items:
            prompt = tokenizer.apply_chat_template(
                build_messages(item), tokenize=False, add_generation_prompt=True
            )
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            target_ids = tokenizer(target_text(item), add_special_tokens=False)["input_ids"]
            target_ids = target_ids + [eos]

            if len(prompt_ids) + len(target_ids) > max_len:
                self.skipped += 1
                continue
            labels = [-100] * len(prompt_ids) + target_ids
            self.rows.append((prompt_ids + target_ids, labels))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        ids, labels = self.rows[i]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class PadCollator:
    def __init__(self, pad_token_id: int):
        self.pad = pad_token_id

    def __call__(self, features: list[dict]) -> dict:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, labels, attn = [], [], []
        for f in features:
            n = max_len - len(f["input_ids"])
            input_ids.append(torch.cat([f["input_ids"], torch.full((n,), self.pad, dtype=torch.long)]))
            labels.append(torch.cat([f["labels"], torch.full((n,), -100, dtype=torch.long)]))
            attn.append(torch.cat([torch.ones(len(f["input_ids"]), dtype=torch.long),
                                   torch.zeros(n, dtype=torch.long)]))
        return {
            "input_ids": torch.stack(input_ids),
            "labels": torch.stack(labels),
            "attention_mask": torch.stack(attn),
        }


def build_model(cfg: dict, dtype: torch.dtype):
    kwargs = _dtype_kwarg(dtype)
    if cfg.get("load_in_4bit"):
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(cfg["model_id"], **kwargs)
    if cfg.get("load_in_4bit"):
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False

    lora = LoraConfig(
        r=int(cfg.get("lora_r", 16)),
        lora_alpha=int(cfg.get("lora_alpha", 32)),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg.get("target_modules") or DEFAULT_TARGETS,
    )
    return get_peft_model(model, lora)


def train(args: argparse.Namespace) -> Path:
    cfg = load_config(
        args.config,
        {
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "max_len": args.max_len,
            "limit": args.limit,
        },
    )
    name = cfg.get("name") or Path(args.config).stem
    data_dir = Path(cfg.get("data_dir", "data"))
    out_dir = Path(cfg.get("out_dir", "runs")) / name
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = pick_dtype()

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    items = load_jsonl(data_dir / "train.jsonl")
    if cfg.get("limit"):
        items = items[: int(cfg["limit"])]
    ds = SFTDataset(items, tokenizer, int(cfg.get("max_len", 1024)))
    print(f"model={cfg['model_id']} train_examples={len(ds)} skipped_long={ds.skipped}")

    model = build_model(cfg, dtype)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"total_params={total/1e6:.2f}M trainable={trainable/1e6:.3f}M ({100*trainable/total:.2f}%)")

    targs = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=float(cfg.get("epochs", 2)),
        per_device_train_batch_size=int(cfg.get("batch_size", 4)),
        gradient_accumulation_steps=int(cfg.get("grad_accum", 4)),
        learning_rate=float(cfg.get("lr", 2e-4)),
        **warmup_kwargs(
            float(cfg.get("warmup_ratio", 0.03)),
            math.ceil(len(ds) / max(1, int(cfg.get("batch_size", 4)) * int(cfg.get("grad_accum", 4))))
            * max(1, int(float(cfg.get("epochs", 2)))),
        ),
        lr_scheduler_type="cosine",
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        logging_steps=int(cfg.get("logging_steps", 20)),
        save_strategy="steps",
        save_steps=int(cfg.get("save_steps", 500)),
        save_total_limit=int(cfg.get("save_total_limit", 2)),
        eval_strategy="no",
        report_to="none",
        seed=int(cfg.get("seed", 42)),
        remove_unused_columns=False,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        fp16=device == "cuda" and dtype == torch.float16,
        bf16=device == "cuda" and dtype == torch.bfloat16,
        dataloader_num_workers=2,
    )

    callbacks = []
    if hub_enabled(cfg):
        ensure_repo(cfg)  # fail early, not at the first checkpoint
        if cfg.get("push_checkpoints", True):
            callbacks.append(checkpoint_callback(cfg, name))

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=PadCollator(tokenizer.pad_token_id),
        callbacks=callbacks,
    )
    trainer.train(resume_from_checkpoint=True if args.resume else None)

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    meta = {
        "track": "generative",
        "config": cfg,
        "total_params": total,
        "trainable_params": trainable,
        "train_examples": len(ds),
        "skipped_long": ds.skipped,
        "train_loss": losses[-1] if losses else None,
    }
    (out_dir / "run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"saved adapter to {out_dir}")

    if hub_enabled(cfg):
        upload_dir(cfg, out_dir, f"{name}/adapter", ignore=["checkpoints/*", "checkpoints"])
        upload_files(cfg, {f"{name}/run.json": out_dir / "run.json"})

    if args.export_merged:
        export_merged(cfg["model_id"], out_dir, Path(args.export_merged), dtype)
    return out_dir


def export_merged(base_id: str, adapter_dir: Path, dest: Path, dtype: torch.dtype) -> None:
    """Merge the adapter into the base weights for vLLM / llama.cpp."""
    from peft import PeftModel

    dest.mkdir(parents=True, exist_ok=True)
    base = AutoModelForCausalLM.from_pretrained(base_id, **_dtype_kwarg(dtype))
    merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
    merged.save_pretrained(dest, safe_serialization=True)
    AutoTokenizer.from_pretrained(str(adapter_dir)).save_pretrained(dest)
    print(f"merged model -> {dest}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="LoRA SFT for fkjee generative track")
    ap.add_argument("--config", required=True)
    ap.add_argument("--epochs", type=float, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="smoke test on a subset")
    ap.add_argument("--resume", action="store_true", help="resume from the latest checkpoint")
    ap.add_argument("--export-merged", default=None)
    train(ap.parse_args(argv))


if __name__ == "__main__":
    main()
