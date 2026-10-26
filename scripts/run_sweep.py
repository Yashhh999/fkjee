#!/usr/bin/env python3
"""Train + evaluate every config, then emit one comparison table.

    python scripts/run_sweep.py                 # full sweep
    python scripts/run_sweep.py --limit 500     # smoke test
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fkjee.config import load_config  # noqa: E402


def sh(cmd: list[str], env: dict) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def params_of(run_dir: Path) -> float | None:
    meta = read_json(run_dir / "run.json")
    total = meta.get("total_params")
    return round(total / 1e6, 2) if total else None


def dev_acc_of(run_dir: Path) -> float | None:
    """Best dev accuracy seen during training -- the collapse detector."""
    return read_json(run_dir / "run.json").get("best_eval", {}).get("eval_accuracy")


def extract(name: str, track: str, results_dir: Path) -> dict:
    data = read_json(results_dir / f"{name}.json")
    row = {"name": name, "track": track}
    if track == "ranker":
        runs = data.get("runs", {})
        for key, label in [
            ("holdout/none", "holdout"),
            ("holdout/question", "abl_question"),
            ("holdout/solo", "abl_solo"),
            ("holdout/shuffle", "abl_shuffle"),
            ("jeebench/none", "jeebench"),
        ]:
            row[label] = runs.get(key, {}).get("accuracy")
        row["by_subject"] = runs.get("holdout/none", {}).get("by_subject", {})
        row["by_exam"] = runs.get("holdout/none", {}).get("by_exam", {})
    else:
        files = data.get("files", {})
        row["holdout"] = files.get("test_holdout", {}).get("accuracy")
        row["jeebench"] = files.get("jeebench_mcq", {}).get("accuracy")
        row["by_subject"] = files.get("test_holdout", {}).get("by_subject", {})
        row["by_exam"] = files.get("test_holdout", {}).get("by_exam", {})
    return row


def fmt(v) -> str:
    return "-" if v is None else f"{v:.3f}"


def write_report(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    gen = sorted([r for r in rows if r["track"] == "generative"], key=lambda r: r.get("params") or 0)
    rank = sorted([r for r in rows if r["track"] == "ranker"], key=lambda r: r.get("params") or 0)

    lines = [
        "# fkjee sweep",
        "",
        "Held-out = every paper at/after the holdout year (JEE 2024+, NEET 2026).",
        "JEEBench = 94 single-correct JEE Advanced MCQs (out-of-domain).",
        "Chance = 0.25. `abl_question` blanks the question and keeps only options.",
        "`abl_solo` scores each option against the question alone (no option set in the query).",
        "",
        "## Predict (listwise ranker)",
        "",
        "| model | params (M) | dev | holdout | JEE | NEET | math | phy | chem | bio | abl_question | abl_solo | abl_shuffle | jeebench |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rank:
        s = r.get("by_subject", {})
        e = r.get("by_exam", {})
        lines.append(
            f"| {r['name']} | {r.get('params','-')} | {fmt(r.get('dev'))} | {fmt(r.get('holdout'))} | "
            f"{fmt(e.get('jee',{}).get('accuracy'))} | {fmt(e.get('neet',{}).get('accuracy'))} | "
            f"{fmt(s.get('math',{}).get('accuracy'))} | {fmt(s.get('physics',{}).get('accuracy'))} | "
            f"{fmt(s.get('chemistry',{}).get('accuracy'))} | {fmt(s.get('biology',{}).get('accuracy'))} | "
            f"{fmt(r.get('abl_question'))} | {fmt(r.get('abl_solo'))} | "
            f"{fmt(r.get('abl_shuffle'))} | {fmt(r.get('jeebench'))} |"
        )

    lines += [
        "",
        "## Solve (generative, LoRA)",
        "",
        "| model | params (M) | holdout | JEE | NEET | math | phy | chem | bio | jeebench |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in gen:
        s = r.get("by_subject", {})
        e = r.get("by_exam", {})
        lines.append(
            f"| {r['name']} | {r.get('params','-')} | {fmt(r.get('holdout'))} | "
            f"{fmt(e.get('jee',{}).get('accuracy'))} | {fmt(e.get('neet',{}).get('accuracy'))} | "
            f"{fmt(s.get('math',{}).get('accuracy'))} | {fmt(s.get('physics',{}).get('accuracy'))} | "
            f"{fmt(s.get('chemistry',{}).get('accuracy'))} | {fmt(s.get('biology',{}).get('accuracy'))} | "
            f"{fmt(r.get('jeebench'))} |"
        )

    lines += [
        "",
        "## Reading the ablation columns",
        "",
        "- `abl_question` ~= 0.25 means the model is not exploiting option shape alone.",
        "- `abl_solo` ~= holdout means the option set in the query is doing no work.",
        "- `abl_shuffle` should track `holdout`; a large drop means letter/position bias.",
        "- `dev` is the best 4-way accuracy seen on all of dev during training; the",
        "  saved weights are that best checkpoint, not the last step.",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    (out.parent / "sweep.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("\n" + "\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="configs/*.yaml")
    ap.add_argument("--epochs", type=float, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", nargs="*", default=None, help="substrings of config names")
    ap.add_argument("--results", default="results")
    args = ap.parse_args()

    os.chdir(ROOT)
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    py = sys.executable
    results_dir = Path(args.results)
    results_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for cfg_path in sorted(glob.glob(args.configs)):
        p = Path(cfg_path)
        if p.name == "base.yaml":
            continue
        cfg = load_config(p)
        name = cfg.get("name", p.stem)
        if args.only and not any(s in name for s in args.only):
            continue
        track = cfg.get("track", "generative")
        print(f"\n{'='*70}\n{name} [{track}]\n{'='*70}")

        try:
            if track == "ranker":
                sh([py, "-m", "fkjee.classifier", "train", "--config", str(p)]
                   + (["--epochs", str(args.epochs)] if args.epochs else [])
                   + (["--limit", str(args.limit)] if args.limit else []), env)
                sh([py, "-m", "fkjee.classifier", "eval",
                    "--model-dir", f"runs/{name}",
                    "--ablate", "none", "question", "solo", "shuffle",
                    "--out", str(results_dir / f"{name}.json")]
                   + (["--limit", str(args.limit)] if args.limit else []), env)
                row = extract(name, track, results_dir)
                row["params"] = params_of(Path(f"runs/{name}"))
                row["dev"] = dev_acc_of(Path(f"runs/{name}"))
            else:
                sh([py, "-m", "fkjee.train", "--config", str(p)]
                   + (["--epochs", str(args.epochs)] if args.epochs else [])
                   + (["--limit", str(args.limit)] if args.limit else []), env)
                # Generation is the slow part: 2.3k holdout items x up to 384 new
                # tokens each can outlast a Kaggle session on its own.  Score a
                # fixed slice (500 questions ~= 2% SE) instead of the full set.
                gen_limit = args.limit or cfg.get("eval_limit", 500)
                sh([py, "-m", "fkjee.evaluate", "--run", f"runs/{name}",
                    "--files", "data/test_holdout.jsonl", "data/jeebench_mcq.jsonl",
                    "--sample", str(gen_limit),
                    "--max-new-tokens", str(cfg.get("max_new_tokens", 384)),
                    "--batch-size", str(cfg.get("eval_batch_size", 16)),
                    "--out", str(results_dir / f"{name}.json")], env)
                row = extract(name, track, results_dir)
                row["params"] = params_of(Path(f"runs/{name}"))
            rows.append(row)
        except subprocess.CalledProcessError:
            print(f"[FAIL] {name}")
            traceback.print_exc()

    if rows:
        write_report(rows, results_dir / "sweep.md")
        from fkjee.hub import enabled as hub_enabled, upload_files

        hub_cfg = load_config(Path("configs/base.yaml"))
        if hub_enabled(hub_cfg):
            files = {"sweep.md": results_dir / "sweep.md"}
            for r in rows:
                files[f"{r['name']}/eval.json"] = results_dir / f"{r['name']}.json"
            upload_files(hub_cfg, files)


if __name__ == "__main__":
    main()
