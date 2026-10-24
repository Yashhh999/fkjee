# Run fkjee on Kaggle

One cell. Paste it into a fresh Kaggle notebook with **Settings → Internet ON** and a **GPU**
(T4 ×2 or P100 — the code uses one GPU and never needs the second).

Optional: **Add-ons → Secrets → `HF_TOKEN`** (Hugging Face write token) so checkpoints, the probe
result and `results/sweep.md` land in `Yashhh999/fkjee`. Without it everything still trains; uploads
are skipped with one warning.

Knobs at the top of the cell:

| knob | values | meaning |
|---|---|---|
| `PROBE` | `True` / `False` | run the frozen-feature probe first (~10 min, decides everything) |
| `RUN` | `"rankers"`, `"tinybert"`, `"minilm-l6"`, `"minilm-l12"`, `"generators"`, `"none"` | what to train |
| `LIMIT` | `None`, or an int | use only N train items (smoke test) |
| `EPOCHS` | `None`, or a float | override the config's epoch count |

Timings on one T4 (configs as shipped, `max_len=384`, 4 epochs):

| stage | wall clock |
|---|---|
| probe (3 encoders × 2 query modes × 2 poolings) | ~10 min |
| `tinybert` (4.4M, 6 epochs) | ~20 min |
| `minilm-l6` (22.7M, 4 epochs) | ~35 min |
| `minilm-l12` (33.4M, 4 epochs) | ~50 min |

`RUN="rankers"` is the full predict track (~1 h 45 m including the probe). Set `LIMIT = 400`,
`EPOCHS = 1`, `PROBE = False` for a ~3-minute pipeline smoke test before the real run.

```python
# ============ fkjee: ranker fix run (Kaggle, GPU) ============
# Settings -> Internet ON, Accelerator -> GPU.  Optional secret: HF_TOKEN
import os, pathlib, subprocess, sys

PROBE  = True        # frozen-feature probe first; ~10 min and it decides the run
RUN    = "rankers"   # "rankers" | "tinybert" | "minilm-l6" | "minilm-l12" | "generators" | "none"
LIMIT  = None        # e.g. 400 for a 3-minute smoke test
EPOCHS = None        # e.g. 1 to sanity-check the pipeline
REPO   = "https://github.com/Yashhh999/fkjee"
DEST   = pathlib.Path("/kaggle/working/fkjee")

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# --- code: always take the latest main, even if this notebook is a rerun ---
if DEST.exists():
    subprocess.run(["git", "-C", str(DEST), "fetch", "--depth", "1", "origin", "main"], check=True)
    subprocess.run(["git", "-C", str(DEST), "reset", "--hard", "FETCH_HEAD"], check=True)
else:
    subprocess.run(["git", "clone", "--depth", "1", REPO, str(DEST)], check=True)
os.chdir(DEST)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], check=True)

print(subprocess.run(
    [sys.executable, "-c",
     "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(),"
     "'|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"],
    capture_output=True, text=True).stdout.strip(), flush=True)

try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    from huggingface_hub import HfApi
    print("HF user:", HfApi().whoami()["name"], flush=True)
except Exception as exc:
    print(f"Hub uploads disabled ({type(exc).__name__}); add an HF_TOKEN secret to enable", flush=True)

if not pathlib.Path("data/train.jsonl").exists():
    subprocess.run([sys.executable, "-m", "fkjee.data", "prepare", "--data-dir", "data"], check=True)

def run(cmd):
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=False)          # keep going even if one config dies

# --- 1. probe: is the answer signal in the encoder at all? -----------------
# Prints one line per encoder x query_mode x pooling, ending in a VERDICT.
if PROBE:
    run([sys.executable, "-u", "scripts/ranker_probe.py"])
    probe_path = pathlib.Path("results/ranker_probe.json")
    if probe_path.exists():
        import json
        probe = json.loads(probe_path.read_text())
        print("\n--- probe summary (chance = 0.25, ln4 = 1.386) ---", flush=True)
        for model in sorted({r["model"] for r in probe["rows"]}):
            rows_m = [r for r in probe["rows"] if r["model"] == model]
            best = max(rows_m, key=lambda r: r["dev_acc"])
            print(f"{model.split('/')[-1]:24s} best dev {best['dev_acc']:.3f} | "
                  f"train CE {best['train_ce']:.3f} | {best['query_mode']}/{best['pooling']} "
                  f"[{best['verdict']}]", flush=True)
        try:
            from huggingface_hub import HfApi
            if os.environ.get("HF_TOKEN"):
                HfApi().upload_file(path_or_fileobj=str(probe_path),
                                    path_in_repo="ranker_probe.json",
                                    repo_id="Yashhh999/fkjee")
                print("uploaded ranker_probe.json", flush=True)
        except Exception as exc:
            print(f"probe upload skipped ({type(exc).__name__})", flush=True)

# --- 2. train + evaluate ---------------------------------------------------
GROUPS = {"rankers": ["tinybert", "minilm"], "generators": ["smollm2", "qwen"]}
if RUN != "none":
    cmd = [sys.executable, "-u", "scripts/run_sweep.py", "--results", "results"]
    for sel in GROUPS.get(RUN, [RUN]):
        cmd += ["--only", sel]
    if LIMIT:  cmd += ["--limit", str(LIMIT)]
    if EPOCHS: cmd += ["--epochs", str(EPOCHS)]
    run(cmd)

    report = pathlib.Path("results/sweep.md")
    if report.exists():
        print("\n" + report.read_text(), flush=True)
    else:
        print("\n[!] no results/sweep.md -- scroll up for the first traceback", flush=True)
```

## Reading the output

The whole point of this run is that a collapse is now **visible in two minutes** instead of after
an hour.

**In the probe** — `train CE` is a linear head on frozen features, `ln4 = 1.3863`. Each row is
labelled; only `SIGNAL` justifies a fine-tune:

* `SIGNAL` — fits train **and** beats chance on dev (CE below `ln4 - 0.02`, accuracy above
  `0.25 + 1 SE`). Read which `(model, query_mode, pooling)` won and use it.
* `weak` — dev CE improves but accuracy is inside the noise band. Worth at most one fine-tune.
* `memorises` — fits the train questions and falls back to chance on dev. More capacity will not
  help; this is what a 22M encoder does on JEE.
* `flat` — cannot even fit the train split. Dead end.

If every row is `memorises`/`flat`, move up in size or to the generator track. That is a real,
publishable negative result, not a bug.

**In training** — `Trainer` logs an eval block every 100 steps:

```
{'eval_loss': 1.3863, 'eval_accuracy': 0.25, ...}
```

* `eval_accuracy` climbing off 0.25 within the first 300 steps → learning.
* flat at 0.25 for 300+ steps while `eval_loss` stays put → the model has collapsed to a constant
  score. Kill that config; more epochs will not help.
* `loss` is now the **true** 4-way cross-entropy (the old log multiplied it by
  `gradient_accumulation_steps`, which is why it read 2.773 = ln 16 and looked like a 16-way
  problem). `loss_raw` keeps the old number for comparison.
* `eval_loss` starts **above** 1.3863 for a pretrained cross-encoder: its MS MARCO head emits
  non-zero logits at step 0, so CE is not the random-guess value. What matters is that it *falls*
  and `eval_accuracy` rises. A fixed 0.25 with a falling loss can still mean it only learned the
  option prior; check the `abl_shuffle` and `abl_question` columns.

Afterwards, `results/sweep.md` has `dev | holdout | JEE | NEET | per-subject | all four ablations`
per model, and `results/<name>.preds.jsonl` has every prediction for error analysis.

**What to send back:** the probe summary block, the `eval_loss`/`eval_accuracy` lines from the
first 300 steps of each model, and the final `results/sweep.md` table.

## Resume after a Kaggle timeout

Checkpoints go to `runs/<name>/checkpoints/step-N` locally and to the Hub. In a fresh session:

```bash
cd /kaggle/working/fkjee
python -m fkjee.classifier train --config configs/minilm-l12-clf.yaml --resume
```

`--resume` picks up the newest local checkpoint; the Hub copies are for pulling into a new
session.
