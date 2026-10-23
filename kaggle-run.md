# Run fkjee on Kaggle

One cell. Paste it into a fresh Kaggle notebook with **Settings → Internet ON**.

Optional: **Add-ons → Secrets → `HF_TOKEN`** (Hugging Face write token) so checkpoints and
`results/sweep.md` land in `Yashhh999/fkjee`. Without it, training runs normally and uploads are
skipped.

Knobs at the top of the cell:

| knob | values | meaning |
|---|---|---|
| `RUN` | `"rankers"`, `"generators"`, `"all"`, or a config name like `"qwen2.5-0.5b"` | what to train |
| `LIMIT` | `None`, or an int | use only N train items (smoke test) |
| `EPOCHS` | `None`, or a float | override the config's epoch count |

`RUN="rankers"` finishes in about an hour on one T4. `RUN="generators"` takes 6–9 hours — run it
in its own session. Resume after a timeout with:

```bash
python -m fkjee.train --config configs/qwen2.5-0.5b.yaml --resume
```

```python
# ============ fkjee one-shot (Kaggle, 2xT4) ============
# Settings -> Internet ON.  Optional: Add-ons -> Secrets -> HF_TOKEN
# Without HF_TOKEN everything still trains; Hub uploads are skipped.
import os, pathlib, subprocess, sys

RUN    = "rankers"   # "rankers" | "generators" | "all" | single config name
LIMIT  = None        # e.g. 200 for a 5-minute smoke test
EPOCHS = None        # e.g. 1 to sanity-check the pipeline
REPO   = "https://github.com/Yashhh999/fkjee"
DEST   = pathlib.Path("/kaggle/working/fkjee")

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

if DEST.exists():
    subprocess.run(["git", "-C", str(DEST), "pull", "--ff-only"], check=False)
else:
    subprocess.run(["git", "clone", "--depth", "1", REPO, str(DEST)], check=True)
os.chdir(DEST)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], check=True)

try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    from huggingface_hub import HfApi
    print("HF user:", HfApi().whoami()["name"])
except Exception as exc:
    print(f"Hub uploads disabled ({type(exc).__name__}); add an HF_TOKEN secret to enable")

if not pathlib.Path("data/train.jsonl").exists():
    subprocess.run([sys.executable, "-m", "fkjee.data", "prepare", "--data-dir", "data"], check=True)

GROUPS = {"rankers": ["tinybert", "minilm"],
          "generators": ["smollm2", "qwen"],
          "all": []}
cmd = [sys.executable, "scripts/run_sweep.py", "--results", "results"]
for o in GROUPS.get(RUN, [RUN]):
    cmd += ["--only", o]
if LIMIT:  cmd += ["--limit", str(LIMIT)]
if EPOCHS: cmd += ["--epochs", str(EPOCHS)]
subprocess.run(cmd, check=True)

print(pathlib.Path("results/sweep.md").read_text())
```
