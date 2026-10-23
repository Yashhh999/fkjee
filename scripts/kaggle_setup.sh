#!/usr/bin/env bash
# Kaggle (2xT4) setup. Run once per notebook session.
set -euo pipefail

python -m pip install -q --upgrade pip
python -m pip install -q -e .

python - <<'PY'
import torch, transformers, peft
print("torch      ", torch.__version__, "| cuda:", torch.cuda.is_available(),
      "| devices:", torch.cuda.device_count())
print("transformers", transformers.__version__, "| peft:", peft.__version__)
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  gpu{i}: {p.name} {p.total_memory/1e9:.1f} GB  sm_{p.major}{p.minor}")
PY

# Data: ~1-2 min (downloads are cached under data/raw).
python -m fkjee.data prepare --data-dir data
