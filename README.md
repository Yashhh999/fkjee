# fkjee

Find the **smallest model that can answer Indian entrance-exam questions** — JEE Main, NEET, and
anything else in the same 4-option format — and prove it with honest ablations.

Two tracks, because "solve" and "predict" are different problems:

| Track | What the model does | Params (total) | Trainable (LoRA) |
|---|---|---|---|
| **solve** — generative | reads the question, writes reasoning, emits `\boxed{C}` | 139M / 362M / 494M / 1.5B | ~4.9M at 139M |
| **predict** — listwise ranker | scores the 4 options with a cross-encoder, softmax, pick max | **4.4M / 22.7M / 33.4M** | all |

The ranker is ~100× smaller than the 1.5B generator. Whether it is *better* is an empirical
question this repo answers rather than assumes — see [Ablations](#ablations-why-you-should-not-trust-a-raw-accuracy-number).

## Data

| Source | Exam | Items kept | Role |
|---|---|---:|---|
| [`ruh-ai/grafite-jee-mains-qna-no-img`](https://huggingface.co/datasets/ruh-ai/grafite-jee-mains-qna-no-img) | JEE Main / AIEEE 2002–2024 | 11,391 | train + test, **chapter-tagged** |
| [`geekyrakshit/indian-exam-questions`](https://huggingface.co/datasets/geekyrakshit/indian-exam-questions) | NEET 2025 + 2026 | 1,138 → 385 unique | train + test, adds **biology** |
| [`PhysicsWallahAI/JEE-Main-2025-Math`](https://huggingface.co/datasets/PhysicsWallahAI/JEE-Main-2025-Math) | JEE Main 2025 Maths | 469 | test |
| [`CK0607/2025-Jee-Mains-Question`](https://huggingface.co/datasets/CK0607/2025-Jee-Mains-Question) | JEE Main 2025 Maths | 250 | test |
| [`daman1209arora/jeebench`](https://huggingface.co/datasets/daman1209arora/jeebench) | JEE **Advanced** 2016 | 94 MCQ | out-of-domain test |

After cross-source dedupe: **12,383 unique questions, 90 syllabus modules, 4 subjects**.

```
train  9,512   (JEE 9,315 · NEET 197)   math 3,689 · phy 2,933 · chem 2,797 · bio 93
dev      517
test   2,354   (JEE 2,167 · NEET 187)   math 1,226 · phy  524 · chem  510 · bio 94
```

`data/coverage.md` is generated on every `make data` run and lists train/test counts **per
chapter**, so you can see exactly which modules are covered and which are thin.

### Split protocol

* **Paper-level**, never row-level. Every question from one paper lands on the same side.
* **Holdout by year**: JEE ≥ 2024, NEET ≥ 2026 (`data.py::DEFAULT_HOLDOUT`, CLI-overridable).
  Nothing from those papers is trained on.
* **Dedupe before split.** NEET ships four codes per exam that reuse the same questions; 1,138
  rows collapse to 385 unique. Done *before* splitting, so a duplicate can never straddle the
  boundary. `prepare` asserts `leakage == 0` and warns if not.
* **Figure-dependent questions are dropped.** Anything containing `<image_N>` is unanswerable by
  a text model, so it is excluded rather than silently guessed.
* **No fabricated chain-of-thought.** The model is trained on a rationale only when the source
  actually has one (97% of JEE rows). Otherwise the target is the bare answer.

## Ablations (why you should not trust a raw accuracy number)

A 4-way MCQ has a **25% chance floor**, and option sets leak information (longest option, "all of
the above", topic priors). The ranker is therefore evaluated four ways on the same questions:

| Flag | What it does | What it proves |
|---|---|---|
| `none` | normal | headline accuracy |
| `question` | blanks the question, keeps options | if ≫ 0.25, the model is reading option *shape*, not the problem |
| `options` | blanks option text, keeps question | if ≫ 0.25, it is not using option content at all |
| `shuffle` | permutes options per question | a big drop from `none` means letter/position bias |

A result is only believable when `none` is well above chance **and** both ablations fall to ~0.25.

## Layout

```
src/fkjee/
  data.py        clean/dedupe/split + coverage report (stdlib only)
  sources.py     pluggable question-bank registry (add an exam = one function)
  prompts.py     chat prompt + supervision target
  parsing.py     \boxed{} / "option (C)" / numeric extraction
  metrics.py     exact-match scoring, per-subject/exam/chapter breakdown
  classifier.py  the tiny listwise ranker (predict track)
  train.py       LoRA/QLoRA SFT (solve track)
  evaluate.py    greedy generation + scoring
  hub.py         optional checkpoint/eval upload to the HF Hub
configs/         one YAML per model
scripts/run_sweep.py   train+eval every config, emit results/sweep.md
tests/           50 unit tests, no GPU or torch needed
```

## Quickstart

Data prep and tests need only Python ≥ 3.10 (no torch):

```bash
make data          # ~1-2 min, downloads are cached in data/raw/
make test          # 47 tests
head -30 data/coverage.md
```

Training on Kaggle (2×T4):

```bash
git clone <this-repo> /kaggle/working/fkjee && cd /kaggle/working/fkjee
bash scripts/kaggle_setup.sh
python scripts/run_sweep.py --only tinybert minilm    # rankers first: minutes, not hours
python scripts/run_sweep.py --only smollm2 qwen       # generators: hours
```

Single runs:

```bash
python -m fkjee.classifier train --config configs/minilm-l6-clf.yaml
python -m fkjee.classifier eval  --model-dir runs/minilm-l6-clf \
    --ablate none question options shuffle

python -m fkjee.train    --config configs/qwen2.5-0.5b.yaml
python -m fkjee.evaluate --run runs/qwen2.5-0.5b \
    --files data/test_holdout.jsonl data/jeebench_mcq.jsonl
```

Outputs: `runs/<name>/run.json` (params, loss), `results/sweep.md` (the comparison table),
`results/*.preds.jsonl` (per-question predictions for error analysis).

## Time budget (one T4, 16 GB)

Order-of-magnitude estimates for the configs as shipped. Measure the first 50 steps — the sweep
logs `it/s` — and trust that over this table.

| config | params | train | + eval | total |
|---|---:|---|---:|---:|
| `tinybert-l2-clf` | 4.4M | 15–25 min | 2 min | **~20 min** |
| `minilm-l6-clf` | 22.7M | 20–35 min | 3 min | **~30 min** |
| `minilm-l12-clf` | 33.4M | 30–50 min | 3 min | **~45 min** |
| `smollm2-135m` | 139M | 25–40 min | 10–15 min | **~45 min** |
| `smollm2-360m` | 362M | 1–1.5 h | ~20 min | **~1.5 h** |
| `qwen2.5-0.5b` | 494M | 1.5–2.5 h | ~25 min | **~2.5 h** |
| `qwen2.5-1.5b` | 1.5B | 3–5 h | 45–60 min | **~4–5 h** |

Full sweep ≈ **10–14 GPU-hours**. Kaggle gives ~30 GPU-h/week but caps a session at 12 h, so run
rankers first (minutes), then one generator per session and `--resume` if you get cut off.

## Checkpointing to the Hub

Set a token and everything uploads automatically; with no token, uploads are skipped with a single
warning and training continues locally.

```bash
export HF_TOKEN=hf_xxx        # Kaggle: Add-ons → Secrets → HF_TOKEN
python -m fkjee.train --config configs/qwen2.5-0.5b.yaml
python -m fkjee.train --config configs/qwen2.5-0.5b.yaml --resume   # after a timeout
```

One repo holds all runs:

```
Yashhh999/fkjee/
  sweep.md
  <run-name>/run.json                    params, loss, resolved config
  <run-name>/eval.json                   scores + per-subject/exam breakdown
  <run-name>/adapter/                    final LoRA adapter or ranker weights
  <run-name>/checkpoints/step-500/...    resume-able checkpoints
```

Knobs (in `configs/base.yaml`, overridable per model):

| key | default | meaning |
|---|---|---|
| `hub_repo` | `Yashhh999/fkjee` | target repo; set `null` to disable |
| `hub_private` | `true` | create the repo private |
| `save_steps` | `500` | local save + upload interval |
| `save_total_limit` | `2` | keep only the last N local checkpoints |
| `push_checkpoints` | `true` | upload each checkpoint |

## What to expect

These are **falsifiable predictions**, not measured results — the sweep has not been run yet
(there is no GPU in this workspace).

* 4.4M ranker: likely 27–33% — barely above chance, useful as a floor.
* 22–33M ranker: likely 32–42%. This is where a tiny model starts to be genuinely better than
  guessing on easy/definitional questions.
* 139M generator: likely 25–33%; small generators often collapse to a letter prior.
* 0.5–1.5B generator: likely 35–50%.

JEE Advanced (JEEBench) will be much lower than JEE Main for every model — it is both harder and
out-of-domain. Do not expect any model here to clear 50% overall.

## Limitations

* **Biology is thin** (93 train / 94 test). NEET's codes reuse questions, so only ~385 unique NEET
  items exist across 1,440 rows. Adding more NEET/Biology sources is the single biggest win.
* **Integer questions** are 23% of JEE Main; the ranker cannot answer them at all, and the
  generators answer them with numeric parsing that can be brittle on units.
* **Contamination risk** is real for the 1.5B model: Qwen2.5 may have seen JEE text during
  pretraining. The 2024+ holdout mitigates but does not eliminate this.
* **Explanations are uneven.** Image-only solutions are dropped rather than hallucinated, so
  ~3% of JEE rows train as direct-answer rather than reasoning.
* The ranker is a **different claim**: it picks the right option, it does not derive it. It is a
  strong small-scale baseline, not a tutor.

## Roadmap

1. More sources: BITSAT/KCET/WBJEE/MHT-CET banks, plus Biology to fix the thin module.
2. Distillation: let a large teacher write rationales for questions the sources lack, then train
   the small generator on them (rejection-sampled, verifier-filtered).
3. Numeric head on the ranker so integer questions are covered without a generator.
4. Curriculum by chapter difficulty using `data/coverage.md`.

## Licences

grafite is MIT; JEEBench is MIT; the NEET/PhysicsWallah/CK datasets carry their own licences on
the Hub. Check each before redistribution.
