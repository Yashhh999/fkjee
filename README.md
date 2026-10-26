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
| `solo` | drops the option set from the query; scores question vs one option | if it matches `none`, the other three options contributed nothing |
| `shuffle` | permutes options per question | a big drop from `none` means letter/position bias |

A result is only believable when `none` is well above chance **and** the ablations stay near 0.25.
Note there is no "blank the option text" ablation: replacing all four option texts with the same
placeholder makes the four logits identical, so accuracy is 0.25 **by construction**. It looks like
a clean result and measures nothing.

## Layout

```
src/fkjee/
  data.py        clean/dedupe/split + coverage report (stdlib only)
  sources.py     pluggable question-bank registry (add an exam = one function)
  prompts.py     chat prompt + supervision target
  parsing.py     \boxed{} / "option (C)" / numeric extraction
  metrics.py     exact-match scoring, per-subject/exam/chapter breakdown
  classifier.py  the tiny listwise ranker (predict track)
  lexical.py     8-parameter lexical ranker (the floor)
  train.py       LoRA/QLoRA SFT (solve track)
  evaluate.py    greedy generation + scoring
  hub.py         optional checkpoint/eval upload to the HF Hub
configs/         one YAML per model
scripts/run_sweep.py    train+eval every config, emit results/sweep.md
scripts/ranker_probe.py frozen-feature probe: is the answer signal in the encoder at all?
tests/           50 unit tests, no GPU or torch needed
```

## Quickstart

Data prep and tests need only Python ≥ 3.10 (no torch):

```bash
make data          # ~1-2 min, downloads are cached in data/raw/
make test          # 50 tests
head -30 data/coverage.md
```

Training on Kaggle (2×T4):

```bash
git clone <this-repo> /kaggle/working/fkjee && cd /kaggle/working/fkjee
bash scripts/kaggle_setup.sh
python scripts/ranker_probe.py                         # ~4 min: is the signal there?
python scripts/run_sweep.py --only tinybert minilm     # rankers first: minutes, not hours
python scripts/run_sweep.py --only smollm2 qwen        # generators: hours
```

Or paste the single cell in [`kaggle-run.md`](kaggle-run.md), which does all of the above.

Single runs:

```bash
python -m fkjee.classifier train --config configs/minilm-l6-clf.yaml
python -m fkjee.classifier eval  --model-dir runs/minilm-l6-clf \
    --ablate none question solo shuffle

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
one generator per session and `--resume` if you get cut off. Generator eval in the sweep is a
500-question holdout slice plus all 94 JEEBench questions (generation-bound); override with
`eval_limit` / `max_new_tokens` in the config if you want the full 2,354.

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

## Results so far (measured, not predicted)

Every number below comes from a Kaggle T4 run; `results/sweep.md` and the per-model JSONs are
uploaded to the Hub. Chance is 0.25 for every MCQ number.

### The predict track is at chance

| model | params | dev peak | holdout acc | n |
|---|---:|---:|---:|---:|
| `minilm-l6-clf` (ms-marco cross-encoder) | 22.7M | 0.328 | **0.262** | 1,684 |
| `minilm-l12-clf` (ms-marco cross-encoder) | 33.4M | 0.304 | **0.253** | 1,684 |
| lexical baseline (7 features + bias) | **8 params** | - | **0.259** | 1,684 |
| chance | 0 | 0.25 | 0.250 | 1,684 |

The standard error at n=1,684 is ±1.1 pp. `0.262` is +1.1 pp, i.e. about **1 SE above chance** --
not a result. The 33M model is *below* the 8-parameter lexical baseline.

The above-chance cells in the sweep are all small-n and not reliable:

* chemistry (`minilm-l6` 0.302, n=344, SE ±2.3 pp) — 2.2 SE above chance, the only larger slice
  that even looks interesting; it is one of ~10 slices tested, so it is not significant after
  correcting for that;
* NEET (0.332, n=187, SE ±3.2 pp) and JEEBench (0.308-0.351, n=94, SE ±4.9 pp) — both inside
  ~2.6 SE of chance, and NEET is all of 187 questions;
* the ablations move by less than 1 SE in either direction, so there is no evidence the model uses
  option content, option order, or letter position.

### The frozen-feature probe says the same thing

`scripts/ranker_probe.py` (2,000 train / 400 dev, linear head on frozen features, 3 encoders ×
2 query modes × 2 poolings) printed:

```
ms-marco-MiniLM-L-12-v2  best dev 0.247 | train CE 1.379 | question/cls [flat]
ms-marco-MiniLM-L-6-v2   best dev 0.270 | train CE 1.365 | question/cls [memorises]
ms-marco-TinyBERT-L-2-v2 best dev 0.252 | train CE 1.351 | full/doc       [memorises]
VERDICT: no generalising signal in any encoder tested.
```

`memorises` means it fits the train questions (CE below `ln 4`) and falls back to chance on dev.
That is the signature of a 22M encoder that has seen the words but cannot do the physics, and
fine-tuning cannot supply knowledge that is not in the weights.

### So it is not a training bug

Four independent checks now agree:

* **Loss logging** was genuinely wrong at first (`Trainer` logs the micro-batch sum divided by
  optimizer steps, so `grad_accum=2` doubled it: a flat 2.773 was `ln 16`, not a 16-way problem).
  It is now the true 4-way CE in `loss`, with `loss_raw` kept for comparison; the run that produced
  the table above logged ~1.39 (= `ln 4`) and moved to 1.38 at its best.
* **Truncation** did delete option text on 6.6% of questions, so `encode_pair()` now reserves space
  for the option. Fixing it changed nothing measurable.
* **Learning rate / head initialisation** was the real bug in the bi-encoder version: a fresh
  scoring head given a 50x LR collapsed. The current models use the *pretrained* MS MARCO head with
  one LR (`2e-5`) and load without a "newly initialised" warning.
* **A frozen linear probe**, an 8-parameter lexical model, and full fine-tunes of 4.4M / 22.7M /
  33.4M encoders all land at 0.25-0.26. Three different capacities, one answer.

### What changed in the harness because of this

* the saved weights are now the **best-dev checkpoint**, not the last step (`load_best_model_at_end`
  with `save_steps == eval_steps`; verified byte-identical to `checkpoint-<best>`);
* dev evaluation covers **all ~370 dev MCQ** (`eval_limit: 500`) instead of the first 256 raw rows;
* the broken `options` ablation is replaced by `solo` (question vs one option, no option set in the
  query) -- see [Ablations](#ablations-why-you-should-not-trust-a-raw-accuracy-number);
* generator eval scores an evenly-spaced 500-question slice, so a 1.5B eval cannot eat the session.

### The generator track is the open question

The rankers are measured negative: **below the threshold for JEE-style MCQs**. The 139M-1.5B
generators can at least write a derivation, so they are the honest next test -- but they are
**not yet measured**, and no prediction is quoted as a result. What the first runs will tell us:

* if `smollm2-135m` sits at 0.25 with a majority-letter `pred_distribution`, it learned the format
  and nothing else, and 139M is too small;
* if accuracy rises with scale (0.25 -> 0.3 -> ...) the bottleneck is capacity/knowledge, which is
  a real finding about the "smallest model that can answer JEE" question;
* a high `parsed` rate with low accuracy means the model solves the format, not the physics.

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
