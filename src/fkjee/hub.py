"""Optional Hugging Face Hub checkpointing.

All runs land in one model repo as a zoo:

    <hub_repo>/
      <run-name>/adapter/...              final LoRA adapter or ranker weights
      <run-name>/checkpoints/step-500/... periodic checkpoints (resume-able)
      <run-name>/run.json                 params, loss, config
      <run-name>/eval-<suite>.json        scores
      sweep.md                            the comparison table

Auth is taken from `HF_TOKEN` (Kaggle secret) or a cached `huggingface-cli login`.
With no token everything is skipped after one warning, so local runs still work.
Set `hub_repo: null` in a config to opt out for a single run.
"""

from __future__ import annotations

import os
from pathlib import Path

_READY: dict[str, bool] = {}
_WARNED = False


def token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def repo_id(cfg: dict) -> str | None:
    return cfg.get("hub_repo") or os.environ.get("FKJEE_HUB_REPO") or None


def enabled(cfg: dict) -> bool:
    return bool(repo_id(cfg))


def ensure_repo(cfg: dict) -> bool:
    """Create the model repo if needed. Returns False (once) when unavailable."""
    global _WARNED
    repo = repo_id(cfg)
    if not repo:
        return False
    if repo in _READY:
        return _READY[repo]

    try:
        from huggingface_hub import HfApi

        HfApi(token=token()).create_repo(
            repo_id=repo,
            repo_type="model",
            private=bool(cfg.get("hub_private", True)),
            exist_ok=True,
        )
        _READY[repo] = True
    except Exception as exc:  # no token, no network, no permission...
        if not _WARNED:
            print(f"[hub] disabled ({type(exc).__name__}: {str(exc)[:140]})")
            _WARNED = True
        _READY[repo] = False
    return _READY[repo]


def upload_dir(cfg: dict, local_dir: str | Path, path_in_repo: str,
               ignore: list[str] | None = None) -> bool:
    if not ensure_repo(cfg):
        return False
    from huggingface_hub import HfApi

    HfApi(token=token()).upload_folder(
        repo_id=repo_id(cfg),
        repo_type="model",
        folder_path=str(local_dir),
        path_in_repo=path_in_repo,
        ignore_patterns=ignore,
        commit_message=f"upload {path_in_repo}",
    )
    print(f"[hub] pushed {path_in_repo} -> {repo_id(cfg)}")
    return True


def upload_files(cfg: dict, files: dict[str, str | Path]) -> int:
    if not ensure_repo(cfg):
        return 0
    from huggingface_hub import HfApi

    api = HfApi(token=token())
    sent = 0
    for path_in_repo, local in files.items():
        if Path(local).exists():
            api.upload_file(
                path_or_fileobj=str(local),
                path_in_repo=path_in_repo,
                repo_id=repo_id(cfg),
                repo_type="model",
                commit_message=f"upload {path_in_repo}",
            )
            sent += 1
    if sent:
        print(f"[hub] pushed {sent} file(s) -> {repo_id(cfg)}")
    return sent


def checkpoint_callback(cfg: dict, run_name: str):
    """TrainerCallback that ships each saved checkpoint to the Hub."""
    from transformers import TrainerCallback

    class _PushCheckpoint(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            step = state.global_step
            ckpt = Path(args.output_dir) / f"checkpoint-{step}"
            if ckpt.exists():
                upload_dir(cfg, ckpt, f"{run_name}/checkpoints/step-{step}")
            return control

    return _PushCheckpoint()
