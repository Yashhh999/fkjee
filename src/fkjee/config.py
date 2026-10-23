"""YAML config loading with one level of `base:` inheritance."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path, overrides: dict | None = None) -> dict:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = cfg.pop("base", None)
    if base:
        parent = load_config(path.parent / base)
        parent.update(cfg)
        cfg = parent
    cfg.setdefault("name", path.stem)
    if overrides:
        cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg
