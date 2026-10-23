"""Small shims for transformers version drift (v4 <-> v5)."""

from __future__ import annotations

import inspect
from functools import lru_cache


@lru_cache(maxsize=1)
def training_arg_names() -> set[str]:
    from transformers import TrainingArguments

    return set(inspect.signature(TrainingArguments.__init__).parameters)


def warmup_kwargs(ratio: float, total_steps: int) -> dict:
    """transformers v5 dropped `warmup_ratio`; express it as explicit steps."""
    if "warmup_ratio" in training_arg_names():
        return {"warmup_ratio": ratio}
    return {"warmup_steps": max(1, int(ratio * max(1, total_steps)))}
