"""Prompt and target construction.  Pure stdlib."""

from __future__ import annotations

from .data import LETTERS, Item

SYSTEM_MCQ = (
    "You are an expert JEE (Joint Entrance Examination) tutor. "
    "Solve the problem carefully, then state the final answer on its own line as "
    "\\boxed{X} where X is the option letter (A, B, C or D)."
)

SYSTEM_INTEGER = (
    "You are an expert JEE (Joint Entrance Examination) tutor. "
    "Solve the problem carefully, then state the final numerical answer on its own "
    "line as \\boxed{VALUE}."
)


def format_options(options: list[str]) -> str:
    return "\n".join(f"({L}) {text.strip()}" for L, text in zip(LETTERS, options))


def render_user(item: Item, options: list[str] | None = None) -> str:
    opts = item.options if options is None else options
    if item.kind == "mcq":
        return (
            f"{item.question.strip()}\n\n"
            f"Options:\n{format_options(opts)}\n\n"
            "Answer with the letter of exactly one option."
        )
    return (
        f"{item.question.strip()}\n\n"
        "This question has a numerical answer. Give the value in \\boxed{}."
    )


def system_prompt(item: Item) -> str:
    return SYSTEM_MCQ if item.kind == "mcq" else SYSTEM_INTEGER


def build_messages(item: Item, options: list[str] | None = None) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt(item)},
        {"role": "user", "content": render_user(item, options)},
    ]


def target_text(item: Item, use_rationale: bool = True) -> str:
    """Supervision target.

    When the source has a usable explanation we train `reasoning -> boxed answer`.
    When it does not, we train the bare answer rather than fabricating a chain of
    thought -- fabricated CoT teaches the model to bluff.
    """
    answer = f"\\boxed{{{item.gold}}}"
    unit = ""
    if item.kind == "integer":
        unit = ""
    if use_rationale and item.rationale:
        return f"{item.rationale.strip()}\n\nTherefore, the answer is {answer}.{unit}"
    if item.kind == "mcq":
        return f"The correct option is {answer}."
    return f"The answer is {answer}."
