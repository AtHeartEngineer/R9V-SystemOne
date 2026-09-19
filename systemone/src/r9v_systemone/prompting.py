"""Deterministic prompt formatting with a cache-stable state prefix."""

import json
from collections.abc import Sequence

from .models import (
    ChoiceQuestion,
    CriterionValue,
    NoulQuestion,
    Question,
    ScoreQuestion,
)


_STATE_HEADER = (
    "System-One evaluates shared state against explicit choice criteria.\n"
    "Select exactly one supplied label and output that label only.\n"
    "--- BEGIN SHARED STATE ---\n"
)
_STATE_FOOTER = "\n--- END SHARED STATE ---\n"


def build_state_prefix(state: str) -> str:
    """Return the fixed instructions and verbatim shared state."""

    return _STATE_HEADER + state + _STATE_FOOTER


def question_criteria(question: Question) -> list[tuple[str, CriterionValue]]:
    """Return the semantic candidate anchors in scoring order."""

    if isinstance(question, ChoiceQuestion):
        return list(question.criteria.items())
    if isinstance(question, NoulQuestion):
        descriptions = question.criteria or {}
        return [
            ("false", descriptions.get("false")),
            ("true", descriptions.get("true")),
        ]
    if isinstance(question, ScoreQuestion):
        return [
            (str(index), description)
            for index, description in enumerate(question.criteria)
        ]
    raise TypeError("unsupported question type")


def _render_description(value: CriterionValue) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def build_question_prompt(
    prefix: str, question: Question, labels: Sequence[str]
) -> str:
    """Append one ordered finite-candidate question to a shared prefix."""

    criteria = question_criteria(question)
    if len(labels) != len(criteria):
        raise ValueError(
            f"question has {len(criteria)} criteria but {len(labels)} labels"
        )

    lines = [prefix, "Question:\n", question.instructions, "\nCriteria:\n"]
    for label, (key, description) in zip(labels, criteria, strict=True):
        if isinstance(question, ScoreQuestion):
            line = label
            if description is not None:
                line += f" = {_render_description(description)}"
        else:
            line = f"{label} = {key}"
            if description is not None:
                line += f": {_render_description(description)}"
        lines.extend((line, "\n"))
    lines.append("Answer:")
    return "".join(lines)
