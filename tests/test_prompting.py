import pytest

from r9v_systemone.models import ChoiceQuestion
from r9v_systemone.prompting import build_question_prompt, build_state_prefix


QUESTION_A = ChoiceQuestion(
    type="choice",
    instructions="Is the office occupied?",
    criteria={
        "occupied": "Evidence shows occupancy",
        "empty": "Evidence shows no occupancy",
    },
)
QUESTION_B = ChoiceQuestion(
    type="choice",
    instructions="Should the office light be on?",
    criteria={"on": None, "off": None},
)


def test_questions_share_identical_state_prefix_bytes():
    prefix = build_state_prefix("sensor.office: clear\n")

    first = build_question_prompt(prefix, QUESTION_A, [" A", " B"])
    second = build_question_prompt(prefix, QUESTION_B, [" A", " B"])

    assert first.startswith(prefix)
    assert second.startswith(prefix)
    assert first[: len(prefix)].encode("utf-8") == second[: len(prefix)].encode(
        "utf-8"
    )


def test_state_prefix_preserves_state_verbatim_between_explicit_delimiters():
    state = "sensor.office: café\r\n  detail:  unchanged  "

    prefix = build_state_prefix(state)

    assert prefix == (
        "System-One evaluates shared state against explicit choice criteria.\n"
        "Select exactly one supplied label and output that label only.\n"
        "--- BEGIN SHARED STATE ---\n"
        f"{state}\n"
        "--- END SHARED STATE ---\n"
    )


def test_question_prompt_preserves_criteria_order_and_terminal_boundary():
    question = ChoiceQuestion(
        type="choice",
        instructions="Choose in the order provided.",
        criteria={
            "zeta": "Last alphabetically",
            "alpha": None,
            "middle": "Between the others",
        },
    )
    prefix = build_state_prefix("state")

    prompt = build_question_prompt(prefix, question, [" A", " B", " C"])

    assert prompt == (
        prefix
        + "Question:\n"
        "Choose in the order provided.\n"
        "Criteria:\n"
        " A = zeta: Last alphabetically\n"
        " B = alpha\n"
        " C = middle: Between the others\n"
        "Answer:"
    )
    assert prompt.endswith("Answer:")


def test_question_prompt_requires_one_label_per_criterion():
    prefix = build_state_prefix("state")

    with pytest.raises(ValueError, match="2 criteria but 1 labels"):
        build_question_prompt(prefix, QUESTION_A, [" A"])
