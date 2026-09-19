import math

import pytest
from pydantic import ValidationError

from r9v_systemone.calibration import CalibrationStore
from r9v_systemone.config import Settings
from r9v_systemone.models import (
    CalibrationMetadata,
    ChoiceResult,
    NoulQuestion,
    NoulResult,
    ScoreQuestion,
    ScoreResult,
    SystemOneRequest,
    SystemOneResponse,
)
from r9v_systemone.prompting import build_question_prompt, build_state_prefix
from r9v_systemone.r9v_client import ScoreObservation
from r9v_systemone.service import SystemOneService
from r9v_systemone.tokenizer import DEFAULT_LABEL_POOL


class FakeR9VClient:
    def __init__(self, scores: list[dict[int, float]]) -> None:
        self._scores = scores
        self.score_calls: list[tuple[str, list[int]]] = []

    async def tokenize(self, text: str) -> list[int]:
        for index, label in enumerate(DEFAULT_LABEL_POOL, start=101):
            if text.endswith(label):
                return [10, 20, index]
        return [10, 20]

    async def score(self, prompt: str, token_ids: list[int]) -> ScoreObservation:
        self.score_calls.append((prompt, list(token_ids)))
        return ScoreObservation(
            logprobs=self._scores[len(self.score_calls) - 1],
            request_usages=(),
        )


class FixedCalibrationStore:
    def temperature_for(self, family: str | None) -> tuple[float, bool]:
        return (2.0, True) if family == "condition" else (1.0, False)


def test_noul_uses_fixed_boolean_anchors_and_accepts_structured_descriptions():
    request = SystemOneRequest.model_validate(
        {
            "state": "sensor: on",
            "questions": {
                "present": {
                    "type": "noul",
                    "instructions": "Is the sensor present?",
                    "criteria": {
                        "true": {"meaning": "present", "sources": ["state"]},
                        "false": ["missing", None],
                    },
                }
            },
        }
    )

    question = request.questions["present"]
    assert isinstance(question, NoulQuestion)
    assert question.criteria == {
        "true": {"meaning": "present", "sources": ["state"]},
        "false": ["missing", None],
    }

    prompt = build_question_prompt(
        build_state_prefix("sensor: on"), question, [" A", " B"]
    )
    assert " A = false:" in prompt
    assert " B = true:" in prompt


def test_noul_rejects_non_boolean_criteria_keys():
    with pytest.raises(ValidationError):
        NoulQuestion(
            type="noul",
            instructions="Is it present?",
            criteria={"yes": "present"},
        )


@pytest.mark.parametrize("level_count", [0, 1, 11])
def test_score_requires_two_through_ten_ordered_levels(level_count):
    with pytest.raises(ValidationError):
        ScoreQuestion(
            type="score",
            instructions="Rate the condition.",
            criteria=[None] * level_count,
        )


def test_score_preserves_json_levels_and_renders_them_deterministically():
    question = ScoreQuestion(
        type="score",
        instructions="Rate the state.",
        criteria=[
            "low",
            {"z": 2, "a": [True, None]},
            ["high", {"urgent": False}],
            None,
        ],
    )
    prefix = build_state_prefix("state")

    prompt = build_question_prompt(
        prefix, question, [" A", " B", " C", " D"]
    )

    assert question.criteria == [
        "low",
        {"z": 2, "a": [True, None]},
        ["high", {"urgent": False}],
        None,
    ]
    assert prompt == (
        prefix
        + "Question:\n"
        "Rate the state.\n"
        "Criteria:\n"
        " A = low\n"
        ' B = {"a":[true,null],"z":2}\n'
        ' C = ["high",{"urgent":false}]\n'
        " D\n"
        "Answer:"
    )
    assert all(f" = {index}" not in prompt for index in range(4))


def test_scalar_questions_reject_values_outside_the_json_contract():
    with pytest.raises(ValidationError):
        NoulQuestion(
            type="noul",
            instructions="Is it present?",
            criteria={"true": "  "},
        )
    with pytest.raises(ValidationError):
        ScoreQuestion(
            type="score",
            instructions="Rate it.",
            criteria=["low", 2],
        )


@pytest.mark.parametrize("non_finite", [float("nan"), float("inf"), -float("inf")])
def test_scalar_questions_reject_non_finite_nested_json(non_finite):
    with pytest.raises(ValidationError):
        ScoreQuestion(
            type="score",
            instructions="Rate it.",
            criteria=["low", {"value": non_finite}],
        )


def test_noul_result_is_a_probability_without_jev_confidence():
    result = NoulResult(
        type="noul",
        noul=0.25,
        probability_kind="raw_renormalized",
        calibration=CalibrationMetadata(applied=False, temperature=1.0),
        raw_logprobs={"false": -0.287682072, "true": -1.386294361},
    )

    dumped = result.model_dump(exclude_none=True)
    assert dumped["noul"] == 0.25
    assert set(dumped["raw_logprobs"]) == {"false", "true"}
    assert "confidence" not in dumped


def test_score_result_requires_full_index_distribution_and_expected_value():
    metadata = CalibrationMetadata(applied=False, temperature=1.0)
    result = ScoreResult(
        type="score",
        score=1.3,
        legend={"0": "low", "1": {"label": "neutral"}, "2": ["high"]},
        probabilities={"0": 0.2, "1": 0.3, "2": 0.5},
        probability_kind="raw_renormalized",
        calibration=metadata,
    )

    assert result.score == pytest.approx(1.3)
    assert list(result.legend) == ["0", "1", "2"]
    assert "confidence" not in result.model_dump()

    with pytest.raises(ValidationError, match="expected value"):
        ScoreResult(
            type="score",
            score=1.0,
            legend={"0": "low", "1": "neutral", "2": "high"},
            probabilities={"0": 0.2, "1": 0.3, "2": 0.5},
            probability_kind="raw_renormalized",
            calibration=metadata,
        )
    with pytest.raises(ValidationError, match="contiguous index"):
        ScoreResult(
            type="score",
            score=1.3,
            legend={"0": "low", "2": "high", "3": "extreme"},
            probabilities={"0": 0.2, "2": 0.3, "3": 0.5},
            probability_kind="raw_renormalized",
            calibration=metadata,
        )


def test_response_accepts_mixed_choice_noul_and_score_results():
    metadata = CalibrationMetadata(applied=False, temperature=1.0)
    response = SystemOneResponse(
        results={
            "choice": ChoiceResult(
                choice="yes",
                probabilities={"yes": 0.75, "no": 0.25},
                probability_kind="raw_renormalized",
                calibration=metadata,
            ),
            "noul": NoulResult(
                type="noul",
                noul=0.75,
                probability_kind="raw_renormalized",
                calibration=metadata,
            ),
            "score": ScoreResult(
                type="score",
                score=0.75,
                legend={"0": "low", "1": "high"},
                probabilities={"0": 0.25, "1": 0.75},
                probability_kind="raw_renormalized",
                calibration=metadata,
            ),
        }
    )

    assert isinstance(response.results["choice"], ChoiceResult)
    assert isinstance(response.results["noul"], NoulResult)
    assert isinstance(response.results["score"], ScoreResult)


@pytest.mark.asyncio
async def test_noul_and_score_share_exact_candidate_scoring_and_diagnostics():
    request = SystemOneRequest.model_validate(
        {
            "state": "sensor.condition: stable",
            "questions": {
                "usable": {
                    "type": "noul",
                    "instructions": "Is the sensor usable?",
                    "criteria": {
                        "true": {"condition": "usable"},
                        "false": ["not usable"],
                    },
                    "family": "condition",
                },
                "condition": {
                    "type": "score",
                    "instructions": "Rate sensor condition.",
                    "criteria": [
                        "low",
                        {"label": "neutral"},
                        ["high", None],
                    ],
                    "family": "condition",
                },
            },
            "include_diagnostics": True,
        }
    )
    client = FakeR9VClient(
        [
            {101: math.log(0.75), 102: math.log(0.25)},
            {101: math.log(0.2), 102: math.log(0.3), 103: math.log(0.5)},
        ]
    )
    service = SystemOneService(Settings(), client, FixedCalibrationStore())

    response = await service.evaluate(request)

    noul = response.results["usable"]
    assert isinstance(noul, NoulResult)
    assert noul.noul == pytest.approx(
        math.sqrt(0.25) / (math.sqrt(0.75) + math.sqrt(0.25))
    )
    assert noul.probability_kind == "temperature_calibrated"
    assert noul.calibration.model_dump() == {
        "applied": True,
        "temperature": 2.0,
        "family": "condition",
        "version": 1,
    }
    assert noul.raw_logprobs == {
        "false": pytest.approx(math.log(0.75)),
        "true": pytest.approx(math.log(0.25)),
    }

    score = response.results["condition"]
    assert isinstance(score, ScoreResult)
    expected_weights = [math.sqrt(0.2), math.sqrt(0.3), math.sqrt(0.5)]
    total = sum(expected_weights)
    expected_probabilities = {
        str(index): value / total
        for index, value in enumerate(expected_weights)
    }
    assert score.score == pytest.approx(
        sum(
            index * probability
            for index, probability in enumerate(
                expected_probabilities.values()
            )
        )
    )
    assert score.legend == {
        "0": "low",
        "1": {"label": "neutral"},
        "2": ["high", None],
    }
    assert score.probabilities == pytest.approx(expected_probabilities)
    assert score.probability_kind == "temperature_calibrated"
    assert score.raw_logprobs == {
        "0": pytest.approx(math.log(0.2)),
        "1": pytest.approx(math.log(0.3)),
        "2": pytest.approx(math.log(0.5)),
    }
    assert "confidence" not in score.model_dump()

    noul_prompt, noul_ids = client.score_calls[0]
    score_prompt, score_ids = client.score_calls[1]
    assert noul_ids == [101, 102]
    assert noul_prompt.index(" A = false") < noul_prompt.index(" B = true")
    assert score_ids == [101, 102, 103]
    assert " A = low" in score_prompt
    assert ' B = {"label":"neutral"}' in score_prompt
    assert all(f" = {index}" not in score_prompt for index in range(3))


@pytest.mark.asyncio
async def test_score_keeps_full_distribution_when_expectations_match():
    request = SystemOneRequest.model_validate(
        {
            "state": "state",
            "questions": {
                "spread": {
                    "type": "score",
                    "instructions": "Rate it.",
                    "criteria": ["low", "middle", "high"],
                },
                "centered": {
                    "type": "score",
                    "instructions": "Rate it.",
                    "criteria": ["low", "middle", "high"],
                },
            },
        }
    )
    client = FakeR9VClient(
        [
            {101: math.log(0.5), 102: -1000.0, 103: math.log(0.5)},
            {101: -1000.0, 102: 0.0, 103: -1000.0},
        ]
    )
    service = SystemOneService(Settings(), client, CalibrationStore())

    response = await service.evaluate(request)

    spread = response.results["spread"]
    centered = response.results["centered"]
    assert isinstance(spread, ScoreResult)
    assert isinstance(centered, ScoreResult)
    assert spread.score == pytest.approx(1.0)
    assert centered.score == pytest.approx(1.0)
    assert spread.probabilities != centered.probabilities
