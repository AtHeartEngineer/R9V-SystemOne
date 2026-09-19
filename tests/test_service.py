import asyncio
import json
import logging

import pytest

from r9v_systemone.calibration import CalibrationStore
from r9v_systemone.config import Settings
from r9v_systemone.models import SystemOneRequest
from r9v_systemone.r9v_client import RequestUsage, R9VTimeoutError, ScoreObservation
from r9v_systemone.service import BatchEvaluationError, SystemOneService
from r9v_systemone.tokenizer import DEFAULT_LABEL_POOL


EXAMPLE_REQUEST = SystemOneRequest.model_validate(
    {
        "state": "binary_sensor.office: on\nlight.office: off\n",
        "questions": {
            "occupancy": {
                "type": "choice",
                "instructions": "Is the office occupied?",
                "criteria": {
                    "occupied": "Evidence shows occupancy",
                    "empty": "Evidence shows no occupancy",
                    "uncertain": None,
                },
                "family": "occupancy",
            },
            "lights": {
                "type": "choice",
                "instructions": "Should the office light be on?",
                "criteria": {"on": None, "off": None},
            },
        },
    }
)


class FakeR9VClient:
    def __init__(
        self,
        scores: list[dict[int, float]],
        *,
        fail_score_call: int | None = None,
    ) -> None:
        self._scores = scores
        self._fail_score_call = fail_score_call
        self.score_calls: list[tuple[str, list[int]]] = []
        self.active_scores = 0
        self.maximum_active_scores = 0

    async def tokenize(self, text: str) -> list[int]:
        for index, label in enumerate(DEFAULT_LABEL_POOL, start=101):
            if text.endswith(label):
                return [10, 20, index]
        return [10, 20]

    async def score(self, prompt: str, token_ids: list[int]) -> ScoreObservation:
        call_number = len(self.score_calls) + 1
        self.score_calls.append((prompt, list(token_ids)))
        self.active_scores += 1
        self.maximum_active_scores = max(
            self.maximum_active_scores, self.active_scores
        )
        try:
            await asyncio.sleep(0)
            if call_number == self._fail_score_call:
                raise R9VTimeoutError("private timeout detail")
            logprobs = self._scores[call_number - 1]
            return ScoreObservation(
                logprobs=logprobs,
                request_usages=tuple(
                    RequestUsage(
                        requested_token_id=token_id,
                        prompt_tokens=40,
                        completion_tokens=1,
                        cached_tokens=20,
                    )
                    for token_id in token_ids
                ),
            )
        finally:
            self.active_scores -= 1


class FixedCalibrationStore:
    def temperature_for(self, family: str | None) -> tuple[float, bool]:
        return (2.0, True) if family == "occupancy" else (1.0, False)


@pytest.mark.asyncio
async def test_choice_results_use_semantic_keys_and_candidate_softmax():
    client = FakeR9VClient(
        [
            {101: -2.0, 102: -0.5, 103: -1.5},
            {101: -0.25, 102: -1.25},
        ]
    )
    service = SystemOneService(Settings(), client, CalibrationStore())

    response = await service.evaluate(EXAMPLE_REQUEST)

    occupancy = response.results["occupancy"]
    assert occupancy.choice == "empty"
    assert list(occupancy.probabilities) == ["occupied", "empty", "uncertain"]
    assert sum(occupancy.probabilities.values()) == pytest.approx(1.0)
    assert occupancy.probability_kind == "raw_renormalized"
    assert occupancy.raw_logprobs is None
    assert list(response.results) == ["occupancy", "lights"]
    first_prompt, first_ids = client.score_calls[0]
    second_prompt, second_ids = client.score_calls[1]
    expected_prefix = (
        "System-One evaluates shared state against explicit choice criteria.\n"
        "Select exactly one supplied label and output that label only.\n"
        "--- BEGIN SHARED STATE ---\n"
        "binary_sensor.office: on\nlight.office: off\n\n"
        "--- END SHARED STATE ---\n"
    )
    assert first_prompt.startswith(expected_prefix)
    assert second_prompt.startswith(expected_prefix)
    assert first_ids == [101, 102, 103]
    assert second_ids == [101, 102]
    assert " A = occupied" in first_prompt
    assert " B = empty" in first_prompt


@pytest.mark.asyncio
async def test_diagnostics_and_calibration_metadata_are_explicit():
    request = EXAMPLE_REQUEST.model_copy(update={"include_diagnostics": True})
    client = FakeR9VClient(
        [
            {101: -2.0, 102: -0.5, 103: -1.5},
            {101: -0.25, 102: -1.25},
        ]
    )
    service = SystemOneService(Settings(), client, FixedCalibrationStore())

    response = await service.evaluate(request)

    occupancy = response.results["occupancy"]
    assert occupancy.probability_kind == "temperature_calibrated"
    assert occupancy.calibration.model_dump() == {
        "applied": True,
        "temperature": 2.0,
        "family": "occupancy",
        "version": 1,
    }
    assert occupancy.raw_logprobs == {
        "occupied": -2.0,
        "empty": -0.5,
        "uncertain": -1.5,
    }
    assert response.results["lights"].probability_kind == "raw_renormalized"
    assert response.results["lights"].calibration.applied is False


@pytest.mark.asyncio
async def test_equal_scores_choose_the_first_criterion_deterministically():
    request = SystemOneRequest.model_validate(
        {
            "state": "state",
            "questions": {
                "tie": {
                    "type": "choice",
                    "instructions": "Choose one.",
                    "criteria": {"first": None, "second": None},
                }
            },
        }
    )
    client = FakeR9VClient([{101: -1.0, 102: -1.0}])
    service = SystemOneService(Settings(), client, CalibrationStore())

    response = await service.evaluate(request)

    assert response.results["tie"].choice == "first"


@pytest.mark.asyncio
async def test_default_concurrency_serializes_question_scoring():
    client = FakeR9VClient(
        [
            {101: -2.0, 102: -0.5, 103: -1.5},
            {101: -0.25, 102: -1.25},
        ]
    )
    service = SystemOneService(Settings(), client, CalibrationStore())

    await service.evaluate(EXAMPLE_REQUEST)

    assert client.maximum_active_scores == 1


@pytest.mark.asyncio
async def test_configured_concurrency_is_a_hard_upper_bound():
    client = FakeR9VClient(
        [
            {101: -2.0, 102: -0.5, 103: -1.5},
            {101: -0.25, 102: -1.25},
        ]
    )
    service = SystemOneService(
        Settings(concurrency=2), client, CalibrationStore()
    )

    await service.evaluate(EXAMPLE_REQUEST)

    assert client.maximum_active_scores == 2


@pytest.mark.asyncio
async def test_upstream_failure_raises_one_typed_batch_error_without_results():
    client = FakeR9VClient(
        [
            {101: -2.0, 102: -0.5, 103: -1.5},
            {101: -0.25, 102: -1.25},
        ],
        fail_score_call=2,
    )
    service = SystemOneService(Settings(), client, CalibrationStore())

    with pytest.raises(BatchEvaluationError) as raised:
        await service.evaluate(EXAMPLE_REQUEST)

    assert raised.value.failure_category == "upstream_timeout"
    assert isinstance(raised.value.__cause__, R9VTimeoutError)
    assert "private timeout detail" not in str(raised.value)


@pytest.mark.asyncio
async def test_first_failure_cancels_queued_questions_without_task_warnings():
    request = SystemOneRequest.model_validate(
        {
            "state": "shared state",
            "questions": {
                "first": {
                    "type": "choice",
                    "instructions": "First question",
                    "criteria": {"yes": None, "no": None},
                },
                "second": {
                    "type": "choice",
                    "instructions": "Second question",
                    "criteria": {"yes": None, "no": None},
                },
                "third": {
                    "type": "choice",
                    "instructions": "Third question",
                    "criteria": {"yes": None, "no": None},
                },
            },
        }
    )
    client = FakeR9VClient(
        [
            {101: -0.25, 102: -1.25},
            {101: -0.25, 102: -1.25},
            {101: -0.25, 102: -1.25},
        ],
        fail_score_call=1,
    )
    service = SystemOneService(Settings(), client, CalibrationStore())
    loop = asyncio.get_running_loop()
    unhandled: list[dict[str, object]] = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: unhandled.append(context))
    try:
        with pytest.raises(BatchEvaluationError) as raised:
            await service.evaluate(request)
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(previous_handler)

    assert raised.value.question_name == "first"
    assert raised.value.failure_category == "upstream_timeout"
    assert len(client.score_calls) == 1
    assert client.active_scores == 0
    assert unhandled == []


@pytest.fixture
def service_caplog(caplog):
    logger = logging.getLogger("r9v_systemone")
    caplog.set_level(logging.INFO, logger="r9v_systemone")
    logger.addHandler(caplog.handler)
    try:
        yield caplog
    finally:
        logger.removeHandler(caplog.handler)


@pytest.mark.asyncio
async def test_service_logs_aggregate_token_counts_without_private_values(
    service_caplog,
):
    caplog = service_caplog
    private_request = EXAMPLE_REQUEST.model_copy(
        update={"state": "private-state-value"}
    )
    client = FakeR9VClient(
        [
            {101: -2.0, 102: -0.5, 103: -1.5},
            {101: -0.25, 102: -1.25},
        ]
    )
    service = SystemOneService(Settings(), client, CalibrationStore())

    await service.evaluate(private_request, request_id="r-complete")

    documents = [json.loads(record.message) for record in caplog.records]
    assert "latency_ms" not in documents[0]
    assert "latency_ms" in documents[1]
    assert documents[1]["prompt_tokens"] == 200
    assert documents[1]["completion_tokens"] == 5
    assert documents[1]["cached_tokens"] == 100
    assert "physical_request_count" not in documents[1]
    assert "private-state-value" not in caplog.text
    assert "Is the office occupied?" not in caplog.text


@pytest.mark.asyncio
async def test_service_failure_log_contains_category_not_exception_detail(
    service_caplog,
):
    caplog = service_caplog
    private_request = EXAMPLE_REQUEST.model_copy(
        update={"state": "private-failure-state"}
    )
    client = FakeR9VClient(
        [
            {101: -2.0, 102: -0.5, 103: -1.5},
            {101: -0.25, 102: -1.25},
        ],
        fail_score_call=2,
    )
    service = SystemOneService(Settings(), client, CalibrationStore())

    with pytest.raises(BatchEvaluationError):
        await service.evaluate(private_request, request_id="r-failed")

    documents = [json.loads(record.message) for record in caplog.records]
    assert documents[-1]["failure_category"] == "upstream_timeout"
    assert "private timeout detail" not in caplog.text
    assert "private-failure-state" not in caplog.text
