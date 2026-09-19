"""Opt-in serialized acceptance evidence against the existing local R9V."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import httpx
import pytest

from r9v_systemone.app import create_app
from r9v_systemone.calibration import CalibrationStore
from r9v_systemone.config import Settings
from r9v_systemone.models import SystemOneRequest, SystemOneResponse
from r9v_systemone.prompting import build_state_prefix
from r9v_systemone.r9v_client import R9VClient, ScoreObservation
from r9v_systemone.service import SystemOneService


LIVE_ENV = "SYSTEMONE_RUN_LIVE_API"
ARTIFACT_ROOT = Path("/home/atheartengineer/r9v-systemone-artifacts/cache")
MODEL_NAME = "qwen3.8-flash-next"
RUN_ID = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
EXPECTED_METRICS = (
    ("vllm:prefix_cache_queries_total", None),
    ("vllm:prefix_cache_hits_total", None),
    ("vllm:prompt_tokens_by_source_total", "local_compute"),
    ("vllm:prompt_tokens_by_source_total", "local_cache_hit"),
    ("vllm:prompt_tokens_cached_total", None),
)

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.getenv(LIVE_ENV) != "1",
        reason=f"set {LIVE_ENV}=1 to contact the existing local R9V",
    ),
]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_json(stem: str, document: object) -> Path:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    destination = ARTIFACT_ROOT / f"{stem}-{RUN_ID}.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)
    return destination


@pytest.fixture(scope="module", autouse=True)
def serialize_live_module() -> AsyncIterator[None]:
    """Serialize this expensive suite even when pytest uses multiple workers."""

    lock_path = ARTIFACT_ROOT / ".live-suite.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


class RecordingR9VClient(R9VClient):
    """Retain per-logical and per-physical evidence without prompt contents."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.condition = "unassigned"
        self.logical_names: tuple[str, ...] = ()
        self.prefix_token_count = 0
        self.logical_records: list[dict[str, object]] = []
        self.physical_records: list[dict[str, object]] = []
        self._logical_index = 0
        self._active_logical_name: str | None = None

    def begin_condition(
        self,
        condition: str,
        logical_names: Sequence[str],
        prefix_token_count: int,
    ) -> None:
        self.condition = condition
        self.logical_names = tuple(logical_names)
        self.prefix_token_count = prefix_token_count
        self.logical_records = []
        self.physical_records = []
        self._logical_index = 0
        self._active_logical_name = None

    async def score(
        self, prompt: str, token_ids: Sequence[int]
    ) -> ScoreObservation:
        logical_name = self.logical_names[self._logical_index]
        self._logical_index += 1
        self._active_logical_name = logical_name
        physical_start = len(self.physical_records)
        started_at = _utc_now()
        started = perf_counter()
        try:
            observation = await super().score(prompt, token_ids)
        finally:
            self._active_logical_name = None
        latency_ms = (perf_counter() - started) * 1000
        prompt_token_counts = [
            usage.prompt_tokens for usage in observation.request_usages
        ]
        suffix_token_counts = [
            count - self.prefix_token_count for count in prompt_token_counts
        ]
        assert all(count >= 0 for count in suffix_token_counts)
        record = {
            "condition": self.condition,
            "logical_question": logical_name,
            "started_at": started_at,
            "completed_at": _utc_now(),
            "latency_ms": latency_ms,
            "prompt_sha256": _sha256(prompt),
            "prompt_character_count": len(prompt),
            "shared_prefix_token_count": self.prefix_token_count,
            "prompt_token_counts": prompt_token_counts,
            "suffix_token_counts": suffix_token_counts,
            "completion_tokens": observation.completion_tokens,
            "physical_request_count": observation.physical_request_count,
            "requested_token_ids": list(token_ids),
            "logprobs": {str(key): value for key, value in observation.logprobs.items()},
            "request_usages": [
                {
                    "requested_token_id": usage.requested_token_id,
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "cached_tokens": usage.cached_tokens,
                }
                for usage in observation.request_usages
            ],
            "physical_record_indexes": list(
                range(physical_start, len(self.physical_records))
            ),
        }
        self.logical_records.append(record)
        return observation

    async def _request(
        self, method: str, path: str, *, json: dict[str, object] | None = None
    ) -> httpx.Response:
        started_at = _utc_now()
        started = perf_counter()
        response = await super()._request(method, path, json=json)
        latency_ms = (perf_counter() - started) * 1000
        if method == "POST" and path == "v1/completions":
            assert json is not None
            prompt = json["prompt"]
            assert isinstance(prompt, str)
            self.physical_records.append(
                {
                    "condition": self.condition,
                    "logical_question": self._active_logical_name,
                    "started_at": started_at,
                    "completed_at": _utc_now(),
                    "latency_ms": latency_ms,
                    "status_code": response.status_code,
                    "request": {
                        key: value
                        for key, value in json.items()
                        if key != "prompt"
                    }
                    | {
                        "prompt_sha256": _sha256(prompt),
                        "prompt_character_count": len(prompt),
                    },
                    "response": response.json(),
                }
            )
        return response


@asynccontextmanager
async def _live_app_client(
    settings: Settings, recording: RecordingR9VClient
) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(settings, client_factory=lambda _: recording)
    transport = httpx.ASGITransport(app=app)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            timeout=settings.timeout_seconds,
        ) as client:
            yield client


def _assert_complete_result(
    response: SystemOneResponse,
    question_name: str,
    expected_choice: str,
    expected_keys: set[str],
) -> None:
    result = response.results[question_name]
    assert result.choice == expected_choice
    assert set(result.probabilities) == expected_keys
    assert all(math.isfinite(value) for value in result.probabilities.values())
    assert all(0.0 <= value <= 1.0 for value in result.probabilities.values())
    assert math.fsum(result.probabilities.values()) == pytest.approx(1.0, abs=1e-6)
    assert result.probability_kind == "raw_renormalized"
    assert result.calibration.model_dump() == {
        "applied": False,
        "temperature": 1.0,
        "family": None,
        "version": None,
    }


def _known_answer_request() -> dict[str, object]:
    return {
        "state": (
            "arithmetic_fact: 10 is greater than 3\n"
            "user_request: turn on the kitchen light\n"
            "binary_sensor.office_occupancy: on\n"
            "sensor.outdoor_rain: unknown\n"
            "device.oven area: kitchen\n"
        ),
        "questions": {
            "numeric_comparison": {
                "type": "choice",
                "instructions": "Which number is numerically larger, 10 or 3?",
                "criteria": {
                    "ten": "10 is larger",
                    "three": "3 is larger",
                },
            },
            "intent_routing": {
                "type": "choice",
                "instructions": "Route the explicit user request to one intent.",
                "criteria": {
                    "turn_on_light": "turn on a light",
                    "turn_off_light": "turn off a light",
                    "query_status": "ask for current status",
                },
            },
            "binary_occupancy": {
                "type": "choice",
                "instructions": (
                    "Does binary_sensor.office_occupancy indicate occupancy?"
                ),
                "criteria": {
                    "occupied": "the sensor is on",
                    "unoccupied": "the sensor is off",
                },
            },
            "three_way_uncertainty": {
                "type": "choice",
                "instructions": "Is the outdoor rain state known true, known false, or unknown?",
                "criteria": {
                    "known_true": "rain is explicitly true",
                    "known_false": "rain is explicitly false",
                    "uncertain": "the state is unknown",
                },
            },
            "five_way_category": {
                "type": "choice",
                "instructions": "Which area contains device.oven?",
                "criteria": {
                    "kitchen": "kitchen area",
                    "bathroom": "bathroom area",
                    "bedroom": "bedroom area",
                    "office": "office area",
                    "garage": "garage area",
                },
            },
        },
        "include_diagnostics": False,
    }


@pytest.mark.asyncio
async def test_live_known_answers_return_top_choices_and_complete_metadata():
    settings = Settings(timeout_seconds=900.0, concurrency=1)
    recording = RecordingR9VClient(settings)
    request = _known_answer_request()
    prefix_count = len(await recording.tokenize(build_state_prefix(request["state"])))
    questions = request["questions"]
    assert isinstance(questions, dict)
    recording.begin_condition("known_answers", list(questions), prefix_count)
    started_at = _utc_now()
    started = perf_counter()
    async with _live_app_client(settings, recording) as client:
        raw_response = await client.post("/v1/systemone", json=request)
    total_ms = (perf_counter() - started) * 1000

    assert raw_response.status_code == 200, raw_response.text
    response = SystemOneResponse.model_validate(raw_response.json())
    expectations = {
        "numeric_comparison": ("ten", {"ten", "three"}),
        "intent_routing": (
            "turn_on_light",
            {"turn_on_light", "turn_off_light", "query_status"},
        ),
        "binary_occupancy": ("occupied", {"occupied", "unoccupied"}),
        "three_way_uncertainty": (
            "uncertain",
            {"known_true", "known_false", "uncertain"},
        ),
        "five_way_category": (
            "kitchen",
            {"kitchen", "bathroom", "bedroom", "office", "garage"},
        ),
    }
    for name, (expected, keys) in expectations.items():
        _assert_complete_result(response, name, expected, keys)

    expected_physical = sum(len(question["criteria"]) for question in questions.values())
    assert len(recording.logical_records) == len(questions)
    assert len(recording.physical_records) == expected_physical
    _write_json(
        "known-answer-results",
        {
            "run_id": RUN_ID,
            "started_at": started_at,
            "completed_at": _utc_now(),
            "total_latency_ms": total_ms,
            "logical_question_count": len(questions),
            "physical_upstream_request_count": len(recording.physical_records),
            "expected_physical_upstream_request_count": expected_physical,
            "response": raw_response.json(),
            "logical_requests": recording.logical_records,
            "physical_requests": recording.physical_records,
        },
    )


def _structured_state(seed: int, record_count: int, namespace: str) -> str:
    modes = ("idle", "active", "standby", "offline")
    areas = ("atrium", "lab", "library", "studio", "utility")
    records: list[dict[str, object]] = []
    for index in range(record_count):
        base = seed + index * 37
        records.append(
            {
                "entity_id": f"sensor.{namespace}_{index:04d}",
                "state": f"{(base * 17) % 10000 / 100:.2f}",
                "attributes": {
                    "area": areas[(base + index) % len(areas)],
                    "battery": (base * 13) % 101,
                    "mode": modes[(base * 3) % len(modes)],
                    "sequence": index,
                    "unit": "synthetic_units",
                },
                "history": [
                    {
                        "minute": offset * 5,
                        "value": (base + offset * 19) % 997,
                    }
                    for offset in range(4)
                ],
            }
        )
    return json.dumps(
        {
            "schema_version": 1,
            "seed": seed,
            "snapshot_id": f"{namespace}-snapshot",
            "records": records,
        },
        separators=(",", ":"),
        sort_keys=True,
    )


async def _tune_state(
    client: R9VClient, *, seed: int, namespace: str
) -> tuple[str, int, int]:
    low = 1
    high = 160
    while low <= high:
        count = (low + high) // 2
        state = _structured_state(seed, count, namespace)
        token_count = len(await client.tokenize(state))
        if token_count < 4_900:
            low = count + 1
        elif token_count > 5_100:
            high = count - 1
        else:
            return state, token_count, count
    raise AssertionError("deterministic state generator could not reach 4,900-5,100 tokens")


_SAMPLE_RE = re.compile(
    r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)"
    r"(?:\{(?P<labels>[^}]*)\})?\s+"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)$"
)
_LABEL_RE = re.compile(r'(?P<name>[A-Za-z_][A-Za-z0-9_]*)="(?P<value>[^"]*)"')


def _parse_relevant_metrics(raw_metrics: str) -> dict[str, float]:
    selected: dict[str, float] = {}
    expected_names = {name for name, _ in EXPECTED_METRICS}
    for line in raw_metrics.splitlines():
        match = _SAMPLE_RE.match(line)
        if match is None or match.group("name") not in expected_names:
            continue
        labels = {
            label.group("name"): label.group("value")
            for label in _LABEL_RE.finditer(match.group("labels") or "")
        }
        if labels.get("model_name") != MODEL_NAME:
            continue
        source = labels.get("source")
        if match.group("name") == "vllm:prompt_tokens_by_source_total" and source not in {
            "local_compute",
            "local_cache_hit",
        }:
            continue
        key = match.group("name")
        if source is not None:
            key += f'{{source="{source}"}}'
        selected[key] = float(match.group("value"))
    return selected


def _expected_metric_keys() -> set[str]:
    return {
        name if source is None else f'{name}{{source="{source}"}}'
        for name, source in EXPECTED_METRICS
    }


async def _metric_snapshot(
    client: R9VClient, condition: str, phase: str
) -> dict[str, object]:
    raw_metrics = await client.metrics()
    parsed = _parse_relevant_metrics(raw_metrics)
    snapshot = {
        "run_id": RUN_ID,
        "captured_at": _utc_now(),
        "condition": condition,
        "phase": phase,
        "model_name": MODEL_NAME,
        "selected_samples": parsed,
        "missing_expected_samples": sorted(_expected_metric_keys() - parsed.keys()),
        "raw_metrics": raw_metrics,
    }
    _write_json(f"metrics-{condition}-{phase}", snapshot)
    return snapshot


def _metric_delta(
    before: dict[str, object], after: dict[str, object]
) -> dict[str, float | None]:
    before_samples = before["selected_samples"]
    after_samples = after["selected_samples"]
    assert isinstance(before_samples, dict)
    assert isinstance(after_samples, dict)
    return {
        key: (
            float(after_samples[key]) - float(before_samples[key])
            if key in before_samples and key in after_samples
            else None
        )
        for key in sorted(_expected_metric_keys())
    }


def _cache_questions() -> dict[str, dict[str, object]]:
    return {
        f"comparison_{index:02d}": {
            "type": "choice",
            "instructions": (
                f"Which integer is larger, {10_000 + index} or {100 + index}?"
            ),
            "criteria": {
                f"value_{10_000 + index}": "the larger integer",
                f"value_{100 + index}": "the smaller integer",
            },
        }
        for index in range(20)
    }


async def _run_cache_condition(
    client: RecordingR9VClient,
    *,
    condition: str,
    state: str,
    question_batches: Sequence[dict[str, dict[str, object]]],
) -> dict[str, object]:
    prefix_count = len(await client.tokenize(build_state_prefix(state)))
    logical_names = [
        f"batch_{batch_index}:{name}"
        for batch_index, questions in enumerate(question_batches, start=1)
        for name in questions
    ]
    expected_physical = sum(
        len(question["criteria"])
        for questions in question_batches
        for question in questions.values()
    )
    client.begin_condition(condition, logical_names, prefix_count)
    service = SystemOneService(Settings(timeout_seconds=900.0, concurrency=1), client, CalibrationStore())
    before = await _metric_snapshot(client, condition, "before")
    started_at = _utc_now()
    started = perf_counter()
    responses = []
    for batch_index, questions in enumerate(question_batches, start=1):
        request = SystemOneRequest.model_validate(
            {"state": state, "questions": questions, "include_diagnostics": True}
        )
        response = await service.evaluate(
            request, request_id=f"cache-{condition}-{batch_index}"
        )
        responses.append(response.model_dump(mode="json", exclude_none=True))
    total_ms = (perf_counter() - started) * 1000
    after = await _metric_snapshot(client, condition, "after")

    assert len(client.logical_records) == len(logical_names)
    assert len(client.physical_records) == expected_physical
    result = {
        "run_id": RUN_ID,
        "condition": condition,
        "started_at": started_at,
        "completed_at": _utc_now(),
        "state_sha256": _sha256(state),
        "shared_prefix_token_count": prefix_count,
        "logical_question_count": len(logical_names),
        "physical_upstream_request_count": len(client.physical_records),
        "expected_physical_upstream_request_count": expected_physical,
        "total_latency_ms": total_ms,
        "metric_delta": _metric_delta(before, after),
        "responses": responses,
        "logical_requests": client.logical_records,
        "physical_requests": client.physical_records,
    }
    _write_json(f"condition-{condition}", result)
    return result


@pytest.mark.asyncio
async def test_live_shared_prefix_cache_conditions_retain_raw_evidence():
    settings = Settings(timeout_seconds=900.0, concurrency=1)
    client = RecordingR9VClient(settings)
    try:
        state_a, state_a_tokens, state_a_records = await _tune_state(
            client, seed=117, namespace="alpha"
        )
        state_a_changed = state_a.replace(
            '"snapshot_id":"alpha-snapshot"',
            '"snapshot_id":"alpha-changed_"',
            1,
        )
        assert state_a_changed != state_a
        state_a_changed_tokens = len(await client.tokenize(state_a_changed))
        state_b, state_b_tokens, state_b_records = await _tune_state(
            client, seed=911, namespace="omega"
        )
        assert 4_900 <= state_a_tokens <= 5_100
        assert 4_900 <= state_a_changed_tokens <= 5_100
        assert 4_900 <= state_b_tokens <= 5_100

        _write_json(
            "synthetic-states",
            {
                "run_id": RUN_ID,
                "generated_at": _utc_now(),
                "generator": "_structured_state",
                "states": {
                    "state_a": {
                        "seed": 117,
                        "namespace": "alpha",
                        "record_count": state_a_records,
                        "state": state_a,
                        "sha256": _sha256(state_a),
                        "token_count": state_a_tokens,
                    },
                    "state_a_changed": {
                        "derivation": "state_a snapshot_id replacement",
                        "state": state_a_changed,
                        "sha256": _sha256(state_a_changed),
                        "token_count": state_a_changed_tokens,
                    },
                    "state_b": {
                        "seed": 911,
                        "namespace": "omega",
                        "record_count": state_b_records,
                        "state": state_b,
                        "sha256": _sha256(state_b),
                        "token_count": state_b_tokens,
                    },
                },
            },
        )

        twenty_questions = _cache_questions()
        first_question = {"comparison_00": twenty_questions["comparison_00"]}
        repeated_question = {"repeat": twenty_questions["comparison_00"]}
        conditions = []
        conditions.append(
            await _run_cache_condition(
                client,
                condition="cold-state-a",
                state=state_a,
                question_batches=[first_question],
            )
        )
        conditions.append(
            await _run_cache_condition(
                client,
                condition="twenty-shared-state-a",
                state=state_a,
                question_batches=[twenty_questions],
            )
        )
        conditions.append(
            await _run_cache_condition(
                client,
                condition="changed-state-a",
                state=state_a_changed,
                question_batches=[twenty_questions],
            )
        )
        conditions.append(
            await _run_cache_condition(
                client,
                condition="repeated-identical-question",
                state=state_a,
                question_batches=[repeated_question, repeated_question],
            )
        )
        conditions.append(
            await _run_cache_condition(
                client,
                condition="similar-length-unrelated-state-b",
                state=state_b,
                question_batches=[first_question],
            )
        )

        summary = {
            "run_id": RUN_ID,
            "completed_at": _utc_now(),
            "model_name": MODEL_NAME,
            "state_token_counts": {
                "state_a": state_a_tokens,
                "state_a_changed": state_a_changed_tokens,
                "state_b": state_b_tokens,
            },
            "state_sha256": {
                "state_a": _sha256(state_a),
                "state_a_changed": _sha256(state_a_changed),
                "state_b": _sha256(state_b),
            },
            "conditions": [
                {
                    "condition": result["condition"],
                    "logical_question_count": result["logical_question_count"],
                    "physical_upstream_request_count": result[
                        "physical_upstream_request_count"
                    ],
                    "total_latency_ms": result["total_latency_ms"],
                    "metric_delta": result["metric_delta"],
                }
                for result in conditions
            ],
            "logical_question_count": sum(
                int(result["logical_question_count"]) for result in conditions
            ),
            "physical_upstream_request_count": sum(
                int(result["physical_upstream_request_count"])
                for result in conditions
            ),
            "interpretation_policy": (
                "Cache reuse is concluded only from cache counters/token-source "
                "metrics; latency alone is not treated as cache evidence."
            ),
        }
        _write_json("cache-summary", summary)
    finally:
        await client.aclose()
