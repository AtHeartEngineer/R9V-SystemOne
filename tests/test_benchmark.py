import asyncio
import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import perf_counter

import pytest
from pydantic import ValidationError

from r9v_systemone.benchmark import (
    BenchmarkConfig,
    BenchmarkCounterResetError,
    BenchmarkIdentity,
    BenchmarkReport,
    ContainerSnapshot,
    MemorySnapshot,
    MetricSnapshot,
    load_corpus,
    load_home_workload,
    run_benchmark,
)
from r9v_systemone.r9v_client import (
    GenerationObservation,
    RequestUsage,
    R9VTimeoutError,
    ScoreObservation,
)
from r9v_systemone.tokenizer import DEFAULT_LABEL_POOL


ROOT = Path(__file__).parents[1]
IDENTITY = BenchmarkIdentity(
    r9v_version="9d5026c",
    image_id="sha256:580bbce",
    profile_id="qwen38-flash-next-256k-mtp2",
    model="qwen3.8-flash-next",
)


def _write_corpus(path: Path, *, extra: dict[str, object] | None = None) -> bytes:
    document: dict[str, object] = {
        "version": 1,
        "cases": [
            {
                "id": f"case-{index:02d}",
                "state": "left=10; right=3",
                "question": "Which value is larger?",
                "criteria": {"left": "the left value", "right": "the right value"},
                "expected": "left",
                "family": "numeric_comparison",
            }
            for index in range(30)
        ],
    }
    if extra:
        document.update(extra)
    raw = (json.dumps(document, indent=2) + "\n").encode()
    path.write_bytes(raw)
    return raw


class FakeR9V:
    def __init__(
        self,
        *,
        generations: list[str] | None = None,
        fail_generation_call: int | None = None,
        fail_score_call: int | None = None,
        delay: bool = False,
    ) -> None:
        self.generations = generations or ["left"]
        self.fail_generation_call = fail_generation_call
        self.fail_score_call = fail_score_call
        self.delay = delay
        self.score_calls: list[tuple[str, list[int]]] = []
        self.generation_calls: list[tuple[str, int]] = []
        self.events: list[str] = []
        self.score_candidate_attempts: list[int] = []
        self.active = 0
        self.maximum_active = 0

    async def tokenize(self, text: str) -> list[int]:
        for index, label in enumerate(DEFAULT_LABEL_POOL, start=101):
            if text.endswith(label):
                return [10, 20, index]
        return [10, 20]

    async def score(self, prompt: str, token_ids: list[int]) -> ScoreObservation:
        call = len(self.score_calls) + 1
        self.score_calls.append((prompt, list(token_ids)))
        self.events.append("systemone")
        await self._enter()
        try:
            if call == self.fail_score_call:
                self.score_candidate_attempts.append(min(2, len(token_ids)))
                raise R9VTimeoutError("failed during a private candidate")
            self.score_candidate_attempts.append(len(token_ids))
            usages = tuple(
                RequestUsage(
                    requested_token_id=token_id,
                    prompt_tokens=40 + offset,
                    completion_tokens=1,
                    cached_tokens=20,
                )
                for offset, token_id in enumerate(token_ids)
            )
            return ScoreObservation(
                logprobs={
                    token_id: (-0.1 if offset == 0 else -2.0)
                    for offset, token_id in enumerate(token_ids)
                },
                request_usages=usages,
            )
        finally:
            self.active -= 1

    async def generate(
        self, prompt: str, *, max_tokens: int = 8
    ) -> GenerationObservation:
        call = len(self.generation_calls) + 1
        self.generation_calls.append((prompt, max_tokens))
        self.events.append("generation")
        await self._enter()
        try:
            if call == self.fail_generation_call:
                raise R9VTimeoutError("private timeout")
            text = self.generations[(call - 1) % len(self.generations)]
            return GenerationObservation(
                text=text,
                prompt_tokens=27,
                completion_tokens=2,
                cached_tokens=11,
            )
        finally:
            self.active -= 1

    async def _enter(self) -> None:
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        if self.delay:
            await asyncio.sleep(0.001)
        else:
            await asyncio.sleep(0)


class FakeSnapshots:
    def __init__(self, metrics: list[dict[str, float]] | None = None) -> None:
        self.metrics = iter(
            metrics
            or [
                {"requests_total": 10.0},
                {"requests_total": 12.0},
                {"requests_total": 12.0},
                {"requests_total": 14.0},
            ]
        )
        self.index = 0

    async def metric_snapshot(self) -> MetricSnapshot:
        return MetricSnapshot(
            captured_at=datetime(2026, 9, 19, tzinfo=UTC),
            counters=next(self.metrics),
        )

    async def memory_snapshot(self) -> MemorySnapshot:
        self.index += 1
        return MemorySnapshot(
            captured_at=datetime(2026, 9, 19, tzinfo=UTC),
            host_available_bytes=1_000_000 - self.index,
            gpu_used_bytes={"card0": 100 + self.index},
        )

    async def container_snapshot(self) -> ContainerSnapshot:
        return ContainerSnapshot(
            captured_at=datetime(2026, 9, 19, tzinfo=UTC),
            container_id="container-1",
            status="running",
            restart_count=0,
            oom_killed=False,
        )


class FakeClock:
    def __init__(self) -> None:
        self.current = datetime(2026, 9, 19, tzinfo=UTC)
        self.monotonic_value = 10.0

    def now(self) -> datetime:
        result = self.current
        self.current += timedelta(milliseconds=1)
        return result

    def monotonic(self) -> float:
        result = self.monotonic_value
        self.monotonic_value += 0.025
        return result


def _config(
    corpus_path: Path,
    client: FakeR9V,
    snapshots: FakeSnapshots | None = None,
    *,
    configured_concurrency: int = 1,
    logical_concurrency: int = 1,
    method_order: tuple[str, str] = ("systemone", "generation"),
    cache_condition: str = "fixture_warm",
) -> BenchmarkConfig:
    clock = FakeClock()
    return BenchmarkConfig(
        corpus_path=corpus_path,
        client=client,
        snapshots=snapshots or FakeSnapshots(),
        identity=IDENTITY,
        configured_concurrency=configured_concurrency,
        logical_concurrency=logical_concurrency,
        generation_max_tokens=12,
        method_order=method_order,
        cache_condition=cache_condition,
        now=clock.now,
        monotonic=clock.monotonic,
    )


def test_corpus_hash_uses_exact_bytes_and_validation_is_strict(tmp_path: Path):
    path = tmp_path / "corpus.json"
    raw = _write_corpus(path)

    loaded = load_corpus(path)

    assert loaded.sha256 == hashlib.sha256(raw).hexdigest()
    assert len(loaded.cases) == 30
    assert loaded.cases[0].expected == "left"

    _write_corpus(path, extra={"unexpected": True})
    with pytest.raises(ValidationError, match="unexpected"):
        load_corpus(path)


def test_corpus_rejects_duplicate_ids(tmp_path: Path):
    path = tmp_path / "corpus.json"
    _write_corpus(path)
    document = json.loads(path.read_text())
    document["cases"][1]["id"] = document["cases"][0]["id"]
    path.write_text(json.dumps(document))

    with pytest.raises(ValidationError, match="case ids must be unique"):
        load_corpus(path)


def test_corpus_rejects_answers_outside_criteria(tmp_path: Path):
    path = tmp_path / "corpus.json"
    _write_corpus(path)
    document = json.loads(path.read_text())
    document["cases"][2]["expected"] = "missing"
    path.write_text(json.dumps(document))

    with pytest.raises(ValidationError, match="expected must name one criterion"):
        load_corpus(path)


@pytest.mark.asyncio
async def test_report_separates_methods_and_sums_physical_candidate_usage(
    tmp_path: Path,
):
    path = tmp_path / "corpus.json"
    raw = _write_corpus(path)
    client = FakeR9V()

    report = await run_benchmark(_config(path, client))

    assert set(report.methods) == {"systemone", "generation"}
    assert report.corpus_sha256 == hashlib.sha256(raw).hexdigest()
    scoring = report.methods["systemone"]
    generation = report.methods["generation"]
    assert scoring.case_count == 30
    assert scoring.known_completion_tokens == 60
    assert scoring.known_physical_upstream_request_count == 60
    assert all(item.physical_upstream_request_count == 2 for item in scoring.cases)
    assert generation.known_completion_tokens == 60
    assert generation.known_physical_upstream_request_count == 30
    assert all(item.raw_candidate_logprobs is not None for item in scoring.cases)
    assert all(item.normalized_scores is not None for item in scoring.cases)
    assert all(item.calibration_kind == "not_applicable" for item in generation.cases)
    assert report.status == "pass"
    assert scoring.metric_counters.delta == {"requests_total": 2.0}
    assert generation.metric_counters.delta == {"requests_total": 2.0}
    assert client.generation_calls[0][1] == 12


@pytest.mark.asyncio
async def test_systemone_decode_accounting_uses_each_cases_candidate_count(
    tmp_path: Path,
):
    path = tmp_path / "corpus.json"
    _write_corpus(path)
    document = json.loads(path.read_text())
    document["cases"][0]["criteria"] = {
        "left": "the left value",
        "right": "the right value",
        "equal": "the values are equal",
        "unknown": "the values are unavailable",
    }
    path.write_text(json.dumps(document))

    report = await run_benchmark(_config(path, FakeR9V()))

    scoring = report.methods["systemone"]
    assert scoring.cases[0].physical_upstream_request_count == 4
    assert scoring.cases[0].completion_tokens == 4
    assert scoring.known_physical_upstream_request_count == 62
    assert scoring.known_completion_tokens == 62


@pytest.mark.asyncio
async def test_malformed_generation_is_not_retried_or_repaired(tmp_path: Path):
    path = tmp_path / "corpus.json"
    _write_corpus(path)
    client = FakeR9V(generations=["The answer is left."])

    report = await run_benchmark(_config(path, client))

    generation = report.methods["generation"]
    assert len(client.generation_calls) == 30
    assert generation.parse_failure_count == 30
    assert generation.correct_count == 0
    assert generation.status == "fail"
    assert report.status == "fail"
    assert all(item.actual_choice is None for item in generation.cases)
    assert all(item.parse_failure for item in generation.cases)


@pytest.mark.asyncio
async def test_failed_request_is_visible_in_aggregate_status(tmp_path: Path):
    path = tmp_path / "corpus.json"
    _write_corpus(path)
    client = FakeR9V(fail_generation_call=2)

    report = await run_benchmark(_config(path, client))

    generation = report.methods["generation"]
    failed = generation.cases[1]
    assert generation.request_failure_count == 1
    assert generation.status == "fail"
    assert failed.failure_category == "upstream_timeout"
    assert failed.physical_upstream_request_count is None
    assert failed.prompt_tokens is None
    assert generation.usage_complete is False
    assert "private timeout" not in report.model_dump_json()


@pytest.mark.asyncio
async def test_counter_reset_is_rejected(tmp_path: Path):
    path = tmp_path / "corpus.json"
    _write_corpus(path)
    snapshots = FakeSnapshots(
        [
            {"requests_total": 10.0},
            {"requests_total": 9.0},
            {"requests_total": 9.0},
            {"requests_total": 11.0},
        ]
    )

    with pytest.raises(BenchmarkCounterResetError, match="requests_total"):
        await run_benchmark(_config(path, FakeR9V(), snapshots))


@pytest.mark.asyncio
async def test_configured_and_logical_concurrency_are_independent(tmp_path: Path):
    path = tmp_path / "corpus.json"
    _write_corpus(path)
    client = FakeR9V(delay=True)

    report = await run_benchmark(
        _config(
            path,
            client,
            configured_concurrency=8,
            logical_concurrency=3,
        )
    )

    assert report.configured_concurrency == 8
    assert report.logical_concurrency == 3
    assert client.maximum_active == 3


@pytest.mark.asyncio
async def test_method_wall_latency_and_throughput_do_not_sum_overlapping_cases(
    tmp_path: Path,
):
    path = tmp_path / "corpus.json"
    _write_corpus(path)
    client = FakeR9V(delay=True)
    config = _config(path, client, logical_concurrency=3)
    config = replace(config, monotonic=perf_counter, now=lambda: datetime.now(UTC))

    report = await run_benchmark(config)

    scoring = report.methods["systemone"]
    assert scoring.wall_latency_ms > 0
    assert scoring.sum_case_latency_ms > scoring.wall_latency_ms * 1.5
    assert scoring.logical_throughput_per_second == pytest.approx(
        30 / (scoring.wall_latency_ms / 1_000)
    )
    assert scoring.physical_throughput_per_second == pytest.approx(
        60 / (scoring.wall_latency_ms / 1_000)
    )
    assert scoring.started_at <= scoring.completed_at


@pytest.mark.asyncio
async def test_methods_share_ordered_candidate_and_canonical_input_identity(
    tmp_path: Path,
):
    path = tmp_path / "corpus.json"
    _write_corpus(path)

    report = await run_benchmark(_config(path, FakeR9V()))

    scoring = report.methods["systemone"].cases
    generation = report.methods["generation"].cases
    scoring_identity = [
        (case.case_id, case.input_sha256, case.expected_choice, case.candidate_keys)
        for case in scoring
    ]
    generation_identity = [
        (case.case_id, case.input_sha256, case.expected_choice, case.candidate_keys)
        for case in generation
    ]
    assert scoring_identity == generation_identity
    assert scoring[0].candidate_keys == ("left", "right")
    assert scoring[0].input_sha256 == (
        "2eac1202c95ab36e913ca68c5a612db5b614d33523405d669f3ad8b25aeb0773"
    )


@pytest.mark.asyncio
async def test_candidate_two_failure_marks_usage_unknown_not_zero(tmp_path: Path):
    path = tmp_path / "corpus.json"
    _write_corpus(path)
    client = FakeR9V(fail_score_call=2)

    report = await run_benchmark(_config(path, client))

    failed = report.methods["systemone"].cases[1]
    aggregate = report.methods["systemone"]
    assert client.score_candidate_attempts[1] == 2
    assert failed.usage_complete is False
    assert failed.prompt_tokens is None
    assert failed.completion_tokens is None
    assert failed.cached_tokens is None
    assert failed.physical_upstream_request_count is None
    assert aggregate.usage_complete is False
    assert aggregate.known_prompt_tokens == 2_349
    assert aggregate.known_completion_tokens == 58
    assert aggregate.known_physical_upstream_request_count == 58
    assert aggregate.physical_throughput_per_second is None
    assert aggregate.status == "fail"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method_order",
    [
        ("systemone", "generation"),
        ("generation", "systemone"),
    ],
)
async def test_method_order_and_cache_condition_are_configured_and_recorded(
    tmp_path: Path, method_order: tuple[str, str]
):
    path = tmp_path / "corpus.json"
    _write_corpus(path)
    client = FakeR9V()

    report = await run_benchmark(
        _config(
            path,
            client,
            method_order=method_order,
            cache_condition="counterbalanced_fixture",
        )
    )

    assert report.method_order == method_order
    assert report.cache_condition == "counterbalanced_fixture"
    assert client.events[:30] == [method_order[0]] * 30
    assert client.events[30:] == [method_order[1]] * 30


def test_committed_corpora_meet_literal_size_and_shared_home_contract():
    corpus = load_corpus(ROOT / "benchmarks" / "corpus.json")
    home = load_home_workload(ROOT / "benchmarks" / "home_state.json")

    assert 30 <= len(corpus.cases) <= 100
    assert {
        "binary",
        "classification",
        "routing",
        "intent",
        "numeric_comparison",
        "home_assistant",
    } <= {case.family for case in corpus.cases}
    assert 4_900 <= home.measured_state_tokens <= 5_100
    assert 20 <= len(home.questions) <= 50
    assert home.state_sha256 == hashlib.sha256(home.state.encode()).hexdigest()
    question_ids = {question.id for question in home.questions}
    assert {
        "presence",
        "rooms_lights",
        "sleep",
        "media",
        "hvac",
        "doors",
        "weather",
        "anomalies",
    } <= {question.category for question in home.questions}
    assert len(question_ids) == len(home.questions)


@pytest.mark.asyncio
async def test_shared_home_workload_is_executable_by_the_same_runner():
    client = FakeR9V()
    config = _config(ROOT / "benchmarks" / "home_state.json", client)

    report = await run_benchmark(replace(config, workload_kind="home"))

    home = load_home_workload(ROOT / "benchmarks" / "home_state.json")
    expected_physical = sum(len(question.criteria) for question in home.questions)
    assert report.case_count == 24
    assert (
        report.methods["systemone"].known_physical_upstream_request_count
        == expected_physical
    )
    assert all(home.state in prompt for prompt, _ in client.score_calls)


def test_report_schema_rejects_unknown_fields(tmp_path: Path):
    path = tmp_path / "corpus.json"
    _write_corpus(path)
    report = asyncio.run(run_benchmark(_config(path, FakeR9V())))
    document = report.model_dump(mode="json")
    document["unknown"] = True

    with pytest.raises(ValidationError, match="unknown"):
        BenchmarkReport.model_validate(document)


def test_report_schema_rejects_inconsistent_aggregate_totals(tmp_path: Path):
    path = tmp_path / "corpus.json"
    _write_corpus(path)
    report = asyncio.run(run_benchmark(_config(path, FakeR9V())))
    document = json.loads(report.model_dump_json())
    document["methods"]["systemone"][
        "known_physical_upstream_request_count"
    ] = 1

    with pytest.raises(ValidationError, match="physical request total"):
        BenchmarkReport.model_validate_json(json.dumps(document))


def test_report_schema_rejects_one_physical_request_for_multicandidate_scoring(
    tmp_path: Path,
):
    path = tmp_path / "corpus.json"
    _write_corpus(path)
    report = asyncio.run(run_benchmark(_config(path, FakeR9V())))
    document = json.loads(report.model_dump_json())
    scoring = document["methods"]["systemone"]
    scoring["cases"][0]["physical_upstream_request_count"] = 1
    scoring["cases"][0]["completion_tokens"] = 1
    scoring["known_physical_upstream_request_count"] -= 1
    scoring["known_completion_tokens"] -= 1

    with pytest.raises(ValidationError, match="candidate count"):
        BenchmarkReport.model_validate_json(json.dumps(document))


def test_report_schema_rejects_inconsistent_metric_delta(tmp_path: Path):
    path = tmp_path / "corpus.json"
    _write_corpus(path)
    report = asyncio.run(run_benchmark(_config(path, FakeR9V())))
    document = json.loads(report.model_dump_json())
    document["methods"]["generation"]["metric_counters"]["delta"][
        "requests_total"
    ] = 99

    with pytest.raises(ValidationError, match="metric delta"):
        BenchmarkReport.model_validate_json(json.dumps(document))


def test_report_schema_rejects_actual_choice_outside_ordered_candidates(
    tmp_path: Path,
):
    path = tmp_path / "corpus.json"
    _write_corpus(path)
    report = asyncio.run(run_benchmark(_config(path, FakeR9V())))
    document = json.loads(report.model_dump_json())
    document["methods"]["generation"]["cases"][0]["actual_choice"] = "absent"

    with pytest.raises(ValidationError, match="actual choice must name a candidate"):
        BenchmarkReport.model_validate_json(json.dumps(document))


def test_report_schema_rejects_cross_method_case_identity_drift(tmp_path: Path):
    path = tmp_path / "corpus.json"
    _write_corpus(path)
    report = asyncio.run(run_benchmark(_config(path, FakeR9V())))
    document = json.loads(report.model_dump_json())
    document["methods"]["generation"]["cases"][0]["input_sha256"] = "0" * 64

    with pytest.raises(ValidationError, match="identical ordered case identities"):
        BenchmarkReport.model_validate_json(json.dumps(document))


def test_report_schema_rejects_method_order_metadata_drift(tmp_path: Path):
    path = tmp_path / "corpus.json"
    _write_corpus(path)
    report = asyncio.run(run_benchmark(_config(path, FakeR9V())))
    document = json.loads(report.model_dump_json())
    document["method_order"] = ["generation", "systemone"]

    with pytest.raises(ValidationError, match="method_order must match execution"):
        BenchmarkReport.model_validate_json(json.dumps(document))


def test_report_schema_round_trips_when_json_object_keys_are_sorted(tmp_path: Path):
    path = tmp_path / "corpus.json"
    _write_corpus(path)
    report = asyncio.run(run_benchmark(_config(path, FakeR9V())))
    sorted_json = json.dumps(json.loads(report.model_dump_json()), sort_keys=True)

    restored = BenchmarkReport.model_validate_json(sorted_json)

    assert restored.method_order == ("systemone", "generation")
    assert set(restored.methods) == {"systemone", "generation"}
    assert restored.methods["systemone"].cases[0].candidate_keys == (
        "left",
        "right",
    )


@pytest.mark.parametrize(
    "invalid_order",
    [
        ("systemone", "generation", "systemone"),
        ["systemone", "generation"],
    ],
)
def test_config_rejects_nonexact_method_order_before_work(
    tmp_path: Path, invalid_order: object
):
    path = tmp_path / "corpus.json"
    _write_corpus(path)
    client = FakeR9V()
    snapshots = FakeSnapshots()
    config = _config(path, client, snapshots)

    with pytest.raises(ValueError, match="method_order must be a two-item tuple"):
        replace(config, method_order=invalid_order)

    assert client.events == []
    assert snapshots.index == 0


@pytest.mark.parametrize(
    ("target", "field"),
    [("report", "started_at"), ("case", "completed_at")],
)
def test_report_schema_rejects_naive_timestamps(
    tmp_path: Path, target: str, field: str
):
    path = tmp_path / "corpus.json"
    _write_corpus(path)
    report = asyncio.run(run_benchmark(_config(path, FakeR9V())))
    document = json.loads(report.model_dump_json())
    if target == "report":
        document[field] = "2026-09-19T00:00:00"
    else:
        document["methods"]["systemone"]["cases"][0][field] = (
            "2026-09-19T00:00:00"
        )

    with pytest.raises(ValidationError, match="timezone"):
        BenchmarkReport.model_validate_json(json.dumps(document))
