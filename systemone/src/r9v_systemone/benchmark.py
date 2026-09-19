"""Versioned, reproducible benchmark corpora and offline-testable runner."""

import asyncio
import hashlib
import json
import math
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .calibration import CalibrationStore, normalize_logprobs
from .models import ChoiceQuestion
from .prompting import build_question_prompt, build_state_prefix
from .r9v_client import (
    GenerationObservation,
    R9VHTTPError,
    R9VProtocolError,
    R9VTimeoutError,
    R9VUnavailableError,
    ScoreObservation,
)
from .tokenizer import DEFAULT_LABEL_POOL, CandidateLabel, allocate_labels


FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
NonnegativeFloat = Annotated[float, Field(ge=0, allow_inf_nan=False)]
MethodName = Literal["systemone", "generation"]
CalibrationKind = Literal[
    "raw_renormalized", "temperature_calibrated", "not_applicable"
]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


def _require_timezone(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must include a timezone")
    return value


class CorpusCase(_StrictModel):
    id: str = Field(min_length=1)
    state: str = Field(min_length=1)
    question: str = Field(min_length=1)
    criteria: dict[str, str | None] = Field(min_length=2, max_length=16)
    expected: str = Field(min_length=1)
    family: str = Field(min_length=1)

    @field_validator("id", "state", "question", "expected", "family")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text fields must not be blank")
        return value

    @field_validator("criteria")
    @classmethod
    def reject_invalid_criteria(
        cls, value: dict[str, str | None]
    ) -> dict[str, str | None]:
        for key, description in value.items():
            if not key.strip():
                raise ValueError("criterion keys must not be blank")
            if description is not None and not description.strip():
                raise ValueError("criterion descriptions must not be blank")
        return value

    @model_validator(mode="after")
    def expected_is_a_criterion(self) -> "CorpusCase":
        if self.expected not in self.criteria:
            raise ValueError("expected must name one criterion")
        return self


class BenchmarkCorpus(_StrictModel):
    version: Literal[1]
    cases: tuple[CorpusCase, ...] = Field(min_length=30, max_length=100)

    @model_validator(mode="after")
    def ids_are_unique(self) -> "BenchmarkCorpus":
        ids = [case.id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("case ids must be unique")
        return self


class HomeQuestion(_StrictModel):
    id: str = Field(min_length=1)
    category: Literal[
        "presence",
        "rooms_lights",
        "sleep",
        "media",
        "hvac",
        "doors",
        "weather",
        "anomalies",
    ]
    question: str = Field(min_length=1)
    criteria: dict[str, str | None] = Field(min_length=2, max_length=16)
    expected: str = Field(min_length=1)
    family: str = Field(min_length=1)

    @model_validator(mode="after")
    def expected_is_a_criterion(self) -> "HomeQuestion":
        if self.expected not in self.criteria:
            raise ValueError("expected must name one criterion")
        return self


class HomeWorkload(_StrictModel):
    version: Literal[1]
    state: str = Field(min_length=1)
    state_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    measured_state_tokens: int = Field(ge=4_900, le=5_100)
    questions: tuple[HomeQuestion, ...] = Field(min_length=20, max_length=50)

    @model_validator(mode="after")
    def validate_content_identity(self) -> "HomeWorkload":
        digest = hashlib.sha256(self.state.encode()).hexdigest()
        if digest != self.state_sha256:
            raise ValueError("state_sha256 does not match state")
        ids = [question.id for question in self.questions]
        if len(ids) != len(set(ids)):
            raise ValueError("home question ids must be unique")
        return self


@dataclass(frozen=True, slots=True)
class LoadedCorpus:
    cases: tuple[CorpusCase, ...]
    sha256: str


class BenchmarkIdentity(_StrictModel):
    r9v_version: str = Field(min_length=1)
    image_id: str = Field(min_length=1)
    profile_id: str = Field(min_length=1)
    model: str = Field(min_length=1)


class MetricSnapshot(_StrictModel):
    captured_at: datetime
    counters: dict[str, NonnegativeFloat]

    @field_validator("captured_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        return _require_timezone(value, "captured_at")


class MemorySnapshot(_StrictModel):
    captured_at: datetime
    host_available_bytes: int = Field(ge=0)
    gpu_used_bytes: dict[str, int]

    @field_validator("captured_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        return _require_timezone(value, "captured_at")

    @field_validator("gpu_used_bytes")
    @classmethod
    def require_nonnegative_gpu_values(cls, value: dict[str, int]) -> dict[str, int]:
        if any(not key or amount < 0 for key, amount in value.items()):
            raise ValueError("GPU memory keys and values must be valid")
        return value


class ContainerSnapshot(_StrictModel):
    captured_at: datetime
    container_id: str = Field(min_length=1)
    status: str = Field(min_length=1)
    restart_count: int = Field(ge=0)
    oom_killed: bool

    @field_validator("captured_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        return _require_timezone(value, "captured_at")


class MetricCounterDelta(_StrictModel):
    before: dict[str, NonnegativeFloat]
    after: dict[str, NonnegativeFloat]
    delta: dict[str, NonnegativeFloat]

    @model_validator(mode="after")
    def validate_delta(self) -> "MetricCounterDelta":
        if not (
            self.before.keys() == self.after.keys() == self.delta.keys()
        ):
            raise ValueError("metric delta keys must match both snapshots")
        for name, before_value in self.before.items():
            difference = self.after[name] - before_value
            if difference < 0 or not math.isclose(
                self.delta[name], difference, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError("metric delta must equal after minus before")
        return self


class BenchmarkCaseResult(_StrictModel):
    case_id: str = Field(min_length=1)
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_keys: tuple[str, ...] = Field(min_length=2, max_length=16)
    method: MethodName
    started_at: datetime
    completed_at: datetime
    expected_choice: str = Field(min_length=1)
    actual_choice: str | None
    correct: bool
    parse_failure: bool
    failure_category: str | None
    latency_ms: NonnegativeFloat
    usage_complete: bool
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    cached_tokens: int | None = Field(default=None, ge=0)
    physical_upstream_request_count: int | None = Field(default=None, ge=0)
    raw_candidate_logprobs: dict[str, FiniteFloat] | None
    normalized_scores: dict[str, NonnegativeFloat] | None
    calibration_kind: CalibrationKind

    @field_validator("started_at", "completed_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        return _require_timezone(value, "case timestamp")

    @field_validator("candidate_keys")
    @classmethod
    def require_unique_candidate_keys(
        cls, value: tuple[str, ...]
    ) -> tuple[str, ...]:
        if any(not key.strip() for key in value):
            raise ValueError("candidate keys must not be blank")
        if len(value) != len(set(value)):
            raise ValueError("candidate keys must be unique")
        return value

    @model_validator(mode="after")
    def validate_case_contract(self) -> "BenchmarkCaseResult":
        if self.completed_at < self.started_at:
            raise ValueError("case completion must not precede its start")
        if self.expected_choice not in self.candidate_keys:
            raise ValueError("expected choice must name a candidate")
        if (
            self.actual_choice is not None
            and self.actual_choice not in self.candidate_keys
        ):
            raise ValueError("actual choice must name a candidate")
        usage_values = (
            self.prompt_tokens,
            self.completion_tokens,
            self.physical_upstream_request_count,
        )
        if self.usage_complete and any(value is None for value in usage_values):
            raise ValueError("complete usage requires every required usage value")
        if self.failure_category is not None:
            if self.correct or self.parse_failure or self.actual_choice is not None:
                raise ValueError("failed cases cannot be correct, parsed, or chosen")
            if self.usage_complete:
                raise ValueError("failed cases cannot claim complete usage")
            if (
                self.raw_candidate_logprobs is not None
                or self.normalized_scores is not None
            ):
                raise ValueError("failed cases cannot contain candidate scores")
            return self
        if not self.usage_complete:
            raise ValueError("completed requests must include complete usage")
        if self.parse_failure:
            if (
                self.method != "generation"
                or self.actual_choice is not None
                or self.correct
            ):
                raise ValueError("parse failure must be an unchosen generation result")
        elif self.actual_choice is None:
            raise ValueError("successful cases must have an actual choice")
        elif self.correct != (self.actual_choice == self.expected_choice):
            raise ValueError("correct must match expected and actual choices")

        if self.method == "systemone":
            if self.raw_candidate_logprobs is None or self.normalized_scores is None:
                raise ValueError("successful scoring cases require candidate scores")
            if set(self.raw_candidate_logprobs) != set(self.candidate_keys):
                raise ValueError("raw candidate keys must match ordered candidates")
            if set(self.normalized_scores) != set(self.candidate_keys):
                raise ValueError("normalized candidate keys must match candidates")
            if not math.isclose(
                math.fsum(self.normalized_scores.values()),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise ValueError("normalized candidate scores must sum to one")
            candidate_count = len(self.raw_candidate_logprobs)
            if self.physical_upstream_request_count != candidate_count:
                raise ValueError("physical request count must equal candidate count")
            if self.completion_tokens != candidate_count:
                raise ValueError("scoring decode tokens must equal candidate count")
            if self.calibration_kind == "not_applicable":
                raise ValueError("scoring cases require a scoring calibration kind")
        elif (
            self.raw_candidate_logprobs is not None
            or self.normalized_scores is not None
            or self.calibration_kind != "not_applicable"
        ):
            raise ValueError("generation cases cannot contain candidate scores")
        elif self.physical_upstream_request_count != 1:
            raise ValueError("successful generation must use one physical request")
        return self


class BenchmarkMethodResult(_StrictModel):
    method: MethodName
    started_at: datetime
    completed_at: datetime
    status: Literal["pass", "fail"]
    case_count: int = Field(ge=1)
    correct_count: int = Field(ge=0)
    parse_failure_count: int = Field(ge=0)
    request_failure_count: int = Field(ge=0)
    accuracy: NonnegativeFloat = Field(le=1)
    wall_latency_ms: NonnegativeFloat = Field(gt=0)
    sum_case_latency_ms: NonnegativeFloat
    logical_throughput_per_second: NonnegativeFloat
    physical_throughput_per_second: NonnegativeFloat | None
    usage_complete: bool
    known_prompt_tokens: int = Field(ge=0)
    known_completion_tokens: int = Field(ge=0)
    known_cached_tokens: int = Field(ge=0)
    cached_tokens_complete: bool
    known_physical_upstream_request_count: int = Field(ge=0)
    metric_counters: MetricCounterDelta
    memory_before: MemorySnapshot
    memory_after: MemorySnapshot
    container_before: ContainerSnapshot
    container_after: ContainerSnapshot
    cases: tuple[BenchmarkCaseResult, ...] = Field(min_length=1)

    @field_validator("started_at", "completed_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        return _require_timezone(value, "method timestamp")

    @model_validator(mode="after")
    def aggregate_matches_cases(self) -> "BenchmarkMethodResult":
        if self.case_count != len(self.cases):
            raise ValueError("case count does not match cases")
        if any(case.method != self.method for case in self.cases):
            raise ValueError("case methods must match aggregate method")
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("method case ids must be unique")
        correct_count = sum(case.correct for case in self.cases)
        parse_failure_count = sum(case.parse_failure for case in self.cases)
        request_failure_count = sum(
            case.failure_category is not None for case in self.cases
        )
        if self.correct_count != correct_count:
            raise ValueError("correct count does not match cases")
        if self.parse_failure_count != parse_failure_count:
            raise ValueError("parse failure count does not match cases")
        if self.request_failure_count != request_failure_count:
            raise ValueError("request failure count does not match cases")
        if not math.isclose(
            self.accuracy,
            correct_count / len(self.cases),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("accuracy does not match cases")
        expected_status = "pass" if correct_count == len(self.cases) else "fail"
        if self.status != expected_status:
            raise ValueError("method status does not match cases")
        known_prompt_tokens = sum(
            value
            for case in self.cases
            if (value := case.prompt_tokens) is not None
        )
        if self.known_prompt_tokens != known_prompt_tokens:
            raise ValueError("prompt token total does not match cases")
        known_completion_tokens = sum(
            value
            for case in self.cases
            if (value := case.completion_tokens) is not None
        )
        if self.known_completion_tokens != known_completion_tokens:
            raise ValueError("completion token total does not match cases")
        physical_total = sum(
            value
            for case in self.cases
            if (value := case.physical_upstream_request_count) is not None
        )
        if self.known_physical_upstream_request_count != physical_total:
            raise ValueError("physical request total does not match cases")
        expected_usage_complete = all(case.usage_complete for case in self.cases)
        if self.usage_complete != expected_usage_complete:
            raise ValueError("usage completeness does not match cases")
        if not math.isclose(
            self.sum_case_latency_ms,
            math.fsum(case.latency_ms for case in self.cases),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("latency total does not match cases")
        cached_values = [
            case.cached_tokens
            for case in self.cases
            if case.cached_tokens is not None
        ]
        if self.known_cached_tokens != sum(cached_values):
            raise ValueError("cached token total does not match cases")
        if self.cached_tokens_complete != (
            len(cached_values) == len(self.cases)
        ):
            raise ValueError("cached token completeness does not match cases")
        if self.completed_at < self.started_at:
            raise ValueError("method completion must not precede its start")
        seconds = self.wall_latency_ms / 1_000
        if not math.isclose(
            self.logical_throughput_per_second,
            len(self.cases) / seconds,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("logical throughput does not match wall latency")
        if self.usage_complete:
            if self.physical_throughput_per_second is None or not math.isclose(
                self.physical_throughput_per_second,
                physical_total / seconds,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError("physical throughput does not match wall latency")
        elif self.physical_throughput_per_second is not None:
            raise ValueError("physical throughput must be null for incomplete usage")
        return self


class BenchmarkReport(_StrictModel):
    schema_version: Literal[1]
    started_at: datetime
    completed_at: datetime
    status: Literal["pass", "fail"]
    identity: BenchmarkIdentity
    corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    case_count: int = Field(ge=1)
    configured_concurrency: int = Field(ge=1)
    logical_concurrency: int = Field(ge=1)
    cache_condition: str = Field(min_length=1)
    method_order: tuple[MethodName, MethodName]
    request_parameters: dict[str, int]
    methods: dict[MethodName, BenchmarkMethodResult]

    @field_validator("started_at", "completed_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        return _require_timezone(value, "report timestamp")

    @field_validator("cache_condition")
    @classmethod
    def reject_blank_cache_condition(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("cache_condition must not be blank")
        return value

    @field_validator("method_order")
    @classmethod
    def require_each_method_once(
        cls, value: tuple[MethodName, MethodName]
    ) -> tuple[MethodName, MethodName]:
        if set(value) != {"systemone", "generation"}:
            raise ValueError("method_order must contain each method once")
        return value

    @model_validator(mode="after")
    def require_both_methods(self) -> "BenchmarkReport":
        if set(self.methods) != {"systemone", "generation"}:
            raise ValueError("methods must contain systemone and generation")
        if any(name != result.method for name, result in self.methods.items()):
            raise ValueError("method keys must match aggregate methods")
        first = self.methods[self.method_order[0]]
        second = self.methods[self.method_order[1]]
        if first.completed_at > second.started_at:
            raise ValueError("method_order must match execution timestamps")
        if any(
            result.case_count != self.case_count
            for result in self.methods.values()
        ):
            raise ValueError("report case count must match both methods")
        expected_status = (
            "pass"
            if all(result.status == "pass" for result in self.methods.values())
            else "fail"
        )
        if self.status != expected_status:
            raise ValueError("report status must match method statuses")
        if self.completed_at < self.started_at:
            raise ValueError("report completion must not precede its start")
        scoring_cases = self.methods["systemone"].cases
        generation_cases = self.methods["generation"].cases
        scoring_identity = tuple(
            (
                case.case_id,
                case.input_sha256,
                case.expected_choice,
                case.candidate_keys,
            )
            for case in scoring_cases
        )
        generation_identity = tuple(
            (
                case.case_id,
                case.input_sha256,
                case.expected_choice,
                case.candidate_keys,
            )
            for case in generation_cases
        )
        if scoring_identity != generation_identity:
            raise ValueError("methods must contain identical ordered case identities")
        return self


class SupportsBenchmarkClient(Protocol):
    async def tokenize(self, text: str) -> list[int]: ...

    async def score(
        self, prompt: str, token_ids: Sequence[int]
    ) -> ScoreObservation: ...

    async def generate(
        self, prompt: str, *, max_tokens: int = 8
    ) -> GenerationObservation: ...


class SupportsSnapshots(Protocol):
    async def metric_snapshot(self) -> MetricSnapshot: ...

    async def memory_snapshot(self) -> MemorySnapshot: ...

    async def container_snapshot(self) -> ContainerSnapshot: ...


class SupportsCalibration(Protocol):
    def temperature_for(self, family: str | None) -> tuple[float, bool]: ...


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    corpus_path: Path
    client: SupportsBenchmarkClient
    snapshots: SupportsSnapshots
    identity: BenchmarkIdentity
    configured_concurrency: int = 1
    logical_concurrency: int = 1
    generation_max_tokens: int = 12
    workload_kind: Literal["corpus", "home"] = "corpus"
    cache_condition: str = "uncontrolled"
    method_order: tuple[MethodName, MethodName] = (
        "systemone",
        "generation",
    )
    calibration: SupportsCalibration = field(default_factory=CalibrationStore)
    now: Callable[[], datetime] = lambda: datetime.now(UTC)
    monotonic: Callable[[], float] = perf_counter

    def __post_init__(self) -> None:
        for name, value in (
            ("configured_concurrency", self.configured_concurrency),
            ("logical_concurrency", self.logical_concurrency),
            ("generation_max_tokens", self.generation_max_tokens),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not self.cache_condition.strip():
            raise ValueError("cache_condition must not be blank")
        if not isinstance(self.method_order, tuple) or len(self.method_order) != 2:
            raise ValueError("method_order must be a two-item tuple")
        if set(self.method_order) != {"systemone", "generation"}:
            raise ValueError("method_order must contain each method once")


class BenchmarkCounterResetError(RuntimeError):
    """A cumulative metric decreased during a benchmark condition."""


class BenchmarkMetricMismatchError(RuntimeError):
    """Metric snapshots cannot be compared without identical counter keys."""


def load_corpus(path: Path) -> LoadedCorpus:
    raw = path.read_bytes()
    document = BenchmarkCorpus.model_validate_json(raw)
    return LoadedCorpus(
        cases=document.cases,
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def load_home_workload(path: Path) -> HomeWorkload:
    return HomeWorkload.model_validate_json(path.read_bytes())


def _load_configured_workload(config: BenchmarkConfig) -> LoadedCorpus:
    if config.workload_kind == "corpus":
        return load_corpus(config.corpus_path)
    raw = config.corpus_path.read_bytes()
    home = HomeWorkload.model_validate_json(raw)
    cases = tuple(
        CorpusCase(
            id=question.id,
            state=home.state,
            question=question.question,
            criteria=question.criteria,
            expected=question.expected,
            family=question.family,
        )
        for question in home.questions
    )
    return LoadedCorpus(cases=cases, sha256=hashlib.sha256(raw).hexdigest())


def _counter_delta(
    before: MetricSnapshot, after: MetricSnapshot
) -> MetricCounterDelta:
    if before.counters.keys() != after.counters.keys():
        raise BenchmarkMetricMismatchError(
            "metric snapshots must contain identical counter keys"
        )
    delta: dict[str, float] = {}
    for name in before.counters:
        difference = after.counters[name] - before.counters[name]
        if difference < 0:
            raise BenchmarkCounterResetError(
                f"metric counter reset during benchmark: {name}"
            )
        delta[name] = difference
    return MetricCounterDelta(
        before=before.counters,
        after=after.counters,
        delta=delta,
    )


def _failure_category(error: Exception) -> str:
    if isinstance(error, R9VTimeoutError):
        return "upstream_timeout"
    if isinstance(error, R9VUnavailableError):
        return "upstream_unavailable"
    if isinstance(error, R9VProtocolError):
        return "upstream_protocol"
    if isinstance(error, R9VHTTPError):
        return "upstream_http"
    return "benchmark_error"


def _input_sha256(case: CorpusCase, model: str) -> str:
    canonical = json.dumps(
        {
            "candidate_keys": list(case.criteria),
            "criteria": [list(item) for item in case.criteria.items()],
            "model": model,
            "question": case.question,
            "state": case.state,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


async def _stable_labeled_prompt(
    client: SupportsBenchmarkClient, case: CorpusCase
) -> tuple[str, list[CandidateLabel]]:
    question = ChoiceQuestion(
        type="choice",
        instructions=case.question,
        criteria=case.criteria,
        family=case.family,
    )
    prefix = build_state_prefix(case.state)
    label_texts = list(DEFAULT_LABEL_POOL[: len(case.criteria)])
    seen: set[tuple[str, ...]] = set()
    while tuple(label_texts) not in seen:
        seen.add(tuple(label_texts))
        prompt = build_question_prompt(prefix, question, label_texts)
        labels = await allocate_labels(client, prompt, len(case.criteria))
        allocated = [label.text for label in labels]
        if allocated == label_texts:
            return prompt, labels
        label_texts = allocated
    raise ValueError("candidate labels were not stable at the prompt boundary")


def _generation_prompt(case: CorpusCase) -> str:
    lines = [
        build_state_prefix(case.state),
        "Question:\n",
        case.question,
        "\nCriteria:\n",
    ]
    for key, description in case.criteria.items():
        line = key if description is None else f"{key}: {description}"
        lines.extend((line, "\n"))
    lines.extend(
        (
            "Output exactly one semantic key from: ",
            ", ".join(case.criteria),
            ".\nAnswer:",
        )
    )
    return "".join(lines)


async def _run_scoring_case(
    config: BenchmarkConfig, case: CorpusCase
) -> BenchmarkCaseResult:
    started_at = config.now()
    started = config.monotonic()
    try:
        prompt, labels = await _stable_labeled_prompt(config.client, case)
        observation = await config.client.score(
            prompt, [label.token_id for label in labels]
        )
        raw = {
            key: observation.logprobs[label.token_id]
            for key, label in zip(case.criteria, labels, strict=True)
        }
        temperature, calibrated = config.calibration.temperature_for(case.family)
        normalized = normalize_logprobs(raw, temperature)
        actual = max(normalized, key=normalized.__getitem__)
        cached_values = [
            usage.cached_tokens
            for usage in observation.request_usages
            if usage.cached_tokens is not None
        ]
        cached = (
            sum(cached_values)
            if len(cached_values) == len(observation.request_usages)
            else None
        )
        return BenchmarkCaseResult(
            case_id=case.id,
            input_sha256=_input_sha256(case, config.identity.model),
            candidate_keys=tuple(case.criteria),
            method="systemone",
            started_at=started_at,
            completed_at=config.now(),
            expected_choice=case.expected,
            actual_choice=actual,
            correct=actual == case.expected,
            parse_failure=False,
            failure_category=None,
            latency_ms=(config.monotonic() - started) * 1_000,
            usage_complete=True,
            prompt_tokens=observation.prompt_tokens,
            completion_tokens=observation.completion_tokens,
            cached_tokens=cached,
            physical_upstream_request_count=observation.physical_request_count,
            raw_candidate_logprobs=raw,
            normalized_scores=normalized,
            calibration_kind=(
                "temperature_calibrated" if calibrated else "raw_renormalized"
            ),
        )
    except Exception as error:
        return _failed_case(config, case, "systemone", started_at, started, error)


async def _run_generation_case(
    config: BenchmarkConfig, case: CorpusCase
) -> BenchmarkCaseResult:
    started_at = config.now()
    started = config.monotonic()
    try:
        observation = await config.client.generate(
            _generation_prompt(case), max_tokens=config.generation_max_tokens
        )
        parsed = observation.text.strip()
        actual = parsed if parsed in case.criteria else None
        return BenchmarkCaseResult(
            case_id=case.id,
            input_sha256=_input_sha256(case, config.identity.model),
            candidate_keys=tuple(case.criteria),
            method="generation",
            started_at=started_at,
            completed_at=config.now(),
            expected_choice=case.expected,
            actual_choice=actual,
            correct=actual == case.expected,
            parse_failure=actual is None,
            failure_category=None,
            latency_ms=(config.monotonic() - started) * 1_000,
            usage_complete=True,
            prompt_tokens=observation.prompt_tokens,
            completion_tokens=observation.completion_tokens,
            cached_tokens=observation.cached_tokens,
            physical_upstream_request_count=observation.physical_request_count,
            raw_candidate_logprobs=None,
            normalized_scores=None,
            calibration_kind="not_applicable",
        )
    except Exception as error:
        return _failed_case(config, case, "generation", started_at, started, error)


def _failed_case(
    config: BenchmarkConfig,
    case: CorpusCase,
    method: MethodName,
    started_at: datetime,
    started: float,
    error: Exception,
) -> BenchmarkCaseResult:
    return BenchmarkCaseResult(
        case_id=case.id,
        input_sha256=_input_sha256(case, config.identity.model),
        candidate_keys=tuple(case.criteria),
        method=method,
        started_at=started_at,
        completed_at=config.now(),
        expected_choice=case.expected,
        actual_choice=None,
        correct=False,
        parse_failure=False,
        failure_category=_failure_category(error),
        latency_ms=(config.monotonic() - started) * 1_000,
        usage_complete=False,
        prompt_tokens=None,
        completion_tokens=None,
        cached_tokens=None,
        physical_upstream_request_count=None,
        raw_candidate_logprobs=None,
        normalized_scores=None,
        calibration_kind=(
            "not_applicable" if method == "generation" else "raw_renormalized"
        ),
    )


async def _bounded_cases(
    cases: tuple[CorpusCase, ...],
    concurrency: int,
    operation: Callable[[CorpusCase], Awaitable[BenchmarkCaseResult]],
) -> tuple[BenchmarkCaseResult, ...]:
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded(case: CorpusCase) -> BenchmarkCaseResult:
        async with semaphore:
            return await operation(case)

    return tuple(await asyncio.gather(*(bounded(case) for case in cases)))


async def _run_method(
    config: BenchmarkConfig,
    cases: tuple[CorpusCase, ...],
    method: MethodName,
) -> BenchmarkMethodResult:
    metrics_before = await config.snapshots.metric_snapshot()
    memory_before = await config.snapshots.memory_snapshot()
    container_before = await config.snapshots.container_snapshot()
    operation = (
        (lambda case: _run_scoring_case(config, case))
        if method == "systemone"
        else (lambda case: _run_generation_case(config, case))
    )
    started_at = config.now()
    wall_started = config.monotonic()
    results = await _bounded_cases(cases, config.logical_concurrency, operation)
    wall_latency_ms = (config.monotonic() - wall_started) * 1_000
    completed_at = config.now()
    metrics_after = await config.snapshots.metric_snapshot()
    memory_after = await config.snapshots.memory_snapshot()
    container_after = await config.snapshots.container_snapshot()
    counters = _counter_delta(metrics_before, metrics_after)

    correct_count = sum(result.correct for result in results)
    request_failure_count = sum(
        result.failure_category is not None for result in results
    )
    parse_failure_count = sum(result.parse_failure for result in results)
    known_prompt_tokens = sum(
        value
        for result in results
        if (value := result.prompt_tokens) is not None
    )
    known_completion_tokens = sum(
        value
        for result in results
        if (value := result.completion_tokens) is not None
    )
    known_cached_tokens = sum(
        value
        for result in results
        if (value := result.cached_tokens) is not None
    )
    known_physical_requests = sum(
        value
        for result in results
        if (value := result.physical_upstream_request_count) is not None
    )
    status = (
        "pass"
        if correct_count == len(results)
        and request_failure_count == 0
        and parse_failure_count == 0
        else "fail"
    )
    return BenchmarkMethodResult(
        method=method,
        started_at=started_at,
        completed_at=completed_at,
        status=status,
        case_count=len(results),
        correct_count=correct_count,
        parse_failure_count=parse_failure_count,
        request_failure_count=request_failure_count,
        accuracy=correct_count / len(results),
        wall_latency_ms=wall_latency_ms,
        sum_case_latency_ms=math.fsum(result.latency_ms for result in results),
        logical_throughput_per_second=(
            len(results) / (wall_latency_ms / 1_000)
        ),
        physical_throughput_per_second=(
            known_physical_requests / (wall_latency_ms / 1_000)
            if all(result.usage_complete for result in results)
            else None
        ),
        usage_complete=all(result.usage_complete for result in results),
        known_prompt_tokens=known_prompt_tokens,
        known_completion_tokens=known_completion_tokens,
        known_cached_tokens=known_cached_tokens,
        cached_tokens_complete=all(
            result.cached_tokens is not None for result in results
        ),
        known_physical_upstream_request_count=known_physical_requests,
        metric_counters=counters,
        memory_before=memory_before,
        memory_after=memory_after,
        container_before=container_before,
        container_after=container_after,
        cases=results,
    )


async def run_benchmark(config: BenchmarkConfig) -> BenchmarkReport:
    """Run scoring and ordinary generation without retries or output repair."""

    corpus = _load_configured_workload(config)
    started_at = config.now()
    methods: dict[MethodName, BenchmarkMethodResult] = {}
    for method in config.method_order:
        methods[method] = await _run_method(config, corpus.cases, method)
    return BenchmarkReport(
        schema_version=1,
        started_at=started_at,
        completed_at=config.now(),
        status=(
            "pass"
            if all(result.status == "pass" for result in methods.values())
            else "fail"
        ),
        identity=config.identity,
        corpus_sha256=corpus.sha256,
        case_count=len(corpus.cases),
        configured_concurrency=config.configured_concurrency,
        logical_concurrency=config.logical_concurrency,
        cache_condition=config.cache_condition,
        method_order=config.method_order,
        request_parameters={
            "systemone_max_tokens_per_candidate": 1,
            "generation_max_tokens": config.generation_max_tokens,
        },
        methods=methods,
    )
