"""Bounded all-or-error orchestration for finite-candidate scoring."""

import asyncio
import math
from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol, cast
from uuid import uuid4

from .calibration import normalize_logprobs
from .config import Settings
from .models import (
    CalibrationMetadata,
    ChoiceQuestion,
    ChoiceResult,
    NoulQuestion,
    NoulResult,
    Question,
    Result,
    ScoreQuestion,
    ScoreResult,
    SystemOneRequest,
    SystemOneResponse,
)
from .logging import (
    log_request_completed,
    log_request_failed,
    log_request_started,
)
from .prompting import (
    build_question_prompt,
    build_state_prefix,
    question_criteria,
)
from .r9v_client import (
    R9VHTTPError,
    R9VProtocolError,
    R9VTimeoutError,
    R9VUnavailableError,
    ScoreObservation,
)
from .tokenizer import (
    DEFAULT_LABEL_POOL,
    CandidateLabel,
    LabelPoolExhausted,
    allocate_labels,
)


class SupportsR9V(Protocol):
    async def tokenize(self, text: str) -> list[int]: ...

    async def score(
        self, prompt: str, token_ids: Sequence[int]
    ) -> ScoreObservation: ...


class SupportsCalibration(Protocol):
    def temperature_for(self, family: str | None) -> tuple[float, bool]: ...


class BatchEvaluationError(RuntimeError):
    """One question failed, so no partial batch response is valid."""

    def __init__(self, question_name: str, failure_category: str) -> None:
        self.question_name = question_name
        self.failure_category = failure_category
        super().__init__(f"question evaluation failed ({failure_category})")


@dataclass(frozen=True, slots=True)
class _EvaluatedQuestion:
    result: Result
    observation: ScoreObservation


def _failure_category(error: Exception) -> str:
    if isinstance(error, LabelPoolExhausted):
        return "label_boundary_exhausted"
    if isinstance(error, R9VTimeoutError):
        return "upstream_timeout"
    if isinstance(error, R9VUnavailableError):
        return "upstream_unavailable"
    if isinstance(error, R9VProtocolError):
        return "upstream_protocol"
    if isinstance(error, R9VHTTPError):
        return "upstream_http"
    return "evaluation_error"


class SystemOneService:
    """Evaluate finite-candidate questions against one shared state prefix."""

    def __init__(
        self,
        settings: Settings,
        r9v: SupportsR9V,
        calibration: SupportsCalibration,
    ) -> None:
        self._r9v = r9v
        self._calibration = calibration
        self._concurrency = settings.concurrency
        self._semaphore = asyncio.Semaphore(settings.concurrency)

    async def evaluate(
        self, request: SystemOneRequest, *, request_id: str | None = None
    ) -> SystemOneResponse:
        effective_request_id = request_id or str(uuid4())
        question_count = len(request.questions)
        started_at = perf_counter()
        log_request_started(
            request.state,
            question_count=question_count,
            request_id=effective_request_id,
        )
        prefix = build_state_prefix(request.state)
        results: dict[str, Result] = {}
        observations: list[ScoreObservation] = []
        try:
            evaluated = await self._evaluate_all(
                list(request.questions.items()),
                prefix,
                request.include_diagnostics,
            )
            for name, evaluated_question in zip(
                request.questions, evaluated, strict=True
            ):
                results[name] = evaluated_question.result
                observations.append(evaluated_question.observation)
        except BatchEvaluationError as error:
            log_request_failed(
                request.state,
                question_count=question_count,
                request_id=effective_request_id,
                latency_ms=(perf_counter() - started_at) * 1000,
                failure_category=error.failure_category,
            )
            raise

        cached_values = [
            usage.cached_tokens
            for observation in observations
            for usage in observation.request_usages
            if usage.cached_tokens is not None
        ]
        log_request_completed(
            request.state,
            question_count=question_count,
            request_id=effective_request_id,
            latency_ms=(perf_counter() - started_at) * 1000,
            prompt_tokens=sum(item.prompt_tokens for item in observations),
            completion_tokens=sum(
                item.completion_tokens for item in observations
            ),
            cached_tokens=sum(cached_values) if cached_values else None,
        )
        return SystemOneResponse(results=results)

    async def _evaluate_all(
        self,
        questions: list[tuple[str, Question]],
        prefix: str,
        include_diagnostics: bool,
    ) -> list[_EvaluatedQuestion]:
        evaluated: list[_EvaluatedQuestion | None] = [None] * len(questions)
        running: dict[asyncio.Task[_EvaluatedQuestion], int] = {}
        next_index = 0

        def start(index: int) -> None:
            name, question = questions[index]
            task = asyncio.create_task(
                self._evaluate_bounded(
                    name, question, prefix, include_diagnostics
                )
            )
            running[task] = index

        while next_index < min(self._concurrency, len(questions)):
            start(next_index)
            next_index += 1

        try:
            while running:
                done, _ = await asyncio.wait(
                    running, return_when=asyncio.FIRST_COMPLETED
                )
                failures: list[tuple[int, BaseException]] = []
                for task in done:
                    index = running.pop(task)
                    try:
                        evaluated[index] = task.result()
                    except BaseException as error:
                        failures.append((index, error))
                if failures:
                    failures.sort(key=lambda item: item[0])
                    raise failures[0][1]
                while (
                    next_index < len(questions)
                    and len(running) < self._concurrency
                ):
                    start(next_index)
                    next_index += 1
        finally:
            for task in running:
                task.cancel()
            if running:
                await asyncio.gather(*running, return_exceptions=True)

        return [cast(_EvaluatedQuestion, item) for item in evaluated]

    async def _evaluate_bounded(
        self,
        name: str,
        question: Question,
        prefix: str,
        include_diagnostics: bool,
    ) -> _EvaluatedQuestion:
        try:
            async with self._semaphore:
                return await self._evaluate_question(
                    question, prefix, include_diagnostics
                )
        except Exception as error:
            raise BatchEvaluationError(name, _failure_category(error)) from error

    async def _evaluate_question(
        self,
        question: Question,
        prefix: str,
        include_diagnostics: bool,
    ) -> _EvaluatedQuestion:
        prompt, labels = await self._build_labeled_prompt(prefix, question)
        observation = await self._r9v.score(
            prompt, [label.token_id for label in labels]
        )
        criteria = question_criteria(question)
        semantic_logprobs = {
            key: observation.logprobs[label.token_id]
            for (key, _description), label in zip(criteria, labels, strict=True)
        }
        temperature, calibrated = self._calibration.temperature_for(question.family)
        probabilities = normalize_logprobs(semantic_logprobs, temperature)
        probability_kind = (
            "temperature_calibrated" if calibrated else "raw_renormalized"
        )
        calibration = CalibrationMetadata(
            applied=calibrated,
            temperature=temperature,
            family=question.family,
            version=1 if calibrated else None,
        )
        raw_logprobs = semantic_logprobs if include_diagnostics else None
        if isinstance(question, ChoiceQuestion):
            result: Result = ChoiceResult(
                choice=max(probabilities, key=probabilities.__getitem__),
                probabilities=probabilities,
                probability_kind=probability_kind,
                calibration=calibration,
                raw_logprobs=raw_logprobs,
            )
        elif isinstance(question, NoulQuestion):
            result = NoulResult(
                type="noul",
                noul=probabilities["true"],
                probability_kind=probability_kind,
                calibration=calibration,
                raw_logprobs=raw_logprobs,
            )
        elif isinstance(question, ScoreQuestion):
            result = ScoreResult(
                type="score",
                score=math.fsum(
                    index * probability
                    for index, probability in enumerate(probabilities.values())
                ),
                legend={key: description for key, description in criteria},
                probabilities=probabilities,
                probability_kind=probability_kind,
                calibration=calibration,
                raw_logprobs=raw_logprobs,
            )
        else:
            raise TypeError("unsupported question type")
        return _EvaluatedQuestion(
            result=result,
            observation=observation,
        )

    async def _build_labeled_prompt(
        self, prefix: str, question: Question
    ) -> tuple[str, list[CandidateLabel]]:
        candidate_count = len(question_criteria(question))
        label_texts = list(DEFAULT_LABEL_POOL[:candidate_count])
        seen: set[tuple[str, ...]] = set()
        while tuple(label_texts) not in seen:
            seen.add(tuple(label_texts))
            prompt = build_question_prompt(prefix, question, label_texts)
            labels = await allocate_labels(
                self._r9v, prompt, candidate_count
            )
            allocated_texts = [label.text for label in labels]
            if allocated_texts == label_texts:
                return prompt, labels
            label_texts = allocated_texts
        raise ValueError("candidate labels were not stable at the prompt boundary")
