import json
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

import httpx
import pytest
from fastapi import FastAPI

from r9v_systemone.app import app as default_app
from r9v_systemone.app import create_app
from r9v_systemone.config import Settings
from r9v_systemone.logging import log_request_started
from r9v_systemone.models import SystemOneRequest
from r9v_systemone.r9v_client import (
    RequestUsage,
    R9VHTTPError,
    R9VProtocolError,
    R9VTimeoutError,
    R9VUnavailableError,
    ScoreObservation,
)
from r9v_systemone.tokenizer import DEFAULT_LABEL_POOL


VALID_REQUEST = {
    "state": "binary_sensor.office: on",
    "questions": {
        "occupancy": {
            "type": "choice",
            "instructions": "Is the office occupied?",
            "criteria": {"occupied": None, "empty": None},
        }
    },
}


def test_module_exposes_a_default_uvicorn_application():
    assert isinstance(default_app, FastAPI)


def test_openapi_exposes_all_question_types_and_sanitized_422_contract():
    document = create_app(Settings()).openapi()
    operation = document["paths"]["/v1/systemone"]["post"]
    schemas = document["components"]["schemas"]

    request_schema = operation["requestBody"]["content"]["application/json"][
        "schema"
    ]
    assert request_schema == {"$ref": "#/components/schemas/SystemOneRequest"}

    question_schema = schemas["SystemOneRequest"]["properties"]["questions"][
        "additionalProperties"
    ]
    assert question_schema["discriminator"] == {
        "propertyName": "type",
        "mapping": {
            "choice": "#/components/schemas/ChoiceQuestion",
            "noul": "#/components/schemas/NoulQuestion",
            "score": "#/components/schemas/ScoreQuestion",
        },
    }
    assert {item["$ref"] for item in question_schema["oneOf"]} == {
        "#/components/schemas/ChoiceQuestion",
        "#/components/schemas/NoulQuestion",
        "#/components/schemas/ScoreQuestion",
    }

    error_schema = operation["responses"]["422"]["content"][
        "application/json"
    ]["schema"]
    assert error_schema == {
        "$ref": "#/components/schemas/RequestErrorResponse"
    }
    assert schemas["RequestErrorResponse"]["properties"]["detail"][
        "enum"
    ] == ["invalid_request", "label_boundary_exhausted"]


def test_framework_schema_validation_defers_the_settings_candidate_limit():
    payload = {
        "state": "state",
        "questions": {
            "many": {
                "type": "choice",
                "instructions": "Choose one.",
                "criteria": {
                    f"choice-{index}": None for index in range(17)
                },
            }
        },
    }

    request = SystemOneRequest.model_validate(payload)

    assert len(request.questions["many"].criteria) == 17


class FakeR9VClient:
    def __init__(
        self,
        *,
        health_error: Exception | None = None,
        score_error: Exception | None = None,
        boundary_labels_available: bool = True,
        health_results: list[bool | Exception] | None = None,
    ) -> None:
        self.health_error = health_error
        self.score_error = score_error
        self.boundary_labels_available = boundary_labels_available
        self.health_results = list(health_results or [])
        self.health_calls = 0
        self.score_calls = 0
        self.tokenize_calls = 0
        self.closed = False

    async def tokenize(self, text: str) -> list[int]:
        self.tokenize_calls += 1
        if self.boundary_labels_available:
            for index, label in enumerate(DEFAULT_LABEL_POOL, start=101):
                if text.endswith(label):
                    return [10, 20, index]
        return [10, 20]

    async def score(
        self, prompt: str, token_ids: list[int]
    ) -> ScoreObservation:
        self.score_calls += 1
        if self.score_error is not None:
            raise self.score_error
        return ScoreObservation(
            logprobs={
                token_id: -0.25 - index
                for index, token_id in enumerate(token_ids)
            },
            request_usages=tuple(
                RequestUsage(
                    requested_token_id=token_id,
                    prompt_tokens=25,
                    completion_tokens=1,
                    cached_tokens=10,
                )
                for token_id in token_ids
            ),
        )

    async def health(self) -> bool:
        self.health_calls += 1
        if self.health_results:
            result = self.health_results.pop(0)
            if isinstance(result, Exception):
                raise result
            return result
        if self.health_error is not None:
            raise self.health_error
        return True

    async def aclose(self) -> None:
        self.closed = True


@asynccontextmanager
async def app_client(
    fake: FakeR9VClient,
    settings: Settings | None = None,
    **app_options: object,
) -> AsyncIterator[tuple[httpx.AsyncClient, Callable[[], int]]]:
    factory_calls = 0

    def factory(_: Settings) -> FakeR9VClient:
        nonlocal factory_calls
        factory_calls += 1
        return fake

    app_options.setdefault("startup_health_attempts", 1)
    app_options.setdefault("startup_health_retry_seconds", 0.0)
    app = create_app(
        settings or Settings(), client_factory=factory, **app_options
    )
    transport = httpx.ASGITransport(app=app)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            yield client, lambda: factory_calls


@pytest.mark.asyncio
async def test_valid_request_returns_semantic_result_and_hides_raw_logprobs():
    fake = FakeR9VClient()

    async with app_client(fake) as (client, factory_calls):
        response = await client.post("/v1/systemone", json=VALID_REQUEST)
        second = await client.post("/v1/systemone", json=VALID_REQUEST)
        assert factory_calls() == 1
        assert fake.closed is False

    assert response.status_code == 200
    assert response.json() == second.json()
    result = response.json()["results"]["occupancy"]
    assert result["choice"] == "occupied"
    assert result["probability_kind"] == "raw_renormalized"
    assert set(result["probabilities"]) == {"occupied", "empty"}
    assert "raw_logprobs" not in result
    assert fake.closed is True


@pytest.mark.asyncio
async def test_requested_diagnostics_expose_only_semantic_raw_logprobs():
    fake = FakeR9VClient()
    request = {**VALID_REQUEST, "include_diagnostics": True}

    async with app_client(fake) as (client, _):
        response = await client.post("/v1/systemone", json=request)

    assert response.status_code == 200
    assert response.json()["results"]["occupancy"]["raw_logprobs"] == {
        "occupied": -0.25,
        "empty": -1.25,
    }


@pytest.mark.asyncio
async def test_api_returns_compatible_noul_and_score_shapes_without_confidence():
    fake = FakeR9VClient()
    request = {
        "state": "device: ready",
        "questions": {
            "ready": {
                "type": "noul",
                "instructions": "Is the device ready?",
            },
            "quality": {
                "type": "score",
                "instructions": "Rate device quality.",
                "criteria": ["low", {"label": "high"}],
            },
        },
        "include_diagnostics": True,
    }

    async with app_client(fake) as (client, _):
        response = await client.post("/v1/systemone", json=request)

    assert response.status_code == 200
    ready = response.json()["results"]["ready"]
    quality = response.json()["results"]["quality"]
    assert ready["type"] == "noul"
    assert ready["noul"] == pytest.approx(0.2689414214)
    assert ready["raw_logprobs"] == {"false": -0.25, "true": -1.25}
    assert "confidence" not in ready
    assert quality["type"] == "score"
    assert quality["score"] == pytest.approx(0.2689414214)
    assert quality["legend"] == {"0": "low", "1": {"label": "high"}}
    assert quality["probabilities"] == pytest.approx(
        {"0": 0.7310585786, "1": 0.2689414214}
    )
    assert quality["raw_logprobs"] == {"0": -0.25, "1": -1.25}
    assert "confidence" not in quality


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "question",
    [
        {
            "type": "noul",
            "instructions": "Is it ready?",
            "criteria": {"unknown": "not a boolean anchor"},
        },
        {
            "type": "score",
            "instructions": "Rate it.",
            "criteria": ["only one"],
        },
        {
            "type": "score",
            "instructions": "Rate it.",
            "criteria": [str(index) for index in range(11)],
        },
    ],
)
async def test_api_rejects_invalid_scalar_contracts_before_scoring(question):
    fake = FakeR9VClient()
    request = {
        "state": "private scalar state",
        "questions": {"scalar": question},
    }

    async with app_client(fake) as (client, _):
        response = await client.post("/v1/systemone", json=request)

    assert response.status_code == 422
    assert response.json() == {"detail": "invalid_request"}
    assert "private scalar state" not in response.text
    assert fake.score_calls == 0


@pytest.mark.asyncio
async def test_request_validation_forbids_unknown_fields_without_echoing_input():
    fake = FakeR9VClient()
    request = {
        **VALID_REQUEST,
        "state": "private validation state",
        "unknown": True,
    }

    async with app_client(fake) as (client, _):
        response = await client.post("/v1/systemone", json=request)

    assert response.status_code == 422
    assert response.json() == {"detail": "invalid_request"}
    assert "private validation state" not in response.text
    assert fake.score_calls == 0


@pytest.mark.asyncio
async def test_malformed_json_uses_the_same_secret_free_request_error():
    fake = FakeR9VClient()

    async with app_client(fake) as (client, _):
        response = await client.post(
            "/v1/systemone",
            content='{"state":"private malformed state"',
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 422
    assert response.json() == {"detail": "invalid_request"}
    assert "private malformed state" not in response.text
    assert fake.score_calls == 0


@pytest.mark.asyncio
async def test_request_validation_enforces_configured_maximum_choices():
    fake = FakeR9VClient()
    request = {
        **VALID_REQUEST,
        "questions": {
            "occupancy": {
                **VALID_REQUEST["questions"]["occupancy"],
                "criteria": {"yes": None, "no": None, "unknown": None},
            }
        },
    }

    async with app_client(fake, Settings(max_choices=2)) as (client, _):
        response = await client.post("/v1/systemone", json=request)

    assert response.status_code == 422
    assert response.json() == {"detail": "invalid_request"}
    assert fake.score_calls == 0


@pytest.mark.asyncio
async def test_boundary_label_exhaustion_is_a_safe_request_error():
    fake = FakeR9VClient(boundary_labels_available=False)

    async with app_client(fake) as (client, _):
        response = await client.post("/v1/systemone", json=VALID_REQUEST)

    assert response.status_code == 422
    assert response.json() == {"detail": "label_boundary_exhausted"}
    assert fake.score_calls == 0


@pytest.mark.asyncio
async def test_upstream_http_failure_maps_to_503_without_private_detail():
    fake = FakeR9VClient(
        score_error=R9VHTTPError("HTTP 503 private upstream response")
    )

    async with app_client(fake) as (client, _):
        response = await client.post("/v1/systemone", json=VALID_REQUEST)

    assert response.status_code == 503
    assert response.json() == {"detail": "r9v_unavailable"}
    assert "private upstream response" not in response.text


@pytest.mark.asyncio
async def test_upstream_timeout_and_protocol_failures_have_distinct_statuses():
    cases = [
        (R9VTimeoutError("private timeout"), 504, "r9v_timeout"),
        (R9VProtocolError("private response"), 502, "r9v_protocol_error"),
        (R9VUnavailableError("private network"), 503, "r9v_unavailable"),
    ]

    for error, status, detail in cases:
        fake = FakeR9VClient(score_error=error)
        async with app_client(fake) as (client, _):
            response = await client.post("/v1/systemone", json=VALID_REQUEST)
        assert response.status_code == status
        assert response.json() == {"detail": detail}
        assert "private" not in response.text


@pytest.mark.asyncio
async def test_health_checks_existing_r9v_and_reports_ready():
    fake = FakeR9VClient()

    async with app_client(fake) as (client, _):
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "r9v": "ready"}
    assert fake.health_calls == 2
    assert fake.score_calls == 0
    assert fake.tokenize_calls == 0


@pytest.mark.asyncio
async def test_health_reports_degraded_when_r9v_is_unavailable():
    fake = FakeR9VClient(
        health_error=R9VUnavailableError("private network address")
    )

    async with app_client(fake) as (client, _):
        response = await client.get("/health")

    assert response.status_code == 503
    assert response.json() == {"status": "degraded", "r9v": "unavailable"}
    assert "private network address" not in response.text


@pytest.mark.asyncio
async def test_lifespan_checks_upstream_readiness_before_serving():
    fake = FakeR9VClient()

    async with app_client(fake) as (_client, _):
        assert fake.health_calls == 1


@pytest.mark.asyncio
async def test_lifespan_retries_readiness_with_injected_fast_sleep(caplog):
    fake = FakeR9VClient(
        health_results=[
            R9VUnavailableError("private first failure"),
            False,
            True,
        ]
    )
    delays: list[float] = []

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    logger = logging.getLogger("r9v_systemone")
    caplog.set_level(logging.INFO, logger="r9v_systemone")
    logger.addHandler(caplog.handler)
    try:
        async with app_client(
            fake,
            startup_health_attempts=4,
            startup_health_retry_seconds=0.25,
            startup_sleep=record_sleep,
        ) as (_client, _):
            assert fake.health_calls == 3
    finally:
        logger.removeHandler(caplog.handler)

    assert delays == [0.25, 0.25]
    readiness = [
        json.loads(record.message)
        for record in caplog.records
        if '"event":"startup_readiness"' in record.message
    ]
    assert readiness == [
        {"attempts": 3, "event": "startup_readiness", "status": "ready"}
    ]
    assert "private first failure" not in caplog.text


@pytest.mark.asyncio
async def test_lifespan_serves_degraded_after_bounded_readiness_attempts():
    fake = FakeR9VClient(
        health_results=[False, False],
        health_error=R9VUnavailableError("private persistent failure"),
    )
    delays: list[float] = []

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    logger = logging.getLogger("r9v_systemone")
    records: list[logging.LogRecord] = []

    class RecordHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = RecordHandler()
    logger.addHandler(handler)
    try:
        async with app_client(
            fake,
            startup_health_attempts=2,
            startup_health_retry_seconds=0.25,
            startup_sleep=record_sleep,
        ) as (client, _):
            response = await client.get("/health")
    finally:
        logger.removeHandler(handler)

    assert delays == [0.25]
    assert fake.health_calls == 3
    assert response.status_code == 503
    assert response.json() == {"status": "degraded", "r9v": "unavailable"}
    assert "private persistent failure" not in response.text
    readiness = [
        json.loads(record.message)
        for record in records
        if '"event":"startup_readiness"' in record.message
    ]
    assert readiness == [
        {"attempts": 2, "event": "startup_readiness", "status": "degraded"}
    ]
    assert "private persistent failure" not in "\n".join(
        record.message for record in records
    )


@pytest.mark.asyncio
async def test_actual_app_startup_applies_scoped_log_level_without_duplicates(
    capsys,
):
    logger = logging.getLogger("r9v_systemone")
    original_handlers = list(logger.handlers)
    original_level = logger.level
    original_propagate = logger.propagate
    logger.handlers = []
    try:
        warning_app = create_app(
            Settings(log_level="WARNING"),
            client_factory=lambda _settings: FakeR9VClient(),
        )
        async with warning_app.router.lifespan_context(warning_app):
            log_request_started(
                "private warning state",
                question_count=1,
                request_id="hidden-info",
            )

        info_app = create_app(
            Settings(log_level="INFO"),
            client_factory=lambda _settings: FakeR9VClient(),
        )
        async with info_app.router.lifespan_context(info_app):
            log_request_started(
                "private info state",
                question_count=1,
                request_id="visible-info",
            )

        documents = [
            json.loads(line)
            for line in capsys.readouterr().err.splitlines()
            if line.startswith("{")
        ]
        request_events = [
            document for document in documents if "request_id" in document
        ]
        assert [event["request_id"] for event in request_events] == [
            "visible-info"
        ]
        assert logger.level == logging.INFO
        assert logger.propagate is False
        assert "private warning state" not in json.dumps(documents)
        assert "private info state" not in json.dumps(documents)
    finally:
        new_handlers = [
            handler for handler in logger.handlers if handler not in original_handlers
        ]
        logger.handlers = original_handlers
        logger.setLevel(original_level)
        logger.propagate = original_propagate
        for handler in new_handlers:
            handler.close()
