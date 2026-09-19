import hashlib
import io
import json
import logging

import pytest

from r9v_systemone.logging import (
    log_request_completed,
    log_request_failed,
    log_request_started,
)


PRIVATE_STATE = (
    "secret state; full prompt; criterion text; api-key-value; "
    "raw upstream response"
)


@pytest.fixture
def log_output():
    logger = logging.getLogger("r9v_systemone")
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    original_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        yield stream
    finally:
        logger.removeHandler(handler)
        logger.setLevel(original_level)
        handler.close()


def event_documents(log_output) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in log_output.getvalue().splitlines()
        if line
    ]


def test_log_event_contains_hash_not_state(log_output):
    log_request_started(PRIVATE_STATE, question_count=2, request_id="r1")

    text = log_output.getvalue()
    assert PRIVATE_STATE not in text
    assert hashlib.sha256(PRIVATE_STATE.encode()).hexdigest()[:16] in text
    assert event_documents(log_output) == [
        {
            "question_count": 2,
            "request_id": "r1",
            "state_sha256_prefix": hashlib.sha256(
                PRIVATE_STATE.encode()
            ).hexdigest()[:16],
        }
    ]


def test_completed_and_failed_logs_use_only_approved_structured_fields(log_output):
    log_request_completed(
        PRIVATE_STATE,
        question_count=3,
        request_id="r2",
        latency_ms=12.5,
        prompt_tokens=120,
        completion_tokens=3,
        cached_tokens=80,
    )
    log_request_failed(
        PRIVATE_STATE,
        question_count=3,
        request_id="r3",
        latency_ms=7.25,
        failure_category="upstream_protocol",
    )

    documents = event_documents(log_output)
    assert set(documents[0]) == {
        "request_id",
        "state_sha256_prefix",
        "question_count",
        "latency_ms",
        "prompt_tokens",
        "completion_tokens",
        "cached_tokens",
    }
    assert set(documents[1]) == {
        "request_id",
        "state_sha256_prefix",
        "question_count",
        "latency_ms",
        "failure_category",
    }
    text = log_output.getvalue()
    assert PRIVATE_STATE not in text
    assert "full prompt" not in text
    assert "criterion text" not in text
    assert "api-key-value" not in text
    assert "raw upstream response" not in text
