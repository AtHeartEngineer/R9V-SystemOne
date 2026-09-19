import asyncio
import json
import math
import os

import httpx
import pytest
from pydantic import SecretStr

from r9v_systemone.config import Settings
from r9v_systemone.r9v_client import (
    R9VClient,
    R9VHTTPError,
    R9VProtocolError,
    R9VTimeoutError,
    R9VUnavailableError,
)


def run(coroutine):
    return asyncio.run(coroutine)


def completion_response(token_id: int, logprob: float) -> dict[str, object]:
    return {
        "choices": [
            {
                "token_ids": [token_id],
                "logprobs": {"token_logprobs": [logprob]},
                "text": "ignored",
            }
        ],
        "usage": {
            "prompt_tokens": 21,
            "completion_tokens": 1,
            "total_tokens": 22,
            "prompt_tokens_details": None,
        },
    }


async def score_with_transport(handler, token_ids: list[int] | None = None):
    client = R9VClient(Settings(), transport=httpx.MockTransport(handler))
    try:
        return await client.score("Answer:", [31] if token_ids is None else token_ids)
    finally:
        await client.aclose()


def test_score_sends_one_safe_singleton_request_per_candidate_in_order():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        body = json.loads(request.content)
        token_id = body["allowed_token_ids"][0]
        return httpx.Response(200, json=completion_response(token_id, -token_id / 10))

    client = R9VClient(Settings(), transport=httpx.MockTransport(handler))
    observation = run(client.score("Answer:", [31, 32]))
    run(client.aclose())

    bodies = [json.loads(request.content) for request in requests]
    assert bodies == [
        {
            "model": "qwen3.8-flash-next",
            "prompt": "Answer:",
            "max_tokens": 1,
            "logprobs": 0,
            "temperature": 0,
            "allowed_token_ids": [31],
            "return_token_ids": True,
            "stream": False,
        },
        {
            "model": "qwen3.8-flash-next",
            "prompt": "Answer:",
            "max_tokens": 1,
            "logprobs": 0,
            "temperature": 0,
            "allowed_token_ids": [32],
            "return_token_ids": True,
            "stream": False,
        },
    ]
    assert all("logprob_token_ids" not in body for body in bodies)
    assert observation.logprobs == {31: -3.1, 32: -3.2}


def test_score_aggregates_actual_usage_and_retains_each_cached_count():
    responses = {
        31: {
            **completion_response(31, -0.4),
            "usage": {
                "prompt_tokens": 42,
                "completion_tokens": 1,
                "total_tokens": 43,
                "prompt_tokens_details": {"cached_tokens": 30},
            },
        },
        32: {
            **completion_response(32, -2.2),
            "usage": {
                "prompt_tokens": 42,
                "completion_tokens": 1,
                "total_tokens": 43,
                "prompt_tokens_details": None,
            },
        },
    }

    def handler(request: httpx.Request) -> httpx.Response:
        token_id = json.loads(request.content)["allowed_token_ids"][0]
        return httpx.Response(200, json=responses[token_id])

    observation = run(score_with_transport(handler, [31, 32]))

    assert observation.logprobs == {31: -0.4, 32: -2.2}
    assert observation.prompt_tokens == 84
    assert observation.completion_tokens == 2
    assert observation.physical_request_count == 2
    assert [usage.cached_tokens for usage in observation.request_usages] == [30, None]
    assert [usage.requested_token_id for usage in observation.request_usages] == [
        31,
        32,
    ]


@pytest.mark.parametrize(
    ("token_ids", "message"),
    [
        ([], "missing returned token ID"),
        ([31, 31], "duplicate returned token ID"),
        ([99], "unexpected returned token ID"),
    ],
)
def test_score_rejects_missing_duplicate_and_unexpected_returned_ids(
    token_ids, message
):
    response = completion_response(31, -0.4)
    response["choices"][0]["token_ids"] = token_ids

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=response)

    with pytest.raises(R9VProtocolError, match=message):
        run(score_with_transport(handler))


@pytest.mark.parametrize(
    "value", [math.nan, math.inf, -math.inf, 10**400, "-0.4", None]
)
def test_score_rejects_nonfinite_or_non_numeric_logprobs(value):
    response = completion_response(31, -0.4)
    response["choices"][0]["logprobs"]["token_logprobs"] = [value]

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=json.dumps(response, allow_nan=True))

    with pytest.raises(R9VProtocolError, match="finite numeric logprob"):
        run(score_with_transport(handler))


def test_score_rejects_invalid_json():
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json")

    with pytest.raises(R9VProtocolError, match="invalid JSON"):
        run(score_with_transport(handler))


def test_score_rejects_http_error_without_retry():
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503, json={"private": "must not leak"})

    with pytest.raises(R9VHTTPError, match="HTTP 503") as raised:
        run(score_with_transport(handler))

    assert calls == 1
    assert "private" not in str(raised.value)


def test_score_translates_timeout_without_retry():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("timed out with private request", request=request)

    with pytest.raises(R9VTimeoutError, match="timed out") as raised:
        run(score_with_transport(handler))

    assert calls == 1
    assert "private request" not in str(raised.value)


def test_score_translates_connection_failure_without_retry():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("private host details", request=request)

    with pytest.raises(R9VUnavailableError, match="request failed") as raised:
        run(score_with_transport(handler))

    assert calls == 1
    assert "private host" not in str(raised.value)


@pytest.mark.parametrize("token_ids", [[], [31, 31], [-1], [True], ["31"]])
def test_score_rejects_invalid_candidate_id_sequences_before_http(token_ids):
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=completion_response(31, -0.4))

    with pytest.raises(ValueError):
        run(score_with_transport(handler, token_ids))

    assert calls == 0


def test_shared_client_exposes_tokenize_health_and_metrics_with_bearer_auth():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/tokenize":
            return httpx.Response(
                200,
                json={
                    "count": 2,
                    "tokens": [10, 20],
                    "token_strs": ["a", "b"],
                    "max_model_len": 262144,
                },
            )
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {
                            "id": "qwen3.8-flash-next",
                            "object": "model",
                            "created": 1,
                            "owned_by": "vllm",
                        }
                    ],
                },
            )
        if request.url.path == "/metrics":
            return httpx.Response(200, text="vllm:prompt_tokens_cached_total 12\n")
        raise AssertionError(f"unexpected path: {request.url.path}")

    client = R9VClient(
        Settings(api_key=SecretStr("test-secret"), timeout_seconds=7.5),
        transport=httpx.MockTransport(handler),
    )

    assert run(client.tokenize("ab")) == [10, 20]
    assert run(client.health()) is True
    assert run(client.metrics()) == "vllm:prompt_tokens_cached_total 12\n"
    run(client.aclose())

    assert json.loads(requests[0].content) == {
        "model": "qwen3.8-flash-next",
        "prompt": "ab",
    }
    assert all(
        request.headers["Authorization"] == "Bearer test-secret"
        for request in requests
    )
    assert all(request.extensions["timeout"]["read"] == 7.5 for request in requests)


def test_health_requires_successful_liveness_and_configured_model_discovery():
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path == "/health":
            return httpx.Response(200, content=b"")
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {
                            "id": "qwen3.8-flash-next",
                            "object": "model",
                            "created": 1,
                            "owned_by": "vllm",
                        }
                    ],
                },
            )
        raise AssertionError(f"unexpected path: {request.url.path}")

    client = R9VClient(Settings(), transport=httpx.MockTransport(handler))
    try:
        assert run(client.health()) is True
    finally:
        run(client.aclose())

    assert paths == ["/health", "/v1/models"]


def test_health_reports_not_ready_when_configured_model_is_missing():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, content=b"")
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {
                            "id": "another-model",
                            "object": "model",
                            "created": 1,
                            "owned_by": "vllm",
                        }
                    ],
                },
            )
        raise AssertionError(f"unexpected path: {request.url.path}")

    client = R9VClient(Settings(), transport=httpx.MockTransport(handler))
    try:
        assert run(client.health()) is False
    finally:
        run(client.aclose())


@pytest.mark.parametrize(
    "document",
    [
        [],
        {},
        {"object": "list", "data": "not-a-list"},
        {"object": "list", "data": [None]},
        {"object": "list", "data": [{"id": 17}]},
    ],
)
def test_health_rejects_malformed_model_discovery(document):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, content=b"")
        if request.url.path == "/v1/models":
            return httpx.Response(200, json=document)
        raise AssertionError(f"unexpected path: {request.url.path}")

    client = R9VClient(Settings(), transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(R9VProtocolError, match="model discovery"):
            run(client.health())
    finally:
        run(client.aclose())


def test_tokenize_rejects_an_inconsistent_token_count():
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "count": 2,
                "tokens": [10],
                "token_strs": ["a"],
                "max_model_len": 262144,
            },
        )

    client = R9VClient(Settings(), transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(R9VProtocolError, match="token counts do not match"):
            run(client.tokenize("a"))
    finally:
        run(client.aclose())


def test_generate_uses_an_ordinary_completion_and_returns_actual_usage():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [{"text": "occupied"}],
                "usage": {
                    "prompt_tokens": 17,
                    "completion_tokens": 2,
                    "total_tokens": 19,
                    "prompt_tokens_details": {"cached_tokens": 8},
                },
            },
        )

    client = R9VClient(Settings(), transport=httpx.MockTransport(handler))
    observation = run(client.generate("Classify:", max_tokens=6))
    run(client.aclose())

    assert json.loads(requests[0].content) == {
        "model": "qwen3.8-flash-next",
        "prompt": "Classify:",
        "max_tokens": 6,
        "temperature": 0,
        "stream": False,
    }
    assert observation.text == "occupied"
    assert observation.prompt_tokens == 17
    assert observation.completion_tokens == 2
    assert observation.cached_tokens == 8
    assert observation.physical_request_count == 1


@pytest.mark.parametrize("max_tokens", [0, -1, True, 1.5])
def test_generate_rejects_invalid_max_tokens_before_http(max_tokens):
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={})

    client = R9VClient(Settings(), transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(ValueError, match="max_tokens"):
            run(client.generate("prompt", max_tokens=max_tokens))
    finally:
        run(client.aclose())

    assert calls == 0


@pytest.mark.live
@pytest.mark.skipif(
    os.getenv("SYSTEMONE_RUN_LIVE_R9V_CLIENT") != "1",
    reason="set SYSTEMONE_RUN_LIVE_R9V_CLIENT=1 for the verified singleton probe",
)
def test_live_known_safe_ten_versus_three_singleton_scoring():
    prompt = (
        "Choose the correct option.\n\n"
        "Question:\nWhich number is larger?\n\n"
        "A: 10\nB: 3\n\nAnswer:\n"
    )

    async def score_and_close():
        client = R9VClient(Settings())
        try:
            return await client.score(prompt, [32, 33])
        finally:
            await client.aclose()

    observation = run(score_and_close())

    assert list(observation.logprobs) == [32, 33]
    assert observation.logprobs[32] > observation.logprobs[33]
    assert observation.physical_request_count == 2
