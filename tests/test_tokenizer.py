import asyncio
import json
import os
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from r9v_systemone.tokenizer import (
    DEFAULT_LABEL_POOL,
    LabelPoolExhausted,
    TokenizerClient,
    TokenizerError,
    allocate_labels,
    token_added_at_boundary,
)


class FakeTokenizer:
    """Async fake whose complete outputs model boundary-sensitive tokenization."""

    def __init__(self, tokenizations: dict[str, list[int]]) -> None:
        self._tokenizations = tokenizations

    async def tokenize(self, text: str) -> list[int]:
        return list(self._tokenizations[text])


@contextmanager
def tokenizer_server(response_document: object):
    received: list[tuple[str, dict[str, object]]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            content_length = int(self.headers["Content-Length"])
            request_document = json.loads(self.rfile.read(content_length))
            received.append((self.path, request_document))
            response_body = json.dumps(response_document).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}", received
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def run(coroutine):
    return asyncio.run(coroutine)


def test_label_must_add_exactly_one_token_at_boundary():
    tokenizer = FakeTokenizer(
        {
            "prompt": [10, 20],
            "prompt A": [10, 20, 31],
            "prompt B": [10, 99, 32],
        }
    )

    labels = run(allocate_labels(tokenizer, "prompt", 1, pool=[" A", " B"]))

    assert [(item.text, item.token_id) for item in labels] == [(" A", 31)]


def test_boundary_check_rejects_zero_or_multiple_added_tokens():
    tokenizer = FakeTokenizer(
        {
            "prompt": [10, 20],
            "promptX": [10, 20],
            "promptY": [10, 20, 30, 40],
        }
    )

    assert run(token_added_at_boundary(tokenizer, "prompt", "X")) is None
    assert run(token_added_at_boundary(tokenizer, "prompt", "Y")) is None


def test_allocator_skips_duplicate_token_ids_and_remaps_deterministically():
    tokenizer = FakeTokenizer(
        {
            "prompt": [10],
            "prompt A": [10, 31],
            "prompt B": [10, 31],
            "prompt C": [10, 33],
        }
    )

    labels = run(
        allocate_labels(tokenizer, "prompt", 2, pool=[" A", " B", " C"])
    )

    assert [(item.text, item.token_id) for item in labels] == [
        (" A", 31),
        (" C", 33),
    ]


def test_allocator_reports_requested_and_available_counts():
    tokenizer = FakeTokenizer(
        {
            "prompt": [10],
            "prompt A": [10, 31],
            "prompt B": [10, 99, 32],
        }
    )

    with pytest.raises(
        LabelPoolExhausted,
        match="requested 2 unique boundary-safe labels, but only 1 available",
    ) as raised:
        run(allocate_labels(tokenizer, "prompt", 2, pool=[" A", " B"]))

    assert raised.value.requested == 2
    assert raised.value.available == 1


def test_default_pool_can_supply_the_configured_ordinary_maximum():
    assert len(DEFAULT_LABEL_POOL) >= 16
    assert len(set(DEFAULT_LABEL_POOL)) == len(DEFAULT_LABEL_POOL)


def test_tokenizer_client_uses_live_contract_and_returns_token_ids():
    response = {
        "count": 2,
        "tokens": [10, 20],
        "token_strs": ["prompt", " suffix"],
        "max_model_len": 262144,
    }
    with tokenizer_server(response) as (base_url, received):
        client = TokenizerClient(base_url, model="served-model")

        tokens = run(client.tokenize("prompt suffix"))

    assert tokens == [10, 20]
    assert received == [
        ("/tokenize", {"prompt": "prompt suffix", "model": "served-model"})
    ]


def test_tokenizer_client_accepts_null_token_strings_from_live_contract():
    response = {
        "count": 2,
        "tokens": [10, 20],
        "token_strs": None,
        "max_model_len": 262144,
    }
    with tokenizer_server(response) as (base_url, _):
        client = TokenizerClient(base_url)

        tokens = run(client.tokenize("prompt suffix"))

    assert tokens == [10, 20]


def test_tokenizer_client_rejects_inconsistent_contract_counts():
    response = {
        "count": 2,
        "tokens": [10],
        "token_strs": ["prompt"],
        "max_model_len": 262144,
    }
    with tokenizer_server(response) as (base_url, _):
        client = TokenizerClient(base_url)

        with pytest.raises(TokenizerError, match="token counts do not match"):
            run(client.tokenize("prompt"))


@pytest.mark.live
@pytest.mark.skipif(
    os.getenv("SYSTEMONE_RUN_LIVE_TOKENIZER") != "1",
    reason="set SYSTEMONE_RUN_LIVE_TOKENIZER=1 to use the running R9V tokenizer",
)
def test_live_r9v_tokenizer_has_at_least_sixteen_boundary_safe_labels():
    client = TokenizerClient(
        base_url=os.getenv("SYSTEMONE_R9V_BASE_URL", "http://127.0.0.1:8000"),
        model=os.getenv("SYSTEMONE_R9V_MODEL", "qwen3.8-flash-next"),
    )
    labels = run(allocate_labels(client, "Answer:", 16))

    assert len(labels) == 16
    assert len({label.token_id for label in labels}) == 16
