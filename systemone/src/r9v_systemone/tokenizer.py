"""Served-tokenizer access and prompt-boundary-safe candidate labels."""

import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_LABEL_POOL = tuple(
    [f" {character}" for character in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
    + [f" {character}" for character in "abcdefghijklmnopqrstuvwxyz"]
    + [f" {digit}" for digit in "0123456789"]
    + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
)


class TokenizerError(RuntimeError):
    """The served tokenizer could not return a valid token sequence."""


class LabelPoolExhausted(ValueError):
    """The pool lacks enough unique labels valid at one prompt boundary."""

    def __init__(self, requested: int, available: int) -> None:
        self.requested = requested
        self.available = available
        super().__init__(
            f"requested {requested} unique boundary-safe labels, "
            f"but only {available} available"
        )


@dataclass(frozen=True, slots=True)
class CandidateLabel:
    text: str
    token_id: int


class SupportsTokenize(Protocol):
    async def tokenize(self, text: str) -> list[int]: ...


class TokenizerClient:
    """Small asynchronous client for R9V's ``/tokenize`` endpoint."""

    def __init__(
        self,
        base_url: str,
        *,
        model: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._endpoint = f"{base_url.rstrip('/')}/tokenize"
        self._model = model
        self._timeout_seconds = timeout_seconds

    async def tokenize(self, text: str) -> list[int]:
        return await asyncio.to_thread(self._tokenize_sync, text)

    def _tokenize_sync(self, text: str) -> list[int]:
        payload: dict[str, object] = {"prompt": text}
        if self._model is not None:
            payload["model"] = self._model
        request = Request(
            self._endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:
                body = response.read()
        except (HTTPError, URLError, TimeoutError) as error:
            raise TokenizerError(f"R9V tokenizer request failed: {error}") from error

        try:
            document = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise TokenizerError("R9V tokenizer returned invalid JSON") from error
        return self._parse_response(document)

    @staticmethod
    def _parse_response(document: object) -> list[int]:
        if not isinstance(document, dict):
            raise TokenizerError("R9V tokenizer response must be a JSON object")

        count = document.get("count")
        tokens = document.get("tokens")
        token_strs = document.get("token_strs")
        max_model_len = document.get("max_model_len")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise TokenizerError("R9V tokenizer response has an invalid count")
        if not isinstance(tokens, list) or any(
            isinstance(token, bool) or not isinstance(token, int) for token in tokens
        ):
            raise TokenizerError("R9V tokenizer response has invalid tokens")
        if token_strs is not None:
            if not isinstance(token_strs, list) or any(
                not isinstance(token, str) for token in token_strs
            ):
                raise TokenizerError("R9V tokenizer response has invalid token_strs")
        if (
            isinstance(max_model_len, bool)
            or not isinstance(max_model_len, int)
            or max_model_len <= 0
        ):
            raise TokenizerError("R9V tokenizer response has an invalid max_model_len")
        if count != len(tokens) or (
            isinstance(token_strs, list) and count != len(token_strs)
        ):
            raise TokenizerError("R9V tokenizer response token counts do not match")
        return list(tokens)


async def token_added_at_boundary(
    client: SupportsTokenize, prompt: str, label: str
) -> int | None:
    before = await client.tokenize(prompt)
    after = await client.tokenize(prompt + label)
    if len(after) == len(before) + 1 and after[:-1] == before:
        return after[-1]
    return None


async def allocate_labels(
    client: SupportsTokenize,
    prompt: str,
    count: int,
    *,
    pool: Sequence[str] = DEFAULT_LABEL_POOL,
) -> list[CandidateLabel]:
    """Select the first unique pool labels valid at the exact prompt boundary."""

    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer")

    allocated: list[CandidateLabel] = []
    used_token_ids: set[int] = set()
    for label in pool:
        token_id = await token_added_at_boundary(client, prompt, label)
        if token_id is None or token_id in used_token_ids:
            continue
        allocated.append(CandidateLabel(text=label, token_id=token_id))
        used_token_ids.add(token_id)
        if len(allocated) == count:
            return allocated

    raise LabelPoolExhausted(requested=count, available=len(allocated))
