"""Strict asynchronous adapter for the existing R9V HTTP service."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import httpx

from r9v_systemone.config import Settings


@dataclass(frozen=True, slots=True)
class RequestUsage:
    requested_token_id: int
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int | None


@dataclass(frozen=True, slots=True)
class ScoreObservation:
    logprobs: dict[int, float]
    request_usages: tuple[RequestUsage, ...]

    @property
    def prompt_tokens(self) -> int:
        return sum(usage.prompt_tokens for usage in self.request_usages)

    @property
    def completion_tokens(self) -> int:
        return sum(usage.completion_tokens for usage in self.request_usages)

    @property
    def physical_request_count(self) -> int:
        return len(self.request_usages)


@dataclass(frozen=True, slots=True)
class GenerationObservation:
    text: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int | None

    @property
    def physical_request_count(self) -> int:
        return 1


class R9VError(RuntimeError):
    """Base error for an R9V operation."""


class R9VHTTPError(R9VError):
    """R9V returned a non-success HTTP status."""


class R9VTimeoutError(R9VError):
    """R9V did not respond within the configured bound."""


class R9VUnavailableError(R9VError):
    """R9V could not be reached."""


class R9VProtocolError(R9VError):
    """R9V returned a response that violates the expected contract."""


class R9VClient:
    """Share one bounded HTTP client across R9V operations."""

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        headers = {"Content-Type": "application/json"}
        if settings.api_key is not None:
            headers["Authorization"] = f"Bearer {settings.api_key.get_secret_value()}"
        self._model = settings.r9v_model
        self._client = httpx.AsyncClient(
            base_url=str(settings.r9v_base_url),
            headers=headers,
            timeout=httpx.Timeout(settings.timeout_seconds),
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "R9VClient":
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()

    async def score(
        self, prompt: str, token_ids: Sequence[int]
    ) -> ScoreObservation:
        requested_ids = self._validate_requested_ids(token_ids)
        logprobs: dict[int, float] = {}
        usages: list[RequestUsage] = []
        for token_id in requested_ids:
            response = await self._post_json(
                "v1/completions",
                {
                    "model": self._model,
                    "prompt": prompt,
                    "max_tokens": 1,
                    "logprobs": 0,
                    "temperature": 0,
                    "allowed_token_ids": [token_id],
                    "return_token_ids": True,
                    "stream": False,
                },
            )
            logprob, usage = self._parse_score_response(response, token_id)
            logprobs[token_id] = logprob
            usages.append(usage)
        return ScoreObservation(logprobs=logprobs, request_usages=tuple(usages))

    async def tokenize(self, text: str) -> list[int]:
        document = await self._post_json(
            "tokenize", {"model": self._model, "prompt": text}
        )
        root = self._require_object(document, "tokenizer response")
        count = self._require_nonnegative_int(root.get("count"), "token count")
        tokens = root.get("tokens")
        if not isinstance(tokens, list) or any(
            isinstance(token, bool) or not isinstance(token, int) for token in tokens
        ):
            raise R9VProtocolError("R9V tokenizer response has invalid tokens")
        token_strs = root.get("token_strs")
        if token_strs is not None and (
            not isinstance(token_strs, list)
            or any(not isinstance(token, str) for token in token_strs)
        ):
            raise R9VProtocolError("R9V tokenizer response has invalid token_strs")
        max_model_len = root.get("max_model_len")
        if (
            isinstance(max_model_len, bool)
            or not isinstance(max_model_len, int)
            or max_model_len <= 0
        ):
            raise R9VProtocolError("R9V tokenizer response has invalid max_model_len")
        if count != len(tokens) or (
            isinstance(token_strs, list) and count != len(token_strs)
        ):
            raise R9VProtocolError("R9V tokenizer response token counts do not match")
        return list(tokens)

    async def health(self) -> bool:
        await self._request("GET", "health")
        response = await self._request("GET", "v1/models")
        document = self._parse_json(response)
        root = self._require_object(document, "model discovery response")
        models = root.get("data")
        if not isinstance(models, list):
            raise R9VProtocolError("R9V model discovery response has invalid data")
        model_ids: list[str] = []
        for model in models:
            if not isinstance(model, dict):
                raise R9VProtocolError(
                    "R9V model discovery response has invalid model entry"
                )
            model_id = model.get("id")
            if not isinstance(model_id, str) or not model_id:
                raise R9VProtocolError(
                    "R9V model discovery response has invalid model ID"
                )
            model_ids.append(model_id)
        return self._model in model_ids

    async def metrics(self) -> str:
        response = await self._request("GET", "metrics")
        return response.text

    async def generate(
        self, prompt: str, *, max_tokens: int = 8
    ) -> GenerationObservation:
        if (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or max_tokens < 1
        ):
            raise ValueError("max_tokens must be a positive integer")
        document = await self._post_json(
            "v1/completions",
            {
                "model": self._model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0,
                "stream": False,
            },
        )
        root = self._require_object(document, "response")
        choices = root.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise R9VProtocolError("R9V response must contain exactly one choice")
        choice = self._require_object(choices[0], "choice")
        text = choice.get("text")
        if not isinstance(text, str):
            raise R9VProtocolError("R9V response has invalid generated text")
        prompt_tokens, completion_tokens, cached_tokens = self._parse_usage_fields(
            root.get("usage")
        )
        return GenerationObservation(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
        )

    async def _post_json(self, path: str, payload: dict[str, object]) -> Any:
        response = await self._request("POST", path, json=payload)
        return self._parse_json(response)

    async def _request(
        self, method: str, path: str, *, json: dict[str, object] | None = None
    ) -> httpx.Response:
        try:
            response = await self._client.request(method, path, json=json)
        except httpx.TimeoutException as error:
            raise R9VTimeoutError("R9V request timed out") from error
        except httpx.RequestError as error:
            raise R9VUnavailableError("R9V request failed") from error
        if not response.is_success:
            raise R9VHTTPError(f"R9V returned HTTP {response.status_code}")
        return response

    @staticmethod
    def _parse_json(response: httpx.Response) -> Any:
        try:
            return response.json()
        except ValueError as error:
            raise R9VProtocolError("R9V returned invalid JSON") from error

    @staticmethod
    def _validate_requested_ids(token_ids: Sequence[int]) -> tuple[int, ...]:
        requested = tuple(token_ids)
        if not requested:
            raise ValueError("token_ids must not be empty")
        if any(
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or token_id < 0
            for token_id in requested
        ):
            raise ValueError("token_ids must contain nonnegative integers")
        if len(set(requested)) != len(requested):
            raise ValueError("token_ids must be unique")
        return requested

    @classmethod
    def _parse_score_response(
        cls, document: object, requested_token_id: int
    ) -> tuple[float, RequestUsage]:
        root = cls._require_object(document, "response")
        choices = root.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise R9VProtocolError("R9V response must contain exactly one choice")
        choice = cls._require_object(choices[0], "choice")
        returned_ids = choice.get("token_ids")
        if not isinstance(returned_ids, list):
            raise R9VProtocolError("R9V response has invalid returned token IDs")
        if not returned_ids:
            raise R9VProtocolError("R9V response is missing returned token ID")
        if any(
            isinstance(token_id, bool) or not isinstance(token_id, int)
            for token_id in returned_ids
        ):
            raise R9VProtocolError("R9V response has invalid returned token IDs")
        if len(set(returned_ids)) != len(returned_ids):
            raise R9VProtocolError("R9V response has duplicate returned token ID")
        if returned_ids != [requested_token_id]:
            raise R9VProtocolError("R9V response has unexpected returned token ID")

        logprobs = cls._require_object(choice.get("logprobs"), "choice logprobs")
        values = logprobs.get("token_logprobs")
        if not isinstance(values, list) or len(values) != 1:
            raise R9VProtocolError("R9V response must contain one token logprob")
        value = values[0]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise R9VProtocolError("R9V response must contain a finite numeric logprob")
        try:
            numeric_value = float(value)
        except (OverflowError, TypeError, ValueError) as error:
            raise R9VProtocolError(
                "R9V response must contain a finite numeric logprob"
            ) from error
        if not math.isfinite(numeric_value):
            raise R9VProtocolError("R9V response must contain a finite numeric logprob")

        usage = cls._parse_usage(root.get("usage"), requested_token_id)
        return numeric_value, usage

    @classmethod
    def _parse_usage(cls, value: object, requested_token_id: int) -> RequestUsage:
        prompt_tokens, completion_tokens, cached_tokens = cls._parse_usage_fields(value)
        return RequestUsage(
            requested_token_id=requested_token_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
        )

    @classmethod
    def _parse_usage_fields(cls, value: object) -> tuple[int, int, int | None]:
        usage = cls._require_object(value, "usage")
        prompt_tokens = cls._require_nonnegative_int(
            usage.get("prompt_tokens"), "prompt_tokens"
        )
        completion_tokens = cls._require_nonnegative_int(
            usage.get("completion_tokens"), "completion_tokens"
        )
        details = usage.get("prompt_tokens_details")
        cached_tokens: int | None = None
        if details is not None:
            details_object = cls._require_object(details, "prompt_tokens_details")
            if details_object.get("cached_tokens") is not None:
                cached_tokens = cls._require_nonnegative_int(
                    details_object["cached_tokens"], "cached_tokens"
                )
        return prompt_tokens, completion_tokens, cached_tokens

    @staticmethod
    def _require_object(value: object, name: str) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise R9VProtocolError(f"R9V response has invalid {name}")
        return value

    @staticmethod
    def _require_nonnegative_int(value: object, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise R9VProtocolError(f"R9V response has invalid {name}")
        return value
