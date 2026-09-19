"""Privacy-safe structured events for System-One requests."""

import hashlib
import json
import logging


_LOGGER = logging.getLogger("r9v_systemone")
_OUTPUT_HANDLER_MARKER = "_r9v_systemone_output_handler"


def configure_logging(level: str) -> None:
    """Configure one private structured-output path for the service logger."""

    handlers = [
        handler
        for handler in _LOGGER.handlers
        if getattr(handler, _OUTPUT_HANDLER_MARKER, False)
    ]
    if handlers:
        output = handlers[0]
        for duplicate in handlers[1:]:
            _LOGGER.removeHandler(duplicate)
            duplicate.close()
    else:
        output = logging.StreamHandler()
        setattr(output, _OUTPUT_HANDLER_MARKER, True)
        _LOGGER.addHandler(output)
    output.setLevel(logging.NOTSET)
    output.setFormatter(logging.Formatter("%(message)s"))
    _LOGGER.setLevel(level)
    _LOGGER.disabled = False
    _LOGGER.propagate = False


def _state_hash_prefix(state: str) -> str:
    return hashlib.sha256(state.encode("utf-8")).hexdigest()[:16]


def _emit(document: dict[str, object]) -> None:
    _LOGGER.info(json.dumps(document, separators=(",", ":"), sort_keys=True))


def log_startup_readiness(*, status: str, attempts: int) -> None:
    _emit(
        {
            "attempts": attempts,
            "event": "startup_readiness",
            "status": status,
        }
    )


def log_request_started(
    state: str, *, question_count: int, request_id: str
) -> None:
    _emit(
        {
            "request_id": request_id,
            "state_sha256_prefix": _state_hash_prefix(state),
            "question_count": question_count,
        }
    )


def log_request_completed(
    state: str,
    *,
    question_count: int,
    request_id: str,
    latency_ms: float,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int | None,
) -> None:
    document: dict[str, object] = {
        "request_id": request_id,
        "state_sha256_prefix": _state_hash_prefix(state),
        "question_count": question_count,
        "latency_ms": latency_ms,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }
    if cached_tokens is not None:
        document["cached_tokens"] = cached_tokens
    _emit(document)


def log_request_failed(
    state: str,
    *,
    question_count: int,
    request_id: str,
    latency_ms: float,
    failure_category: str,
) -> None:
    _emit(
        {
            "request_id": request_id,
            "state_sha256_prefix": _state_hash_prefix(state),
            "question_count": question_count,
            "latency_ms": latency_ms,
            "failure_category": failure_category,
        }
    )
