"""FastAPI entry point for the localhost System-One service."""

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from .calibration import CalibrationStore
from .config import Settings
from .logging import configure_logging, log_startup_readiness
from .models import RequestErrorResponse, SystemOneRequest, SystemOneResponse
from .r9v_client import R9VClient, R9VError
from .service import BatchEvaluationError, SystemOneService


ClientFactory = Callable[[Settings], R9VClient]
StartupSleep = Callable[[float], Awaitable[None]]


async def _check_startup_readiness(
    r9v: R9VClient,
    *,
    attempts: int,
    retry_seconds: float,
    sleep: StartupSleep,
) -> None:
    for attempt in range(1, attempts + 1):
        try:
            if await r9v.health():
                log_startup_readiness(status="ready", attempts=attempt)
                return
        except R9VError:
            pass
        if attempt < attempts:
            await sleep(retry_seconds)
    log_startup_readiness(status="degraded", attempts=attempts)


def _batch_error_response(error: BatchEvaluationError) -> HTTPException:
    responses = {
        "label_boundary_exhausted": (422, "label_boundary_exhausted"),
        "upstream_timeout": (504, "r9v_timeout"),
        "upstream_protocol": (502, "r9v_protocol_error"),
        "upstream_http": (503, "r9v_unavailable"),
        "upstream_unavailable": (503, "r9v_unavailable"),
    }
    status_code, detail = responses.get(
        error.failure_category, (500, "evaluation_failed")
    )
    return HTTPException(status_code=status_code, detail=detail)


def create_app(
    settings: Settings,
    *,
    client_factory: ClientFactory = R9VClient,
    startup_health_attempts: int = 3,
    startup_health_retry_seconds: float = 1.0,
    startup_sleep: StartupSleep = asyncio.sleep,
) -> FastAPI:
    """Create one app whose shared dependencies live for its lifespan."""

    if startup_health_attempts < 1:
        raise ValueError("startup_health_attempts must be positive")
    if startup_health_retry_seconds < 0:
        raise ValueError("startup_health_retry_seconds must be nonnegative")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging(settings.log_level)
        r9v = client_factory(settings)
        try:
            calibration = CalibrationStore(settings.calibration_path)
            app.state.r9v = r9v
            app.state.systemone = SystemOneService(settings, r9v, calibration)
            await _check_startup_readiness(
                r9v,
                attempts=startup_health_attempts,
                retry_seconds=startup_health_retry_seconds,
                sleep=startup_sleep,
            )
            yield
        finally:
            await r9v.aclose()

    app = FastAPI(lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(
        _request: Request, _error: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422, content={"detail": "invalid_request"}
        )

    @app.get("/health")
    async def health() -> JSONResponse:
        try:
            ready = await app.state.r9v.health()
        except R9VError:
            ready = False
        if not ready:
            return JSONResponse(
                status_code=503,
                content={"status": "degraded", "r9v": "unavailable"},
            )
        return JSONResponse(content={"status": "ok", "r9v": "ready"})

    @app.post(
        "/v1/systemone",
        response_model=SystemOneResponse,
        response_model_exclude_none=True,
        responses={
            422: {
                "model": RequestErrorResponse,
                "description": (
                    "Invalid request or no boundary-safe candidate labels"
                ),
            }
        },
    )
    async def evaluate(
        payload: SystemOneRequest = Body(...),
    ) -> SystemOneResponse:
        try:
            request = SystemOneRequest.model_validate_with_max_choices(
                payload.model_dump(), max_choices=settings.max_choices
            )
        except ValidationError as error:
            raise HTTPException(
                status_code=422, detail="invalid_request"
            ) from error
        try:
            return await app.state.systemone.evaluate(request)
        except BatchEvaluationError as error:
            raise _batch_error_response(error) from error

    return app


app = create_app(Settings())
