"""Strict request, response, and calibration data contracts."""

import math
from typing import Annotated, Any, Literal, Self, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    ValidationInfo,
    field_validator,
    model_validator,
)

HARD_MAX_CHOICES = 64

Probability = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
CriterionValue: TypeAlias = (
    str | dict[str, JsonValue] | list[JsonValue] | None
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ChoiceQuestion(StrictModel):
    type: Literal["choice"]
    instructions: str = Field(min_length=1)
    criteria: dict[str, str | None] = Field(
        min_length=2, max_length=HARD_MAX_CHOICES
    )
    family: str | None = None

    @field_validator("instructions")
    @classmethod
    def require_nonempty_instructions(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("instructions must not be blank")
        return value

    @field_validator("criteria")
    @classmethod
    def require_valid_criteria(
        cls, value: dict[str, str | None]
    ) -> dict[str, str | None]:
        for key, description in value.items():
            if not key.strip():
                raise ValueError("criterion keys must not be blank")
            if description is not None and not description.strip():
                raise ValueError("criterion descriptions must not be blank")
        return value

    @field_validator("family")
    @classmethod
    def require_nonempty_family(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("family must not be blank")
        return value


def _require_nonblank_criterion(value: CriterionValue) -> CriterionValue:
    if isinstance(value, str) and not value.strip():
        raise ValueError("criterion descriptions must not be blank")
    _require_finite_json(value)
    return value


def _require_finite_json(value: JsonValue) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("criterion descriptions must contain finite JSON values")
    if isinstance(value, dict):
        for nested in value.values():
            _require_finite_json(nested)
    elif isinstance(value, list):
        for nested in value:
            _require_finite_json(nested)


class NoulQuestion(StrictModel):
    type: Literal["noul"]
    instructions: str = Field(min_length=1)
    criteria: dict[Literal["false", "true"], CriterionValue] | None = None
    family: str | None = None

    @field_validator("instructions")
    @classmethod
    def require_nonempty_instructions(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("instructions must not be blank")
        return value

    @field_validator("criteria")
    @classmethod
    def require_valid_criteria(
        cls,
        value: dict[Literal["false", "true"], CriterionValue] | None,
    ) -> dict[Literal["false", "true"], CriterionValue] | None:
        if value is not None:
            for description in value.values():
                _require_nonblank_criterion(description)
        return value

    @field_validator("family")
    @classmethod
    def require_nonempty_family(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("family must not be blank")
        return value


class ScoreQuestion(StrictModel):
    type: Literal["score"]
    instructions: str = Field(min_length=1)
    criteria: list[CriterionValue] = Field(min_length=2, max_length=10)
    family: str | None = None

    @field_validator("instructions")
    @classmethod
    def require_nonempty_instructions(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("instructions must not be blank")
        return value

    @field_validator("criteria")
    @classmethod
    def require_valid_criteria(
        cls, value: list[CriterionValue]
    ) -> list[CriterionValue]:
        for description in value:
            _require_nonblank_criterion(description)
        return value

    @field_validator("family")
    @classmethod
    def require_nonempty_family(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("family must not be blank")
        return value


Question = Annotated[
    ChoiceQuestion | NoulQuestion | ScoreQuestion, Field(discriminator="type")
]


class SystemOneRequest(StrictModel):
    state: str = Field(min_length=1)
    questions: dict[str, Question] = Field(min_length=1)
    include_diagnostics: bool = False

    @field_validator("state")
    @classmethod
    def require_nonempty_state(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("state must not be blank")
        return value

    @field_validator("questions")
    @classmethod
    def require_question_names(
        cls, value: dict[str, Question]
    ) -> dict[str, Question]:
        if any(not key.strip() for key in value):
            raise ValueError("question names must not be blank")
        return value

    @model_validator(mode="after")
    def enforce_configured_choice_limit(self, info: ValidationInfo) -> Self:
        context = info.context or {}
        if "max_choices" not in context:
            return self
        max_choices = context["max_choices"]
        if (
            isinstance(max_choices, bool)
            or not isinstance(max_choices, int)
            or not 2 <= max_choices <= HARD_MAX_CHOICES
        ):
            raise ValueError("max_choices must be an integer from 2 through 64")
        for name, question in self.questions.items():
            candidate_count = (
                2
                if isinstance(question, NoulQuestion)
                else len(question.criteria)
            )
            if candidate_count > max_choices:
                raise ValueError(
                    f"question {name!r} exceeds the configured maximum of "
                    f"{max_choices} choices"
                )
        return self

    @classmethod
    def model_validate_with_max_choices(
        cls, value: Any, *, max_choices: int
    ) -> Self:
        """Validate input against one immutable settings-derived choice limit."""

        return cls.model_validate(value, context={"max_choices": max_choices})


class CalibrationMetadata(StrictModel):
    applied: bool
    temperature: FiniteFloat = Field(gt=0.0)
    family: str | None = None
    version: int | None = Field(default=None, ge=1)

    @field_validator("family")
    @classmethod
    def require_nonempty_family(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("family must not be blank")
        return value


class ChoiceResult(StrictModel):
    choice: str = Field(min_length=1)
    probabilities: dict[str, Probability] = Field(
        min_length=2, max_length=HARD_MAX_CHOICES
    )
    probability_kind: Literal["raw_renormalized", "temperature_calibrated"]
    calibration: CalibrationMetadata
    raw_logprobs: dict[str, FiniteFloat] | None = None

    @field_validator("choice")
    @classmethod
    def require_nonempty_choice(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("choice must not be blank")
        return value

    @field_validator("probabilities")
    @classmethod
    def require_probability_keys(
        cls, value: dict[str, Probability]
    ) -> dict[str, Probability]:
        if any(not key.strip() for key in value):
            raise ValueError("probability keys must not be blank")
        return value

    @model_validator(mode="after")
    def require_consistent_distribution(self) -> Self:
        if self.choice not in self.probabilities:
            raise ValueError("choice must name one of the probability keys")
        if not math.isclose(
            math.fsum(self.probabilities.values()), 1.0, rel_tol=0.0, abs_tol=1e-6
        ):
            raise ValueError("probabilities must sum to 1")
        if self.raw_logprobs is not None:
            if any(not key.strip() for key in self.raw_logprobs):
                raise ValueError("raw logprob keys must not be blank")
            if self.raw_logprobs.keys() != self.probabilities.keys():
                raise ValueError("raw logprob keys must match probability keys")
        return self


class NoulResult(StrictModel):
    type: Literal["noul"]
    noul: Probability
    probability_kind: Literal["raw_renormalized", "temperature_calibrated"]
    calibration: CalibrationMetadata
    raw_logprobs: dict[Literal["false", "true"], FiniteFloat] | None = None

    @model_validator(mode="after")
    def require_complete_diagnostics(self) -> Self:
        if self.raw_logprobs is not None and list(self.raw_logprobs) != [
            "false",
            "true",
        ]:
            raise ValueError("raw logprobs must contain false then true")
        return self


class ScoreResult(StrictModel):
    type: Literal["score"]
    score: FiniteFloat = Field(ge=0.0)
    legend: dict[str, CriterionValue] = Field(min_length=2, max_length=10)
    probabilities: dict[str, Probability] = Field(min_length=2, max_length=10)
    probability_kind: Literal["raw_renormalized", "temperature_calibrated"]
    calibration: CalibrationMetadata
    raw_logprobs: dict[str, FiniteFloat] | None = None

    @model_validator(mode="after")
    def require_consistent_distribution(self) -> Self:
        expected_keys = [str(index) for index in range(len(self.legend))]
        if (
            list(self.legend) != expected_keys
            or list(self.probabilities) != expected_keys
        ):
            raise ValueError("legend and probabilities must use contiguous index keys")
        if not math.isclose(
            math.fsum(self.probabilities.values()), 1.0, rel_tol=0.0, abs_tol=1e-6
        ):
            raise ValueError("probabilities must sum to 1")
        if (
            self.raw_logprobs is not None
            and list(self.raw_logprobs) != expected_keys
        ):
            raise ValueError("raw logprob keys must match contiguous index keys")
        expected_score = math.fsum(
            index * probability
            for index, probability in enumerate(self.probabilities.values())
        )
        if not math.isclose(
            self.score, expected_score, rel_tol=0.0, abs_tol=1e-6
        ):
            raise ValueError(
                "score must equal the probability-weighted expected value"
            )
        return self


Result: TypeAlias = ChoiceResult | NoulResult | ScoreResult


class SystemOneResponse(StrictModel):
    results: dict[str, Result] = Field(min_length=1)

    @field_validator("results")
    @classmethod
    def require_result_names(
        cls, value: dict[str, Result]
    ) -> dict[str, Result]:
        if any(not key.strip() for key in value):
            raise ValueError("result names must not be blank")
        return value


class RequestErrorResponse(StrictModel):
    detail: Literal["invalid_request", "label_boundary_exhausted"]


class CalibrationEntry(StrictModel):
    temperature: FiniteFloat = Field(gt=0.0)


class CalibrationFile(StrictModel):
    version: Literal[1]
    families: dict[str, CalibrationEntry]

    @field_validator("families")
    @classmethod
    def require_family_names(
        cls, value: dict[str, CalibrationEntry]
    ) -> dict[str, CalibrationEntry]:
        if any(not key.strip() for key in value):
            raise ValueError("calibration family names must not be blank")
        return value
