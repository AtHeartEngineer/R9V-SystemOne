import pytest
from pydantic import ValidationError

from r9v_systemone.models import (
    CalibrationFile,
    CalibrationMetadata,
    ChoiceQuestion,
    ChoiceResult,
    SystemOneRequest,
    SystemOneResponse,
)


EXAMPLE_REQUEST = {
    "state": "binary_sensor.office: on\nlight.office: off\n",
    "questions": {
        "occupancy": {
            "type": "choice",
            "instructions": "Is the office occupied?",
            "criteria": {
                "occupied": "Evidence shows occupancy",
                "empty": "Evidence shows no occupancy",
                "uncertain": None,
            },
        },
        "lights": {
            "type": "choice",
            "instructions": "Should the office light be on?",
            "criteria": {"on": None, "off": None},
        },
    },
}


def test_choice_question_requires_two_criteria():
    with pytest.raises(ValidationError):
        ChoiceQuestion(
            type="choice", instructions="Occupied?", criteria={"yes": None}
        )


def test_request_preserves_question_and_criteria_order():
    request = SystemOneRequest.model_validate(EXAMPLE_REQUEST)

    assert list(request.questions) == ["occupancy", "lights"]
    assert list(request.questions["occupancy"].criteria) == [
        "occupied",
        "empty",
        "uncertain",
    ]


def test_request_rejects_empty_state_and_instructions():
    empty_state = {**EXAMPLE_REQUEST, "state": " \n"}
    empty_instructions = {
        **EXAMPLE_REQUEST,
        "questions": {
            "occupancy": {
                **EXAMPLE_REQUEST["questions"]["occupancy"],
                "instructions": "  ",
            }
        },
    }

    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate(empty_state)
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate(empty_instructions)


def test_request_forbids_unknown_fields_and_uses_strict_types():
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate({**EXAMPLE_REQUEST, "unexpected": True})
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate(
            {**EXAMPLE_REQUEST, "include_diagnostics": "yes"}
        )


def test_request_enforces_the_supplied_choice_limit_without_global_state():
    with pytest.raises(ValidationError, match="configured maximum of 2"):
        SystemOneRequest.model_validate_with_max_choices(
            EXAMPLE_REQUEST, max_choices=2
        )

    request = SystemOneRequest.model_validate_with_max_choices(
        EXAMPLE_REQUEST, max_choices=3
    )
    assert len(request.questions["occupancy"].criteria) == 3


def test_request_diagnostics_default_is_false():
    assert SystemOneRequest.model_validate(EXAMPLE_REQUEST).include_diagnostics is False


def test_choice_result_requires_normalized_probabilities_and_matching_choice():
    calibration = CalibrationMetadata(applied=False, temperature=1.0)

    with pytest.raises(ValidationError, match="sum to 1"):
        ChoiceResult(
            choice="occupied",
            probabilities={"occupied": 0.4, "empty": 0.4},
            probability_kind="raw_renormalized",
            calibration=calibration,
        )
    with pytest.raises(ValidationError, match="choice must name"):
        ChoiceResult(
            choice="uncertain",
            probabilities={"occupied": 0.5, "empty": 0.5},
            probability_kind="raw_renormalized",
            calibration=calibration,
        )


def test_response_preserves_results_and_exposes_calibration_metadata():
    result = ChoiceResult(
        choice="empty",
        probabilities={"occupied": 0.25, "empty": 0.75},
        probability_kind="temperature_calibrated",
        calibration=CalibrationMetadata(
            applied=True, family="occupancy", temperature=1.25, version=1
        ),
        raw_logprobs={"occupied": -2.0, "empty": -0.9},
    )

    response = SystemOneResponse(results={"occupancy": result})

    assert list(response.results) == ["occupancy"]
    assert response.results["occupancy"].calibration.temperature == 1.25
    assert response.results["occupancy"].probability_kind == "temperature_calibrated"


def test_calibration_file_is_versioned_strict_and_positive():
    calibration = CalibrationFile.model_validate(
        {"version": 1, "families": {"occupancy": {"temperature": 1.25}}}
    )
    assert calibration.families["occupancy"].temperature == 1.25

    with pytest.raises(ValidationError):
        CalibrationFile.model_validate(
            {"version": 2, "families": {"occupancy": {"temperature": 1.25}}}
        )
    with pytest.raises(ValidationError):
        CalibrationFile.model_validate(
            {"version": 1, "families": {"occupancy": {"temperature": 0.0}}}
        )
    with pytest.raises(ValidationError):
        CalibrationFile.model_validate(
            {
                "version": 1,
                "families": {
                    "occupancy": {"temperature": 1.25, "unexpected": True}
                },
            }
        )
