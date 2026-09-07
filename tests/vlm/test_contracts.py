import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from wam_drones.vlm.contracts import (
    AnswerResponse,
    CaptionResponse,
    CountAnswer,
    PointResponse,
    PreprocessingMetadata,
    SourceFrame,
    VLMInferenceRecord,
    VLMResponseContract,
    VLMSupervisionRecord,
    parse_vlm_response,
    pixel_to_relative_0_1000_xy,
    relative_0_1000_xy_to_pixel,
)


def point_payload(**changes: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "vlm_response_v1",
        "type": "point",
        "status": "found",
        "description": "The requested car is left of the road centre.",
        "point": {
            "x": 238,
            "y": 604,
            "coordinate_system": "relative_0_1000_xy",
            "semantics": "target_center",
        },
    }
    payload.update(changes)
    return payload


def test_response_union_round_trips_all_variants() -> None:
    responses = (
        CaptionResponse(type="caption", status="ok", text="Two cars are visible."),
        AnswerResponse(
            type="answer",
            status="ok",
            answer=CountAnswer(kind="count", value=2),
        ),
        PointResponse.model_validate(point_payload()),
    )

    for response in responses:
        assert parse_vlm_response(response.model_dump_json()) == response


@pytest.mark.parametrize(
    "payload",
    [
        point_payload(unexpected=True),
        point_payload(status="invented"),
        point_payload(status="not_visible"),
        point_payload(point={"y": 238, "x_coordinate": 604}),
        point_payload(point={"x": 1001, "y": 0, "semantics": "target_center"}),
        point_payload(point={"x": float("nan"), "y": 0, "semantics": "target_center"}),
    ],
)
def test_response_rejects_malformed_or_semantically_invalid_generation(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        parse_vlm_response(json.dumps(payload))


def test_abstentions_cannot_include_model_output_content() -> None:
    with pytest.raises(ValidationError, match="caption text is forbidden"):
        CaptionResponse(type="caption", status="unknown", text="probably a car")
    with pytest.raises(ValidationError, match="answer is forbidden"):
        AnswerResponse(
            type="answer",
            status="ambiguous",
            answer=CountAnswer(kind="count", value=1),
        )
    with pytest.raises(ValidationError, match="point is forbidden"):
        PointResponse.model_validate(point_payload(status="no_candidate"))


def test_pixel_grid_conversion_uses_xy_order_and_inclusive_edges() -> None:
    assert pixel_to_relative_0_1000_xy(
        0, 0, image_width_px=1920, image_height_px=1080
    ) == (0, 0)
    assert pixel_to_relative_0_1000_xy(
        1919, 1079, image_width_px=1920, image_height_px=1080
    ) == (1000, 1000)
    assert relative_0_1000_xy_to_pixel(
        1000, 0, image_width_px=1920, image_height_px=1080
    ) == (1919, 0)
    with pytest.raises(ValueError, match="outside"):
        pixel_to_relative_0_1000_xy(1920, 0, image_width_px=1920, image_height_px=1080)


def _audit_record(
    *,
    parse_errors: tuple[str, ...] = (),
    response: CaptionResponse | AnswerResponse | PointResponse | None = None,
) -> VLMInferenceRecord:
    return VLMInferenceRecord(
        inference_id="smoke-001",
        image_sha256="a" * 64,
        source_frame=SourceFrame(source_id="det-val-0001"),
        model_revision="model@abc123",
        prompt_revision="v1",
        preprocessing=PreprocessingMetadata(
            source_width_px=1920,
            source_height_px=1080,
            model_width_px=896,
            model_height_px=896,
        ),
        raw_generation="not JSON",
        parse_errors=parse_errors,
        response=response,
    )


def test_inference_audit_distinguishes_parse_failures_from_valid_responses() -> None:
    failed = _audit_record(parse_errors=("invalid JSON",))
    assert failed.response is None
    with pytest.raises(ValidationError, match="parse errors are required"):
        _audit_record()
    with pytest.raises(ValidationError, match="parse errors are forbidden"):
        _audit_record(
            parse_errors=("invalid JSON",),
            response=CaptionResponse(type="caption", status="ok", text="road"),
        )


def test_supervision_record_keeps_expected_response_in_canonical_evidence() -> None:
    record = VLMSupervisionRecord(
        sample_id="mot-0001-count-cars",
        image_sha256="b" * 64,
        source_frame=SourceFrame(
            source_id="mot-val", sequence_id="uav0000086_00000_v", frame_number=1
        ),
        dataset="VisDrone2019-MOT",
        source_split="val",
        split="train",
        task_family="count",
        question="How many cars are visible?",
        expected_response=AnswerResponse(
            type="answer", status="ok", answer=CountAnswer(kind="count", value=3)
        ),
        prompt_template_id="count-v1",
        teacher_source="ground_truth",
        teacher_evidence={"box_count": 3},
        label_confidence=1.0,
    )
    assert record.expected_response.type == "answer"


def test_checked_in_vlm_schemas_match_models() -> None:
    schema_dir = Path(__file__).parents[2] / "interfaces" / "schemas"
    expected = {
        "vlm_response_v1.schema.json": VLMResponseContract.model_json_schema(),
        "vlm_supervision_record_v1.schema.json": (
            VLMSupervisionRecord.model_json_schema()
        ),
        "vlm_inference_record_v1.schema.json": VLMInferenceRecord.model_json_schema(),
    }
    for filename, schema in expected.items():
        assert json.loads((schema_dir / filename).read_text(encoding="utf-8")) == schema
