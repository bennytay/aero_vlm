from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest
import yaml
from pydantic import ValidationError

from wam_drones.dataset import DatasetManifest, DatasetSample
from wam_drones.vocabulary import TARGET_IDS_BY_LABEL


def valid_sample() -> DatasetSample:
    return DatasetSample(
        image_path="data/raw/v0/example.jpg",
        image_sha256=f"sha256:{'a' * 64}",
        split="val",
        source="fixture",
        source_id="scene-1",
        session_id="fixture:scene-1",
        source_url="https://example.invalid/image",
        licence="CC BY 4.0",
        no_target=False,
        target_label="bicycle",
        target_id=7,
        bbox_xyxy_normalized=(0.1, 0.2, 0.7, 0.8),
    )


def test_sources_cover_exact_frozen_vocabulary() -> None:
    root = Path(__file__).parents[1]
    sources = cast(
        dict[str, Any],
        yaml.safe_load(
            (root / "data" / "manifests" / "sources_v0.yaml").read_text(
                encoding="utf-8"
            )
        ),
    )
    labels = [source["target_label"] for source in sources["sources"]]

    assert labels == list(TARGET_IDS_BY_LABEL)
    assert len(labels) == 10


def test_manifest_schema_roundtrip() -> None:
    manifest = DatasetManifest(
        created_utc=datetime(2026, 8, 28, tzinfo=UTC),
        samples=(valid_sample(),),
    )

    restored = DatasetManifest.model_validate_json(manifest.model_dump_json())
    assert restored == manifest


def test_no_target_is_a_flag_not_a_label() -> None:
    values = valid_sample().model_dump()
    values.update(
        no_target=True,
        target_label=None,
        target_id=None,
        bbox_xyxy_normalized=None,
    )

    no_target = DatasetSample.model_validate(values)
    assert no_target.no_target
    assert no_target.target_label is None


def test_manifest_rejects_wrong_frozen_id() -> None:
    values = valid_sample().model_dump()
    values["target_id"] = 9

    with pytest.raises(ValidationError, match="vocabulary_v0"):
        DatasetSample.model_validate(values)
