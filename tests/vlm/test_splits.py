import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from wam_drones.vlm.splits import (
    LockedSplit,
    assert_no_held_out_leakage,
    default_locked_split_path,
    load_locked_split,
)


def test_default_locked_split_path_is_checked_in() -> None:
    assert default_locked_split_path().is_file()


def test_load_locked_split_matches_phase4_tracking_split() -> None:
    split = load_locked_split()
    assert split.dataset == "VisDrone2019-MOT official validation split"
    assert set(split.development_sequences) == {
        "uav0000086_00000_v",
        "uav0000117_02622_v",
        "uav0000137_00458_v",
    }
    assert set(split.held_out_sequences) == {
        "uav0000182_00000_v",
        "uav0000268_05773_v",
        "uav0000305_00000_v",
        "uav0000339_00001_v",
    }


def test_load_locked_split_rejects_unsupported_schema_version(
    tmp_path: Path,
) -> None:
    payload = {
        "schema_version": "vlm_locked_split_v2",
        "dataset": "x",
        "source": "x",
        "development_sequences": ["a"],
        "held_out_sequences": ["b"],
        "note": "x",
    }
    path = tmp_path / "split.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported locked split schema_version"):
        load_locked_split(path)


def test_load_locked_split_rejects_sequence_listed_in_both_splits(
    tmp_path: Path,
) -> None:
    payload = {
        "schema_version": "vlm_locked_split_v1",
        "dataset": "x",
        "source": "x",
        "development_sequences": ["a", "shared"],
        "held_out_sequences": ["shared", "b"],
        "note": "x",
    }
    path = tmp_path / "split.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="listed in both splits"):
        load_locked_split(path)


def test_locked_split_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        LockedSplit(
            schema_version="vlm_locked_split_v1",
            dataset="x",
            source="x",
            development_sequences=("a",),
            held_out_sequences=("b",),
            note="x",
            unexpected=True,
        )


def test_assert_no_held_out_leakage_passes_development_sequences() -> None:
    assert_no_held_out_leakage(["uav0000086_00000_v", "uav0000117_02622_v"])


def test_assert_no_held_out_leakage_rejects_held_out_sequences() -> None:
    with pytest.raises(ValueError, match="uav0000182_00000_v"):
        assert_no_held_out_leakage(["uav0000086_00000_v", "uav0000182_00000_v"])
