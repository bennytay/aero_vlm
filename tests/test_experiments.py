import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from wam_drones.experiments.record import ExperimentRecord, initialise_experiment


def prepare_repository(root: Path) -> None:
    template_dir = root / "evaluation" / "experiments"
    template_dir.mkdir(parents=True)
    (template_dir / "TEMPLATE.md").write_text(
        "# Experiment <ID>: <title>\n", encoding="utf-8"
    )
    config_dir = root / "configs"
    config_dir.mkdir()
    (config_dir / "example.yaml").write_text("value: 1\n", encoding="utf-8")


def test_experiment_helper_writes_valid_meta_json(tmp_path: Path) -> None:
    prepare_repository(tmp_path)
    created = datetime(2026, 8, 28, tzinfo=UTC)

    meta_path = initialise_experiment(
        "exp_20260828_helper_test",
        tmp_path,
        question="Does the helper write its metadata?",
        hypothesis="A schema-valid JSON file is created.",
        created_utc=created,
    )

    record = ExperimentRecord.model_validate_json(meta_path.read_text(encoding="utf-8"))
    assert record.experiment_id == "exp_20260828_helper_test"
    assert record.git_commit == "UNKNOWN"
    assert record.config_hash.startswith("sha256:")
    assert record.created_utc == created
    assert json.loads(meta_path.read_text(encoding="utf-8"))["metrics"] == {}
    assert meta_path.with_name("report.md").exists()


def test_experiment_helper_refuses_bad_name(tmp_path: Path) -> None:
    prepare_repository(tmp_path)

    with pytest.raises(ValueError, match="exp_YYYYMMDD_short_name"):
        initialise_experiment("phase0", tmp_path)
