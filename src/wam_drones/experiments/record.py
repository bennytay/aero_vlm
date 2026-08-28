"""Small, local, reproducible experiment record implementation."""

import json
import re
import subprocess
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

EXPERIMENT_ID_PATTERN = re.compile(r"^exp_\d{8}_[a-z0-9]+(?:_[a-z0-9]+)*$")


class ExperimentRecord(BaseModel):
    """Metadata needed to identify and reproduce one experiment."""

    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    question: str = Field(min_length=1)
    hypothesis: str = Field(min_length=1)
    git_commit: str = Field(min_length=1)
    config_hash: str = Field(min_length=1)
    dataset_manifest_hash: str | None
    random_seed: int
    hardware: str = Field(min_length=1)
    firmware_version: str | None
    model_hash: str | None
    metrics: dict[str, Any]
    artefact_paths: list[str]
    created_utc: datetime


def git_commit(repo_root: Path) -> str:
    """Return HEAD for a Git worktree, or UNKNOWN outside one."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "UNKNOWN"
    return result.stdout.strip() or "UNKNOWN"


def hash_paths(paths: Sequence[Path], repo_root: Path) -> str:
    """Hash relative names and contents in a stable order."""
    digest = sha256()
    existing_paths = sorted(
        (path for path in paths if path.is_file()),
        key=lambda path: path.relative_to(repo_root).as_posix(),
    )
    if not existing_paths:
        digest.update(b"NO_CONFIG_FILES")
    for path in existing_paths:
        digest.update(path.relative_to(repo_root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def repository_config_hash(repo_root: Path) -> str:
    """Hash all versionable files under the repository's config directory."""
    config_root = repo_root / "configs"
    return hash_paths(tuple(config_root.rglob("*")), repo_root)


def initialise_experiment(
    experiment_id: str,
    repo_root: Path,
    *,
    question: str = "TBD: what single uncertainty does this experiment resolve?",
    hypothesis: str = "TBD: state the expected measurable result.",
    random_seed: int = 0,
    hardware: str = "desktop environment; exact hardware TBD",
    metrics: Mapping[str, Any] | None = None,
    artefact_paths: Sequence[str] = (),
    created_utc: datetime | None = None,
) -> Path:
    """Create a metadata record and report stub, refusing to overwrite either."""
    if EXPERIMENT_ID_PATTERN.fullmatch(experiment_id) is None:
        raise ValueError("experiment ID must match exp_YYYYMMDD_short_name")

    experiment_dir = repo_root / "evaluation" / "experiments" / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)
    meta_path = experiment_dir / "meta.json"
    report_path = experiment_dir / "report.md"
    if meta_path.exists() or report_path.exists():
        raise FileExistsError(f"experiment already exists: {experiment_id}")

    record = ExperimentRecord(
        experiment_id=experiment_id,
        question=question,
        hypothesis=hypothesis,
        git_commit=git_commit(repo_root),
        config_hash=repository_config_hash(repo_root),
        dataset_manifest_hash=None,
        random_seed=random_seed,
        hardware=hardware,
        firmware_version=None,
        model_hash=None,
        metrics=dict(metrics or {}),
        artefact_paths=list(artefact_paths),
        created_utc=created_utc or datetime.now(UTC),
    )
    meta_path.write_text(
        json.dumps(record.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    template_path = repo_root / "evaluation" / "experiments" / "TEMPLATE.md"
    template = template_path.read_text(encoding="utf-8")
    report_path.write_text(
        template.replace("<ID>", experiment_id).replace("<title>", "TBD"),
        encoding="utf-8",
    )
    return meta_path
