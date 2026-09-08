"""Tests for the small, source-aware Step 4 demo manifest."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _showcase_module() -> object:
    script = Path(__file__).resolve().parents[2] / "scripts/render_vlm_showcase.py"
    spec = importlib.util.spec_from_file_location("render_vlm_showcase", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _valid_manifest() -> dict[str, object]:
    source = {
        "id": "open-footage-v1",
        "url": "https://example.invalid/source",
        "license": "CC BY 4.0",
        "downloaded_on": "2026-09-08",
        "permitted_use": "public showcase with attribution",
        "sha256": "a" * 64,
    }
    return {
        "manifest_version": "vlm_demo_source_v1",
        "sources": [source],
        "cases": [
            {
                "id": kind,
                "kind": kind,
                "source_id": source["id"],
                "image": f"{kind}.jpg",
                "question": (
                    "Describe the scene."
                    if kind == "caption"
                    else "Point to the requested target."
                    if kind in {"point", "abstention"}
                    else "Is a vehicle visible?"
                ),
            }
            for kind in ("caption", "answer", "point", "abstention")
        ],
    }


def test_showcase_manifest_requires_four_kinds(tmp_path: Path) -> None:
    manifest = tmp_path / "source_manifest.json"
    manifest.write_text(json.dumps(_valid_manifest()), encoding="utf-8")
    module = _showcase_module()
    cases, sources = module.load_manifest(manifest)  # type: ignore[attr-defined]
    assert len(cases) == 4
    assert sources["open-footage-v1"]["license"] == "CC BY 4.0"


def test_showcase_manifest_rejects_missing_abstention(tmp_path: Path) -> None:
    payload = _valid_manifest()
    cases = payload["cases"]
    assert isinstance(cases, list)
    cases[-1]["kind"] = "point"
    manifest = tmp_path / "source_manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="case kinds"):
        _showcase_module().load_manifest(manifest)  # type: ignore[attr-defined]


def test_abstention_requires_a_pointing_question(tmp_path: Path) -> None:
    payload = _valid_manifest()
    cases = payload["cases"]
    assert isinstance(cases, list)
    cases[-1]["question"] = "Is a vehicle visible?"
    manifest = tmp_path / "source_manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="abstention cases"):
        _showcase_module().load_manifest(manifest)  # type: ignore[attr-defined]
