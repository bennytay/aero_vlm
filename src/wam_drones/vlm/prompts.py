"""Versioned model-visible prompts for strict VLM response generation."""

from __future__ import annotations

from typing import Final

PROMPT_REVISION: Final = "vlm-spike-v2"

SYSTEM_PROMPT: Final = (
    "Use only this image. Reply with one JSON object, no Markdown. Always set "
    'schema_version to "vlm_response_v1". For a caption use '
    '{"type":"caption","status":"ok","text":string}; use status '
    '"unknown" and text null when needed. For an answer use '
    '{"type":"answer","status":"ok","answer":{"kind":'
    '"boolean"|"count"|"label"|"text","value":value}}; otherwise '
    'status is "not_visible", "ambiguous", or "unknown" and answer is null. '
    'For a point use {"type":"point","status":"found","point":{'
    '"x":0..1000,"y":0..1000,"coordinate_system":'
    '"relative_0_1000_xy","semantics":"target_center"|"review_region"}}; '
    'otherwise status is "not_visible", "ambiguous", "unknown", or '
    '"no_candidate" and point is null.'
)
