"""Versioned model-visible prompts for strict VLM response generation."""

from __future__ import annotations

from typing import Final

PROMPT_REVISION: Final = "vlm-spike-v3"

SYSTEM_PROMPT: Final = (
    "Use only this image. Reply with exactly one JSON object and no Markdown. "
    'schema_version is always "vlm_response_v1". '
)


def render_prompt(question: str) -> str:
    """Use the smallest strict response template compatible with the question."""
    normalized = question.lower().strip()
    if normalized.startswith("describe"):
        template = (
            '{"type":"caption","status":"ok","text":"..."}; use status '
            '"unknown" and text null when unsupported.'
        )
    elif normalized.startswith("how many"):
        template = (
            '{"type":"answer","status":"ok","answer":{"kind":"count",'
            '"value":0}}; use "not_visible", "ambiguous", or "unknown" with '
            "answer null otherwise."
        )
    elif normalized.startswith(("is ", "are ")):
        template = (
            '{"type":"answer","status":"ok","answer":{"kind":"boolean",'
            '"value":true}}; use "not_visible", "ambiguous", or "unknown" with '
            "answer null otherwise."
        )
    elif normalized.startswith("point to"):
        semantics = "review_region" if "review" in normalized else "target_center"
        template = (
            '{"type":"point","status":"found","point":{"x":0,"y":0,'
            '"coordinate_system":"relative_0_1000_xy",'
            f'"semantics":"{semantics}"}}; use "not_visible", "ambiguous", '
            '"unknown", or "no_candidate" with point null otherwise.'
        )
    else:
        template = (
            '{"type":"answer","status":"ok","answer":{"kind":"text",'
            '"value":"..."}}; use "not_visible", "ambiguous", or "unknown" '
            "with answer null otherwise."
        )
    return f"{SYSTEM_PROMPT}Required form: {template}\nUser question: {question}"
