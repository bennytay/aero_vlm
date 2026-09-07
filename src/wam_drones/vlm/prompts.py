"""Versioned model-visible prompts for strict VLM response generation."""

from __future__ import annotations

from typing import Final, Literal

PROMPT_REVISION: Final = "vlm-spike-v4"

ResponseKind = Literal["caption", "answer", "point"]


def response_kind_for(question: str) -> ResponseKind:
    """Classify which response contract a question is routed to.

    Shared between the prompt template below and the constrained-decoding
    schema selection, so both stay in sync on what a question expects.
    """
    normalized = question.lower().strip()
    if normalized.startswith("describe"):
        return "caption"
    if normalized.startswith("point to"):
        return "point"
    return "answer"


SYSTEM_PROMPT: Final = (
    "Use only this image. Reply with exactly one JSON object and no Markdown. "
    'schema_version is always "vlm_response_v1". An ordinary, unremarkable '
    "photograph is not a reason to abstain: abstention statuses are only for "
    "when the image or question genuinely prevents an answer, never a "
    "default choice. "
)


def render_prompt(question: str) -> str:
    """Use the smallest strict response template compatible with the question."""
    normalized = question.lower().strip()
    kind = response_kind_for(question)
    if kind == "caption":
        template = (
            'If you can describe the scene: {"type":"caption","status":"ok",'
            '"text":"<one sentence>"}. Only if the image itself is blank, '
            'corrupted, or unviewable: {"type":"caption","status":"unknown",'
            '"text":null}.'
        )
    elif normalized.startswith("how many"):
        template = (
            'If you can count: {"type":"answer","status":"ok","answer":'
            '{"kind":"count","value":<integer, 0 if none are visible>}}. '
            'Only if the whole area is obscured: {"type":"answer",'
            '"status":"not_visible","answer":null}. Only if several disjoint '
            'groups could be meant and you cannot tell which: '
            '{"type":"answer","status":"ambiguous","answer":null}. Only if '
            'the question does not apply to this image: {"type":"answer",'
            '"status":"unknown","answer":null}.'
        )
    elif normalized.startswith(("is ", "are ")):
        template = (
            'If you can tell: {"type":"answer","status":"ok","answer":'
            '{"kind":"boolean","value":<true or false>}}. Only if the '
            'relevant area is obscured: {"type":"answer","status":'
            '"not_visible","answer":null}. Only if it is genuinely unclear '
            'either way: {"type":"answer","status":"ambiguous","answer":null}. '
            'Only if the question does not apply to this image: '
            '{"type":"answer","status":"unknown","answer":null}.'
        )
    elif normalized.startswith("point to"):
        semantics = "review_region" if "review" in normalized else "target_center"
        template = (
            "If exactly one matching target exists: "
            '{"type":"point","status":"found","point":{"x":<int 0-1000>,'
            '"y":<int 0-1000>,"coordinate_system":"relative_0_1000_xy",'
            f'"semantics":"{semantics}"}}}}. '
            f'The semantics value must be exactly "{semantics}" — never a '
            "description of the object. Look carefully before giving up: "
            'only use {"type":"point","status":"no_candidate","point":null} '
            "when nothing matching the question exists anywhere in the "
            'image. Only use {"type":"point","status":"ambiguous",'
            '"point":null} when two or more equally valid candidates exist '
            'and you cannot pick one. Only use {"type":"point","status":'
            '"not_visible","point":null} when a target you would expect is '
            'hidden or out of frame. Only use {"type":"point","status":'
            '"unknown","point":null} when the question itself cannot be '
            "evaluated from this image. Whenever status is not \"found\", "
            "point must be the JSON literal null, never coordinates."
        )
    else:
        template = (
            'If you can answer: {"type":"answer","status":"ok","answer":'
            '{"kind":"text","value":"<short answer>"}}. Only if the '
            'relevant area is obscured: {"type":"answer","status":'
            '"not_visible","answer":null}. Only if several equally valid '
            'answers exist and you cannot pick one: {"type":"answer",'
            '"status":"ambiguous","answer":null}. Only if the question does '
            'not apply to this image: {"type":"answer","status":"unknown",'
            '"answer":null}.'
        )
    return f"{SYSTEM_PROMPT}Required form: {template}\nUser question: {question}"
