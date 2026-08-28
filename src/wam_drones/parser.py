"""Pure parser for the frozen L1 command grammar."""

from enum import StrEnum
from hashlib import sha256
from math import isfinite

from pydantic import BaseModel, ConfigDict

from wam_drones.contracts import Behaviour, CommandState
from wam_drones.vocabulary import TARGET_IDS_BY_LABEL

_RESERVED_BEHAVIOURS = frozenset({"FOLLOW", "CIRCLE", "INSPECT"})


class CommandErrorReason(StrEnum):
    """Stable reason codes returned for rejected commands."""

    UNSUPPORTED_BEHAVIOUR = "UNSUPPORTED_BEHAVIOUR"
    UNKNOWN_TARGET = "UNKNOWN_TARGET"
    MALFORMED = "MALFORMED"


class CommandError(BaseModel):
    """A parse failure that is safe to pass across an interface boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reason_code: CommandErrorReason


def _command_id(text: str) -> str:
    return f"cmd_{sha256(text.encode('ascii')).hexdigest()[:16]}"


def _positive_distance(value: str) -> float | None:
    try:
        distance = float(value)
    except ValueError:
        return None
    if not isfinite(distance) or distance <= 0:
        return None
    return distance


def parse_command(text: str) -> CommandState | CommandError:
    """Parse one exact L1 command without I/O or mutable state."""
    if (
        not text
        or text != text.strip(" ")
        or "  " in text
        or any(character.isspace() and character != " " for character in text)
    ):
        return CommandError(reason_code=CommandErrorReason.MALFORMED)

    behaviour_text, separator, remainder = text.partition(" ")
    if not separator or not remainder:
        return CommandError(reason_code=CommandErrorReason.MALFORMED)
    if behaviour_text in _RESERVED_BEHAVIOURS:
        return CommandError(reason_code=CommandErrorReason.UNSUPPORTED_BEHAVIOUR)
    try:
        behaviour = Behaviour(behaviour_text)
    except ValueError:
        return CommandError(reason_code=CommandErrorReason.MALFORMED)

    for label, target_id in TARGET_IDS_BY_LABEL.items():
        if remainder == label:
            return CommandState(
                command_id=_command_id(text),
                behaviour=behaviour,
                target_label=label,
                target_id=target_id,
            )
        label_prefix = f"{label} "
        if remainder.startswith(label_prefix):
            distance_text = remainder.removeprefix(label_prefix)
            if " " in distance_text:
                return CommandError(reason_code=CommandErrorReason.MALFORMED)
            distance = _positive_distance(distance_text)
            if distance is None:
                return CommandError(reason_code=CommandErrorReason.MALFORMED)
            return CommandState(
                command_id=_command_id(text),
                behaviour=behaviour,
                target_label=label,
                target_id=target_id,
                desired_distance_m=distance,
            )

    return CommandError(reason_code=CommandErrorReason.UNKNOWN_TARGET)
