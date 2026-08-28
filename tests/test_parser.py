import pytest

from wam_drones.contracts import Behaviour, CommandState
from wam_drones.parser import CommandError, CommandErrorReason, parse_command


@pytest.mark.parametrize(
    ("text", "behaviour", "target_label", "target_id", "distance"),
    [
        ("FIND red backpack", Behaviour.FIND, "red backpack", 1, None),
        ("APPROACH water bottle 2.5", Behaviour.APPROACH, "water bottle", 2, 2.5),
        ("HOLD black vehicle", Behaviour.HOLD, "black vehicle", 4, None),
    ],
)
def test_parse_success(
    text: str,
    behaviour: Behaviour,
    target_label: str,
    target_id: int,
    distance: float | None,
) -> None:
    result = parse_command(text)

    assert isinstance(result, CommandState)
    assert result.behaviour == behaviour
    assert result.target_label == target_label
    assert result.target_id == target_id
    assert result.desired_distance_m == distance
    assert result.command_timeout_ms is None


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("find red backpack", CommandErrorReason.MALFORMED),
        ("FIND Red Backpack", CommandErrorReason.UNKNOWN_TARGET),
        ("FIND bottle", CommandErrorReason.UNKNOWN_TARGET),
        ("FIND red bag", CommandErrorReason.UNKNOWN_TARGET),
        ("FOLLOW bicycle", CommandErrorReason.UNSUPPORTED_BEHAVIOUR),
        ("APPROACH something red", CommandErrorReason.UNKNOWN_TARGET),
        ("FIND  red backpack", CommandErrorReason.MALFORMED),
        ("FIND red backpack 0", CommandErrorReason.MALFORMED),
        ("FIND red backpack NaN", CommandErrorReason.MALFORMED),
    ],
)
def test_parse_failure(text: str, reason: CommandErrorReason) -> None:
    result = parse_command(text)

    assert isinstance(result, CommandError)
    assert result.reason_code == reason


def test_parse_is_deterministic() -> None:
    assert parse_command("FIND bicycle") == parse_command("FIND bicycle")
