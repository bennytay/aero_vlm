import json
from collections.abc import Callable
from pathlib import Path

from wam_drones.contracts import CommandState, TargetObservation
from wam_drones.parser import parse_command


def test_command_state_schema_roundtrip() -> None:
    command = parse_command("APPROACH water bottle 2.5")
    assert isinstance(command, CommandState)

    restored = CommandState.model_validate_json(command.model_dump_json())
    assert restored == command


def test_target_observation_schema_roundtrip(
    target_observation_factory: Callable[[], TargetObservation],
) -> None:
    observation = target_observation_factory()

    restored = TargetObservation.model_validate_json(observation.model_dump_json())
    assert restored == observation


def test_checked_in_schemas_match_models() -> None:
    schema_dir = Path(__file__).parents[1] / "interfaces" / "schemas"
    expected = {
        "command_state.schema.json": CommandState.model_json_schema(),
        "target_observation.schema.json": TargetObservation.model_json_schema(),
    }
    for filename, schema in expected.items():
        checked_in = json.loads((schema_dir / filename).read_text(encoding="utf-8"))
        assert checked_in == schema
