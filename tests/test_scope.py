from pathlib import Path
from typing import Any, cast

import yaml

from wam_drones.scope import NO_TARGET, PARKED_LABELS, SUPPORTED_LABELS


def test_scope_v0_is_exactly_frozen() -> None:
    document = cast(
        dict[str, Any],
        yaml.safe_load(
            (Path(__file__).parents[1] / "configs" / "scope_v0.yaml").read_text(
                encoding="utf-8"
            )
        ),
    )

    assert tuple(document["supported"]) == SUPPORTED_LABELS
    assert tuple(document["parked"]) == PARKED_LABELS
    assert document["reject_flag"] == NO_TARGET
    assert set(SUPPORTED_LABELS).isdisjoint(PARKED_LABELS)
