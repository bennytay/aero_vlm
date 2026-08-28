from pathlib import Path
from typing import Any, cast

import yaml

from wam_drones.vocabulary import VOCABULARY

EXPECTED_LABELS = [
    {"id": 1, "name": "red backpack"},
    {"id": 2, "name": "water bottle"},
    {"id": 3, "name": "person in high-vis"},
    {"id": 4, "name": "black vehicle"},
    {"id": 5, "name": "solar panel"},
    {"id": 6, "name": "wheelie bin"},
    {"id": 7, "name": "bicycle"},
    {"id": 8, "name": "cardboard box"},
    {"id": 9, "name": "sports ball"},
    {"id": 10, "name": "orange bucket"},
]


def test_vocabulary_v0_is_exactly_frozen() -> None:
    path = Path(__file__).parents[1] / "configs" / "vocabulary_v0.yaml"
    document = cast(dict[str, Any], yaml.safe_load(path.read_text(encoding="utf-8")))

    assert document == {"version": "v0", "labels": EXPECTED_LABELS}
    assert list(VOCABULARY) == [
        (label["id"], label["name"]) for label in EXPECTED_LABELS
    ]
    assert len(document["labels"]) == 10
