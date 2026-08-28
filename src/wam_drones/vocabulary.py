"""Frozen version 0 target vocabulary."""

from typing import Final

VOCABULARY_VERSION: Final = "v0"
VOCABULARY: Final[tuple[tuple[int, str], ...]] = (
    (1, "red backpack"),
    (2, "water bottle"),
    (3, "person in high-vis"),
    (4, "black vehicle"),
    (5, "solar panel"),
    (6, "wheelie bin"),
    (7, "bicycle"),
    (8, "cardboard box"),
    (9, "sports ball"),
    (10, "orange bucket"),
)
TARGET_IDS_BY_LABEL: Final[dict[str, int]] = {
    label: target_id for target_id, label in VOCABULARY
}
