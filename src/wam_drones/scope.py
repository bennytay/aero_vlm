"""Frozen supported and parked target scope for Phase 2."""

from typing import Final

SUPPORTED_LABELS: Final[tuple[str, ...]] = (
    "water bottle",
    "black vehicle",
    "bicycle",
    "cardboard box",
    "sports ball",
)
PARKED_LABELS: Final[tuple[str, ...]] = (
    "red backpack",
    "person in high-vis",
    "solar panel",
    "wheelie bin",
    "orange bucket",
)
NO_TARGET: Final = "no_target"
