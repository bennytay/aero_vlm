"""Deterministic Phase 1 fixture preparation."""

import shutil
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png"}


def extract_visdrone_fixtures(
    archive: Path, output_dir: Path, count: int = 10
) -> list[Path]:
    """Extract the first images in lexical order, without retaining the archive tree."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive) as source:
        members = sorted(
            name
            for name in source.namelist()
            if PurePosixPath(name).suffix.lower() in SUPPORTED_SUFFIXES
            and "images" in PurePosixPath(name).parts
        )[:count]
        if len(members) != count:
            raise ValueError(f"archive contains only {len(members)} usable images")
        extracted: list[Path] = []
        for index, member in enumerate(members):
            suffix = PurePosixPath(member).suffix.lower()
            destination = output_dir / f"visdrone_{index:02d}{suffix}"
            with source.open(member) as input_file, destination.open("wb") as output:
                shutil.copyfileobj(input_file, output)
            extracted.append(destination)
    return extracted
