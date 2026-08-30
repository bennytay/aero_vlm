from pathlib import Path
from zipfile import ZipFile

import pytest

from wam_drones.detection.fixtures import extract_visdrone_fixtures


def test_extracts_ten_images_in_stable_order(tmp_path: Path) -> None:
    archive = tmp_path / "visdrone.zip"
    with ZipFile(archive, "w") as output:
        for index in reversed(range(12)):
            output.writestr(f"VisDrone/images/{index:02d}.jpg", bytes([index]))
        output.writestr("VisDrone/annotations/00.txt", b"annotation")

    extracted = extract_visdrone_fixtures(archive, tmp_path / "fixtures")

    assert [path.name for path in extracted] == [
        f"visdrone_{index:02d}.jpg" for index in range(10)
    ]
    assert [path.read_bytes() for path in extracted] == [
        bytes([index]) for index in range(10)
    ]


def test_rejects_archive_with_too_few_images(tmp_path: Path) -> None:
    archive = tmp_path / "visdrone.zip"
    with ZipFile(archive, "w") as output:
        output.writestr("VisDrone/images/00.jpg", b"image")

    with pytest.raises(ValueError, match="only 1"):
        extract_visdrone_fixtures(archive, tmp_path / "fixtures")
