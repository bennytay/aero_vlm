"""Run frozen MobileCLIP2-S0 over an image folder and emit observations."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from wam_drones.teacher.model import PROMPT_TEMPLATES, MobileClipTeacher, run_target

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})


def image_paths(folder: Path) -> list[Path]:
    return sorted(
        path for path in folder.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_folder", type=Path)
    parser.add_argument(
        "--vocabulary",
        type=Path,
        default=Path("configs/vocabulary_v0.yaml"),
        help="Frozen vocabulary file; validated for discoverability before inference.",
    )
    parser.add_argument("--template", choices=PROMPT_TEMPLATES, default="photo")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not args.vocabulary.is_file():
        parser.error(f"vocabulary not found: {args.vocabulary}")
    backend = MobileClipTeacher(device=args.device)
    rows = [
        {
            "image_path": str(path),
            "observation": run_target(
                path, backend, template_name=args.template
            ).model_dump(mode="json"),
        }
        for path in image_paths(args.image_folder)
    ]
    payload = "\n".join(json.dumps(row) for row in rows) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
