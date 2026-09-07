"""Image-only VLM spike command line interface."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from importlib import import_module
from pathlib import Path
from typing import Any

from wam_drones.vlm.contracts import SourceFrame
from wam_drones.vlm.inference import TransformersVLMBackend, infer_image, infer_rgb
from wam_drones.vlm.model_config import VLMModelConfig, load_model_config

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _peak_memory_bytes() -> int | None:
    try:
        torch = import_module("torch")
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return int(torch.cuda.max_memory_allocated())


def _reset_peak_memory() -> None:
    try:
        torch = import_module("torch")
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _write_run(
    output_dir: Path, records: list[tuple[Any, float]], *, semantic_failures: int = 0
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "audit.jsonl").write_text(
        "\n".join(record.model_dump_json() for record, _ in records)
        + ("\n" if records else ""),
        encoding="utf-8",
    )
    latencies = [latency for _, latency in records]
    successful = sum(record.response is not None for record, _ in records)
    metrics = {
        "frames": len(records),
        "parse_successes": successful,
        "parse_success_rate": successful / len(records) if records else None,
        "semantic_failures": semantic_failures,
        "mean_latency_ms": sum(latencies) / len(latencies) if latencies else None,
        "max_latency_ms": max(latencies, default=None),
        "peak_memory_bytes": _peak_memory_bytes(),
        "preprocessing": [
            record.preprocessing.model_dump(mode="json") for record, _ in records
        ],
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )


def _infer_video(
    args: argparse.Namespace, backend: TransformersVLMBackend
) -> list[tuple[Any, float]]:
    try:
        cv2 = import_module("cv2")
        from PIL import Image
    except ImportError as error:
        raise RuntimeError(
            "run `uv sync --group vlm-inference` for video inference"
        ) from error
    capture = cv2.VideoCapture(str(args.source))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {args.source}")
    records: list[tuple[Any, float]] = []
    index = 0
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            if index % args.sample_every == 0:
                rgb_array = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                rgb = Image.fromarray(rgb_array, mode="RGB")
                digest = hashlib.sha256(rgb.tobytes()).hexdigest()
                records.append(
                    infer_rgb(
                        rgb,
                        image_digest=digest,
                        backend=backend,
                        question=args.question,
                        source_frame=SourceFrame(
                            source_id=f"{args.source}#frame-{index + 1}",
                            sequence_id=str(args.source),
                            frame_number=index + 1,
                        ),
                        prompt_revision=args.prompt_revision,
                    )
                )
                if args.frame_limit is not None and len(records) >= args.frame_limit:
                    break
            index += 1
    finally:
        capture.release()
    return records


def run_infer(args: argparse.Namespace, config: VLMModelConfig) -> int:
    _reset_peak_memory()
    backend = TransformersVLMBackend(
        config, device=args.device, decode_mode=args.decode_mode
    )
    if args.source.suffix.lower() in IMAGE_EXTENSIONS:
        records = [
            infer_image(
                args.source,
                backend,
                args.question,
                prompt_revision=args.prompt_revision,
            )
        ]
    elif args.source.suffix.lower() in VIDEO_EXTENSIONS:
        records = _infer_video(args, backend)
    else:
        raise ValueError(f"unsupported source: {args.source}")
    _write_run(args.output_dir, records)
    print(args.output_dir / "audit.jsonl")
    return 0


def run_smoke(args: argparse.Namespace, config: VLMModelConfig) -> int:
    """Run the checked-in task-balanced suite against images supplied by the user."""
    cases = json.loads(args.suite.read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise ValueError("smoke suite must be a JSON array")
    _reset_peak_memory()
    backend = TransformersVLMBackend(
        config, device=args.device, decode_mode=args.decode_mode
    )
    records: list[tuple[Any, float]] = []
    semantic_failures = 0
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("smoke suite cases must be objects")
        image = args.images_dir / str(case["image"])
        record, latency = infer_image(
            image, backend, str(case["question"]), prompt_revision=args.prompt_revision
        )
        records.append((record, latency))
        if (
            record.response is None
            or record.response.type != case["expected_type"]
            or record.response.status != case["expected_status"]
        ):
            semantic_failures += 1
    _write_run(args.output_dir, records, semantic_failures=semantic_failures)
    print(args.output_dir / "metrics.json")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wam-vlm")
    parser.add_argument(
        "--model-config",
        type=Path,
        default=repository_root() / "configs/vlm/models/qwen3_vl_4b.yaml",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    infer = subparsers.add_parser(
        "infer", help="infer one image or sampled video frames"
    )
    infer.add_argument("source", type=Path)
    infer.add_argument("--question", required=True)
    infer.add_argument("--output-dir", type=Path, default=Path("runs/vlm_infer"))
    infer.add_argument("--device", default="auto")
    infer.add_argument(
        "--decode-mode", choices=("unconstrained", "schema"), default="unconstrained"
    )
    infer.add_argument("--prompt-revision", default="vlm-spike-v1")
    infer.add_argument("--sample-every", type=int, default=30)
    infer.add_argument("--frame-limit", type=int)
    smoke = subparsers.add_parser("smoke", help="run the 36-case task-balanced suite")
    smoke.add_argument("--images-dir", type=Path, required=True)
    smoke.add_argument(
        "--suite",
        type=Path,
        default=repository_root() / "configs/vlm/smoke_suite_v1.json",
    )
    smoke.add_argument("--output-dir", type=Path, default=Path("runs/vlm_smoke"))
    smoke.add_argument("--device", default="auto")
    smoke.add_argument(
        "--decode-mode", choices=("unconstrained", "schema"), default="unconstrained"
    )
    smoke.add_argument("--prompt-revision", default="vlm-spike-v1")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "infer" and args.sample_every < 1:
        parser.error("--sample-every must be at least 1")
    config = load_model_config(args.model_config)
    if args.command == "infer":
        return run_infer(args, config)
    return run_smoke(args, config)


if __name__ == "__main__":
    raise SystemExit(main())
