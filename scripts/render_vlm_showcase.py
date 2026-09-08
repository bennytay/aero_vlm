#!/usr/bin/env python3
"""Run honest single-frame VLM demo cases and render a simple still showcase.

The script deliberately launches the existing ``wam-vlm infer`` implementation
for every case. It does not call a detector or tracker and never changes a
model response before retaining or displaying it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

REQUIRED_KINDS = frozenset({"caption", "answer", "point", "abstention"})
DISPLAY_ORDER = (0, 2, 1)
VIDEO_CASES = (
    (
        0,
        "city_analysis",
        "Describe the traffic flow around the roundabout and identify areas where "
        "vehicles and pedestrians may conflict. This is for a human operator's "
        "situational-awareness briefing.",
    ),
    (
        1,
        "waterfront_analysis",
        "Describe the boat activity and shoreline conditions that a human operator "
        "should note during a coastal monitoring patrol.",
    ),
    (
        2,
        "neighborhood_analysis",
        "Describe the street layout, dense vegetation, and prominent landmarks that "
        "could help orient a ground response team approaching this neighborhood.",
    ),
    (
        2,
        "white_roof_point",
        "Point to the large white-roofed building that a ground response team could "
        "use as a visual reference.",
    ),
)


def load_manifest(
    path: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Load and validate the deliberately small showcase manifest."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON manifest: {path}") from error
    if (
        not isinstance(payload, dict)
        or payload.get("manifest_version") != "vlm_demo_source_v1"
    ):
        raise ValueError("manifest_version must be 'vlm_demo_source_v1'")
    sources = payload.get("sources")
    cases = payload.get("cases")
    if not isinstance(sources, list) or not isinstance(cases, list):
        raise ValueError("manifest requires 'sources' and 'cases' arrays")
    source_by_id: dict[str, dict[str, Any]] = {}
    required_source_fields = {
        "id",
        "url",
        "license",
        "downloaded_on",
        "permitted_use",
        "sha256",
    }
    for source in sources:
        if not isinstance(source, dict) or not required_source_fields <= source.keys():
            raise ValueError(
                "each source needs id, url, license, downloaded_on, permitted_use, "
                "sha256"
            )
        source_id = source["id"]
        if not isinstance(source_id, str) or not source_id or source_id in source_by_id:
            raise ValueError("source ids must be unique non-empty strings")
        source_hash = source["sha256"]
        if not isinstance(source_hash, str) or len(source_hash) != 64:
            raise ValueError("source sha256 must be a 64-character string")
        source_by_id[source_id] = source
    case_kinds: set[str] = set()
    case_ids: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("each case must be an object")
        required_case_fields = {"id", "kind", "source_id", "image", "question"}
        if not required_case_fields <= case.keys():
            raise ValueError("each case needs id, kind, source_id, image, question")
        case_id, kind = case["id"], case["kind"]
        if not isinstance(case_id, str) or not case_id or case_id in case_ids:
            raise ValueError("case ids must be unique non-empty strings")
        if not isinstance(kind, str):
            raise ValueError("case kinds must be strings")
        if kind not in REQUIRED_KINDS:
            raise ValueError(f"unsupported case kind: {kind!r}")
        source_id = case["source_id"]
        if not isinstance(source_id, str) or source_id not in source_by_id:
            raise ValueError(f"case {case_id!r} references an unknown source")
        if not isinstance(case["image"], str) or not isinstance(case["question"], str):
            raise ValueError("case image and question must be strings")
        question = case["question"].lower().strip()
        if kind == "caption" and not question.startswith("describe"):
            raise ValueError(
                "caption cases must use a question beginning with 'Describe'"
            )
        if kind in {"point", "abstention"} and not question.startswith("point to"):
            raise ValueError(
                "point and abstention cases must use a question beginning with "
                "'Point to'"
            )
        case_ids.add(case_id)
        case_kinds.add(kind)
    if case_kinds != REQUIRED_KINDS:
        raise ValueError(
            f"manifest must have exactly these case kinds: {sorted(REQUIRED_KINDS)}"
        )
    if len(cases) != len(REQUIRED_KINDS):
        raise ValueError("the MVP manifest must contain exactly four cases")
    return cases, source_by_id


def run_case(args: argparse.Namespace, image: Path, case_dir: Path) -> dict[str, Any]:
    """Execute the production CLI and return its single immutable audit row."""
    command = [
        sys.executable,
        "-m",
        "wam_drones.vlm.cli",
        "--model-config",
        str(args.model_config),
        "infer",
        str(image),
        "--question",
        args.question,
        "--output-dir",
        str(case_dir),
        "--device",
        args.device,
        "--decode-mode",
        args.decode_mode,
    ]
    subprocess.run(command, check=True)
    rows = case_dir.joinpath("audit.jsonl").read_text(encoding="utf-8").splitlines()
    if len(rows) != 1:
        raise RuntimeError(
            f"expected exactly one audit row for {image}, found {len(rows)}"
        )
    record = json.loads(rows[0])
    if not isinstance(record, dict):
        raise RuntimeError("CLI audit row must be a JSON object")
    return record


def _font(size: int) -> Any:
    from PIL import ImageFont

    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _gloss(record: dict[str, Any]) -> str:
    response = record.get("response")
    if not isinstance(response, dict):
        return "Model response did not validate; raw output is shown."
    status = response.get("status")
    if status not in {"ok", "found"}:
        return f"The model abstained: {status}."
    if response.get("type") == "caption":
        return str(response.get("text"))
    if response.get("type") == "point":
        return "The marker shows the returned target location."
    answer = response.get("answer")
    return str(answer.get("value")) if isinstance(answer, dict) else "Validated answer."


def is_abstention(record: dict[str, Any]) -> bool:
    """Return whether a validated response declined a caption/answer/point."""
    response = record.get("response")
    if not isinstance(response, dict):
        return False
    return response.get("status") not in {"ok", "found"}


def render_case(
    image_path: Path,
    case: dict[str, Any],
    record: dict[str, Any],
    output: Path,
) -> None:
    """Render source RGB, question, raw validated JSON, and one plain gloss."""
    from PIL import Image, ImageDraw

    canvas = Image.new("RGB", (1600, 900), "#101418")
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    image.thumbnail((900, 780))
    image_x = (900 - image.width) // 2
    image_y = 60 + (780 - image.height) // 2
    canvas.paste(image, (image_x, image_y))
    draw = ImageDraw.Draw(canvas)
    title_font, body_font, small_font = _font(30), _font(20), _font(16)
    draw.text(
        (30, 18),
        "Offline research — not flight control",
        fill="#f3f5f7",
        font=title_font,
    )
    response = record.get("response")
    if (
        isinstance(response, dict)
        and response.get("type") == "point"
        and response.get("status") == "found"
    ):
        point = response.get("point")
        if isinstance(point, dict):
            x = image_x + round(int(point["x"]) * (image.width - 1) / 1000)
            y = image_y + round(int(point["y"]) * (image.height - 1) / 1000)
            draw.ellipse((x - 12, y - 12, x + 12, y + 12), outline="#ff4d4f", width=5)
            draw.line((x - 19, y, x + 19, y), fill="#ff4d4f", width=3)
            draw.line((x, y - 19, x, y + 19), fill="#ff4d4f", width=3)
    panel_x = 940
    draw.text(
        (panel_x, 70),
        f"Question\n{case['question']}",
        fill="#ffffff",
        font=body_font,
        spacing=8,
    )
    display_response = response
    if display_response is None:
        display_response = {"raw_generation": record.get("raw_generation")}
    raw = json.dumps(display_response, indent=2)
    draw.multiline_text((panel_x, 180), raw, fill="#9fdfb0", font=small_font, spacing=6)
    draw.multiline_text(
        (panel_x, 650),
        f"Plain-language gloss\n{_gloss(record)}",
        fill="#ffffff",
        font=body_font,
        spacing=8,
    )
    draw.text(
        (panel_x, 820),
        f"Case: {case['id']}  •  source: {case['source_id']}",
        fill="#abb6c2",
        font=small_font,
    )
    canvas.save(output)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_video_frame(video: Path, output: Path, *, at_seconds: float) -> None:
    import cv2

    capture = cv2.VideoCapture(str(video))
    try:
        capture.set(cv2.CAP_PROP_POS_MSEC, at_seconds * 1000)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        raise ValueError(f"could not decode a frame at {at_seconds}s from {video}")
    if not cv2.imwrite(str(output), frame):
        raise RuntimeError(f"could not write extracted frame: {output}")


def _natural_response(record: dict[str, Any]) -> str:
    """Turn a validated contract response into the only response shown on video."""
    response = record.get("response")
    if not isinstance(response, dict):
        return "I couldn't return a validated answer for this frame."
    status = response.get("status")
    if status not in {"ok", "found"}:
        if status == "no_candidate":
            return "I couldn't find that in this frame."
        if status == "ambiguous":
            return "There are several possible matches, so I won't guess."
        if status == "not_visible":
            return "I can't see enough of the relevant area to answer."
        return "I can't determine that from this frame."
    if response.get("type") == "caption":
        return str(response.get("text"))
    if response.get("type") == "point":
        return "The requested building is highlighted by the arrow."
    answer = response.get("answer")
    if not isinstance(answer, dict):
        return "I couldn't return a validated answer for this frame."
    value = answer.get("value")
    if answer.get("kind") == "boolean":
        return "Yes." if value else "No."
    if answer.get("kind") == "count":
        return f"I can see {value}."
    return str(value)


def _wrap(text: str, *, width: int = 42) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    return lines + ([current] if current else [])


def _draw_panel(
    frame: Any,
    *,
    question: str,
    response: str | None,
    record: dict[str, Any],
    show_marker: bool,
) -> Any:
    import cv2

    height, width = frame.shape[:2]
    model_response = record.get("response")
    point_response = (
        isinstance(model_response, dict) and model_response.get("status") == "found"
    )
    canvas = frame.copy()
    panel_top, panel_bottom = 42, height - 42
    if point_response:
        panel_left, panel_right = 42, round(width * 0.43)
    else:
        panel_left, panel_right = round(width * 0.57), width - 42
    overlay = canvas.copy()
    cv2.rectangle(
        overlay,
        (panel_left, panel_top),
        (panel_right, panel_bottom),
        (18, 24, 28),
        thickness=-1,
    )
    cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, canvas)
    text_x = panel_left + 30
    cv2.putText(
        canvas,
        "OFFLINE RESEARCH - NOT FLIGHT CONTROL",
        (text_x, panel_top + 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    y = panel_top + 108
    cv2.putText(
        canvas,
        "QUERY",
        (text_x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (113, 192, 255),
        2,
    )
    for line in _wrap(question, width=39):
        y += 36
        cv2.putText(
            canvas,
            line,
            (text_x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
        )
    if response is not None:
        y += 80
        cv2.putText(
            canvas,
            "RESPONSE",
            (text_x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (126, 225, 163),
            2,
        )
        response_lines = _wrap(response, width=39)
        line_height = min(36, max(24, (panel_bottom - y - 30) // len(response_lines)))
        font_scale = max(0.46, min(0.66, line_height * 0.018))
        for line in response_lines:
            y += line_height
            cv2.putText(
                canvas,
                line,
                (text_x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                2,
            )
    if point_response and show_marker:
        point = model_response.get("point")
        if isinstance(point, dict):
            x = round(int(point["x"]) * (width - 1) / 1000)
            y = round(int(point["y"]) * (height - 1) / 1000)
            arrow_start = (max(42, x - 230), max(42, y - 190))
            cv2.arrowedLine(
                canvas,
                arrow_start,
                (x, y),
                (50, 50, 255),
                thickness=8,
                tipLength=0.16,
            )
            cv2.circle(canvas, (x, y), 31, (50, 50, 255), 6)
            cv2.putText(
                canvas,
                "TARGET",
                (arrow_start[0], max(35, arrow_start[1] - 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (50, 50, 255),
                2,
                cv2.LINE_AA,
            )
    return canvas


def _render_video_segment(
    video: Path,
    writer: Any,
    *,
    seconds: float,
    questions: list[tuple[str, dict[str, Any]]],
    static_second_question_frame: Path | None = None,
) -> None:
    import cv2

    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        capture.release()
        raise ValueError(f"could not read frame rate from {video}")
    target_frames = round(seconds * 30)
    source_index = 0
    frozen_frame = None
    if static_second_question_frame is not None:
        frozen_frame = cv2.imread(str(static_second_question_frame))
        if frozen_frame is None:
            capture.release()
            raise ValueError(
                f"could not decode pointing frame: {static_second_question_frame}"
            )
        frozen_frame = cv2.resize(frozen_frame, (1920, 1080))
    try:
        for output_index in range(target_frames):
            source_position = output_index / 30 * fps
            while source_index <= round(source_position):
                ok, source = capture.read()
                if not ok:
                    raise ValueError(f"{video} is shorter than {seconds} seconds")
                source_index += 1
            source = cv2.resize(source, (1920, 1080))
            phase = output_index / target_frames
            question_index = 0 if len(questions) == 1 or phase < 0.70 else 1
            local_phase = phase if question_index == 0 else (phase - 0.70) / 0.30
            if question_index == 1 and frozen_frame is not None:
                source = frozen_frame
            question, record = questions[question_index]
            response = None
            response_starts, response_complete = 0.12, 0.55
            if local_phase >= response_starts:
                full_response = _natural_response(record)
                progress = min(
                    1.0,
                    (local_phase - response_starts)
                    / (response_complete - response_starts),
                )
                words = full_response.split()
                visible_words = max(1, round(len(words) * progress))
                response = " ".join(words[:visible_words])
                if progress < 1:
                    response += " |"
            writer.write(
                _draw_panel(
                    source,
                    question=question,
                    response=response,
                    record=record,
                    show_marker=response is not None,
                )
            )
    finally:
        capture.release()


def run_video_showcase(args: argparse.Namespace) -> int:
    import cv2

    videos = [video.resolve() for video in args.videos]
    if len(videos) != 3 or any(not video.is_file() for video in videos):
        raise ValueError("--videos needs exactly three existing video files")
    if args.output_dir.exists():
        raise ValueError(f"output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    frames_dir = args.output_dir / "source_frames"
    frames_dir.mkdir()
    records: list[str] = []
    queries_by_video: dict[int, list[tuple[str, dict[str, Any]]]] = {
        0: [],
        1: [],
        2: [],
    }
    for video_index, case_id, question in VIDEO_CASES:
        frame = frames_dir / f"{case_id}.jpg"
        _extract_video_frame(videos[video_index], frame, at_seconds=2.5)
        args.question = question
        record = run_case(args, frame, args.output_dir / "cli_audits" / case_id)
        records.append(json.dumps(record, separators=(",", ":")))
        queries_by_video[video_index].append((question, record))
    args.output_dir.joinpath("predictions.jsonl").write_text(
        "\n".join(records) + "\n", encoding="utf-8"
    )
    provenance = {
        "source_type": "user-supplied local media",
        "permitted_use": "demo use authorized by workspace user",
        "clips": [
            {
                "path": str(video),
                "sha256": _file_sha256(video),
                "used_seconds": args.clip_seconds,
            }
            for video in videos
        ],
    }
    args.output_dir.joinpath("source_manifest.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    output = args.output_dir / "showcase.mp4"
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1920, 1080)
    )
    if not writer.isOpened():
        raise RuntimeError("could not create MP4 video writer")
    try:
        for index in DISPLAY_ORDER:
            pointing_frame = (
                frames_dir / "white_roof_point.jpg"
                if len(queries_by_video[index]) > 1
                else None
            )
            _render_video_segment(
                videos[index],
                writer,
                seconds=args.clip_seconds,
                questions=queries_by_video[index],
                static_second_question_frame=pointing_frame,
            )
    finally:
        writer.release()
    print(output)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--videos", type=Path, nargs=3)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/vlm/models/qwen3_vl_2b.yaml"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--decode-mode",
        choices=("unconstrained", "schema"),
        default="schema",
    )
    parser.add_argument("--clip-seconds", type=float, default=10)
    args = parser.parse_args()
    if (args.manifest is None) == (args.videos is None):
        parser.error("provide exactly one of --manifest or --videos")
    if args.videos is not None:
        if args.clip_seconds <= 0:
            parser.error("--clip-seconds must be greater than zero")
        return run_video_showcase(args)
    assert args.manifest is not None
    cases, _ = load_manifest(args.manifest)
    manifest_dir = args.manifest.parent.resolve()
    if args.output_dir.exists():
        raise ValueError(f"output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    shutil.copy2(args.manifest, args.output_dir / "source_manifest.json")
    records: list[str] = []
    abstention_seen = False
    for case in cases:
        image = (manifest_dir / case["image"]).resolve()
        if not image.is_file() or manifest_dir not in image.parents:
            raise ValueError(
                "case image must be a file inside the manifest directory: "
                f"{case['image']}"
            )
        args.question = case["question"]
        record = run_case(args, image, args.output_dir / "cli_audits" / case["id"])
        records.append(json.dumps(record, separators=(",", ":")))
        render_case(image, case, record, args.output_dir / f"{case['id']}.png")
        abstention_seen = abstention_seen or (
            case["kind"] == "abstention" and is_abstention(record)
        )
    args.output_dir.joinpath("predictions.jsonl").write_text(
        "\n".join(records) + "\n", encoding="utf-8"
    )
    print(args.output_dir / "predictions.jsonl")
    if not abstention_seen:
        print(
            "The abstention case did not produce an abstention; inspect the rendered "
            "result and select a different honest case.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
