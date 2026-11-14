"""Webcam dataset capture CLI and frame saving helpers for Dataset A."""

from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

VALID_SPLITS: tuple[str, ...] = ("train", "val", "test")
IMAGE_EXTENSIONS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".webp"})


def _sanitize_tag(tag: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", tag.strip()).strip("_")
    if not cleaned:
        raise ValueError("Capture tag must contain at least one alphanumeric character.")
    return cleaned


def build_capture_filename(split: str, tag: str, index: int, ext: str = ".png") -> str:
    """Build a deterministic split-and-tag-prefixed image filename."""
    if split not in VALID_SPLITS:
        raise ValueError(f"Invalid split {split!r}; expected one of {VALID_SPLITS}.")
    if index < 0:
        raise ValueError(f"Capture index must be non-negative, got {index}.")
    clean_tag = _sanitize_tag(tag)
    clean_ext = ext if ext.startswith(".") else f".{ext}"
    return f"{split}_{clean_tag}_{index:05d}{clean_ext}"


def _next_available_index(images_split_dir: Path, split: str, tag: str, ext: str = ".png") -> int:
    existing = [
        p for p in images_split_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    idx = len(existing)
    while (images_split_dir / build_capture_filename(split, tag, idx, ext=ext)).exists():
        idx += 1
    return idx


def save_labeled_frame(
    frame: np.ndarray,
    output_root: Path,
    split: str,
    tag: str,
    index: int | None = None,
) -> Path:
    """Save an image under `output_root/images/<split>/` and append JSONL capture metadata."""
    if split not in VALID_SPLITS:
        raise ValueError(f"Invalid split {split!r}; expected one of {VALID_SPLITS}.")
    if frame is None or frame.ndim < 2 or frame.size == 0:
        raise ValueError("Cannot save an empty image frame.")

    root = Path(output_root)
    images_split_dir = root / "images" / split
    labels_split_dir = root / "labels" / split
    images_split_dir.mkdir(parents=True, exist_ok=True)
    labels_split_dir.mkdir(parents=True, exist_ok=True)

    resolved_index = (
        _next_available_index(images_split_dir, split, tag) if index is None else index
    )
    filename = build_capture_filename(split=split, tag=tag, index=resolved_index)
    image_path = images_split_dir / filename

    if not cv2.imwrite(str(image_path), frame):
        raise RuntimeError(f"Failed to write captured image to {image_path}")

    height, width = int(frame.shape[0]), int(frame.shape[1])
    record = {
        "timestamp_iso": datetime.now(UTC).isoformat(),
        "split": split,
        "tag": tag,
        "filename": filename,
        "relative_path": str(image_path.relative_to(root)),
        "width": width,
        "height": height,
    }
    metadata_path = root / "capture_metadata.jsonl"
    with metadata_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")

    return image_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive overhead webcam capture tool for Dataset A.",
    )
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index (default: 0).")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/dataset_a_overhead"),
        help="Root directory where images/<split>/ and capture_metadata.jsonl are stored.",
    )
    parser.add_argument(
        "--split",
        choices=list(VALID_SPLITS),
        default="train",
        help="Initial dataset split (train, val, or test).",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="AD_vapes",
        help="Device/scene tag (e.g. AD_mix, E_unseen, neg_airpods_pens).",
    )
    parser.add_argument("--width", type=int, default=1280, help="Requested capture width.")
    parser.add_argument("--height", type=int, default=720, help="Requested capture height.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run live OpenCV preview: Space saves frame, 1/2/3 switches split, q exits."""
    args = _parse_args(argv)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(args.width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(args.height))

    current_split: str = args.split
    saved_count = 0
    window_name = "Leukerbad Dataset A Capture"

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            preview = frame.copy()
            hud = (
                f"Split: {current_split} | Tag: {args.tag} | Saved: {saved_count} | "
                "[SPACE] Save  [1/2/3] train/val/test  [q] Quit"
            )
            cv2.rectangle(preview, (0, 0), (preview.shape[1], 40), (20, 20, 20), -1)
            cv2.putText(
                preview,
                hud,
                (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 180),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, preview)

            key = cv2.waitKey(15) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("1"):
                current_split = "train"
            elif key == ord("2"):
                current_split = "val"
            elif key == ord("3"):
                current_split = "test"
            elif key == ord(" "):
                saved_path = save_labeled_frame(
                    frame=frame,
                    output_root=args.output,
                    split=current_split,
                    tag=args.tag,
                )
                saved_count += 1
                print(f"[saved #{saved_count}] {saved_path}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
