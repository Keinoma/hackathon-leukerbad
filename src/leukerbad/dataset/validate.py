"""Dataset A integrity validator (split hash leakage, unseen device tags, YOLO-seg polygons)."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

SPLITS: tuple[str, ...] = ("train", "val", "test")
IMAGE_EXTENSIONS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".webp"})

# Matches held-out Unseen Vape E tags such as vapeE, vape_E, test_E, E_unseen, unseen_E
_HELD_OUT_DEVICE_RE = re.compile(
    r"(?:^|[_\-])(?:vape[_\-]?e|test[_\-]?e|e[_\-]?unseen|unseen[_\-]?e|design[_\-]?e)"
    r"(?:$|[_\-\.]|\d)",
    re.IGNORECASE,
)


@dataclass
class ValidationReport:
    """Summary of YOLO instance segmentation dataset hygiene checks."""

    is_valid: bool
    split_counts: dict[str, int]
    polygon_count: int
    bbox_only_count: int
    duplicate_hashes: list[str] = field(default_factory=list)
    leaked_device_files: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_held_out_device_filename(filename: str) -> bool:
    return _HELD_OUT_DEVICE_RE.search(filename) is not None


def validate_yolo_seg_dataset(dataset_root: Path) -> ValidationReport:
    """Validate split uniqueness, unseen-device isolation, and YOLO-seg polygon formatting."""
    root = Path(dataset_root)
    split_counts: dict[str, int] = {split: 0 for split in SPLITS}
    polygon_count = 0
    bbox_only_count = 0
    duplicate_hashes: list[str] = []
    leaked_device_files: list[str] = []
    errors: list[str] = []

    if not root.exists():
        errors.append(f"Dataset root does not exist: {root}")
        return ValidationReport(
            is_valid=False,
            split_counts=split_counts,
            polygon_count=0,
            bbox_only_count=0,
            duplicate_hashes=[],
            leaked_device_files=[],
            errors=errors,
        )

    hash_to_locations: dict[str, list[tuple[str, Path]]] = {}

    for split in SPLITS:
        images_dir = root / "images" / split
        labels_dir = root / "labels" / split

        if images_dir.exists():
            image_files = sorted(
                p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )
        else:
            image_files = []

        split_counts[split] = len(image_files)

        for img_path in image_files:
            rel_img = str(img_path.relative_to(root))
            if split in ("train", "val") and _is_held_out_device_filename(img_path.name):
                leaked_device_files.append(rel_img)
                errors.append(f"Held-out unseen device file leaked into {split}: {rel_img}")

            file_hash = _sha256_file(img_path)
            hash_to_locations.setdefault(file_hash, []).append((split, img_path))

        if labels_dir.exists():
            label_files = sorted(p for p in labels_dir.glob("*.txt") if p.is_file())
        else:
            label_files = []

        for lbl_path in label_files:
            rel_lbl = str(lbl_path.relative_to(root))
            if split in ("train", "val") and _is_held_out_device_filename(lbl_path.name):
                if rel_lbl not in leaked_device_files:
                    leaked_device_files.append(rel_lbl)
                errors.append(f"Held-out unseen device label leaked into {split}: {rel_lbl}")

            lines = lbl_path.read_text(encoding="utf-8").splitlines()
            for line_num, raw_line in enumerate(lines, start=1):
                stripped = raw_line.strip()
                if not stripped:
                    continue
                tokens = stripped.split()
                try:
                    class_id = int(tokens[0])
                except ValueError:
                    errors.append(f"{rel_lbl}:{line_num}: invalid integer class ID {tokens[0]!r}")
                    continue

                if class_id < 0:
                    errors.append(f"{rel_lbl}:{line_num}: negative class ID {class_id}")
                    continue

                coord_tokens = tokens[1:]
                try:
                    coords = [float(tok) for tok in coord_tokens]
                except ValueError:
                    errors.append(f"{rel_lbl}:{line_num}: non-numeric coordinate values")
                    continue

                if not all(math.isfinite(val) for val in coords):
                    errors.append(f"{rel_lbl}:{line_num}: non-finite coordinate values")
                    continue

                out_of_bounds = [val for val in coords if val < 0.0 or val > 1.0]
                if out_of_bounds:
                    errors.append(
                        f"{rel_lbl}:{line_num}: coordinates outside [0.0, 1.0]: {out_of_bounds[:3]}"
                    )
                    continue

                if len(coords) == 4:
                    bbox_only_count += 1
                    errors.append(
                        f"{rel_lbl}:{line_num}: 4-value bounding box found instead of polygon mask"
                    )
                    continue

                if len(coords) < 6 or len(coords) % 2 != 0:
                    errors.append(
                        f"{rel_lbl}:{line_num}: polygon requires even coordinate count >= 6, "
                        f"got {len(coords)}"
                    )
                    continue

                polygon_count += 1

    for digest, entries in sorted(hash_to_locations.items()):
        if len(entries) > 1:
            duplicate_hashes.append(digest)
            involved_splits = sorted({s for s, _ in entries})
            rel_paths = [str(p.relative_to(root)) for _, p in entries]
            errors.append(
                f"Duplicate image SHA-256 {digest[:12]} across splits {involved_splits}: "
                f"{rel_paths}"
            )

    if sum(split_counts.values()) == 0:
        errors.append(f"No images found under {root / 'images'}")

    is_valid = len(errors) == 0
    return ValidationReport(
        is_valid=is_valid,
        split_counts=split_counts,
        polygon_count=polygon_count,
        bbox_only_count=bbox_only_count,
        duplicate_hashes=duplicate_hashes,
        leaked_device_files=leaked_device_files,
        errors=errors,
    )


def write_yolo_dataset_yaml(dataset_root: Path, class_names: list[str]) -> Path:
    """Write an Ultralytics YOLO segmentation `data.yaml` configuration file."""
    if not class_names:
        raise ValueError("class_names must contain at least one class name.")

    root = Path(dataset_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    yaml_path = root / "data.yaml"

    lines = [
        f"path: {json.dumps(str(root))}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {len(class_names)}",
        "names:",
    ]
    for idx, name in enumerate(class_names):
        lines.append(f"  {idx}: {json.dumps(str(name))}")

    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path
