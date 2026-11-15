"""SAM 2 bbox-to-polygon label upgrader and 1-click interactive point annotator."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

IMAGE_EXTENSIONS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".webp"})


@dataclass(frozen=True)
class ParsedYoloLine:
    """Parsed representation of either a 4-value YOLO bbox or >=6-value polygon line."""

    kind: Literal["bbox", "polygon"]
    class_id: int
    values: tuple[float, ...]


def parse_yolo_label_line(line: str) -> ParsedYoloLine:
    """Parse a single YOLO label line into either a `bbox` or `polygon` specification."""
    stripped = line.strip()
    if not stripped:
        raise ValueError("Cannot parse an empty YOLO label line.")

    tokens = stripped.split()
    if len(tokens) < 5:
        raise ValueError(f"Expected at least 5 tokens in YOLO label line, got {len(tokens)}.")

    try:
        class_id = int(tokens[0])
    except ValueError as exc:
        raise ValueError(f"Invalid integer class ID {tokens[0]!r}.") from exc

    if class_id < 0:
        raise ValueError(f"Class ID must be non-negative, got {class_id}.")

    try:
        coords = tuple(float(tok) for tok in tokens[1:])
    except ValueError as exc:
        raise ValueError("Non-numeric coordinate in YOLO label line.") from exc

    if not all(math.isfinite(val) for val in coords):
        raise ValueError("YOLO coordinates must be finite numbers.")

    if len(coords) == 4:
        return ParsedYoloLine(kind="bbox", class_id=class_id, values=coords)

    if len(coords) >= 6 and len(coords) % 2 == 0:
        return ParsedYoloLine(kind="polygon", class_id=class_id, values=coords)

    raise ValueError(
        f"Invalid coordinate count {len(coords)}; expected 4 (bbox) or even count >= 6 (polygon)."
    )


def yolo_bbox_to_xyxy_px(
    cx: float,
    cy: float,
    w: float,
    h: float,
    img_w: int,
    img_h: int,
) -> tuple[float, float, float, float]:
    """Convert normalized YOLO `(cx, cy, w, h)` into clamped pixel `(x1, y1, x2, y2)`."""
    if img_w <= 0 or img_h <= 0:
        raise ValueError(f"Image dimensions must be positive, got ({img_w}, {img_h}).")

    half_w = w / 2.0
    half_h = h / 2.0
    x1 = float(np.clip((cx - half_w) * img_w, 0.0, float(img_w)))
    y1 = float(np.clip((cy - half_h) * img_h, 0.0, float(img_h)))
    x2 = float(np.clip((cx + half_w) * img_w, 0.0, float(img_w)))
    y2 = float(np.clip((cy + half_h) * img_h, 0.0, float(img_h)))
    return (x1, y1, x2, y2)


def polygon_px_to_yolo_seg_line(
    class_id: int,
    polygon_px: np.ndarray,
    img_w: int,
    img_h: int,
) -> str:
    """Format a pixel-space `(N, 2)` polygon into a normalized `[0.0, 1.0]` YOLO-seg line."""
    if class_id < 0:
        raise ValueError(f"Class ID must be non-negative, got {class_id}.")
    if img_w <= 0 or img_h <= 0:
        raise ValueError(f"Image dimensions must be positive, got ({img_w}, {img_h}).")

    pts = np.asarray(polygon_px, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 3:
        raise ValueError(f"Polygon requires at least 3 vertices, got {pts.shape[0]}.")

    norm = np.empty_like(pts, dtype=np.float64)
    norm[:, 0] = np.clip(pts[:, 0] / float(img_w), 0.0, 1.0)
    norm[:, 1] = np.clip(pts[:, 1] / float(img_h), 0.0, 1.0)

    coord_str = " ".join(f"{val:.6f}" for val in norm.ravel())
    return f"{class_id} {coord_str}"


def _load_sam_model(sam_weights: str) -> Any:
    from ultralytics import SAM

    return SAM(sam_weights)


def _extract_polygon_from_sam_results(results: Any) -> np.ndarray | None:
    """Extract the largest valid `(N, 2)` pixel polygon from Ultralytics SAM prediction output."""
    if not results:
        return None
    first = results[0]
    masks = getattr(first, "masks", None)
    if masks is None:
        return None

    xy_list = getattr(masks, "xy", None)
    if xy_list:
        valid_polys: list[np.ndarray] = []
        for poly in xy_list:
            arr = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
            if arr.shape[0] >= 3:
                valid_polys.append(arr)
        if valid_polys:
            return max(valid_polys, key=lambda p: float(cv2.contourArea(p)))

    raw_data = getattr(masks, "data", None)
    if raw_data is not None:
        if hasattr(raw_data, "cpu"):
            mask_arr = raw_data.cpu().numpy()
        else:
            mask_arr = np.asarray(raw_data)
        if mask_arr.ndim == 3 and mask_arr.shape[0] > 0:
            binary = (mask_arr[0] > 0.5).astype(np.uint8) * 255
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid = [c.reshape(-1, 2).astype(np.float32) for c in contours if len(c) >= 3]
            if valid:
                return max(valid, key=lambda p: float(cv2.contourArea(p)))

    return None


def _extract_polygons_for_bboxes(results: Any, expected_count: int) -> list[np.ndarray | None]:
    """Extract one `(N, 2)` pixel polygon per prompted bbox from Ultralytics SAM output."""
    if expected_count <= 0:
        return []
    if expected_count == 1:
        return [_extract_polygon_from_sam_results(results)]
    if not results:
        return [None] * expected_count

    first = results[0]
    masks = getattr(first, "masks", None)
    if masks is None:
        return [None] * expected_count

    xy_list = getattr(masks, "xy", None)
    if xy_list is not None and len(xy_list) == expected_count:
        extracted: list[np.ndarray | None] = []
        for poly in xy_list:
            arr = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
            extracted.append(arr if arr.shape[0] >= 3 else None)
        return extracted

    raw_data = getattr(masks, "data", None)
    if raw_data is not None:
        mask_arr = raw_data.cpu().numpy() if hasattr(raw_data, "cpu") else np.asarray(raw_data)
        if mask_arr.ndim == 3 and mask_arr.shape[0] == expected_count:
            out: list[np.ndarray | None] = []
            for idx in range(expected_count):
                binary = (mask_arr[idx] > 0.5).astype(np.uint8) * 255
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                valid = [c.reshape(-1, 2).astype(np.float32) for c in contours if len(c) >= 3]
                out.append(max(valid, key=lambda p: float(cv2.contourArea(p))) if valid else None)
            return out

    return [None] * expected_count


def upgrade_bbox_labels_with_sam(
    images_dir: Path,
    labels_dir: Path,
    output_labels_dir: Path,
    sam_weights: str = "sam2.1_b.pt",
    sam_model: Any | None = None,
    log_every: int = 250,
) -> dict[str, int]:
    """Upgrade 4-number YOLO bboxes to tight SAM instance polygons, keeping existing polygons."""
    img_root = Path(images_dir)
    lbl_root = Path(labels_dir)
    out_root = Path(output_labels_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    stats: dict[str, int] = {
        "processed_images": 0,
        "upgraded_bboxes": 0,
        "kept_polygons": 0,
        "fallback_boxes": 0,
        "background_images": 0,
    }

    if not img_root.exists():
        raise FileNotFoundError(f"Images directory does not exist: {img_root}")

    image_files = sorted(
        p for p in img_root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    total_images = len(image_files)
    model = sam_model

    for img_idx, img_path in enumerate(image_files, start=1):
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        img_h, img_w = int(image.shape[0]), int(image.shape[1])
        stats["processed_images"] += 1

        src_lbl_path = lbl_root / f"{img_path.stem}.txt"
        dst_lbl_path = out_root / f"{img_path.stem}.txt"

        if not src_lbl_path.exists():
            stats["background_images"] += 1
            continue

        raw_lines = [ln.strip() for ln in src_lbl_path.read_text(encoding="utf-8").splitlines()]
        non_empty_lines = [ln for ln in raw_lines if ln]
        if not non_empty_lines:
            dst_lbl_path.write_text("", encoding="utf-8")
            stats["background_images"] += 1
            continue

        parsed_items = [parse_yolo_label_line(raw_line) for raw_line in non_empty_lines]
        out_slots: list[str | None] = [None] * len(parsed_items)
        pending_bbox_indices: list[int] = []
        pending_bboxes_px: list[list[float]] = []

        for slot_idx, parsed in enumerate(parsed_items):
            if parsed.kind == "polygon":
                clamped = [min(max(val, 0.0), 1.0) for val in parsed.values]
                coord_str = " ".join(f"{val:.6f}" for val in clamped)
                out_slots[slot_idx] = f"{parsed.class_id} {coord_str}"
                stats["kept_polygons"] += 1
            else:
                cx, cy, bw, bh = parsed.values
                x1, y1, x2, y2 = yolo_bbox_to_xyxy_px(cx, cy, bw, bh, img_w, img_h)
                pending_bbox_indices.append(slot_idx)
                pending_bboxes_px.append([x1, y1, x2, y2])

        if pending_bboxes_px:
            if model is None:
                model = _load_sam_model(sam_weights)
            sam_results = model.predict(image, bboxes=pending_bboxes_px, verbose=False)
            polys = _extract_polygons_for_bboxes(sam_results, len(pending_bboxes_px))

            if len(pending_bboxes_px) > 1 and all(p is None for p in polys):
                polys = [
                    _extract_polygon_from_sam_results(
                        model.predict(image, bboxes=[bbox_px], verbose=False)
                    )
                    for bbox_px in pending_bboxes_px
                ]

            for slot_idx, bbox_px, poly_px in zip(
                pending_bbox_indices, pending_bboxes_px, polys, strict=True
            ):
                cls_id = parsed_items[slot_idx].class_id
                if poly_px is not None and poly_px.shape[0] >= 3:
                    out_slots[slot_idx] = polygon_px_to_yolo_seg_line(cls_id, poly_px, img_w, img_h)
                    stats["upgraded_bboxes"] += 1
                else:
                    x1, y1, x2, y2 = bbox_px
                    box_poly = np.array(
                        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                        dtype=np.float32,
                    )
                    out_slots[slot_idx] = polygon_px_to_yolo_seg_line(
                        cls_id, box_poly, img_w, img_h
                    )
                    stats["fallback_boxes"] += 1

        out_lines = [line for line in out_slots if line is not None]
        dst_lbl_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

        if log_every > 0 and img_idx % log_every == 0:
            print(
                f"[SAM 2] {img_idx}/{total_images} images | "
                f"upgraded_bboxes={stats['upgraded_bboxes']} background={stats['background_images']}"
            )

    return stats


def is_class_key_allowed(cls_id: int, num_classes: int) -> bool:
    """Return True if interactive class key `cls_id` is permitted under `num_classes`."""
    if num_classes < 1:
        raise ValueError(f"num_classes must be >= 1, got {num_classes}.")
    return 0 <= cls_id < num_classes


def _format_class_hud_keys(num_classes: int) -> str:
    if num_classes < 1:
        raise ValueError(f"num_classes must be >= 1, got {num_classes}.")
    if num_classes == 1:
        return "[0] vape"
    return "[0] vape [1] non_vape"


def run_interactive_annotator(
    images_dir: Path,
    labels_dir: Path,
    sam_weights: str = "sam2.1_b.pt",
    sam_model: Any | None = None,
    num_classes: int = 1,
) -> int:
    """Interactive 1-click SAM point annotator: click object -> 0=vape (and 1=non_vape if nc>=2)."""
    if num_classes < 1:
        raise ValueError(f"num_classes must be >= 1, got {num_classes}.")

    img_root = Path(images_dir)
    lbl_root = Path(labels_dir)
    lbl_root.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        p for p in img_root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_files:
        raise FileNotFoundError(f"No images found in {img_root}")

    model = sam_model if sam_model is not None else _load_sam_model(sam_weights)
    window_name = "Leukerbad 1-Click SAM Annotator"
    class_keys_hud = _format_class_hud_keys(num_classes)

    pending_poly_px: np.ndarray | None = None
    click_xy: tuple[int, int] | None = None
    current_image: np.ndarray | None = None

    def _on_mouse(event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        nonlocal pending_poly_px, click_xy
        if event == cv2.EVENT_LBUTTONDOWN and current_image is not None:
            click_xy = (x, y)
            res = model.predict(current_image, points=[[x, y]], labels=[1], verbose=False)
            pending_poly_px = _extract_polygon_from_sam_results(res)

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, _on_mouse)

    idx = 0
    try:
        while 0 <= idx < len(image_files):
            img_path = image_files[idx]
            loaded = cv2.imread(str(img_path))
            if loaded is None:
                idx += 1
                continue
            current_image = loaded
            img_h, img_w = int(loaded.shape[0]), int(loaded.shape[1])
            lbl_path = lbl_root / f"{img_path.stem}.txt"

            saved_lines: list[str] = []
            if lbl_path.exists():
                saved_lines = [
                    ln.strip()
                    for ln in lbl_path.read_text(encoding="utf-8").splitlines()
                    if ln.strip()
                ]
            pending_poly_px = None
            click_xy = None
            warning_banner = ""

            while True:
                canvas = loaded.copy()
                if pending_poly_px is not None and len(pending_poly_px) >= 3:
                    overlay = canvas.copy()
                    pts_int = np.round(pending_poly_px).astype(np.int32).reshape(-1, 1, 2)
                    cv2.fillPoly(overlay, [pts_int], (0, 220, 100))
                    cv2.addWeighted(overlay, 0.4, canvas, 0.6, 0.0, dst=canvas)
                    cv2.polylines(canvas, [pts_int], True, (0, 255, 120), 2)
                if click_xy is not None:
                    cv2.circle(canvas, click_xy, 5, (0, 80, 255), -1)

                hud = (
                    f"[{idx + 1}/{len(image_files)}] {img_path.name} (saved={len(saved_lines)}) | "
                    f"Click->SAM | {class_keys_hud} [b] empty_bg [u] undo [n/p] nav [q] quit"
                )
                cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 38), (20, 20, 20), -1)
                cv2.putText(
                    canvas,
                    hud,
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 240, 120),
                    1,
                    cv2.LINE_AA,
                )
                if warning_banner:
                    cv2.rectangle(canvas, (0, 38), (canvas.shape[1], 68), (20, 20, 120), -1)
                    cv2.putText(
                        canvas,
                        warning_banner,
                        (10, 58),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (120, 200, 255),
                        1,
                        cv2.LINE_AA,
                    )
                cv2.imshow(window_name, canvas)

                key = cv2.waitKey(20) & 0xFF
                if key in (ord("q"), 27):
                    return 0
                if key in (ord("0"), ord("1")) and pending_poly_px is not None:
                    cls_id = 0 if key == ord("0") else 1
                    if not is_class_key_allowed(cls_id, num_classes):
                        warning_banner = (
                            f"Warning: key [{cls_id}] refused (num_classes={num_classes}); "
                            "use --num-classes 2 for non_vape."
                        )
                        print(warning_banner)
                        continue
                    warning_banner = ""
                    line = polygon_px_to_yolo_seg_line(cls_id, pending_poly_px, img_w, img_h)
                    saved_lines.append(line)
                    lbl_path.write_text("\n".join(saved_lines) + "\n", encoding="utf-8")
                    pending_poly_px = None
                    click_xy = None
                elif key == ord("b"):
                    warning_banner = ""
                    saved_lines = []
                    lbl_path.write_text("", encoding="utf-8")
                    pending_poly_px = None
                elif key == ord("u") and saved_lines:
                    warning_banner = ""
                    saved_lines.pop()
                    content = "\n".join(saved_lines) + ("\n" if saved_lines else "")
                    lbl_path.write_text(content, encoding="utf-8")
                elif key in (ord("n"), ord(" ")):
                    idx += 1
                    break
                elif key == ord("p"):
                    idx = max(0, idx - 1)
                    break
    finally:
        cv2.destroyAllWindows()

    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SAM 2 YOLO bbox-to-polygon upgrader and 1-click point annotator.",
    )
    parser.add_argument("--images", type=Path, required=True, help="Directory of input images.")
    parser.add_argument("--labels", type=Path, required=True, help="Directory of YOLO .txt labels.")
    parser.add_argument(
        "--output-labels",
        type=Path,
        default=None,
        help="Output labels directory for batch bbox upgrade (defaults to --labels in-place).",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="sam2.1_b.pt",
        help="Ultralytics SAM weights checkpoint (default: sam2.1_b.pt).",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch interactive 1-click OpenCV point annotator (0=vape, 1=non_vape with nc=2).",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=1,
        help=(
            "Number of classes for interactive annotation (default: 1 = vape only, "
            "lookalikes left as background; 2 = allow key 1 for non_vape)."
        ),
    )
    args = parser.parse_args(argv)
    if args.num_classes < 1:
        parser.error("--num-classes must be >= 1.")
    return args


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for interactive point annotation or batch bbox->polygon conversion."""
    args = _parse_args(argv)
    if args.interactive:
        return run_interactive_annotator(
            images_dir=args.images,
            labels_dir=args.labels,
            sam_weights=args.weights,
            num_classes=args.num_classes,
        )

    out_dir = args.output_labels if args.output_labels is not None else args.labels
    stats = upgrade_bbox_labels_with_sam(
        images_dir=args.images,
        labels_dir=args.labels,
        output_labels_dir=out_dir,
        sam_weights=args.weights,
    )
    print(f"SAM label upgrade complete: {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
