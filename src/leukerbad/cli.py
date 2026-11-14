"""Command-line entry point for Leukerbad vape & waste detection and grasp visualization."""

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from leukerbad.detection.base import Detection, Detector
from leukerbad.detection.visualize import draw_detections
from leukerbad.detection.yolo_seg import YoloSegDetector


@dataclass(frozen=True)
class _SyntheticBoxes:
    xyxy: np.ndarray
    conf: np.ndarray
    cls: np.ndarray

    def __len__(self) -> int:
        return int(self.xyxy.shape[0])


@dataclass(frozen=True)
class _SyntheticMasks:
    xy: list[np.ndarray]


@dataclass(frozen=True)
class _SyntheticResult:
    boxes: _SyntheticBoxes
    masks: _SyntheticMasks
    names: dict[int, str]


class _SyntheticYoloModel:
    """Offline Ultralytics-compatible stub model backing `--synthetic-demo`."""

    def __init__(
        self,
        polygons: list[np.ndarray],
        confidences: list[float],
        class_ids: list[int],
        names: dict[int, str],
    ) -> None:
        self._polygons = polygons
        self._confidences = confidences
        self._class_ids = class_ids
        self._names = names

    def predict(
        self,
        source: Any,
        conf: float = 0.25,
        iou: float = 0.45,
        device: str | None = None,
        verbose: bool = False,
    ) -> list[_SyntheticResult]:
        del source, iou, device, verbose
        kept_xyxy: list[list[float]] = []
        kept_conf: list[float] = []
        kept_cls: list[int] = []
        kept_polys: list[np.ndarray] = []

        for poly, score, cls_id in zip(
            self._polygons, self._confidences, self._class_ids, strict=True
        ):
            if score < conf:
                continue
            x_min, y_min = poly.min(axis=0)
            x_max, y_max = poly.max(axis=0)
            kept_xyxy.append([float(x_min), float(y_min), float(x_max), float(y_max)])
            kept_conf.append(score)
            kept_cls.append(cls_id)
            kept_polys.append(poly)

        xyxy_arr = (
            np.asarray(kept_xyxy, dtype=np.float32)
            if kept_xyxy
            else np.zeros((0, 4), dtype=np.float32)
        )
        boxes = _SyntheticBoxes(
            xyxy=xyxy_arr,
            conf=np.asarray(kept_conf, dtype=np.float32),
            cls=np.asarray(kept_cls, dtype=np.int64),
        )
        masks = _SyntheticMasks(xy=kept_polys)
        return [_SyntheticResult(boxes=boxes, masks=masks, names=self._names)]


def _rotated_capsule_polygon(
    center: tuple[float, float],
    length: float,
    width: float,
    angle_deg: float,
) -> np.ndarray:
    """Create a 12-vertex rounded-end cylinder polygon rotated by `angle_deg`."""
    hl = length / 2.0
    hw = width / 2.0
    chamfer = min(hw * 0.45, hl * 0.15)
    local_pts = np.array(
        [
            [-hl + chamfer, -hw],
            [hl - chamfer, -hw],
            [hl, -hw * 0.55],
            [hl, hw * 0.55],
            [hl - chamfer, hw],
            [-hl + chamfer, hw],
            [-hl, hw * 0.55],
            [-hl, -hw * 0.55],
        ],
        dtype=np.float32,
    )
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    return local_pts @ rot.T + np.asarray(center, dtype=np.float32)


def _create_synthetic_demo_scene() -> tuple[np.ndarray, _SyntheticYoloModel]:
    """Generate an offline synthetic conveyor tabletop frame and matching segmentation model."""
    height, width = 480, 640
    frame = np.full((height, width, 3), (54, 58, 62), dtype=np.uint8)

    for x in range(0, width, 40):
        cv2.line(frame, (x, 0), (x, height), (62, 66, 70), 1)
    for y in range(0, height, 40):
        cv2.line(frame, (0, y), (width, y), (62, 66, 70), 1)

    specs: list[
        tuple[tuple[float, float], float, float, float, tuple[int, int, int], float, int]
    ] = [
        ((195.0, 205.0), 148.0, 30.0, -32.0, (190, 95, 45), 0.94, 0),
        ((445.0, 175.0), 100.0, 26.0, 52.0, (90, 175, 235), 0.89, 1),
        ((335.0, 365.0), 156.0, 16.0, 14.0, (175, 180, 175), 0.82, 2),
    ]
    names = {0: "disposable_vape", 1: "vape_pod", 2: "non_vape_pen"}

    polygons: list[np.ndarray] = []
    confidences: list[float] = []
    class_ids: list[int] = []

    for center, length, obj_w, angle_deg, body_bgr, score, cls_id in specs:
        poly = _rotated_capsule_polygon(center, length, obj_w, angle_deg)
        shadow = np.round(poly + np.array([4.0, 5.0], dtype=np.float32)).astype(np.int32)
        cv2.fillPoly(frame, [shadow.reshape(-1, 1, 2)], (34, 36, 38), lineType=cv2.LINE_AA)

        pts_i32 = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(frame, [pts_i32], body_bgr, lineType=cv2.LINE_AA)
        cv2.polylines(frame, [pts_i32], isClosed=True, color=(30, 30, 30), thickness=1)

        rad = math.radians(angle_deg)
        tip_cx = round(center[0] + math.cos(rad) * (length * 0.42))
        tip_cy = round(center[1] + math.sin(rad) * (length * 0.42))
        cv2.circle(frame, (tip_cx, tip_cy), max(3, int(obj_w * 0.22)), (40, 40, 45), -1)

        polygons.append(poly)
        confidences.append(score)
        class_ids.append(cls_id)

    model = _SyntheticYoloModel(
        polygons=polygons,
        confidences=confidences,
        class_ids=class_ids,
        names=names,
    )
    return frame, model


def build_parser() -> argparse.ArgumentParser:
    """Create the `leukerbad-detect` command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="leukerbad-detect",
        description="Detect discarded vapes/waste and render SO-101 2D grasp overlays.",
    )
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--image", type=Path, help="Path to an input image file.")
    source_group.add_argument("--camera", type=int, help="OpenCV webcam device index (e.g. 0).")
    source_group.add_argument(
        "--synthetic-demo",
        action="store_true",
        help="Run offline on a generated synthetic tabletop scene with rotated vapes.",
    )

    parser.add_argument("--save", type=Path, help="Save the annotated output frame to this path.")
    parser.add_argument(
        "--weights",
        type=str,
        default="yolo11s-seg.pt",
        help="Ultralytics segmentation checkpoint path or name (default: yolo11s-seg.pt).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Minimum confidence threshold (default: 0.25).",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Optional comma-separated whitelist of class labels.",
    )
    parser.add_argument(
        "--mm-per-px",
        type=float,
        default=None,
        help="Optional tabletop scale (mm per pixel) to compute SO-101 jaw opening in mm.",
    )
    parser.add_argument(
        "--verifier",
        choices=["none", "gemini"],
        default="none",
        help="Optional Stage-2 VLM verifier backend (default: none).",
    )
    parser.add_argument(
        "--only-vapes",
        action="store_true",
        default=False,
        help="Filter out non-vape detections.",
    )
    return parser


def _parse_allowed_labels(raw_labels: str | None) -> set[str] | None:
    if raw_labels is None:
        return None
    parsed = {item.strip() for item in raw_labels.split(",") if item.strip()}
    return parsed or None


def _filter_only_vapes(detections: list[Detection]) -> list[Detection]:
    return [
        det
        for det in detections
        if "vape" in det.label.lower()
        and not det.label.lower().startswith(("non_vape", "non-vape"))
    ]


def _print_detection_summary(detections: list[Detection]) -> None:
    print(f"Detected {len(detections)} object(s):")
    for idx, det in enumerate(detections, start=1):
        if det.grasp is not None:
            cx, cy = det.grasp.center_px
            extra = (
                f"center=({cx:.1f}, {cy:.1f})px, "
                f"wrist={det.grasp.gripper_angle_deg:.1f}deg, "
                f"width={det.grasp.width_px:.1f}px"
            )
            if "gripper_opening_mm" in det.metadata:
                extra += f", opening={det.metadata['gripper_opening_mm']:.1f}mm"
        else:
            extra = f"bbox={det.bounding_box}"
        print(f"  [{idx}] {det.label} ({det.confidence:.2f}) | {extra}")


def _save_frame(output_path: Path, frame: np.ndarray) -> bool:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = bool(cv2.imwrite(str(output_path), frame))
    if ok:
        print(f"Saved annotated image to {output_path}")
    else:
        print(f"Error: failed to write image to {output_path}", file=sys.stderr)
    return ok


def _build_detector(
    args: argparse.Namespace,
    injected_model: Any | None = None,
) -> Detector:
    allowed_labels = _parse_allowed_labels(args.labels)
    detector: Detector = YoloSegDetector(
        weights=args.weights,
        conf_threshold=args.conf,
        allowed_labels=allowed_labels,
        mm_per_px=args.mm_per_px,
        model=injected_model,
    )
    if args.verifier == "gemini":
        from leukerbad.detection.vlm_verifier import VlmVerifiedDetector

        detector = VlmVerifiedDetector(detector, reject_non_vapes=args.only_vapes)
    return detector


def _run_camera_loop(detector: Detector, args: argparse.Namespace) -> int:
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: unable to open camera index {args.camera}.", file=sys.stderr)
        return 1

    print("Live camera started — press 'q' to quit, 's' to save snapshot.")
    snapshot_count = 0
    last_annotated: np.ndarray | None = None
    saved_manually = False

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            detections = detector.detect(frame)
            if args.only_vapes and args.verifier == "none":
                detections = _filter_only_vapes(detections)

            annotated = draw_detections(frame, detections)
            last_annotated = annotated
            cv2.imshow("Leukerbad Vape & Waste Detection", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                snapshot_count += 1
                target = args.save or Path(f"leukerbad_snapshot_{snapshot_count:03d}.png")
                _save_frame(target, annotated)
                saved_manually = True
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if (
        args.save is not None
        and last_annotated is not None
        and not saved_manually
        and not _save_frame(args.save, last_annotated)
    ):
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the Leukerbad detection and grasp visualization CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.image is None and args.camera is None and not args.synthetic_demo:
        print(
            "Leukerbad object detection and robotics CLI ready. "
            "Pass --synthetic-demo, --image <path>, or --camera <index> (see --help)."
        )
        return 0

    if args.camera is not None:
        detector = _build_detector(args)
        return _run_camera_loop(detector, args)

    if args.synthetic_demo:
        frame, synthetic_model = _create_synthetic_demo_scene()
        injected = None if Path(args.weights).is_file() else synthetic_model
        detector = _build_detector(args, injected_model=injected)
    else:
        if not args.image.is_file():
            print(f"Error: image file not found: {args.image}", file=sys.stderr)
            return 1
        loaded = cv2.imread(str(args.image))
        if loaded is None:
            print(f"Error: failed to decode image: {args.image}", file=sys.stderr)
            return 1
        frame = loaded
        detector = _build_detector(args)

    detections = detector.detect(frame)
    if args.only_vapes and args.verifier == "none":
        detections = _filter_only_vapes(detections)

    _print_detection_summary(detections)
    annotated = draw_detections(frame, detections)

    if args.save is not None and not _save_frame(args.save, annotated):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
