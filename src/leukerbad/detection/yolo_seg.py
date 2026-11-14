"""Ultralytics YOLO11 instance segmentation detector with analytical grasp extraction."""

from pathlib import Path
from typing import Any

import numpy as np

from leukerbad.detection.base import Detection
from leukerbad.detection.geometry import compute_grasp_from_polygon, estimate_gripper_opening_mm


def select_best_device(preferred: str | None = None) -> str:
    """Select the best compute device for inference (`mps`, `cuda`, or `cpu`)."""
    if preferred is not None and preferred.strip():
        return preferred.strip()

    try:
        import torch

        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except (ImportError, RuntimeError, AttributeError):
        return "cpu"

    return "cpu"


def _to_numpy(value: Any) -> np.ndarray:
    """Convert a PyTorch tensor, list, or array-like object to a NumPy ndarray."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _bbox_to_polygon(bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Construct a 4-corner float32 polygon from `(x1, y1, x2, y2)`."""
    x1, y1, x2, y2 = bbox
    return np.array(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        dtype=np.float32,
    )


class YoloSegDetector:
    """YOLO instance-segmentation detector producing mask polygons and 2D grasp poses."""

    def __init__(
        self,
        weights: str | Path = "yolo11s-seg.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        allowed_labels: set[str] | None = None,
        device: str | None = None,
        mm_per_px: float | None = None,
        model: Any | None = None,
    ) -> None:
        self.weights = weights
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.allowed_labels = (
            {label.strip().lower() for label in allowed_labels if label.strip()}
            if allowed_labels is not None
            else None
        )
        self.device = select_best_device(device)
        self.mm_per_px = mm_per_px
        self._model: Any | None = model

    def _get_model(self) -> Any:
        """Lazily load the Ultralytics YOLO model on first inference."""
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(str(self.weights))
        return self._model

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run segmentation inference on a BGR image frame and extract grasp candidates."""
        if (
            frame is None
            or not isinstance(frame, np.ndarray)
            or frame.size == 0
            or frame.ndim < 2
            or frame.shape[0] == 0
            or frame.shape[1] == 0
        ):
            return []

        model = self._get_model()
        results = model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []

        detections: list[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue

            xyxy_arr = _to_numpy(boxes.xyxy).reshape(-1, 4)
            conf_arr = _to_numpy(boxes.conf).reshape(-1)
            cls_arr = _to_numpy(boxes.cls).reshape(-1)
            names = getattr(result, "names", {}) or {}

            masks = getattr(result, "masks", None)
            mask_polygons = getattr(masks, "xy", None) if masks is not None else None

            for idx in range(len(xyxy_arr)):
                cls_id = int(cls_arr[idx])
                if isinstance(names, dict):
                    label = str(names.get(cls_id, str(cls_id)))
                else:
                    label = str(names[cls_id])

                if self.allowed_labels is not None and label.lower() not in self.allowed_labels:
                    continue

                confidence = float(conf_arr[idx])
                x1, y1, x2, y2 = (round(float(coord)) for coord in xyxy_arr[idx])
                bounding_box: tuple[int, int, int, int] = (x1, y1, x2, y2)

                polygon: np.ndarray | None = None
                if (
                    mask_polygons is not None
                    and idx < len(mask_polygons)
                    and mask_polygons[idx] is not None
                ):
                    candidate = np.asarray(_to_numpy(mask_polygons[idx]), dtype=np.float32)
                    if candidate.size >= 6:
                        reshaped = candidate.reshape(-1, 2)
                        if reshaped.shape[0] >= 3:
                            polygon = reshaped

                if polygon is not None:
                    grasp = compute_grasp_from_polygon(polygon)
                    if grasp is None:
                        fallback_poly = _bbox_to_polygon(bounding_box)
                        grasp = compute_grasp_from_polygon(fallback_poly)
                else:
                    polygon = _bbox_to_polygon(bounding_box)
                    grasp = compute_grasp_from_polygon(polygon)

                metadata: dict[str, Any] = {
                    "class_id": cls_id,
                    "source_model": str(self.weights),
                }
                if self.mm_per_px is not None and grasp is not None:
                    width_mm, opening_mm = estimate_gripper_opening_mm(
                        grasp.width_px,
                        mm_per_px=self.mm_per_px,
                    )
                    if width_mm is not None and opening_mm is not None:
                        metadata["width_mm"] = width_mm
                        metadata["gripper_opening_mm"] = opening_mm

                detections.append(
                    Detection(
                        label=label,
                        confidence=confidence,
                        bounding_box=bounding_box,
                        polygon=polygon,
                        grasp=grasp,
                        metadata=metadata,
                    )
                )

        detections.sort(key=lambda det: det.confidence, reverse=True)
        return detections
