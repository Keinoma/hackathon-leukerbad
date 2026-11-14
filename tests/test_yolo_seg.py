"""Unit tests for `YoloSegDetector` and `draw_detections` visual overlays."""

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from leukerbad.detection.base import Detection
from leukerbad.detection.geometry import compute_grasp_from_polygon, normalize_axial_angle_deg
from leukerbad.detection.visualize import (
    GENERAL_COLOR_BGR,
    VAPE_COLOR_BGR,
    _build_label_text,
    _select_color_bgr,
    draw_detections,
)
from leukerbad.detection.yolo_seg import YoloSegDetector, select_best_device


class _FakeTensor:
    """Minimal tensor stub mimicking PyTorch `.detach().cpu().numpy()`."""

    def __init__(self, array: np.ndarray) -> None:
        self._array = np.asarray(array)

    def detach(self) -> "_FakeTensor":
        return self

    def cpu(self) -> "_FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self._array

    def __len__(self) -> int:
        return int(self._array.shape[0])


@dataclass
class _MockBoxes:
    xyxy: Any
    conf: Any
    cls: Any

    def __len__(self) -> int:
        return len(self.xyxy)


@dataclass
class _MockMasks:
    xy: list[np.ndarray]


@dataclass
class _MockResult:
    boxes: _MockBoxes | None
    masks: _MockMasks | None
    names: dict[int, str]


class _MockYoloModel:
    def __init__(self, results: list[_MockResult]) -> None:
        self.results = results
        self.calls: list[dict[str, Any]] = []

    def predict(self, **kwargs: Any) -> list[_MockResult]:
        self.calls.append(kwargs)
        return self.results


def _rotated_rect(
    center: tuple[float, float],
    length: float,
    width: float,
    angle_deg: float,
) -> np.ndarray:
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    hl, hw = length / 2.0, width / 2.0
    local = np.array([[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw]], dtype=np.float32)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    return local @ rot.T + np.asarray(center, dtype=np.float32)


def test_select_best_device_respects_preferred_and_auto() -> None:
    assert select_best_device("cpu") == "cpu"
    assert select_best_device("  cuda:0  ") == "cuda:0"
    auto_device = select_best_device(None)
    assert auto_device in {"mps", "cuda", "cpu"}


def test_yolo_seg_detector_extracts_polygons_grasps_and_sorts_by_confidence() -> None:
    vape_poly = _rotated_rect(center=(220.0, 180.0), length=120.0, width=26.0, angle_deg=-30.0)
    pen_poly = _rotated_rect(center=(410.0, 300.0), length=140.0, width=16.0, angle_deg=15.0)

    boxes = _MockBoxes(
        xyxy=_FakeTensor(
            np.array(
                [
                    [340.0, 275.0, 480.0, 325.0],
                    [160.0, 140.0, 280.0, 220.0],
                ],
                dtype=np.float32,
            )
        ),
        conf=_FakeTensor(np.array([0.74, 0.93], dtype=np.float32)),
        cls=_FakeTensor(np.array([1, 0], dtype=np.float32)),
    )
    masks = _MockMasks(xy=[pen_poly, vape_poly])
    result = _MockResult(
        boxes=boxes,
        masks=masks,
        names={0: "disposable_vape", 1: "non_vape_pen"},
    )
    mock_model = _MockYoloModel([result])

    detector = YoloSegDetector(
        weights="custom-vape-seg.pt",
        conf_threshold=0.3,
        iou_threshold=0.5,
        device="cpu",
        mm_per_px=0.5,
        model=mock_model,
    )
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect(frame)

    assert len(mock_model.calls) == 1
    assert mock_model.calls[0]["conf"] == pytest.approx(0.3)
    assert mock_model.calls[0]["iou"] == pytest.approx(0.5)
    assert mock_model.calls[0]["device"] == "cpu"

    assert [d.label for d in detections] == ["disposable_vape", "non_vape_pen"]
    top = detections[0]
    assert top.confidence == pytest.approx(0.93)
    assert top.bounding_box == (160, 140, 280, 220)
    assert top.polygon is not None and top.polygon.shape == (4, 2)
    assert top.grasp is not None
    assert top.grasp.center_px[0] == pytest.approx(220.0, abs=0.5)
    assert top.grasp.center_px[1] == pytest.approx(180.0, abs=0.5)
    assert top.grasp.width_px == pytest.approx(26.0, abs=0.5)
    expected_wrist = normalize_axial_angle_deg(-30.0 + 90.0)
    assert top.grasp.gripper_angle_deg == pytest.approx(expected_wrist, abs=1.0)

    assert top.metadata["class_id"] == 0
    assert top.metadata["source_model"] == "custom-vape-seg.pt"
    assert top.metadata["width_mm"] == pytest.approx(13.0, abs=0.3)
    assert top.metadata["gripper_opening_mm"] == pytest.approx(25.0, abs=0.3)


def test_yolo_seg_detector_filters_allowed_labels_case_insensitively() -> None:
    boxes = _MockBoxes(
        xyxy=np.array([[10, 20, 110, 50], [200, 100, 260, 220]], dtype=np.float32),
        conf=np.array([0.88, 0.91], dtype=np.float32),
        cls=np.array([0, 1], dtype=np.int64),
    )
    result = _MockResult(
        boxes=boxes,
        masks=None,
        names={0: "Disposable_Vape", 1: "bottle"},
    )
    detector = YoloSegDetector(
        allowed_labels={"disposable_vape"},
        device="cpu",
        model=_MockYoloModel([result]),
    )
    detections = detector.detect(np.zeros((300, 400, 3), dtype=np.uint8))

    assert len(detections) == 1
    assert detections[0].label == "Disposable_Vape"


def test_yolo_seg_detector_falls_back_to_bbox_when_masks_missing_or_short() -> None:
    boxes = _MockBoxes(
        xyxy=np.array([[50, 100, 170, 130], [200, 210, 240, 310]], dtype=np.float32),
        conf=np.array([0.85, 0.75], dtype=np.float32),
        cls=np.array([0, 0], dtype=np.int64),
    )
    short_masks = _MockMasks(xy=[np.array([[50.0, 100.0], [60.0, 110.0]], dtype=np.float32)])
    result = _MockResult(boxes=boxes, masks=short_masks, names={0: "vape"})

    detector = YoloSegDetector(device="cpu", model=_MockYoloModel([result]))
    detections = detector.detect(np.zeros((400, 400, 3), dtype=np.uint8))

    assert len(detections) == 2
    for det in detections:
        assert det.polygon is not None
        assert det.polygon.shape == (4, 2)
        assert det.grasp is not None

    assert detections[0].grasp.length_px == pytest.approx(120.0, abs=0.5)
    assert detections[0].grasp.width_px == pytest.approx(30.0, abs=0.5)


def test_yolo_seg_detector_handles_empty_or_invalid_frames() -> None:
    mock_model = _MockYoloModel([])
    detector = YoloSegDetector(device="cpu", model=mock_model)

    assert detector.detect(np.zeros((0, 0, 3), dtype=np.uint8)) == []
    assert detector.detect(np.zeros((10,), dtype=np.uint8)) == []
    assert detector.detect(None) == []  # type: ignore[arg-type]
    assert len(mock_model.calls) == 0


def test_visualize_color_rules_and_non_mutating_rendering() -> None:
    assert _select_color_bgr("vape") == VAPE_COLOR_BGR
    assert _select_color_bgr("disposable_vape") == VAPE_COLOR_BGR
    assert _select_color_bgr("non_vape") == GENERAL_COLOR_BGR
    assert _select_color_bgr("non_vape_pen") == GENERAL_COLOR_BGR
    assert _select_color_bgr("plastic_bottle") == GENERAL_COLOR_BGR

    frame = np.full((360, 480, 3), 40, dtype=np.uint8)
    snapshot = frame.copy()

    poly = _rotated_rect(center=(240.0, 180.0), length=130.0, width=28.0, angle_deg=25.0)
    grasp = compute_grasp_from_polygon(poly)
    det_vape = Detection(
        label="disposable_vape",
        confidence=0.96,
        bounding_box=(175, 140, 305, 220),
        polygon=poly,
        grasp=grasp,
        metadata={"vlm": {"battery_hazard": True}},
    )
    det_plain = Detection(
        label="non_vape_marker",
        confidence=0.72,
        bounding_box=(20, 20, 90, 60),
    )

    pill = _build_label_text(det_vape)
    assert "disposable_vape 0.96" in pill
    assert "wrist=" in pill
    assert "w=28px" in pill
    assert "VLM:HAZARD" in pill

    rendered = draw_detections(frame, [det_vape, det_plain], alpha=0.4)

    assert np.array_equal(frame, snapshot), "draw_detections must never mutate input frame"
    assert rendered.shape == frame.shape
    assert rendered.dtype == frame.dtype
    assert not np.array_equal(rendered, frame)
