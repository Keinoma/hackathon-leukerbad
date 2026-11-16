"""Validate image-space measurements independently of the model checkpoint."""

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from leukerbad.cli import report
from leukerbad.detection.base import Detection, ObjectGeometry
from leukerbad.detection.yolo import YOLODetector, mask_geometry


@pytest.mark.parametrize("angle", [0, 30, 60, 90, 135, 179, -30])
def test_geometry_reports_long_axis_angle_and_short_width(angle):
    polygon = cv2.boxPoints(((100, 120), (80, 20), angle))
    geometry = mask_geometry(polygon)
    assert geometry is not None
    assert geometry.x_px == pytest.approx(100, abs=0.001)
    assert geometry.y_px == pytest.approx(120, abs=0.001)
    assert geometry.width_px == pytest.approx(20, abs=0.001)
    assert geometry.theta_deg == pytest.approx(angle % 180, abs=0.001)


def test_invalid_mask_does_not_produce_measurements():
    assert mask_geometry(np.zeros((4, 2))) is None
    assert mask_geometry(np.array([[np.nan, 0], [1, 2], [3, 4]])) is None


def test_measurements_stay_associated_with_their_box():
    def box(confidence):
        return SimpleNamespace(
            cls=np.array(0), conf=np.array(confidence),
            xyxy=[SimpleNamespace(cpu=lambda: np.array([1, 2, 200, 200]))],
        )

    result = SimpleNamespace(
        names={0: "vape"}, boxes=[box(0.9), box(0.8)],
        masks=SimpleNamespace(xy=[
            cv2.boxPoints(((100, 120), (80, 20), 30)), np.zeros((4, 2)),
        ]),
    )
    first, second = YOLODetector._detections(result)
    assert first.geometry.theta_deg == pytest.approx(30, abs=0.001)
    assert second.geometry is None
    result.masks = None
    assert all(item.geometry is None for item in YOLODetector._detections(result))


def test_terminal_report_contains_measurements_and_units(capsys):
    report([Detection("vape", 0.9, (1, 2, 100, 100), ObjectGeometry(50, 60, 30, 20))])
    output = capsys.readouterr().out
    assert "x=50.0px y=60.0px theta=30.0deg width=20.0px" in output
