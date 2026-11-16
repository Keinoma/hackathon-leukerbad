"""Verify supplemental masks preserve local detections and expose honest centers."""

from types import SimpleNamespace

import numpy as np
import pytest

from leukerbad.cli import report
from leukerbad.detection.base import Detection, ObjectGeometry
from leukerbad.detection.yolo import YOLODetector, match_masks

POLYGON = np.array([[20, 20], [80, 20], [80, 40], [20, 40]], dtype=np.float32)


def box(confidence, bounds):
    return SimpleNamespace(
        cls=np.array(0), conf=np.array(confidence),
        xyxy=[SimpleNamespace(cpu=lambda: np.array(bounds))],
    )


def detector_with_segmenter(has_mask=True):
    image = np.zeros((100, 320, 3), dtype=np.uint8)
    local = SimpleNamespace(
        names={0: "Vape"}, boxes=[box(0.92, [10, 10, 90, 50])], masks=None,
        plot=lambda **kwargs: image.copy(),
    )
    segmented = SimpleNamespace(
        names={0: "vape"}, boxes=[box(0.43, [20, 20, 80, 40])],
        masks=SimpleNamespace(xy=[POLYGON]) if has_mask else None,
    )
    detector = YOLODetector.__new__(YOLODetector)
    detector.conf, detector.device, detector.imgsz = 0.25, "cpu", 640
    detector._predict = lambda frame: local
    detector.segmenter = SimpleNamespace(predict=lambda **kwargs: [segmented])
    return detector, image


def test_supplemental_mask_preserves_local_class_confidence_and_box():
    detector, image = detector_with_segmenter()
    annotated, detections = detector.annotate(image)
    detection = detections[0]
    assert detection.label == "Vape"
    assert detection.confidence == pytest.approx(0.92)
    assert detection.bounding_box == (10, 10, 90, 50)
    assert detection.geometry == ObjectGeometry(50, 30, 0, 20)
    np.testing.assert_array_equal(annotated[30, 50], [0, 255, 255])
    assert annotated[24, 25, 1] > 0  # translucent green inside the mask
    assert detector.detect(image)[0].geometry == detection.geometry


def test_missing_mask_shows_box_center_and_no_fake_geometry(capsys):
    detector, image = detector_with_segmenter(has_mask=False)
    annotated, detections = detector.annotate(image)
    assert detections[0].geometry is None
    np.testing.assert_array_equal(annotated[30, 50], [0, 165, 255])
    report(detections)
    output = capsys.readouterr().out
    assert "x=50.0px y=30.0px center=box mask=unavailable" in output
    assert "theta=" not in output


def test_mask_is_assigned_once_and_must_overlap_and_match_class():
    geometry = ObjectGeometry(50, 30, 0, 20)
    local = Detection("Vape", 0.9, (10, 10, 90, 50))
    segment = Detection("vape", 0.8, (20, 20, 80, 40), geometry)
    assert len(match_masks([local, local], [segment], [POLYGON])) == 1
    assert match_masks([local], [Detection("person", 0.9, (20, 20, 80, 40), geometry)],
                       [POLYGON]) == {}
    assert match_masks([Detection("Vape", 0.9, (200, 200, 300, 300))],
                       [segment], [POLYGON]) == {}
