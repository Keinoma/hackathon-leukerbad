"""Object detection components."""

from leukerbad.detection.base import Detection, Detector, GraspCandidate
from leukerbad.detection.geometry import (
    compute_grasp_from_polygon,
    estimate_gripper_opening_mm,
    normalize_axial_angle_deg,
)
from leukerbad.detection.visualize import draw_detections
from leukerbad.detection.vlm_verifier import VlmVerdict, VlmVerifiedDetector, crop_detection_roi
from leukerbad.detection.yolo_seg import YoloSegDetector, select_best_device

__all__ = [
    "Detection",
    "Detector",
    "GraspCandidate",
    "VlmVerdict",
    "VlmVerifiedDetector",
    "YoloSegDetector",
    "compute_grasp_from_polygon",
    "crop_detection_roi",
    "draw_detections",
    "estimate_gripper_opening_mm",
    "normalize_axial_angle_deg",
    "select_best_device",
]
