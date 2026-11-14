"""Interfaces shared by object detection implementations."""

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class GraspCandidate:
    """2D tabletop grasp geometry derived from an instance segmentation mask.

    Angles are in degrees in image coordinates (+x right, +y down), wrapped to [-90.0, 90.0):
    - `object_angle_deg`: Principal major-axis angle of the object.
    - `gripper_angle_deg`: Antipodal jaw closing axis, perpendicular to `object_angle_deg`.
    """

    center_px: tuple[float, float]
    object_angle_deg: float
    gripper_angle_deg: float
    width_px: float
    length_px: float
    box_points_px: tuple[tuple[float, float], ...] = ()


@dataclass(frozen=True)
class Detection:
    """A single detected object in image coordinates."""

    label: str
    confidence: float
    bounding_box: tuple[int, int, int, int]
    polygon: np.ndarray | None = None
    grasp: GraspCandidate | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Detector(Protocol):
    """Protocol implemented by model-backed detectors."""

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Return detections for an image frame."""
