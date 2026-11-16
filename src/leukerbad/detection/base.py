"""Interfaces shared by object detection implementations."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class ObjectGeometry:
    """Mask rectangle in pixels; long-axis angle is clockwise modulo 180 degrees."""

    x_px: float
    y_px: float
    theta_deg: float
    width_px: float


@dataclass(frozen=True)
class Detection:
    """A single detected object in image coordinates."""

    label: str
    confidence: float
    bounding_box: tuple[int, int, int, int]
    geometry: ObjectGeometry | None = None


class Detector(Protocol):
    """Protocol implemented by model-backed detectors."""

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Return detections for an image frame."""
