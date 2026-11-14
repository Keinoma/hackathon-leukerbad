"""Interfaces shared by object detection implementations."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Detection:
    """A single detected object in image coordinates."""

    label: str
    confidence: float
    bounding_box: tuple[int, int, int, int]


class Detector(Protocol):
    """Protocol implemented by model-backed detectors."""

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Return detections for an image frame."""
