"""Analytical 2D grasp geometry extraction from segmentation polygons."""

import math

import cv2
import numpy as np

from leukerbad.detection.base import GraspCandidate


def normalize_axial_angle_deg(angle_deg: float) -> float:
    """Wrap an undirected axis angle in degrees into [-90.0, 90.0)."""
    wrapped = ((angle_deg + 90.0) % 180.0) - 90.0
    if math.isclose(wrapped, 90.0, abs_tol=1e-9):
        return -90.0
    return wrapped


def compute_grasp_from_polygon(polygon: np.ndarray) -> GraspCandidate | None:
    """Compute oriented bounding box and antipodal grasp pose from a 2D polygon.

    Args:
        polygon: Array of shape (N, 2) or (N, 1, 2) with pixel coordinates (x, y).

    Returns:
        `GraspCandidate` or `None` if fewer than 3 valid points are given.
    """
    pts = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] < 3:
        return None

    rect = cv2.minAreaRect(pts)
    (rect_cx, rect_cy), _, _ = rect
    box = cv2.boxPoints(rect)  # (4, 2) float32 corners in cyclic order

    edge_01 = box[1] - box[0]
    edge_12 = box[2] - box[1]
    len_01 = float(np.linalg.norm(edge_01))
    len_12 = float(np.linalg.norm(edge_12))

    if max(len_01, len_12) < 1e-6:
        return None

    if len_01 >= len_12:
        major_vec = edge_01
        length_px = len_01
        width_px = len_12
    else:
        major_vec = edge_12
        length_px = len_12
        width_px = len_01

    raw_major_deg = math.degrees(math.atan2(float(major_vec[1]), float(major_vec[0])))
    object_angle_deg = normalize_axial_angle_deg(raw_major_deg)
    gripper_angle_deg = normalize_axial_angle_deg(object_angle_deg + 90.0)

    moments = cv2.moments(pts)
    if abs(moments["m00"]) > 1e-5:
        cx = float(moments["m10"] / moments["m00"])
        cy = float(moments["m01"] / moments["m00"])
    else:
        cx = float(rect_cx)
        cy = float(rect_cy)

    box_tuple = tuple((float(pt[0]), float(pt[1])) for pt in box)
    return GraspCandidate(
        center_px=(cx, cy),
        object_angle_deg=object_angle_deg,
        gripper_angle_deg=gripper_angle_deg,
        width_px=width_px,
        length_px=length_px,
        box_points_px=box_tuple,
    )


def estimate_gripper_opening_mm(
    width_px: float,
    mm_per_px: float | None = None,
    clearance_mm: float = 12.0,
    min_opening_mm: float = 8.0,
    max_opening_mm: float = 75.0,
) -> tuple[float | None, float | None]:
    """Convert pixel width to object width (mm) and safe pre-grasp opening (mm)."""
    if mm_per_px is None or mm_per_px <= 0.0:
        return None, None
    width_mm = width_px * mm_per_px
    opening_mm = min(max(width_mm + clearance_mm, min_opening_mm), max_opening_mm)
    return width_mm, opening_mm
