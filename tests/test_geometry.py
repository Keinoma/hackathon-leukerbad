import math

import numpy as np
import pytest

from leukerbad.detection.geometry import (
    compute_grasp_from_polygon,
    estimate_gripper_opening_mm,
    normalize_axial_angle_deg,
)


def _make_rotated_rect_polygon(
    center: tuple[float, float],
    length: float,
    width: float,
    angle_deg: float,
) -> np.ndarray:
    """Create a synthetic elongated vape-like rectangle polygon rotated by `angle_deg`."""
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    hl, hw = length / 2.0, width / 2.0
    local = np.array([[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw]], dtype=np.float32)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    return local @ rot.T + np.array(center, dtype=np.float32)


def _axial_diff_deg(a: float, b: float) -> float:
    """Smallest undirected angular difference in [0, 90]."""
    diff = abs(a - b) % 180.0
    return min(diff, 180.0 - diff)


@pytest.mark.parametrize("target_angle", [0.0, 30.0, 45.0, -60.0, -90.0])
def test_compute_grasp_from_rotated_vape_polygons(target_angle: float) -> None:
    poly = _make_rotated_rect_polygon(center=(320.0, 240.0), length=120.0, width=24.0, angle_deg=target_angle)
    grasp = compute_grasp_from_polygon(poly)

    assert grasp is not None
    assert grasp.center_px[0] == pytest.approx(320.0, abs=0.5)
    assert grasp.center_px[1] == pytest.approx(240.0, abs=0.5)
    assert grasp.length_px == pytest.approx(120.0, abs=0.5)
    assert grasp.width_px == pytest.approx(24.0, abs=0.5)
    assert _axial_diff_deg(grasp.object_angle_deg, target_angle) < 1.0
    expected_gripper = normalize_axial_angle_deg(target_angle + 90.0)
    assert _axial_diff_deg(grasp.gripper_angle_deg, expected_gripper) < 1.0
    assert len(grasp.box_points_px) == 4


def test_compute_grasp_rejects_degenerate_polygons() -> None:
    assert compute_grasp_from_polygon(np.array([[10.0, 10.0], [20.0, 20.0]])) is None
    assert compute_grasp_from_polygon(np.zeros((4, 2), dtype=np.float32)) is None


def test_estimate_gripper_opening_mm() -> None:
    assert estimate_gripper_opening_mm(30.0, mm_per_px=None) == (None, None)
    width_mm, opening_mm = estimate_gripper_opening_mm(40.0, mm_per_px=0.5, clearance_mm=10.0)
    assert width_mm == pytest.approx(20.0)
    assert opening_mm == pytest.approx(30.0)
