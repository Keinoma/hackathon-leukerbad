"""Measured lens geometry, calibration fitting, and workspace consistency."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import pytest

from leukerbad.lens import LensCorrection, check_workspace_lens, fit_lens, load_lens


def lens_data(distortion=None):
    return {"version": 1, "image_size": [640, 480],
            "camera_matrix": [[600, 0, 320], [0, 610, 240], [0, 0, 1]],
            "dist_coeffs": [0, 0, 0, 0, 0] if distortion is None else distortion}


def test_zero_distortion_preserves_pixels_and_resolution():
    lens = LensCorrection(lens_data())
    image = np.arange(480 * 640 * 3, dtype=np.uint8).reshape(480, 640, 3)
    np.testing.assert_array_equal(lens.apply(image), image)
    with pytest.raises(ValueError, match="resolution"):
        lens.apply(image[:240])


def test_remap_uses_the_correct_radial_tangential_model():
    lens = LensCorrection(lens_data([-.22, .04, .001, -.002, .01]))
    pixels = np.array([[160, 120], [480, 120], [160, 360], [480, 360]], np.float64)
    normalised = np.column_stack(((pixels[:, 0] - 320) / 600,
                                  (pixels[:, 1] - 240) / 610, np.ones(4)))
    distorted, _ = cv2.projectPoints(normalised, np.zeros(3), np.zeros(3),
                                    lens.camera_matrix, lens.distortion)
    mapped = np.array([[lens.maps[0][int(y), int(x)], lens.maps[1][int(y), int(x)]]
                       for x, y in pixels])
    np.testing.assert_allclose(mapped, distorted[:, 0], atol=0.0001)
    recovered = cv2.undistortPoints(distorted, lens.camera_matrix, lens.distortion,
                                   P=lens.camera_matrix)[:, 0]
    np.testing.assert_allclose(recovered, pixels, atol=0.02)


def test_workspace_requires_recalibration_if_lens_changes(tmp_path):
    path = tmp_path / "lens.json"
    data = lens_data()
    path.write_text(json.dumps(data))
    lens = load_lens(path)
    check_workspace_lens({"lens_id": lens.id}, lens)
    check_workspace_lens({}, None)
    assert load_lens(tmp_path / "missing.json") is None
    with pytest.raises(ValueError, match="paper calibration"):
        check_workspace_lens({}, lens)
    with pytest.raises(ValueError, match="paper calibration"):
        check_workspace_lens({"lens_id": lens.id}, None)
    changed = LensCorrection(lens_data([-.1, 0, 0, 0, 0]))
    assert changed.id != lens.id
    with pytest.raises(ValueError, match="paper calibration"):
        check_workspace_lens({"lens_id": lens.id}, changed)


@pytest.mark.parametrize("coefficients", [[float("nan"), 0, 0, 0, 0], [1, 2]])
def test_invalid_lens_coefficients_are_rejected(coefficients):
    with pytest.raises(ValueError):
        LensCorrection(lens_data(coefficients))


def synthetic_views(count=14, tilted=True):
    data = lens_data([-.14, .025, .001, -.001, .003])
    matrix = np.asarray(data["camera_matrix"], np.float64)
    distortion = np.asarray(data["dist_coeffs"], np.float64)
    points = np.zeros((9 * 6, 3), np.float32)
    points[:, :2] = np.mgrid[:9, :6].T.reshape(-1, 2) * 20
    views = []
    for index in range(count):
        rotation = np.array([.35 * np.sin(index), .4 * np.cos(index), .08 * np.sin(index / 2)]
                            if tilted else [0, 0, 0], np.float64)
        translation = np.array([-80 + 9 * index, -60 + 3 * index, 550 + 10 * index]
                               if tilted else [-80, -50, 600], np.float64)
        projected, _ = cv2.projectPoints(points, rotation, translation, matrix, distortion)
        views.append(projected.astype(np.float32))
    return views


def test_intrinsics_are_recovered_from_varied_known_board_poses():
    data = fit_lens(synthetic_views(), (640, 480))
    assert data["samples"] == 14
    assert data["rms_px"] < 0.01
    assert data["camera_matrix"][0][0] == pytest.approx(600, rel=0.01)
    assert data["camera_matrix"][1][1] == pytest.approx(610, rel=0.01)
    np.testing.assert_allclose(data["dist_coeffs"][:4], [-.14, .025, .001, -.001], atol=0.005)


def test_too_few_or_identical_views_are_rejected():
    with pytest.raises(ValueError, match="12"):
        fit_lens(synthetic_views(3), (640, 480))
    with pytest.raises(ValueError):
        fit_lens(synthetic_views(12, tilted=False), (640, 480))


def test_bundled_checkerboard_has_the_expected_internal_corner_count():
    # Rasterise the actual printable asset, rather than a second copy of its layout.
    asset = Path(__file__).resolve().parents[1] / "assets" / "checkerboard.svg"
    root = ET.parse(asset).getroot()
    image = np.full((720, 960), 255, np.uint8)
    for rectangle in root.findall("{http://www.w3.org/2000/svg}rect"):
        if rectangle.get("fill") == "black":
            x, y, width, height = (int(float(rectangle.get(key)) * 4)
                                  for key in ("x", "y", "width", "height"))
            image[y:y + height, x:x + width] = 0
    found, corners = cv2.findChessboardCornersSB(image, (9, 6))
    assert found and len(corners) == 54
