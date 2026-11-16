"""Check measured coordinates and one-target manual JSON snapshots."""

import json
from argparse import Namespace
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from leukerbad.cli import run_camera
from leukerbad.detection.base import Detection
from leukerbad.workspace import (
    PositionProbe,
    calibrate,
    draw_workspace,
    export_target,
    grid_points,
    order_paper_corners,
    orient_frame,
    point_position,
    transform,
)


def calibration():
    return {"image_size": [500, 500], "points_px": [[10, 10], [410, 10],
            [410, 410], [10, 410]], "width_mm": 310, "height_mm": 385}


def test_projected_polygon_produces_correct_mm_angle_and_width(tmp_path):
    data = calibration()
    _, matrix = transform(data, (500, 500))
    world = cv2.boxPoints(((100, 150), (80, 20), 30))
    polygon = cv2.perspectiveTransform(world[None], np.linalg.inv(matrix))[0]
    detection = Detection("vape", 0.95, (0, 0, 200, 200))
    path = tmp_path / "target.json"
    payload = export_target(path, data, (500, 500), [detection], {0: polygon})
    assert payload["state"] == "ready"
    target = payload["target"]
    assert target["x_mm"] == pytest.approx(100, abs=0.01)
    assert target["y_mm"] == pytest.approx(150, abs=0.01)
    assert target["theta_deg"] == pytest.approx(30, abs=0.01)
    assert target["width_mm"] == pytest.approx(20, abs=0.01)
    assert json.loads(path.read_text()) == payload


def test_empty_or_outside_masks_clear_old_snapshot(tmp_path):
    path = tmp_path / "target.json"
    path.write_text('{"state":"ready"}')
    detection = Detection("vape", 0.95, (0, 0, 200, 200))
    payload = export_target(path, calibration(), (500, 500), [detection],
                            {0: np.array([[0, 0], [100, 0], [100, 100]])})
    assert payload["state"] == "empty" and payload["target"] is None
    assert json.loads(path.read_text())["state"] == "empty"


def test_resolution_changes_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="resolution"):
        export_target(tmp_path / "target.json", calibration(), (640, 480), [], {})


def test_one_target_per_keypress_and_a_new_request_id(tmp_path):
    detections = [Detection("vape", 0.9, (0, 0, 200, 200))] * 2
    polygons = {0: cv2.boxPoints(((100, 100), (50, 10), 20)),
                1: cv2.boxPoints(((200, 200), (50, 10), 40))}
    path = tmp_path / "target.json"
    first = export_target(path, calibration(), (500, 500), detections, polygons)
    second = export_target(path, calibration(), (500, 500), detections, polygons)
    assert first["target"] == second["target"]
    assert first["request_id"] != second["request_id"]
    assert first["target"]["y_mm"] < 100


def test_invalid_corner_order_is_rejected():
    data = calibration()
    data["points_px"][1], data["points_px"][2] = data["points_px"][2], data["points_px"][1]
    with pytest.raises(ValueError):
        transform(data, (500, 500))


def centred_calibration():
    return {**calibration(), "origin_px": [210, 210], "rotation_deg": 180, "version": 2}


def test_clicked_centre_is_zero_with_x_up_and_y_left():
    data = centred_calibration()
    _, matrix = transform(data, (500, 500))
    positions = np.array([[[210, 210], [410, 10], [10, 410]]], np.float32)
    projected = cv2.perspectiveTransform(positions, matrix)[0]
    np.testing.assert_allclose(projected, [[0, 0], [192.5, -155], [-192.5, 155]], atol=0.001)
    data["origin_px"] = [200, 200]  # use the actual mark, not an assumed midpoint
    _, matrix = transform(data, (500, 500))
    actual_zero = cv2.perspectiveTransform(np.array([[[200, 200]]], np.float32), matrix)
    np.testing.assert_allclose(actual_zero, 0, atol=0.001)


def test_centred_json_contains_signed_mm_and_correct_world_angle(tmp_path):
    data = centred_calibration()
    _, matrix = transform(data, (500, 500))
    rectangle = cv2.boxPoints(((-60, -50), (80, 20), 30))
    polygon = cv2.perspectiveTransform(rectangle[None], np.linalg.inv(matrix))[0]
    payload = export_target(tmp_path / "target.json", data, (500, 500),
                            [Detection("vape", 0.95, (0, 0, 200, 200))], {0: polygon})
    assert payload["origin"] == "clicked_mark"
    assert payload["axes"] == {"x": "up", "y": "left"}
    assert payload["target"]["x_mm"] == pytest.approx(-60, abs=0.01)
    assert payload["target"]["y_mm"] == pytest.approx(-50, abs=0.01)
    assert payload["target"]["theta_deg"] == pytest.approx(30, abs=0.01)


def test_grid_is_a_projected_centimetre_lattice():
    data = centred_calibration()
    _, matrix = transform(data, (500, 500))
    pixels = grid_points(data, (500, 500))
    assert len(pixels) == 31 * 39
    world = cv2.perspectiveTransform(pixels[None], matrix)[0]
    np.testing.assert_allclose(world / 10, np.rint(world / 10), atol=0.0001)


def test_outside_is_half_brightness_and_origin_is_red():
    image = np.full((500, 500, 3), 200, dtype=np.uint8)
    result = draw_workspace(image, centred_calibration())
    np.testing.assert_array_equal(result[2, 2], [100, 100, 100])
    np.testing.assert_array_equal(result[210, 210], [0, 0, 255])
    assert np.mean(np.all(result[20:400, 20:400] == 200, axis=2)) > 0.8
    np.testing.assert_array_equal(image[2, 2], [200, 200, 200])


@pytest.mark.parametrize("angle,inverse", [(0, 0), (90, 270), (180, 180), (270, 90)])
def test_rotation_can_be_reversed_without_losing_pixels(angle, inverse):
    image = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    np.testing.assert_array_equal(orient_frame(orient_frame(image, angle), inverse), image)


def test_invalid_origin_outside_paper_is_rejected():
    data = centred_calibration()
    data["origin_px"] = [450, 450]
    with pytest.raises(ValueError, match="origin"):
        transform(data, (500, 500))


@pytest.mark.parametrize("corners", [
    [[10, 10], [410, 10], [410, 410], [10, 410]],
    [[10, 10], [10, 410], [410, 410], [410, 10]],
])
def test_calibration_rotation_and_fifth_click_are_saved(monkeypatch, tmp_path, corners):
    callbacks = {}
    keys = iter([ord("t"), 13])
    monkeypatch.setattr(cv2, "namedWindow", lambda *args: None)
    monkeypatch.setattr(cv2, "setMouseCallback", lambda title, fn: callbacks.update(click=fn))
    monkeypatch.setattr(cv2, "imshow", lambda *args: None)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *args: 1)

    def wait(delay):
        key = next(keys)
        if key == 13:
            for x, y in [*corners, [210, 210]]:
                callbacks["click"](cv2.EVENT_LBUTTONDOWN, x, y, None, None)
        return key

    monkeypatch.setattr(cv2, "waitKey", wait)
    path = tmp_path / "workspace.json"
    calibrate(np.zeros((500, 500, 3), np.uint8), 310, 385, path, rotation=180)
    saved = json.loads(path.read_text())
    assert saved["rotation_deg"] == 270
    assert saved["width_mm"] == 385
    assert saved["height_mm"] == 310
    assert saved["origin_px"] == [210, 210]
    assert saved["points_px"] == [[10, 10], [410, 10], [410, 410], [10, 410]]
    assert saved["version"] == 2


@pytest.mark.parametrize("rotation", [0, 180])
@pytest.mark.parametrize("correct_lens", [False, True])
def test_live_view_and_json_use_the_saved_rotation_and_same_masks(
        monkeypatch, tmp_path, rotation, correct_lens):
    from leukerbad.lens import LensCorrection

    image = np.arange(40 * 60 * 3, dtype=np.uint8).reshape(40, 60, 3)
    data = {"image_size": [60, 40], "points_px": [[5, 5], [55, 5], [55, 35], [5, 35]],
            "origin_px": [30, 20], "width_mm": 310, "height_mm": 385,
            "rotation_deg": rotation}
    lens_path = None
    corrected = image
    if correct_lens:
        lens_data = {"version": 1, "image_size": [60, 40],
                     "camera_matrix": [[55, 0, 30], [0, 55, 20], [0, 0, 1]],
                     "dist_coeffs": [-.2, .03, .001, -.001, 0]}
        lens_path = tmp_path / "lens.json"
        lens_path.write_text(json.dumps(lens_data))
        lens = LensCorrection(lens_data)
        data["lens_id"] = lens.id
        corrected = lens.apply(image)
    path = tmp_path / "workspace.json"
    path.write_text(json.dumps(data))
    polygon = np.array([[10, 10], [20, 10], [20, 16], [10, 16]], np.float32)
    calls = []

    class Detector:
        def analyse(self, frame, draw_geometry):
            np.testing.assert_array_equal(frame, orient_frame(corrected, rotation))
            assert draw_geometry is False
            calls.append(1)
            return frame.copy(), [Detection("vape", 0.9, (10, 10, 20, 16))], {0: polygon}

    camera = SimpleNamespace(isOpened=lambda: True, release=lambda: None)
    frames = SimpleNamespace(read=lambda: (True, image), close=lambda: None)
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args: camera)
    monkeypatch.setattr("leukerbad.cli.LatestFrames", lambda camera: frames)
    monkeypatch.setattr(cv2, "namedWindow", lambda *args: None)
    monkeypatch.setattr(cv2, "setMouseCallback", lambda *args: None)
    monkeypatch.setattr(cv2, "imshow", lambda *args: None)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *args: 1)
    keys = iter([ord("j"), ord("q")])
    monkeypatch.setattr(cv2, "waitKey", lambda delay: next(keys))
    output = tmp_path / "target.json"
    run_camera(Namespace(camera=0, calibration=path, json_out=output, rotate=90,
                         lens_file=lens_path), Detector())
    payload = json.loads(output.read_text())
    assert payload["state"] == "empty"
    assert len(calls) == 2
    # Coordinate/angle values are covered by export and click tests.


def test_click_reports_origin_and_centimetre_offsets():
    data = centred_calibration()
    np.testing.assert_allclose(point_position(data, (210, 210)), [0, 0], atol=0.0001)
    np.testing.assert_allclose(point_position(data, (250, 170)), [38.5, -31], atol=0.0001)
    assert point_position(data, (0, 0)) is None


def test_click_uses_the_perspective_corrected_paper_frame():
    data = {**centred_calibration(), "points_px": [[30, 10], [450, 40], [390, 470], [10, 420]]}
    _, matrix = transform(data, (500, 500))
    pixel = cv2.perspectiveTransform(np.array([[[-25, 40]]], np.float32), np.linalg.inv(matrix))[0, 0]
    np.testing.assert_allclose(point_position(data, pixel), [-25, 40], atol=0.001)


def test_probe_displays_and_prints_clicked_position_then_clears_an_outside_click(capsys):
    probe = PositionProbe(centred_calibration())
    probe.on_mouse(cv2.EVENT_MOUSEMOVE, 250, 170, None, None)
    assert probe.message is None
    probe.on_mouse(cv2.EVENT_LBUTTONDOWN, 250, 170, None, None)
    assert "x=38.5 mm  y=-31.0 mm" in capsys.readouterr().out
    image = np.zeros((500, 500, 3), np.uint8)
    probe.draw(image)
    np.testing.assert_array_equal(image[170, 250], [255, 0, 255])
    probe.on_mouse(cv2.EVENT_LBUTTONDOWN, 0, 0, None, None)
    assert probe.position is None and probe.pixel is None
    assert "outside" in probe.message


def test_live_mouse_callback_uses_the_displayed_image_without_writing_json(monkeypatch, tmp_path, capsys):
    data = centred_calibration()
    path = tmp_path / "workspace.json"
    path.write_text(json.dumps(data))
    image = np.full((500, 500, 3), 200, np.uint8)
    detector = SimpleNamespace(analyse=lambda *args, **kwargs: (image.copy(), [], {}))
    camera = SimpleNamespace(isOpened=lambda: True, release=lambda: None)
    frames = SimpleNamespace(read=lambda: (True, image), close=lambda: None)
    callbacks, previews = {}, []
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args: camera)
    monkeypatch.setattr("leukerbad.cli.LatestFrames", lambda camera: frames)
    monkeypatch.setattr(cv2, "namedWindow", lambda *args: None)
    monkeypatch.setattr(cv2, "setMouseCallback", lambda title, fn: callbacks.update(click=fn))
    monkeypatch.setattr(cv2, "imshow", lambda title, frame: previews.append(frame.copy()))
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *args: 1)

    def wait(delay):
        if len(previews) == 1:
            callbacks["click"](cv2.EVENT_LBUTTONDOWN, 250, 170, None, None)
            return -1
        return ord("q")

    monkeypatch.setattr(cv2, "waitKey", wait)
    output = tmp_path / "target.json"
    run_camera(Namespace(camera=0, calibration=path, json_out=output, lens_file=None), detector)
    assert "x=38.5 mm  y=-31.0 mm" in capsys.readouterr().out
    np.testing.assert_array_equal(previews[-1][170, 250], [255, 0, 255])
    assert output.exists()
    assert json.loads(output.read_text())["state"] == "empty"


@pytest.mark.parametrize("pixel,expected", [
    ((210, 170), (38.5, 0)), ((210, 250), (-38.5, 0)),
    ((250, 210), (0, -31)), ((170, 210), (0, 31)),
])
def test_each_axis_direction_has_the_requested_sign(pixel, expected):
    np.testing.assert_allclose(point_position(centred_calibration(), pixel), expected, atol=0.001)


@pytest.mark.parametrize("polygon,theta", [
    ([[200, 160], [220, 160], [220, 260], [200, 260]], 0),
    ([[160, 200], [260, 200], [260, 220], [160, 220]], 90),
])
def test_theta_zero_is_vertical_and_ninety_is_horizontal(tmp_path, polygon, theta):
    payload = export_target(tmp_path / "target.json", centred_calibration(), (500, 500),
                            [Detection("vape", 0.9, (0, 0, 0, 0))], {0: np.array(polygon)})
    assert payload["target"]["theta_deg"] == pytest.approx(theta, abs=0.01)
    assert payload["theta_reference"] == "+X"
    assert payload["theta_direction"] == "anticlockwise"


@pytest.mark.parametrize("order", [
    [0, 1, 2, 3], [0, 3, 2, 1], [1, 2, 3, 0], [1, 0, 3, 2],
    [2, 3, 0, 1], [2, 1, 0, 3], [3, 0, 1, 2], [3, 2, 1, 0],
])
def test_all_cyclic_corner_orders_preserve_axes_and_physical_scale(order):
    original = centred_calibration()
    points = np.asarray(original["points_px"])
    first_edge = set(order[:2])
    width, height = (310, 385) if first_edge in ({0, 1}, {2, 3}) else (385, 310)
    data = {**original, "points_px": points[order].tolist(),
            "width_mm": width, "height_mm": height}
    corners, matrix = transform(data, (500, 500))
    expected_corners, expected = transform(original, (500, 500))
    np.testing.assert_allclose(corners, expected_corners)
    np.testing.assert_allclose(matrix, expected, atol=0.0001)


def test_reported_sideways_axis_calibration_is_corrected_without_changing_scale():
    points = [[324, 60], [291, 1002], [1585, 1015], [1604, 69]]
    ordered, width, height = order_paper_corners(points, 280, 385)
    np.testing.assert_array_equal(ordered, np.asarray(points)[[0, 3, 2, 1]])
    assert (width, height) == (385, 280)
    data = {"points_px": points, "width_mm": 280, "height_mm": 385,
            "image_size": [1920, 1080], "origin_px": [953, 519]}
    _, matrix = transform(data, data["image_size"])
    pixel_axes = cv2.perspectiveTransform(np.array([[[0, 0], [40, 0], [0, 40]]], np.float32),
                                         np.linalg.inv(matrix))[0]
    assert pixel_axes[1, 1] < pixel_axes[0, 1]  # positive X is upward
    assert pixel_axes[2, 0] < pixel_axes[0, 0]  # positive Y is leftward
    np.testing.assert_allclose(point_position(data, data["origin_px"]), [0, 0], atol=.001)


@pytest.mark.parametrize("angle", [-45, -2, 0, 2, 45, 90])
def test_theta_increases_anticlockwise_and_stays_near_zero_for_upright_puffs(tmp_path, angle):
    # Square pixels at 1 mm/pixel: construct the puff physically rotated in the image.
    data = {"image_size": [600, 600], "points_px": [[50, 50], [550, 50], [550, 550], [50, 550]],
            "width_mm": 500, "height_mm": 500, "origin_px": [300, 300]}
    direction = np.array([-np.sin(np.radians(angle)), -np.cos(np.radians(angle))])
    across = np.array([-direction[1], direction[0]])
    centre = np.array([300, 300])
    polygon = np.array([centre + 60 * direction + 10 * across,
                        centre + 60 * direction - 10 * across,
                        centre - 60 * direction - 10 * across,
                        centre - 60 * direction + 10 * across])
    payload = export_target(tmp_path / "target.json", data, (600, 600),
                            [Detection("vape", .9, (0, 0, 0, 0))], {0: polygon})
    assert payload["target"]["theta_deg"] == pytest.approx(angle, abs=0.01)


def test_automatic_state_selects_highest_confidence_and_clears_empty(tmp_path):
    data = centred_calibration()
    detections = [Detection('vape', .6, (0, 0, 0, 0)), Detection('vape', .98, (0, 0, 0, 0))]
    polygons = {0: cv2.boxPoints(((140, 140), (40, 15), 10)),
                1: cv2.boxPoints(((280, 280), (40, 15), 20))}
    path = tmp_path / 'state.json'
    result = export_target(path, data, (500, 500), detections, polygons, frame_id=10)
    assert result['target']['confidence'] == .98
    assert result['frame_id'] == 10 and result['timestamp'] > 0
    result = export_target(path, data, (500, 500), [], {}, frame_id=11)
    assert result['target'] is None and result['frame_id'] == 11
