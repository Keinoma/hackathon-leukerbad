"""Paper calibration, centred coordinates, workspace overlay, and manual JSON export."""

import json
import os
import tempfile
import time
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from leukerbad.detection.yolo import mask_geometry


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, indent=2, allow_nan=False)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def order_paper_corners(points, width, height):
    """Order cyclic clicks TL/TR/BR/BL while preserving their measured edge lengths."""
    points = np.asarray(points, dtype=np.float32)
    centre = points.mean(axis=0)
    order = np.argsort(np.arctan2(points[:, 1] - centre[1], points[:, 0] - centre[0]))
    ordered = points[order]
    start = int(np.argmin(ordered.sum(axis=1)))
    order = np.roll(order, -start)
    top_edge = set(map(int, order[:2]))
    if top_edge not in ({0, 1}, {2, 3}):
        width, height = height, width
    return points[order], width, height


def transform(calibration, image_size):
    if tuple(calibration["image_size"]) != tuple(image_size):
        raise ValueError("Camera resolution changed: calibrate again")
    points = np.asarray(calibration["points_px"], dtype=np.float32)
    width, height = calibration["width_mm"], calibration["height_mm"]
    if (not np.isfinite([width, height]).all() or min(width, height) <= 0
            or points.shape != (4, 2) or not np.isfinite(points).all()
            or not cv2.isContourConvex(points) or cv2.contourArea(points) < 100):
        raise ValueError("Use four distinct rectangle corners in A-B-C-D order and positive mm")
    if (points < 0).any() or (points >= image_size).any():
        raise ValueError("Calibration points must be inside the image")
    if "origin_px" in calibration:
        points, width, height = order_paper_corners(points, width, height)
    destination = np.array([[0, 0], [width, 0], [width, height], [0, height]], np.float32)
    matrix = cv2.getPerspectiveTransform(points, destination)
    if "origin_px" in calibration:
        origin = np.asarray(calibration["origin_px"], dtype=np.float32)
        if (origin.shape != (2,) or not np.isfinite(origin).all()
                or cv2.pointPolygonTest(points, tuple(map(float, origin)), False) <= 0):
            raise ValueError("Click the origin mark inside the calibrated rectangle")
        ox, oy = cv2.perspectiveTransform(origin[None, None], matrix)[0, 0]
        # Robot frame requested by the user: up is +X, left is +Y.
        matrix = np.array([[0, -1, oy], [-1, 0, ox], [0, 0, 1]]) @ matrix
    return points, matrix


def orient_frame(image, degrees):
    rotations = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
                 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    if degrees == 0:
        return image
    if degrees not in rotations:
        raise ValueError("Rotation must be 0, 90, 180, or 270 degrees")
    return cv2.rotate(image, rotations[degrees])


def calibrate(image, width, height, output, rotation=0, lens_id=None):
    points = []
    title = "Click A B C D then origin | t: rotate | Enter: save | r: reset | q: cancel"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)

    def click(event, x, y, flags, parameter):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 5:
            points.append((x, y))

    def calibration_data():
        view = orient_frame(image, rotation)
        plane_width, plane_height = (height, width) if rotation % 180 else (width, height)
        corners, _, _ = order_paper_corners(points[:4], plane_width, plane_height)
        return {"version": 2, "rotation_deg": rotation,
                "lens_id": lens_id,
                "axes": {"x": "up", "y": "left"},
                "image_size": [view.shape[1], view.shape[0]],
                "points_px": corners.tolist(), "origin_px": points[4],
                "corner_order": "top_left_clockwise",
                "width_mm": plane_width, "height_mm": plane_height}

    cv2.setMouseCallback(title, click)
    try:
        while True:
            oriented = orient_frame(image, rotation)
            preview = oriented.copy()
            for label, point in zip(["A", "B", "C", "D", "0,0"], points, strict=False):
                cv2.circle(preview, point, 5, (0, 255, 255), -1)
                cv2.putText(preview, label, (point[0] + 8, point[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            instruction = ("t: robot at bottom | click top-left, top-right, bottom-right, bottom-left, origin"
                           if not points else f"{len(points)}/5 points | t rotates and resets points")
            cv2.putText(preview, instruction, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2)
            plane_width, plane_height = (height, width) if rotation % 180 else (width, height)
            cv2.putText(preview, f"A-B: {plane_width:g} mm | A-D: {plane_height:g} mm",
                        (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            if len(points) == 5:
                data = calibration_data()
                try:
                    preview = draw_workspace(preview, data)
                except ValueError as error:
                    cv2.putText(preview, str(error), (12, 80), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, (0, 0, 255), 2)
            cv2.imshow(title, preview)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                return
            if key == ord("r"):
                points.clear()
            if key == ord("t"):
                rotation = (rotation + 90) % 360
                points.clear()
            if key in (10, 13) and len(points) == 5:
                data = calibration_data()
                try:
                    transform(data, data["image_size"])
                except ValueError as error:
                    print(f"{error}; press r to reset the clicks.", flush=True)
                else:
                    write_json(output, data)
                    print(f"Calibration saved: {Path(output).resolve()}", flush=True)
                    return
    finally:
        cv2.destroyAllWindows()


def measure_targets(calibration, image_size, detections, polygons):
    points, matrix = transform(calibration, image_size)
    targets = []
    for index, polygon in polygons.items():
        polygon = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        if len(polygon) < 3 or not np.isfinite(polygon).all():
            continue
        if any(cv2.pointPolygonTest(points, tuple(map(float, point)), False) < 0
               for point in polygon):
            continue
        plane = cv2.perspectiveTransform(polygon[None], matrix)[0]
        geometry = mask_geometry(plane)
        if geometry is not None:
            if "origin_px" in calibration:
                # X is up and Y is left, so atan2(Y, X) is anticlockwise on screen.
                # Fold the undirected puff axis around vertical.
                theta = round((geometry.theta_deg + 90) % 180 - 90, 2)
                if theta <= -90:
                    theta = 90.0
            else:
                theta = round(geometry.theta_deg % 180, 2) % 180
            targets.append({"index": index, "x_mm": round(geometry.x_px, 2),
                            "y_mm": round(geometry.y_px, 2),
                            "theta_deg": theta,
                            "width_mm": round(geometry.width_px, 2),
                            "confidence": round(detections[index].confidence, 3)})
    return targets


def grid_points(calibration, image_size):
    """Project a 10 mm lattice aligned to the chosen origin onto the camera image."""
    corners, matrix = transform(calibration, image_size)
    plane = cv2.perspectiveTransform(corners[None], matrix)[0]
    xs = np.arange(np.ceil(plane[:, 0].min() / 10) * 10,
                   np.floor(plane[:, 0].max() / 10) * 10 + 0.1, 10)
    ys = np.arange(np.ceil(plane[:, 1].min() / 10) * 10,
                   np.floor(plane[:, 1].max() / 10) * 10 + 0.1, 10)
    grid = np.array(np.meshgrid(xs, ys), dtype=np.float32).reshape(2, -1).T
    if not len(grid):
        return np.empty((0, 2), dtype=np.float32)
    return cv2.perspectiveTransform(grid[None], np.linalg.inv(matrix))[0]


def _label(image, text, position, color):
    (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    x = max(3, min(position[0], image.shape[1] - width - 4))
    y = max(height + 4, min(position[1], image.shape[0] - baseline - 4))
    cv2.rectangle(image, (x - 3, y - height - 3),
                  (x + width + 3, y + baseline + 3), (20, 20, 20), -1)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 1, cv2.LINE_AA)


def point_position(calibration, pixel):
    """Measure a click in the already corrected and rotated camera view."""
    corners, matrix = transform(calibration, calibration["image_size"])
    point = np.asarray(pixel, dtype=np.float32)
    if (point.shape != (2,) or not np.isfinite(point).all()
            or cv2.pointPolygonTest(corners, tuple(map(float, point)), False) < 0):
        return None
    world = cv2.perspectiveTransform(point[None, None], matrix)[0, 0]
    return float(world[0]), float(world[1])


class PositionProbe:
    """Keep the last clicked paper position visible, without exporting a target."""

    def __init__(self, calibration):
        self.calibration = calibration
        self.pixel = None
        self.position = None
        self.message = None

    def on_mouse(self, event, x, y, flags, parameter):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        self.position = point_position(self.calibration, (x, y))
        self.pixel = (x, y) if self.position is not None else None
        if self.position is None:
            self.message = "Click outside calibrated paper"
        else:
            px, py = self.position
            self.message = f"Clicked: x={px:.1f} mm  y={py:.1f} mm"
        print(self.message, flush=True)

    def draw(self, image):
        if self.pixel is not None:
            cv2.drawMarker(image, self.pixel, (255, 0, 255), cv2.MARKER_CROSS, 22, 2)
            cv2.circle(image, self.pixel, 7, (255, 0, 255), 1)
        if self.message is not None:
            _label(image, self.message, (12, image.shape[0] - 14), (255, 160, 255))


def draw_workspace(image, calibration, targets=()):
    size = (image.shape[1], image.shape[0])
    corners, matrix = transform(calibration, size)
    inverse = np.linalg.inv(matrix)
    contour = np.rint(corners).astype(np.int32)
    inside = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(inside, contour, 255)
    result = image.copy()
    result[inside == 0] = (result[inside == 0] * 0.5).astype(np.uint8)
    for point in grid_points(calibration, size):
        cv2.circle(result, tuple(np.rint(point).astype(int)), 1, (150, 95, 20), -1)
    cv2.polylines(result, [contour], True, (255, 210, 0), 2, cv2.LINE_AA)
    axes = cv2.perspectiveTransform(np.array([[[0, 0], [40, 0], [0, 40]]],
                                            dtype=np.float32), inverse)[0]
    origin, x_axis, y_axis = (tuple(np.rint(point).astype(int)) for point in axes)
    cv2.arrowedLine(result, origin, x_axis, (40, 40, 230), 2, tipLength=0.15)
    cv2.arrowedLine(result, origin, y_axis, (40, 180, 40), 2, tipLength=0.15)
    for label, point in [("+X", x_axis), ("+Y", y_axis), ("0,0 mm", origin)]:
        _label(result, label, (point[0] + 6, point[1] - 8), (255, 255, 255))
    cv2.circle(result, origin, 5, (0, 0, 255), -1)
    for target in targets:
        point = cv2.perspectiveTransform(np.array([[[target["x_mm"], target["y_mm"]]]],
                                                 dtype=np.float32), inverse)[0, 0]
        centre = tuple(np.rint(point).astype(int))
        cv2.drawMarker(result, centre, (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
        label = (f"x={target['x_mm']:.1f} y={target['y_mm']:.1f} mm "
                 f"theta={target['theta_deg']:.1f} deg w={target['width_mm']:.1f} mm")
        text_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
        position = (max(0, min(centre[0] + 12, size[0] - text_width - 4)),
                    max(16, min(centre[1] + 24, size[1] - 8)))
        _label(result, label, position, (0, 255, 255))
    return result


def export_target(path, calibration, image_size, detections, polygons, frame_id=None):
    targets = measure_targets(calibration, image_size, detections, polygons)
    target = max(targets, key=lambda item: item["confidence"]) if targets else None
    if target is not None:
        target = {key: value for key, value in target.items() if key != "index"}
    payload = {"request_id": str(uuid4()), "created_at_unix": time.time(),
               "state": "ready" if target is not None else "empty",
               "coordinate_frame": "paper_mm",
               "origin": "clicked_mark" if "origin_px" in calibration else "corner_A",
               "axes": ({"x": "up", "y": "left"} if "origin_px" in calibration
                        else {"x": "right", "y": "down"}),
               "theta_reference": "+X",
               "theta_direction": "anticlockwise" if "origin_px" in calibration else "toward +Y",
               "theta_period_deg": 180,
               "theta_range_deg": [-90, 90] if "origin_px" in calibration else [0, 180],
               "lens_corrected": calibration.get("lens_id") is not None,
               "target": target}
    payload["timestamp"] = payload["created_at_unix"]
    if frame_id is not None:
        payload["frame_id"] = frame_id
    write_json(path, payload)
    return payload
