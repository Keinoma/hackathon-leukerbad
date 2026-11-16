"""Ultralytics adapter and an image-space grasp visualization."""

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from leukerbad.detection.base import Detection, ObjectGeometry


def grasp_line(polygon: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return center and endpoints across a mask's minimum-rectangle short axis.

    This is a 2D visualization heuristic, not a calibrated robot grasp pose.
    """
    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    if len(points) < 3 or not np.isfinite(points).all():
        return None
    rectangle = cv2.minAreaRect(points)
    if min(rectangle[1]) < 1:
        return None
    corners = cv2.boxPoints(rectangle)
    edges = np.roll(corners, -1, axis=0) - corners
    short = edges[np.argmin(np.linalg.norm(edges, axis=1))]
    center = np.asarray(rectangle[0])
    return center, center - short / 2, center + short / 2


def mask_geometry(polygon: np.ndarray) -> ObjectGeometry | None:
    """Measure the long-axis orientation and short-side width of a mask rectangle.

    Image x points right, y points down. Theta is clockwise from horizontal in
    [0, 180): the mask alone does not identify which end is the mouthpiece.
    """
    grasp = grasp_line(polygon)
    if grasp is None:
        return None
    center, first, second = grasp
    short = second - first
    theta = (np.degrees(np.arctan2(short[1], short[0])) + 90.0) % 180.0
    # Floating point noise near the horizontal direction must not display 180.
    if np.isclose(theta, 180.0, atol=1e-5, rtol=0):
        theta = 0.0
    return ObjectGeometry(
        float(center[0]), float(center[1]), float(theta), float(np.linalg.norm(short)),
    )


class YOLODetector:
    def __init__(self, weights: Path, conf=0.25, device="auto", imgsz=640,
                 segment_weights: Path | None = None):
        import torch
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.segmenter = None
        if segment_weights is not None:
            self.segmenter = YOLO(str(segment_weights))
            if self.segmenter.task != "segment":
                raise ValueError("--segment-weights must be a segmentation checkpoint")
        self.conf = conf
        self.imgsz = imgsz
        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else (
                "0" if torch.cuda.is_available() else "cpu"
            )
        self.device = device
        print(f"Loaded {weights} | task={self.model.task} | device={device}", flush=True)
        print(f"Classes: {self.model.names}", flush=True)
        if self.segmenter is not None:
            print(f"Supplementary segmentation: {segment_weights}", flush=True)

    def _predict(self, frame):
        return self.model.predict(
            source=frame, conf=self.conf, device=self.device,
            imgsz=self.imgsz, verbose=False, retina_masks=True,
        )[0]

    @staticmethod
    def _detections(result) -> list[Detection]:
        if result.boxes is None:
            return []
        polygons = result.masks.xy if result.masks is not None else []
        return [
            Detection(
                result.names[int(box.cls.item())], float(box.conf.item()),
                tuple(round(value) for value in box.xyxy[0].cpu().tolist()),
                mask_geometry(polygons[index]) if index < len(polygons) else None,
            )
            for index, box in enumerate(result.boxes)
        ]

    def detect(self, frame: np.ndarray) -> list[Detection]:
        return self._infer(frame)[1]

    def masks(self, frame: np.ndarray):
        """Return detections and associated polygons for a requested snapshot."""
        _, detections, polygons = self._infer(frame)
        return detections, polygons

    def _infer(self, frame):
        result = self._predict(frame)
        detections = self._detections(result)
        polygons = {
            index: polygon for index, polygon in enumerate(result.masks.xy)
            if index < len(detections) and detections[index].geometry is not None
        } if result.masks is not None else {}
        if self.segmenter is not None and detections:
            segmented = self.segmenter.predict(
                source=frame, conf=self.conf, device=self.device,
                imgsz=self.imgsz, verbose=False, retina_masks=False,
            )[0]
            if segmented.masks is not None and segmented.boxes is not None:
                segments = self._detections(segmented)
                matched = match_masks(detections, segments, segmented.masks.xy)
                polygons.update(matched)
                detections = [
                    replace(item, geometry=mask_geometry(polygons[index]))
                    if index in polygons else item
                    for index, item in enumerate(detections)
                ]
        return result, detections, polygons

    def annotate(self, frame: np.ndarray) -> tuple[np.ndarray, list[Detection]]:
        annotated, detections, _ = self.analyse(frame)
        return annotated, detections

    def analyse(self, frame: np.ndarray, draw_geometry=True):
        """Return preview, detections, and masks without repeating inference."""
        result, detections, polygons = self._infer(frame)
        annotated = result.plot(masks=False)
        overlay = annotated.copy()
        for polygon in polygons.values():
            cv2.fillPoly(overlay, [np.rint(polygon).astype(np.int32)], (0, 200, 0))
        if polygons:
            annotated = cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0)
            for polygon in polygons.values():
                cv2.polylines(
                    annotated, [np.rint(polygon).astype(np.int32)], True, (0, 255, 0), 2,
                )
        for detection in detections:
            if not draw_geometry:
                continue
            geometry = detection.geometry
            if geometry is None:
                left, top, right, bottom = detection.bounding_box
                center = ((left + right) // 2, (top + bottom) // 2)
                cv2.drawMarker(
                    annotated, center, (0, 165, 255), cv2.MARKER_CROSS, 22, 2,
                )
                label = f"center=box (no mask) x={center[0]} y={center[1]} px"
                text_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
                height, width = annotated.shape[:2]
                position = (max(0, min(left, width - text_width - 4)),
                            max(16, min(bottom + 18, height - 8)))
                for color, thickness in (((0, 0, 0), 3), ((0, 165, 255), 1)):
                    cv2.putText(annotated, label, position, cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, color, thickness, cv2.LINE_AA)
                continue
            center = np.array([geometry.x_px, geometry.y_px])
            radians = np.radians(geometry.theta_deg)
            axis = np.array([np.cos(radians), np.sin(radians)])
            across = np.array([-axis[1], axis[0]]) * geometry.width_px / 2
            first, second = (np.rint(point).astype(int) for point in (
                center - across, center + across,
            ))
            cv2.line(annotated, tuple(first), tuple(second), (0, 255, 255), 3)
            for endpoint in (first, second):
                cv2.circle(annotated, tuple(endpoint), 4, (0, 255, 255), -1)
            axis_first, axis_second = (np.rint(point).astype(int) for point in (
                center - 30 * axis, center + 30 * axis,
            ))
            cv2.line(annotated, tuple(axis_first), tuple(axis_second), (255, 255, 0), 2)
            cv2.drawMarker(
                annotated, tuple(np.rint(center).astype(int)), (0, 255, 255),
                cv2.MARKER_CROSS, 18, 2,
            )
            lines = [
                f"theta={geometry.theta_deg:.1f} deg  width={geometry.width_px:.1f} px",
                f"x={geometry.x_px:.1f}  y={geometry.y_px:.1f} px",
            ]
            text_width = max(cv2.getTextSize(
                line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1,
            )[0][0] for line in lines)
            height, width = annotated.shape[:2]
            x = max(0, min(detection.bounding_box[0], width - text_width - 4))
            y = max(16, min(detection.bounding_box[3] + 18, height - 24))
            for offset, text in enumerate(lines):
                position = (x, y + 20 * offset)
                for color, thickness in (((0, 0, 0), 3), ((0, 255, 255), 1)):
                    cv2.putText(
                        annotated, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, thickness, cv2.LINE_AA,
                    )
        return annotated, detections, polygons


def match_masks(detections, segments, polygons, minimum_iou=0.3):
    """Match masks to local detections by class and box IoU, using each mask once."""
    candidates = []
    for index, detection in enumerate(detections):
        x1, y1, x2, y2 = detection.bounding_box
        area = max(0, x2 - x1) * max(0, y2 - y1)
        for mask_index, segment in enumerate(segments):
            if mask_index >= len(polygons) or segment.geometry is None:
                continue
            if detection.label.casefold() != segment.label.casefold():
                continue
            a1, b1, a2, b2 = segment.bounding_box
            intersection = max(0, min(x2, a2) - max(x1, a1)) * max(
                0, min(y2, b2) - max(y1, b1),
            )
            union = area + max(0, a2 - a1) * max(0, b2 - b1) - intersection
            iou = intersection / union if union > 0 else 0
            if iou >= minimum_iou:
                candidates.append((iou, index, mask_index))
    matched, used = {}, set()
    for _, index, mask_index in sorted(candidates, reverse=True):
        if index not in matched and mask_index not in used:
            matched[index] = polygons[mask_index]
            used.add(mask_index)
    return matched
