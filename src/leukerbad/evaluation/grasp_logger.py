"""Dataset B physical tabletop grasp validation logger and summary analytics."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from leukerbad.detection.base import Detection, GraspCandidate
from leukerbad.detection.geometry import compute_grasp_from_polygon, normalize_axial_angle_deg


class FailureReason(StrEnum):
    """Categorized outcomes for physical tabletop grasp attempts."""

    NONE = "none"
    SLIP = "slip"
    COLLISION = "collision"
    MISSED_OBJECT = "missed_object"
    WRONG_ANGLE = "wrong_angle"
    FALSE_POSITIVE = "false_positive"
    OTHER = "other"


@dataclass(frozen=True)
class GraspTrialRecord:
    """Complete Dataset B record for a single physical robot grasp trial."""

    # 1. Trial identity & saved visual evidence
    trial_id: str
    timestamp_iso: str
    raw_image_path: str
    overlay_image_path: str
    # 2. Candidate grasp point & orientation
    grasp_center_px: tuple[float, float]
    object_angle_deg: float
    gripper_angle_deg: float
    target_pose_mm: tuple[float, float, float] | None
    # 3. Object design ID (e.g. vape_A .. vape_E)
    object_design_id: str
    # 4. Gripper opening & approximate object width
    approx_width_px: float
    approx_width_mm: float | None
    gripper_opening_mm: float | None
    # 5. Whether the grasp succeeded
    succeeded: bool
    # 6. Failure reason & optional notes
    failure_reason: FailureReason
    notes: str = ""

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize trial record to a JSON-compatible dictionary."""
        return {
            "trial_id": self.trial_id,
            "timestamp_iso": self.timestamp_iso,
            "raw_image_path": self.raw_image_path,
            "overlay_image_path": self.overlay_image_path,
            "grasp_center_px": [self.grasp_center_px[0], self.grasp_center_px[1]],
            "object_angle_deg": self.object_angle_deg,
            "gripper_angle_deg": self.gripper_angle_deg,
            "target_pose_mm": list(self.target_pose_mm) if self.target_pose_mm is not None else None,
            "object_design_id": self.object_design_id,
            "approx_width_px": self.approx_width_px,
            "approx_width_mm": self.approx_width_mm,
            "gripper_opening_mm": self.gripper_opening_mm,
            "succeeded": self.succeeded,
            "failure_reason": self.failure_reason.value,
            "notes": self.notes,
        }

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> GraspTrialRecord:
        """Deserialize a trial record from a parsed JSONL row."""
        center_raw = payload["grasp_center_px"]
        pose_raw = payload.get("target_pose_mm")
        return cls(
            trial_id=str(payload["trial_id"]),
            timestamp_iso=str(payload["timestamp_iso"]),
            raw_image_path=str(payload["raw_image_path"]),
            overlay_image_path=str(payload["overlay_image_path"]),
            grasp_center_px=(float(center_raw[0]), float(center_raw[1])),
            object_angle_deg=float(payload["object_angle_deg"]),
            gripper_angle_deg=float(payload["gripper_angle_deg"]),
            target_pose_mm=(
                (float(pose_raw[0]), float(pose_raw[1]), float(pose_raw[2]))
                if pose_raw is not None
                else None
            ),
            object_design_id=str(payload["object_design_id"]),
            approx_width_px=float(payload["approx_width_px"]),
            approx_width_mm=(
                float(payload["approx_width_mm"])
                if payload.get("approx_width_mm") is not None
                else None
            ),
            gripper_opening_mm=(
                float(payload["gripper_opening_mm"])
                if payload.get("gripper_opening_mm") is not None
                else None
            ),
            succeeded=bool(payload["succeeded"]),
            failure_reason=FailureReason(str(payload["failure_reason"])),
            notes=str(payload.get("notes", "")),
        )


CSV_COLUMNS: tuple[str, ...] = (
    "trial_id",
    "timestamp_iso",
    "object_design_id",
    "succeeded",
    "failure_reason",
    "grasp_cx_px",
    "grasp_cy_px",
    "object_angle_deg",
    "gripper_angle_deg",
    "approx_width_px",
    "approx_width_mm",
    "gripper_opening_mm",
    "target_x_mm",
    "target_y_mm",
    "target_z_mm",
    "raw_image_path",
    "overlay_image_path",
    "notes",
)


def _resolve_grasp_geometry(detection: Detection) -> GraspCandidate:
    if detection.grasp is not None:
        return detection.grasp

    if detection.polygon is not None:
        computed = compute_grasp_from_polygon(detection.polygon)
        if computed is not None:
            return computed

    x1, y1, x2, y2 = detection.bounding_box
    bw = max(float(x2 - x1), 1.0)
    bh = max(float(y2 - y1), 1.0)
    cx = (float(x1) + float(x2)) / 2.0
    cy = (float(y1) + float(y2)) / 2.0
    if bw >= bh:
        obj_angle = 0.0
        length_px, width_px = bw, bh
    else:
        obj_angle = -90.0
        length_px, width_px = bh, bw
    grip_angle = normalize_axial_angle_deg(obj_angle + 90.0)
    box_pts = (
        (float(x1), float(y1)),
        (float(x2), float(y1)),
        (float(x2), float(y2)),
        (float(x1), float(y2)),
    )
    return GraspCandidate(
        center_px=(cx, cy),
        object_angle_deg=obj_angle,
        gripper_angle_deg=grip_angle,
        width_px=width_px,
        length_px=length_px,
        box_points_px=box_pts,
    )


def _render_grasp_overlay(
    frame: np.ndarray,
    detection: Detection,
    grasp: GraspCandidate,
    trial_id: str,
    object_design_id: str,
    succeeded: bool,
    failure_reason: FailureReason,
) -> np.ndarray:
    canvas = frame.copy()
    mask_color = (40, 220, 90) if succeeded else (40, 80, 245)

    if detection.polygon is not None:
        pts = np.asarray(detection.polygon, dtype=np.float32).reshape(-1, 2)
        if pts.shape[0] >= 3:
            pts_i32 = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
            tinted = canvas.copy()
            cv2.fillPoly(tinted, [pts_i32], mask_color)
            cv2.addWeighted(tinted, 0.38, canvas, 0.62, 0.0, dst=canvas)
            cv2.polylines(canvas, [pts_i32], isClosed=True, color=mask_color, thickness=2)
    else:
        x1, y1, x2, y2 = detection.bounding_box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), mask_color, 2)

    if len(grasp.box_points_px) == 4:
        obb_pts = np.round(np.asarray(grasp.box_points_px, dtype=np.float32)).astype(np.int32)
        cv2.polylines(canvas, [obb_pts.reshape(-1, 1, 2)], True, (255, 220, 0), 1, cv2.LINE_AA)

    cx, cy = grasp.center_px
    center_pt = (round(float(cx)), round(float(cy)))

    # Draw major object axis (cyan)
    obj_rad = math.radians(grasp.object_angle_deg)
    half_len = max(grasp.length_px * 0.45, 18.0)
    ox = round(half_len * math.cos(obj_rad))
    oy = round(half_len * math.sin(obj_rad))
    cv2.line(
        canvas,
        (center_pt[0] - ox, center_pt[1] - oy),
        (center_pt[0] + ox, center_pt[1] + oy),
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )

    # Draw antipodal gripper closing axis (magenta) + crosshair
    grip_rad = math.radians(grasp.gripper_angle_deg)
    half_grip = max(grasp.width_px * 0.75, 16.0)
    gx = round(half_grip * math.cos(grip_rad))
    gy = round(half_grip * math.sin(grip_rad))
    p1 = (center_pt[0] - gx, center_pt[1] - gy)
    p2 = (center_pt[0] + gx, center_pt[1] + gy)
    cv2.line(canvas, p1, p2, (255, 60, 255), 2, cv2.LINE_AA)
    cv2.circle(canvas, p1, 4, (255, 60, 255), -1, cv2.LINE_AA)
    cv2.circle(canvas, p2, 4, (255, 60, 255), -1, cv2.LINE_AA)
    cv2.drawMarker(
        canvas,
        center_pt,
        (255, 255, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=14,
        thickness=2,
    )

    outcome_str = "SUCCESS" if succeeded else f"FAIL:{failure_reason.value}"
    banner = (
        f"{trial_id} | {object_design_id} | {outcome_str} | "
        f"grip={grasp.gripper_angle_deg:.1f}deg w={grasp.width_px:.1f}px"
    )
    banner_h = min(36, canvas.shape[0])
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], banner_h), (20, 20, 20), -1)
    cv2.putText(
        canvas,
        banner,
        (8, min(24, banner_h - 4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        mask_color,
        1,
        cv2.LINE_AA,
    )
    return canvas


class GraspTrialLogger:
    """Persists Dataset B physical robot grasp trials to disk (PNGs, JSONL, CSV) and summarizes."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.raw_dir = self.output_dir / "raw"
        self.overlays_dir = self.output_dir / "overlays"
        self.jsonl_path = self.output_dir / "trials.jsonl"
        self.csv_path = self.output_dir / "trials_summary.csv"

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.overlays_dir.mkdir(parents=True, exist_ok=True)

        self._records: list[GraspTrialRecord] = []
        if self.jsonl_path.exists():
            for raw_line in self.jsonl_path.read_text(encoding="utf-8").splitlines():
                stripped = raw_line.strip()
                if stripped:
                    self._records.append(GraspTrialRecord.from_json_dict(json.loads(stripped)))

    @property
    def records(self) -> tuple[GraspTrialRecord, ...]:
        """Return all logged trials in chronological order."""
        return tuple(self._records)

    def _next_trial_id(self) -> str:
        idx = len(self._records) + 1
        existing_ids = {rec.trial_id for rec in self._records}
        while f"trial_{idx:04d}" in existing_ids:
            idx += 1
        return f"trial_{idx:04d}"

    def log_trial(
        self,
        frame: np.ndarray,
        detection: Detection,
        object_design_id: str,
        succeeded: bool,
        failure_reason: FailureReason | str = FailureReason.NONE,
        gripper_opening_mm: float | None = None,
        approx_width_mm: float | None = None,
        target_pose_mm: tuple[float, float, float] | None = None,
        notes: str = "",
    ) -> GraspTrialRecord:
        """Save raw/overlay images and log a physical grasp attempt to JSONL and CSV."""
        if frame is None or frame.ndim < 2 or frame.size == 0:
            raise ValueError("Cannot log a grasp trial with an empty camera frame.")
        if not object_design_id.strip():
            raise ValueError("object_design_id must be a non-empty string.")

        parsed_reason = FailureReason(failure_reason)
        if succeeded:
            resolved_reason = FailureReason.NONE
        elif parsed_reason == FailureReason.NONE:
            resolved_reason = FailureReason.OTHER
        else:
            resolved_reason = parsed_reason

        trial_id = self._next_trial_id()
        timestamp_iso = datetime.now(UTC).isoformat()
        grasp = _resolve_grasp_geometry(detection)

        raw_img_path = self.raw_dir / f"{trial_id}.png"
        overlay_img_path = self.overlays_dir / f"{trial_id}_mask.png"

        if not cv2.imwrite(str(raw_img_path), frame):
            raise RuntimeError(f"Failed to save raw trial image to {raw_img_path}")

        overlay_frame = _render_grasp_overlay(
            frame=frame,
            detection=detection,
            grasp=grasp,
            trial_id=trial_id,
            object_design_id=object_design_id,
            succeeded=succeeded,
            failure_reason=resolved_reason,
        )
        if not cv2.imwrite(str(overlay_img_path), overlay_frame):
            raise RuntimeError(f"Failed to save mask overlay image to {overlay_img_path}")

        record = GraspTrialRecord(
            trial_id=trial_id,
            timestamp_iso=timestamp_iso,
            raw_image_path=str(raw_img_path),
            overlay_image_path=str(overlay_img_path),
            grasp_center_px=grasp.center_px,
            object_angle_deg=grasp.object_angle_deg,
            gripper_angle_deg=grasp.gripper_angle_deg,
            target_pose_mm=target_pose_mm,
            object_design_id=object_design_id,
            approx_width_px=grasp.width_px,
            approx_width_mm=approx_width_mm,
            gripper_opening_mm=gripper_opening_mm,
            succeeded=bool(succeeded),
            failure_reason=resolved_reason,
            notes=notes,
        )

        self._records.append(record)
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_json_dict()) + "\n")

        self._write_summary_csv()
        return record

    def _write_summary_csv(self) -> None:
        with self.csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
            writer.writeheader()
            for rec in self._records:
                tx, ty, tz = (
                    rec.target_pose_mm if rec.target_pose_mm is not None else (None, None, None)
                )
                writer.writerow(
                    {
                        "trial_id": rec.trial_id,
                        "timestamp_iso": rec.timestamp_iso,
                        "object_design_id": rec.object_design_id,
                        "succeeded": rec.succeeded,
                        "failure_reason": rec.failure_reason.value,
                        "grasp_cx_px": f"{rec.grasp_center_px[0]:.2f}",
                        "grasp_cy_px": f"{rec.grasp_center_px[1]:.2f}",
                        "object_angle_deg": f"{rec.object_angle_deg:.2f}",
                        "gripper_angle_deg": f"{rec.gripper_angle_deg:.2f}",
                        "approx_width_px": f"{rec.approx_width_px:.2f}",
                        "approx_width_mm": (
                            f"{rec.approx_width_mm:.2f}" if rec.approx_width_mm is not None else ""
                        ),
                        "gripper_opening_mm": (
                            f"{rec.gripper_opening_mm:.2f}"
                            if rec.gripper_opening_mm is not None
                            else ""
                        ),
                        "target_x_mm": f"{tx:.2f}" if tx is not None else "",
                        "target_y_mm": f"{ty:.2f}" if ty is not None else "",
                        "target_z_mm": f"{tz:.2f}" if tz is not None else "",
                        "raw_image_path": rec.raw_image_path,
                        "overlay_image_path": rec.overlay_image_path,
                        "notes": rec.notes,
                    }
                )

    def summarize(self) -> dict[str, Any]:
        """Compute aggregate success rate, per-design rates, and failure mode breakdown."""
        total = len(self._records)
        success_count = sum(1 for rec in self._records if rec.succeeded)
        success_rate = (success_count / total) if total > 0 else 0.0

        by_design_counts: dict[str, list[bool]] = {}
        failure_breakdown: dict[str, int] = {}

        for rec in self._records:
            by_design_counts.setdefault(rec.object_design_id, []).append(rec.succeeded)
            if not rec.succeeded:
                key = rec.failure_reason.value
                failure_breakdown[key] = failure_breakdown.get(key, 0) + 1

        by_design_id: dict[str, dict[str, int | float]] = {}
        for design_id, outcomes in sorted(by_design_counts.items()):
            design_total = len(outcomes)
            design_succ = sum(1 for ok in outcomes if ok)
            by_design_id[design_id] = {
                "total": design_total,
                "succeeded": design_succ,
                "success_rate": design_succ / design_total,
            }

        return {
            "total_trials": total,
            "success_count": success_count,
            "failure_count": total - success_count,
            "success_rate": success_rate,
            "by_design_id": by_design_id,
            "failure_breakdown": failure_breakdown,
        }
