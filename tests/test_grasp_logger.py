"""Unit tests for the Dataset B physical grasp validation logger."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from leukerbad.detection.base import Detection
from leukerbad.detection.geometry import compute_grasp_from_polygon
from leukerbad.evaluation import FailureReason, GraspTrialLogger


def _make_vape_detection(
    center: tuple[float, float] = (320.0, 240.0),
    with_precomputed_grasp: bool = True,
) -> Detection:
    cx, cy = center
    poly = np.array(
        [
            [cx - 60.0, cy - 12.0],
            [cx + 60.0, cy - 12.0],
            [cx + 60.0, cy + 12.0],
            [cx - 60.0, cy + 12.0],
        ],
        dtype=np.float32,
    )
    grasp = compute_grasp_from_polygon(poly) if with_precomputed_grasp else None
    bbox = (int(cx - 60), int(cy - 12), int(cx + 60), int(cy + 12))
    return Detection(
        label="vape",
        confidence=0.94,
        bounding_box=bbox,
        polygon=poly,
        grasp=grasp,
    )


def _make_tabletop_frame(width: int = 640, height: int = 480) -> np.ndarray:
    frame = np.full((height, width, 3), 180, dtype=np.uint8)
    cv2.rectangle(frame, (260, 228), (380, 252), (50, 120, 210), -1)
    return frame


def test_grasp_trial_logger_persists_images_jsonl_csv_and_summary(tmp_path: Path) -> None:
    logger = GraspTrialLogger(output_dir=tmp_path)
    frame = _make_tabletop_frame()

    det_1 = _make_vape_detection(with_precomputed_grasp=True)
    rec_1 = logger.log_trial(
        frame=frame,
        detection=det_1,
        object_design_id="vape_A",
        succeeded=True,
        gripper_opening_mm=34.0,
        approx_width_mm=22.0,
        target_pose_mm=(180.0, -25.0, 12.0),
        notes="Clean centered lift",
    )

    # Second trial omits precomputed grasp -> logger derives grasp from polygon automatically
    det_2 = _make_vape_detection(center=(280.0, 200.0), with_precomputed_grasp=False)
    rec_2 = logger.log_trial(
        frame=frame,
        detection=det_2,
        object_design_id="vape_A",
        succeeded=False,
        failure_reason=FailureReason.SLIP,
        gripper_opening_mm=34.0,
        approx_width_mm=22.0,
        notes="Glossy casing slipped during lift",
    )

    rec_3 = logger.log_trial(
        frame=frame,
        detection=det_1,
        object_design_id="vape_E",
        succeeded=True,
        gripper_opening_mm=38.0,
        approx_width_mm=26.0,
        target_pose_mm=(200.0, 10.0, 12.0),
        notes="Zero-shot unseen Vape E success",
    )

    rec_4 = logger.log_trial(
        frame=frame,
        detection=det_1,
        object_design_id="vape_E",
        succeeded=False,
        failure_reason="collision",
    )

    rec_5 = logger.log_trial(
        frame=frame,
        detection=det_1,
        object_design_id="vape_B",
        succeeded=False,
        failure_reason=FailureReason.MISSED_OBJECT,
    )

    assert rec_1.trial_id == "trial_0001"
    assert rec_1.failure_reason == FailureReason.NONE
    assert rec_2.failure_reason == FailureReason.SLIP
    assert rec_4.failure_reason == FailureReason.COLLISION
    assert rec_5.failure_reason == FailureReason.MISSED_OBJECT
    assert rec_2.approx_width_px == pytest.approx(24.0, abs=0.5)

    # Verify raw & mask overlay PNGs exist and render visual overlay annotations
    for rec in (rec_1, rec_2, rec_3, rec_4, rec_5):
        raw_path = Path(rec.raw_image_path)
        ov_path = Path(rec.overlay_image_path)
        assert raw_path.exists()
        assert ov_path.exists()
        raw_loaded = cv2.imread(str(raw_path))
        ov_loaded = cv2.imread(str(ov_path))
        assert raw_loaded is not None and raw_loaded.shape == frame.shape
        assert ov_loaded is not None and ov_loaded.shape == frame.shape
        assert not np.array_equal(raw_loaded, ov_loaded)

    # Verify JSONL persistence
    jsonl_rows = [
        json.loads(ln) for ln in (tmp_path / "trials.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(jsonl_rows) == 5
    assert jsonl_rows[0]["target_pose_mm"] == [180.0, -25.0, 12.0]
    assert jsonl_rows[1]["failure_reason"] == "slip"

    # Verify CSV summary file
    with (tmp_path / "trials_summary.csv").open("r", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == 5
    assert csv_rows[2]["object_design_id"] == "vape_E"
    assert csv_rows[2]["succeeded"] == "True"

    # Verify summarize() metrics
    summary = logger.summarize()
    assert summary["total_trials"] == 5
    assert summary["success_count"] == 2
    assert summary["failure_count"] == 3
    assert summary["success_rate"] == pytest.approx(0.4)
    assert summary["by_design_id"]["vape_A"]["success_rate"] == pytest.approx(0.5)
    assert summary["by_design_id"]["vape_E"]["success_rate"] == pytest.approx(0.5)
    assert summary["by_design_id"]["vape_B"]["success_rate"] == pytest.approx(0.0)
    assert summary["failure_breakdown"] == {
        "slip": 1,
        "collision": 1,
        "missed_object": 1,
    }

    # Verify reloading existing directory resumes cleanly at trial_0006
    reloaded = GraspTrialLogger(output_dir=tmp_path)
    assert len(reloaded.records) == 5
    rec_6 = reloaded.log_trial(
        frame=frame,
        detection=det_1,
        object_design_id="vape_E",
        succeeded=True,
    )
    assert rec_6.trial_id == "trial_0006"
    assert reloaded.summarize()["total_trials"] == 6
    assert reloaded.summarize()["success_count"] == 3
