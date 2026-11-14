"""Unit tests for Dataset A capture, SAM bbox-to-polygon masking, validator, and Colab notebook."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from leukerbad.dataset import (
    build_capture_filename,
    parse_yolo_label_line,
    polygon_px_to_yolo_seg_line,
    save_labeled_frame,
    upgrade_bbox_labels_with_sam,
    validate_yolo_seg_dataset,
    write_yolo_dataset_yaml,
    yolo_bbox_to_xyxy_px,
)


class _DummySamModel:
    """Offline mock of Ultralytics SAM returning an octagonal polygon inside each queried bbox."""

    def __init__(self) -> None:
        self.calls: list[list[list[float]]] = []

    def predict(
        self,
        _image: np.ndarray,
        bboxes: list[list[float]] | None = None,
        verbose: bool = False,
    ) -> list[SimpleNamespace]:
        assert verbose is False
        assert bboxes is not None and len(bboxes) == 1
        self.calls.append(bboxes)
        x1, y1, x2, y2 = bboxes[0]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        rx, ry = (x2 - x1) * 0.45, (y2 - y1) * 0.45
        poly = np.array(
            [
                [cx - rx, cy - ry * 0.5],
                [cx + rx, cy - ry * 0.5],
                [cx + rx, cy + ry * 0.5],
                [cx - rx, cy + ry * 0.5],
            ],
            dtype=np.float32,
        )
        masks = SimpleNamespace(xy=[poly], data=None)
        return [SimpleNamespace(masks=masks)]


def _make_distinct_image(seed_val: int, width: int = 200, height: int = 120) -> np.ndarray:
    rng = np.random.default_rng(seed_val)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def test_capture_save_labeled_frame_and_metadata(tmp_path: Path) -> None:
    fname = build_capture_filename("train", "AD_mix", 7)
    assert fname == "train_AD_mix_00007.png"

    with pytest.raises(ValueError, match="Invalid split"):
        build_capture_filename("invalid", "AD_mix", 0)

    frame_a = _make_distinct_image(1, width=160, height=100)
    frame_b = _make_distinct_image(2, width=160, height=100)

    saved_1 = save_labeled_frame(frame_a, tmp_path, split="train", tag="AD_mix")
    saved_2 = save_labeled_frame(frame_b, tmp_path, split="test", tag="E_unseen")

    assert saved_1.exists()
    assert saved_1.parent == tmp_path / "images" / "train"
    assert saved_2.parent == tmp_path / "images" / "test"

    meta_path = tmp_path / "capture_metadata.jsonl"
    rows = [json.loads(line) for line in meta_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert rows[0]["split"] == "train"
    assert rows[0]["tag"] == "AD_mix"
    assert rows[0]["width"] == 160
    assert rows[0]["height"] == 100
    assert rows[1]["split"] == "test"
    assert rows[1]["tag"] == "E_unseen"


def test_parse_yolo_label_line_and_polygon_clamping() -> None:
    bbox_parsed = parse_yolo_label_line("0 0.5 0.5 0.4 0.2")
    assert bbox_parsed.kind == "bbox"
    assert bbox_parsed.class_id == 0
    assert bbox_parsed.values == pytest.approx((0.5, 0.5, 0.4, 0.2))

    poly_parsed = parse_yolo_label_line("1 0.1 0.1 0.3 0.1 0.3 0.4 0.1 0.4")
    assert poly_parsed.kind == "polygon"
    assert poly_parsed.class_id == 1
    assert len(poly_parsed.values) == 8

    with pytest.raises(ValueError):
        parse_yolo_label_line("0 0.1 0.2 0.3 0.4 0.5")  # 5 coords (invalid)

    x1, y1, x2, y2 = yolo_bbox_to_xyxy_px(0.5, 0.5, 0.5, 0.25, img_w=200, img_h=100)
    assert (x1, y1, x2, y2) == pytest.approx((50.0, 37.5, 150.0, 62.5))

    # Out-of-frame vertices should clamp cleanly into [0.0, 1.0]
    raw_poly = np.array([[-15.0, 50.0], [250.0, -5.0], [100.0, 130.0]], dtype=np.float32)
    seg_line = polygon_px_to_yolo_seg_line(class_id=0, polygon_px=raw_poly, img_w=200, img_h=100)
    reparsed = parse_yolo_label_line(seg_line)
    assert reparsed.kind == "polygon"
    assert all(0.0 <= v <= 1.0 for v in reparsed.values)
    assert reparsed.values == pytest.approx((0.0, 0.5, 1.0, 0.0, 0.5, 1.0))


def test_upgrade_bbox_labels_with_mocked_sam(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels_in"
    out_labels_dir = tmp_path / "labels_out"
    images_dir.mkdir()
    labels_dir.mkdir()

    # Image 1: contains one 4-number bbox AND one existing polygon
    img_1 = _make_distinct_image(10, width=200, height=100)
    cv2.imwrite(str(images_dir / "sample_01.png"), img_1)
    (labels_dir / "sample_01.txt").write_text(
        "0 0.5 0.5 0.4 0.2\n1 0.100000 0.100000 0.200000 0.100000 0.200000 0.300000\n",
        encoding="utf-8",
    )

    # Image 2: zero-vape empty background scene
    img_2 = _make_distinct_image(20, width=200, height=100)
    cv2.imwrite(str(images_dir / "bg_empty.png"), img_2)
    (labels_dir / "bg_empty.txt").write_text("", encoding="utf-8")

    mock_sam = _DummySamModel()
    stats = upgrade_bbox_labels_with_sam(
        images_dir=images_dir,
        labels_dir=labels_dir,
        output_labels_dir=out_labels_dir,
        sam_model=mock_sam,
    )

    assert stats["processed_images"] == 2
    assert stats["upgraded_bboxes"] == 1
    assert stats["kept_polygons"] == 1
    assert stats["background_images"] == 1
    assert len(mock_sam.calls) == 1

    out_lines = (out_labels_dir / "sample_01.txt").read_text(encoding="utf-8").strip().splitlines()
    assert len(out_lines) == 2
    first_upgraded = parse_yolo_label_line(out_lines[0])
    second_kept = parse_yolo_label_line(out_lines[1])
    assert first_upgraded.kind == "polygon"
    assert first_upgraded.class_id == 0
    assert len(first_upgraded.values) == 8
    assert second_kept.kind == "polygon"
    assert second_kept.class_id == 1

    assert (out_labels_dir / "bg_empty.txt").read_text(encoding="utf-8") == ""


def _write_dataset_sample(
    root: Path,
    split: str,
    stem: str,
    image: np.ndarray,
    label_text: str | None,
) -> None:
    img_dir = root / "images" / split
    lbl_dir = root / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(img_dir / f"{stem}.png"), image)
    if label_text is not None:
        (lbl_dir / f"{stem}.txt").write_text(label_text, encoding="utf-8")


def test_validate_yolo_seg_dataset_passes_clean_splits(tmp_path: Path) -> None:
    poly_line = "0 0.2 0.2 0.8 0.2 0.8 0.5 0.2 0.5\n"
    _write_dataset_sample(tmp_path, "train", "train_vapeA_001", _make_distinct_image(1), poly_line)
    _write_dataset_sample(tmp_path, "val", "val_vapeB_001", _make_distinct_image(2), poly_line)
    # Held-out vapeE in test split is valid
    _write_dataset_sample(tmp_path, "test", "test_vapeE_001", _make_distinct_image(3), poly_line)

    yaml_path = write_yolo_dataset_yaml(tmp_path, ["vape"])
    assert yaml_path.exists()
    yaml_text = yaml_path.read_text(encoding="utf-8")
    assert "nc: 1" in yaml_text
    assert '"vape"' in yaml_text

    report = validate_yolo_seg_dataset(tmp_path)
    assert report.is_valid is True
    assert report.split_counts == {"train": 1, "val": 1, "test": 1}
    assert report.polygon_count == 3
    assert report.bbox_only_count == 0
    assert report.duplicate_hashes == []
    assert report.leaked_device_files == []
    assert report.errors == []


def test_validate_yolo_seg_dataset_catches_duplicates_bboxes_and_vape_e_leakage(
    tmp_path: Path,
) -> None:
    shared_img = _make_distinct_image(99)
    poly_line = "0 0.2 0.2 0.7 0.2 0.7 0.6 0.2 0.6\n"
    bbox_line = "0 0.5 0.5 0.3 0.2\n"

    # 1. Duplicate image across train and test + bbox-only line in train
    _write_dataset_sample(tmp_path, "train", "train_vapeA_dup", shared_img, bbox_line)
    _write_dataset_sample(tmp_path, "test", "test_vapeE_dup", shared_img, poly_line)

    # 2. Held-out Unseen Vape E accidentally leaked into val split
    _write_dataset_sample(
        tmp_path,
        "val",
        "val_vapeE_leaked_01",
        _make_distinct_image(105),
        poly_line,
    )

    report = validate_yolo_seg_dataset(tmp_path)
    assert report.is_valid is False
    assert report.bbox_only_count == 1
    assert report.polygon_count == 2
    assert len(report.duplicate_hashes) == 1
    assert any("val_vapeE_leaked_01.png" in path for path in report.leaked_device_files)
    assert len(report.errors) >= 3


def test_colab_notebook_is_valid_nbformat_v4() -> None:
    nb_path = Path(__file__).resolve().parents[1] / "notebooks" / "train_vape_seg.ipynb"
    assert nb_path.exists()

    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    assert isinstance(nb["cells"], list)
    code_cells = [cell for cell in nb["cells"] if cell.get("cell_type") == "code"]
    assert len(code_cells) >= 5

    full_source = "\n".join("".join(cell.get("source", [])) for cell in code_cells)
    assert "nvidia-smi" in full_source
    assert "upgrade_bbox_labels_with_sam" in full_source
    assert "yolo11s-seg.pt" in full_source
    assert "degrees=180.0" in full_source
    assert "hsv_h=0.4" in full_source
    assert "copy_paste=0.2" in full_source
    assert "epochs=25" in full_source
    assert "lr0=1e-3" in full_source
    assert 'split="test"' in full_source
