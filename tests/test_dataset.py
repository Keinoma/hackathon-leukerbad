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
from leukerbad.dataset.sam_masker import (
    _format_class_hud_keys,
    _parse_args,
    is_class_key_allowed,
    run_interactive_annotator,
)
from leukerbad.dataset.validate import _sha256_file, compute_dhash, hamming_distance_64


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


def test_validate_yolo_seg_dataset_enforces_num_classes(tmp_path: Path) -> None:
    vape_poly = "0 0.2 0.2 0.8 0.2 0.8 0.5 0.2 0.5\n"
    lookalike_poly = "1 0.1 0.1 0.4 0.1 0.4 0.4 0.1 0.4\n"

    _write_dataset_sample(tmp_path, "train", "train_vapeA_01", _make_distinct_image(11), vape_poly)
    _write_dataset_sample(
        tmp_path, "val", "val_lookalike_01", _make_distinct_image(12), lookalike_poly
    )

    with pytest.raises(ValueError, match="num_classes"):
        validate_yolo_seg_dataset(tmp_path, num_classes=0)

    report_1cls = validate_yolo_seg_dataset(tmp_path, num_classes=1)
    assert report_1cls.is_valid is False
    assert any("num_classes=1" in err and "class ID 1" in err for err in report_1cls.errors)

    report_2cls = validate_yolo_seg_dataset(tmp_path, num_classes=2)
    assert report_2cls.is_valid is True
    assert report_2cls.errors == []
    assert report_2cls.polygon_count == 2


def test_sam_annotator_num_classes_cli_and_key_gating(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    default_args = _parse_args(["--images", "imgs", "--labels", "lbls", "--interactive"])
    assert default_args.num_classes == 1

    two_cls_args = _parse_args(
        ["--images", "imgs", "--labels", "lbls", "--interactive", "--num-classes", "2"]
    )
    assert two_cls_args.num_classes == 2

    assert is_class_key_allowed(0, num_classes=1) is True
    assert is_class_key_allowed(1, num_classes=1) is False
    assert is_class_key_allowed(0, num_classes=2) is True
    assert is_class_key_allowed(1, num_classes=2) is True
    assert "[1] non_vape" not in _format_class_hud_keys(1)
    assert "[1] non_vape" in _format_class_hud_keys(2)

    images_dir = tmp_path / "imgs"
    labels_1cls = tmp_path / "lbls_1cls"
    labels_2cls = tmp_path / "lbls_2cls"
    images_dir.mkdir()
    cv2.imwrite(str(images_dir / "scene_01.png"), _make_distinct_image(42, width=160, height=120))

    class _DummyPointSam:
        def predict(
            self,
            _image: np.ndarray,
            points: list[list[int]] | None = None,
            labels: list[int] | None = None,
            verbose: bool = False,
        ) -> list[SimpleNamespace]:
            assert verbose is False
            assert points is not None and labels == [1]
            poly = np.array([[20.0, 20.0], [80.0, 20.0], [80.0, 70.0]], dtype=np.float32)
            return [SimpleNamespace(masks=SimpleNamespace(xy=[poly], data=None))]

    mouse_cb_holder: dict[str, object] = {}

    def _fake_set_mouse_cb(_win: str, cb: object) -> None:
        mouse_cb_holder["cb"] = cb

    monkeypatch.setattr(cv2, "namedWindow", lambda *_a, **_k: None)
    monkeypatch.setattr(cv2, "setMouseCallback", _fake_set_mouse_cb)
    monkeypatch.setattr(cv2, "imshow", lambda *_a, **_k: None)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)

    # In 1-class mode: click -> press '1' (refused!) -> press '0' (saved as class 0) -> 'q'
    keys_1cls = iter([ord("1"), ord("0"), ord("q")])

    def _wait_1cls(_delay: int) -> int:
        cb = mouse_cb_holder["cb"]
        assert callable(cb)
        k = next(keys_1cls)
        if k == ord("1"):
            cb(cv2.EVENT_LBUTTONDOWN, 40, 40, 0, None)
        return k

    monkeypatch.setattr(cv2, "waitKey", _wait_1cls)
    rc_1 = run_interactive_annotator(
        images_dir=images_dir,
        labels_dir=labels_1cls,
        sam_model=_DummyPointSam(),
        num_classes=1,
    )
    assert rc_1 == 0
    saved_1 = (labels_1cls / "scene_01.txt").read_text(encoding="utf-8").strip().splitlines()
    assert len(saved_1) == 1
    assert parse_yolo_label_line(saved_1[0]).class_id == 0

    # In 2-class mode: click -> press '1' (saved as class 1!) -> click -> press '0' -> 'q'
    keys_2cls = iter([ord("1"), ord("0"), ord("q")])

    def _wait_2cls(_delay: int) -> int:
        cb = mouse_cb_holder["cb"]
        assert callable(cb)
        k = next(keys_2cls)
        if k in (ord("1"), ord("0")):
            cb(cv2.EVENT_LBUTTONDOWN, 50, 50, 0, None)
        return k

    monkeypatch.setattr(cv2, "waitKey", _wait_2cls)
    rc_2 = run_interactive_annotator(
        images_dir=images_dir,
        labels_dir=labels_2cls,
        sam_model=_DummyPointSam(),
        num_classes=2,
    )
    assert rc_2 == 0
    saved_2 = (labels_2cls / "scene_01.txt").read_text(encoding="utf-8").strip().splitlines()
    assert [parse_yolo_label_line(ln).class_id for ln in saved_2] == [1, 0]


def test_validate_yolo_seg_dataset_detects_cross_split_perceptual_near_duplicates(
    tmp_path: Path,
) -> None:
    poly_line = "0 0.2 0.2 0.8 0.2 0.8 0.6 0.2 0.6\n"

    # Build a smooth, textured 120x160 synthetic tabletop scene
    yy, xx = np.meshgrid(
        np.linspace(0.0, 4.0 * np.pi, 120),
        np.linspace(0.0, 6.0 * np.pi, 160),
        indexing="ij",
    )
    pattern = (
        127.0 + 60.0 * np.sin(xx) * np.cos(yy) + 40.0 * np.cos(0.5 * xx + yy)
    ).clip(10, 245)
    base_img = np.stack(
        [pattern, np.roll(pattern, 7, axis=1), np.roll(pattern, 13, axis=0)],
        axis=2,
    )
    train_img = base_img.astype(np.uint8)

    # Tweak 3 pixels by +1 intensity so SHA-256 differs completely while dHash stays near-identical
    val_near_dup = train_img.copy()
    val_near_dup[10, 15, 0] = np.uint8(int(val_near_dup[10, 15, 0]) + 1)
    val_near_dup[60, 80, 1] = np.uint8(int(val_near_dup[60, 80, 1]) + 1)
    val_near_dup[100, 140, 2] = np.uint8(int(val_near_dup[100, 140, 2]) + 1)

    near_root = tmp_path / "near_dup_ds"
    _write_dataset_sample(near_root, "train", "train_scene_01", train_img, poly_line)
    _write_dataset_sample(near_root, "val", "val_scene_01", val_near_dup, poly_line)

    sha_train = _sha256_file(near_root / "images" / "train" / "train_scene_01.png")
    sha_val = _sha256_file(near_root / "images" / "val" / "val_scene_01.png")
    assert sha_train != sha_val
    assert hamming_distance_64(compute_dhash(train_img), compute_dhash(val_near_dup)) <= 6

    report_near = validate_yolo_seg_dataset(near_root)
    assert report_near.is_valid is True
    assert report_near.duplicate_hashes == []
    assert len(report_near.near_duplicates) == 1
    assert "train_scene_01.png" in report_near.near_duplicates[0]
    assert "val_scene_01.png" in report_near.near_duplicates[0]

    # Genuinely distinct textured scenes across splits must produce zero near-duplicate warnings
    distinct_root = tmp_path / "distinct_ds"
    other_img = cv2.flip(255 - train_img, 1)
    # Shift spatial frequency for a genuinely distinct gradient structure
    checker = (((np.arange(120)[:, None] // 15) + (np.arange(160)[None, :] // 20)) % 2) * 200 + 25
    distinct_img = np.stack([checker, other_img[:, :, 1], 255 - checker], axis=2).astype(np.uint8)
    _write_dataset_sample(distinct_root, "train", "train_scene_a", train_img, poly_line)
    _write_dataset_sample(distinct_root, "val", "val_scene_b", distinct_img, poly_line)

    report_distinct = validate_yolo_seg_dataset(distinct_root)
    assert report_distinct.is_valid is True
    assert report_distinct.near_duplicates == []


def test_colab_notebook_is_valid_nbformat_v4() -> None:
    nb_path = Path(__file__).resolve().parents[1] / "notebooks" / "train_vape_seg.ipynb"
    assert nb_path.exists()

    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    assert isinstance(nb["cells"], list)
    code_cells = [cell for cell in nb["cells"] if cell.get("cell_type") == "code"]
    assert len(code_cells) >= 5

    cell2_src = "".join(code_cells[1].get("source", []))
    cell3_src = "".join(code_cells[2].get("source", []))
    cell4_src = "".join(code_cells[3].get("source", []))

    # Cell 2: CLASS_NAMES defined at top of configuration block + num_classes=len(CLASS_NAMES)
    assert 'CLASS_NAMES = ["vape"]' in cell2_src
    assert cell2_src.index('CLASS_NAMES = ["vape"]') < cell2_src.index("DRIVE_ROOT =")
    assert "num_classes=len(CLASS_NAMES)" in cell2_src
    assert "near_duplicates=" in cell2_src

    # Cell 3 & Cell 4: dynamic checkpoint resolution via .trainer.best + exist_ok=True + seed=0
    for stage_src in (cell3_src, cell4_src):
        assert "exist_ok=True" in stage_src
        assert ".trainer.best" in stage_src
        assert "seed=0" in stage_src
        assert 'Path("/content/runs/' not in stage_src

    # Cell 4: explicit AdamW optimizer + lr0=5e-4 + patience=15 + freeze=10
    assert 'optimizer="AdamW"' in cell4_src
    assert "lr0=5e-4" in cell4_src
    assert "patience=15" in cell4_src
    assert "freeze=10" in cell4_src

    full_source = "\n".join("".join(cell.get("source", [])) for cell in code_cells)
    assert "nvidia-smi" in full_source
    assert "upgrade_bbox_labels_with_sam" in full_source
    assert "yolo11s-seg.pt" in full_source
    assert "degrees=180.0" in full_source
    assert "hsv_h=0.4" in full_source
    assert "copy_paste=0.2" in full_source
    assert "epochs=25" in full_source
    assert 'split="test"' in full_source
