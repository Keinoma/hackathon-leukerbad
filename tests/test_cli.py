"""Unit tests for the `leukerbad-detect` CLI entry point."""

import sys
import types
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from leukerbad import cli
from leukerbad.detection.base import Detection
from leukerbad.detection.geometry import compute_grasp_from_polygon


def test_cli_bare_invocation_prints_status_and_returns_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = cli.main([])
    captured = capsys.readouterr()

    assert rc == 0
    assert "Leukerbad" in captured.out
    assert "--synthetic-demo" in captured.out


def test_cli_synthetic_demo_saves_annotated_image(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out_path = tmp_path / "outputs" / "synthetic_annotated.png"
    rc = cli.main(["--synthetic-demo", "--mm-per-px", "0.5", "--save", str(out_path)])
    captured = capsys.readouterr()

    assert rc == 0
    assert out_path.is_file()
    saved = cv2.imread(str(out_path))
    assert saved is not None
    assert saved.shape == (480, 640, 3)
    assert "Detected 3 object(s):" in captured.out
    assert "disposable_vape" in captured.out
    assert "opening=" in captured.out


def test_cli_synthetic_demo_only_vapes_filters_distractors(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = cli.main(["--synthetic-demo", "--only-vapes"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "Detected 2 object(s):" in captured.out
    assert "non_vape_pen" not in captured.out


def test_cli_image_mode_with_mocked_detector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_img = tmp_path / "tabletop.png"
    output_img = tmp_path / "annotated.png"
    raw = np.full((240, 320, 3), 80, dtype=np.uint8)
    assert cv2.imwrite(str(input_img), raw)

    recorded_kwargs: dict[str, Any] = {}

    class _StubDetector:
        def __init__(self, **kwargs: Any) -> None:
            recorded_kwargs.update(kwargs)

        def detect(self, frame: np.ndarray) -> list[Detection]:
            assert frame.shape == (240, 320, 3)
            poly = np.array(
                [[80.0, 90.0], [200.0, 90.0], [200.0, 118.0], [80.0, 118.0]],
                dtype=np.float32,
            )
            return [
                Detection(
                    label="vape",
                    confidence=0.91,
                    bounding_box=(80, 90, 200, 118),
                    polygon=poly,
                    grasp=compute_grasp_from_polygon(poly),
                )
            ]

    monkeypatch.setattr(cli, "YoloSegDetector", _StubDetector)

    rc = cli.main(
        [
            "--image",
            str(input_img),
            "--save",
            str(output_img),
            "--weights",
            "custom.pt",
            "--conf",
            "0.4",
            "--labels",
            "vape, disposable_vape",
            "--mm-per-px",
            "0.6",
        ]
    )
    captured = capsys.readouterr()

    assert rc == 0
    assert output_img.is_file()
    assert recorded_kwargs["weights"] == "custom.pt"
    assert recorded_kwargs["conf_threshold"] == pytest.approx(0.4)
    assert recorded_kwargs["allowed_labels"] == {"vape", "disposable_vape"}
    assert recorded_kwargs["mm_per_px"] == pytest.approx(0.6)
    assert "vape (0.91)" in captured.out


def test_cli_missing_image_returns_nonzero(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = cli.main(["--image", str(tmp_path / "does_not_exist.png")])
    captured = capsys.readouterr()

    assert rc == 1
    assert "not found" in captured.err


def test_cli_gemini_verifier_wraps_detector_lazily(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    wrapped_calls: list[bool] = []

    class _FakeVlmVerifiedDetector:
        def __init__(self, base_detector: Any, reject_non_vapes: bool = False) -> None:
            self._base = base_detector
            wrapped_calls.append(reject_non_vapes)

        def detect(self, frame: np.ndarray) -> list[Detection]:
            return self._base.detect(frame)[:1]

    fake_module = types.ModuleType("leukerbad.detection.vlm_verifier")
    fake_module.VlmVerifiedDetector = _FakeVlmVerifiedDetector  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "leukerbad.detection.vlm_verifier", fake_module)

    rc = cli.main(["--synthetic-demo", "--verifier", "gemini", "--only-vapes"])
    captured = capsys.readouterr()

    assert rc == 0
    assert wrapped_calls == [True]
    assert "Detected 1 object(s):" in captured.out
