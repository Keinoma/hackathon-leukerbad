"""Unit tests for the multimodal VLM lookalike verifier wrapper."""

import importlib.machinery
import importlib.util
import json
import sys
import types
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from leukerbad.detection.base import Detection, GraspCandidate
from leukerbad.detection.vlm_verifier import (
    VlmVerdict,
    VlmVerifiedDetector,
    crop_detection_roi,
)


@dataclass
class _StubDetector:
    detections: list[Detection]

    def detect(self, frame: np.ndarray) -> list[Detection]:
        del frame
        return list(self.detections)


@dataclass
class _FakeResponse:
    text: str


class _FakeModels:
    def __init__(self, responses: list[str | Exception]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def generate_content(
        self,
        *,
        model: str,
        contents: list[Any],
        config: Any,
    ) -> _FakeResponse:
        self.calls.append({"model": model, "contents": contents, "config": config})
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return _FakeResponse(text=item)


class _FakeGenAiClient:
    def __init__(self, responses: list[str | Exception]) -> None:
        self.models = _FakeModels(responses)


def _sample_frame(height: int = 120, width: int = 160) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)


def _sample_grasp() -> GraspCandidate:
    return GraspCandidate(
        center_px=(50.0, 60.0),
        object_angle_deg=15.0,
        gripper_angle_deg=-75.0,
        width_px=18.0,
        length_px=72.0,
        box_points_px=((14.0, 51.0), (86.0, 51.0), (86.0, 69.0), (14.0, 69.0)),
    )


def test_crop_detection_roi_clamps_to_image_boundaries() -> None:
    frame = _sample_frame(height=100, width=120)

    # Top-left boundary overflow
    top_left_crop = crop_detection_roi(frame, (0, 2, 20, 22), padding_ratio=0.25)
    assert top_left_crop.shape == (27, 25, 3)
    np.testing.assert_array_equal(top_left_crop, frame[0:27, 0:25])

    # Bottom-right boundary overflow
    bottom_right_crop = crop_detection_roi(frame, (105, 85, 120, 100), padding_ratio=0.40)
    assert bottom_right_crop.shape == (21, 21, 3)
    np.testing.assert_array_equal(bottom_right_crop, frame[79:100, 99:120])

    # Extreme box beyond right/bottom edge still produces non-empty crop
    edge_crop = crop_detection_roi(frame, (119, 99, 130, 110), padding_ratio=0.20)
    assert edge_crop.shape[0] >= 1
    assert edge_crop.shape[1] >= 1


def test_crop_detection_roi_rejects_empty_frame() -> None:
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="non-empty"):
        crop_detection_roi(empty, (0, 0, 10, 10))


def test_lookalike_relabeled_or_rejected_while_preserving_grasp_and_polygon() -> None:
    polygon = np.array([[20, 30], [80, 30], [80, 70], [20, 70]], dtype=np.int32)
    grasp = _sample_grasp()
    lookalike_det = Detection(
        label="vape",
        confidence=0.62,
        bounding_box=(20, 30, 80, 70),
        polygon=polygon,
        grasp=grasp,
        metadata={"track_id": 7},
    )
    verdict_json = json.dumps(
        {
            "is_vape": False,
            "detected_object": "airpods_case",
            "confidence": 0.94,
            "hazard_note": "Benign non-vape lookalike",
        }
    )

    # Case 1: reject_non_vapes=False -> relabel for Arm 1 general-waste clearer, preserve geometry
    client_keep = _FakeGenAiClient([verdict_json])
    verifier_keep = VlmVerifiedDetector(
        base_detector=_StubDetector([lookalike_det]),
        reject_non_vapes=False,
        client=client_keep,
    )
    kept = verifier_keep.detect(_sample_frame())

    assert len(kept) == 1
    result = kept[0]
    assert result.label == "non_vape:airpods_case"
    assert result.confidence == pytest.approx(0.94)
    assert result.grasp == grasp
    assert result.polygon is not None
    np.testing.assert_array_equal(result.polygon, polygon)
    assert result.metadata["track_id"] == 7
    assert result.metadata["raw_yolo_label"] == "vape"
    assert result.metadata["vlm"]["verified"] is True
    assert result.metadata["vlm"]["is_vape"] is False
    assert result.metadata["vlm"]["detected_object"] == "airpods_case"
    assert len(client_keep.models.calls) == 1

    # Case 2: reject_non_vapes=True -> drop lookalike completely
    client_drop = _FakeGenAiClient([verdict_json])
    verifier_drop = VlmVerifiedDetector(
        base_detector=_StubDetector([lookalike_det]),
        reject_non_vapes=True,
        client=client_drop,
    )
    dropped = verifier_drop.detect(_sample_frame())
    assert dropped == []


def test_true_unseen_vape_promoted_with_blended_confidence_and_hazard_metadata() -> None:
    grasp = _sample_grasp()
    unseen_vape_det = Detection(
        label="cylindrical_object",
        confidence=0.48,
        bounding_box=(15, 25, 95, 65),
        grasp=grasp,
    )
    verdict = VlmVerdict(
        is_vape=True,
        detected_object="disposable_vape",
        confidence=0.96,
        hazard_note="Embedded Li-ion cell + nicotine e-liquid fire hazard",
    )
    client = _FakeGenAiClient(
        [
            json.dumps(
                {
                    "is_vape": verdict.is_vape,
                    "detected_object": verdict.detected_object,
                    "confidence": verdict.confidence,
                    "hazard_note": verdict.hazard_note,
                }
            )
        ]
    )

    verifier = VlmVerifiedDetector(
        base_detector=_StubDetector([unseen_vape_det]),
        model_name="gemini-2.5-flash",
        client=client,
    )
    results = verifier.detect(_sample_frame())

    assert len(results) == 1
    promoted = results[0]
    assert promoted.label == "vape"
    assert promoted.confidence == pytest.approx(round(0.4 * 0.48 + 0.6 * 0.96, 4))
    assert promoted.grasp == grasp
    assert promoted.metadata["raw_yolo_label"] == "cylindrical_object"
    assert promoted.metadata["vlm"] == {
        "verified": True,
        "model": "gemini-2.5-flash",
        "is_vape": True,
        "detected_object": "disposable_vape",
        "confidence": 0.96,
        "hazard_note": "Embedded Li-ion cell + nicotine e-liquid fire hazard",
    }


def _ensure_google_genai_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register stub `google.genai` module specs when running in a bare `--extra dev` env."""
    try:
        has_real = importlib.util.find_spec("google.genai") is not None
    except (ModuleNotFoundError, ValueError):
        has_real = False
    if has_real:
        return

    google_mod = sys.modules.get("google")
    if google_mod is None:
        google_mod = types.ModuleType("google")
        google_mod.__path__ = []  # type: ignore[attr-defined]
        google_mod.__spec__ = importlib.machinery.ModuleSpec(
            "google",
            loader=None,
            is_package=True,
        )
        monkeypatch.setitem(sys.modules, "google", google_mod)

    genai_mod = types.ModuleType("google.genai")
    genai_mod.__spec__ = importlib.machinery.ModuleSpec("google.genai", loader=None)
    monkeypatch.setitem(sys.modules, "google.genai", genai_mod)
    monkeypatch.setattr(google_mod, "genai", genai_mod, raising=False)


def test_confidence_band_filtering_skips_vlm_call() -> None:
    low_det = Detection(label="vape", confidence=0.12, bounding_box=(10, 10, 40, 40))
    high_det = Detection(label="vape", confidence=0.90, bounding_box=(50, 50, 90, 90))
    client = _FakeGenAiClient([])

    # Uses default max_conf_to_verify=0.85 so high-confidence (> 0.85) skips VLM by default
    verifier = VlmVerifiedDetector(
        base_detector=_StubDetector([low_det, high_det]),
        min_conf_to_verify=0.25,
        client=client,
    )
    assert verifier.max_conf_to_verify == pytest.approx(0.85)
    results = verifier.detect(_sample_frame())

    assert len(results) == 2
    assert client.models.calls == []
    assert results[0].metadata["vlm"] == {"verified": False, "reason": "outside_conf_band"}
    assert results[1].metadata["vlm"] == {"verified": False, "reason": "outside_conf_band"}


def test_missing_api_key_and_network_error_fallback_gracefully(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ensure_google_genai_spec(monkeypatch)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    base_det = Detection(
        label="vape",
        confidence=0.55,
        bounding_box=(10, 15, 60, 55),
        grasp=_sample_grasp(),
    )

    # 1. Offline / missing API key fallback
    offline_verifier = VlmVerifiedDetector(
        base_detector=_StubDetector([base_det]),
        api_key=None,
        client=None,
    )
    offline_results = offline_verifier.detect(_sample_frame())
    assert len(offline_results) == 1
    assert offline_results[0].label == "vape"
    assert offline_results[0].confidence == pytest.approx(0.55)
    assert offline_results[0].grasp == base_det.grasp
    assert offline_results[0].metadata["vlm"] == {
        "verified": False,
        "reason": "missing_api_key",
    }

    # 2. Simulated network exception + malformed JSON fallback
    failing_client = _FakeGenAiClient([TimeoutError("Gemini request timed out"), "{bad json"])
    error_verifier = VlmVerifiedDetector(
        base_detector=_StubDetector([base_det, base_det]),
        client=failing_client,
    )
    error_results = error_verifier.detect(_sample_frame())
    assert len(error_results) == 2
    assert error_results[0].label == "vape"
    assert error_results[0].metadata["vlm"] == {
        "verified": False,
        "reason": "error:TimeoutError",
    }
    assert error_results[1].label == "vape"
    assert error_results[1].metadata["vlm"] == {
        "verified": False,
        "reason": "error:JSONDecodeError",
    }


def test_init_raises_import_error_when_google_genai_missing_and_no_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    with pytest.raises(ImportError, match="uv sync --extra vlm"):
        VlmVerifiedDetector(base_detector=_StubDetector([]), client=None)

    def _raise_missing_parent(name: str) -> None:
        raise ModuleNotFoundError(f"No module named '{name.partition('.')[0]}'")

    monkeypatch.setattr(importlib.util, "find_spec", _raise_missing_parent)
    with pytest.raises(ImportError, match="uv sync --extra vlm"):
        VlmVerifiedDetector(base_detector=_StubDetector([]), client=None)


def test_injected_client_works_without_google_genai_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "google.genai", None)
    monkeypatch.setitem(sys.modules, "google.genai.types", None)

    det = Detection(label="vape", confidence=0.50, bounding_box=(10, 10, 50, 50))
    verdict_json = json.dumps(
        {
            "is_vape": False,
            "detected_object": "bic_lighter",
            "confidence": 0.91,
            "hazard_note": "Gas lighter lookalike",
        }
    )
    client = _FakeGenAiClient([verdict_json])
    verifier = VlmVerifiedDetector(
        base_detector=_StubDetector([det]),
        client=client,
    )
    results = verifier.detect(_sample_frame())

    assert len(results) == 1
    assert results[0].label == "non_vape:bic_lighter"
    assert len(client.models.calls) == 1
    image_part = client.models.calls[0]["contents"][0]
    assert image_part.mime_type == "image/jpeg"
    assert isinstance(image_part.data, bytes)


def test_invalid_confidence_bounds_raise_value_error() -> None:
    with pytest.raises(ValueError, match="min_conf_to_verify"):
        VlmVerifiedDetector(
            base_detector=_StubDetector([]),
            min_conf_to_verify=0.90,
            max_conf_to_verify=0.40,
            client=_FakeGenAiClient([]),
        )
