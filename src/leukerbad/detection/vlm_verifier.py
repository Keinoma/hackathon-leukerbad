"""Multimodal VLM verification wrapper for filtering tabletop vape lookalikes."""

import json
import os
from dataclasses import dataclass, replace
from typing import Any

import cv2
import numpy as np

from leukerbad.detection.base import Detection, Detector

_SYSTEM_INSTRUCTION = (
    "You are a hazmat robotics vision verifier inspecting cropped tabletop objects for a "
    "dual-arm waste-sorting cell. Distinguish true disposable vapes, e-cigarettes, puff bars, "
    "and pod vapes (embedded Li-ion battery + e-liquid fire hazard) from benign tabletop "
    "lookalikes. Inspect fine structural cues:\n"
    "- Vapes: contoured/duckbill or cylindrical mouthpiece, bottom airflow holes, LED battery "
    "end-cap, e-liquid window, compact rounded-rectangular or cylindrical puff-bar body.\n"
    "- Benign lookalikes: AirPods/wireless earbud cases (clamshell lid seam, no mouthpiece), "
    "pens/markers/highlighters (pocket clip, pen cap, writing tip), lipsticks/lip balms, "
    "USB flash drives (metal connector plug), BIC/gas lighters (metal wind hood, spark wheel).\n"
    "Respond ONLY with a single JSON object matching:\n"
    '{"is_vape": bool, "detected_object": "<snake_case_name>", '
    '"confidence": <float 0.0..1.0>, "hazard_note": "<concise hazard or lookalike note>"}'
)


@dataclass(frozen=True)
class VlmVerdict:
    """Structured multimodal verdict returned by the VLM lookalike verifier."""

    is_vape: bool
    detected_object: str
    confidence: float
    hazard_note: str


def crop_detection_roi(
    frame: np.ndarray,
    bounding_box: tuple[int, int, int, int],
    padding_ratio: float = 0.20,
) -> np.ndarray:
    """Return a padded, strictly bounds-clamped, non-empty ROI crop from ``frame``."""
    if frame.ndim < 2 or frame.shape[0] == 0 or frame.shape[1] == 0:
        raise ValueError("frame must be a non-empty 2D or 3D image array")

    frame_h, frame_w = frame.shape[:2]
    raw_x1, raw_y1, raw_x2, raw_y2 = bounding_box
    x1, x2 = min(raw_x1, raw_x2), max(raw_x1, raw_x2)
    y1, y2 = min(raw_y1, raw_y2), max(raw_y1, raw_y2)

    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    ratio = max(0.0, float(padding_ratio))
    pad_x = round(box_w * ratio)
    pad_y = round(box_h * ratio)

    cx1 = max(0, min(frame_w - 1, x1 - pad_x))
    cy1 = max(0, min(frame_h - 1, y1 - pad_y))
    cx2 = max(cx1 + 1, min(frame_w, x2 + pad_x))
    cy2 = max(cy1 + 1, min(frame_h, y2 + pad_y))

    return frame[cy1:cy2, cx1:cx2].copy()


def _parse_verdict_json(raw_text: str) -> VlmVerdict:
    """Parse a JSON string from Gemini into a validated ``VlmVerdict``."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    payload = json.loads(cleaned)
    if not isinstance(payload, dict) or "is_vape" not in payload:
        raise ValueError("VLM response JSON must be an object containing 'is_vape'")

    raw_is_vape = payload["is_vape"]
    if isinstance(raw_is_vape, bool):
        is_vape = raw_is_vape
    elif isinstance(raw_is_vape, str) and raw_is_vape.lower() in {"true", "false"}:
        is_vape = raw_is_vape.lower() == "true"
    else:
        raise ValueError("VLM 'is_vape' field must be a boolean")

    raw_obj = str(payload.get("detected_object") or "").strip().lower().replace(" ", "_")
    if not raw_obj:
        raw_obj = "disposable_vape" if is_vape else "unknown_object"

    raw_conf = float(payload.get("confidence", 0.5))
    confidence = max(0.0, min(1.0, raw_conf))

    default_note = (
        "Embedded Li-ion cell + nicotine e-liquid fire hazard"
        if is_vape
        else "Benign non-vape lookalike"
    )
    hazard_note = str(payload.get("hazard_note") or default_note).strip() or default_note

    return VlmVerdict(
        is_vape=is_vape,
        detected_object=raw_obj,
        confidence=confidence,
        hazard_note=hazard_note,
    )


class VlmVerifiedDetector:
    """Two-stage detector that verifies ambiguous YOLO crops with Gemini VLM."""

    def __init__(
        self,
        base_detector: Detector,
        model_name: str = "gemini-2.5-flash",
        api_key: str | None = None,
        min_conf_to_verify: float = 0.20,
        max_conf_to_verify: float = 1.0,
        padding_ratio: float = 0.20,
        reject_non_vapes: bool = False,
        client: Any | None = None,
    ) -> None:
        self.base_detector = base_detector
        self.model_name = model_name
        self.api_key = (
            api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        )
        self.min_conf_to_verify = min_conf_to_verify
        self.max_conf_to_verify = max_conf_to_verify
        self.padding_ratio = padding_ratio
        self.reject_non_vapes = reject_non_vapes
        self._client = client

    def _get_or_create_client(self) -> Any:
        if self._client is not None:
            return self._client
        from google import genai

        self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _verify_crop(self, jpeg_bytes: bytes, det: Detection) -> VlmVerdict:
        from google.genai import types

        client = self._get_or_create_client()
        image_part = types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg")
        user_prompt = (
            f"Stage-1 detector proposed label='{det.label}' (confidence={det.confidence:.2f}). "
            "Inspect mouthpieces, airflow holes, LED caps, seams, and clips to determine "
            "whether this object is a real vape/e-cigarette/puff or a benign lookalike "
            "(AirPods/earbud case, pen, highlighter, marker, lipstick, USB stick, BIC lighter)."
        )
        config = types.GenerateContentConfig(
            system_instruction=_SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            temperature=0.0,
        )
        response = client.models.generate_content(
            model=self.model_name,
            contents=[image_part, user_prompt],
            config=config,
        )
        raw_text = getattr(response, "text", None)
        if not isinstance(raw_text, str) or not raw_text.strip():
            raise ValueError("Empty response text from VLM client")
        return _parse_verdict_json(raw_text)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run stage-1 detection and verify in-band candidates via Gemini VLM."""
        detections = self.base_detector.detect(frame)
        verified_detections: list[Detection] = []

        for det in detections:
            if det.confidence < self.min_conf_to_verify or det.confidence > self.max_conf_to_verify:
                verified_detections.append(
                    replace(
                        det,
                        metadata={
                            **det.metadata,
                            "vlm": {"verified": False, "reason": "outside_conf_band"},
                        },
                    )
                )
                continue

            if self._client is None and not self.api_key:
                verified_detections.append(
                    replace(
                        det,
                        metadata={
                            **det.metadata,
                            "vlm": {"verified": False, "reason": "missing_api_key"},
                        },
                    )
                )
                continue

            try:
                crop = crop_detection_roi(
                    frame,
                    det.bounding_box,
                    padding_ratio=self.padding_ratio,
                )
                encoded_ok, encoded_buf = cv2.imencode(".jpg", crop)
                if not encoded_ok:
                    raise RuntimeError("cv2.imencode failed for ROI crop")
                verdict = self._verify_crop(encoded_buf.tobytes(), det)
            except Exception as exc:  # noqa: BLE001
                verified_detections.append(
                    replace(
                        det,
                        metadata={
                            **det.metadata,
                            "vlm": {
                                "verified": False,
                                "reason": f"error:{exc.__class__.__name__}",
                            },
                        },
                    )
                )
                continue

            vlm_meta = {
                "verified": True,
                "model": self.model_name,
                "is_vape": verdict.is_vape,
                "detected_object": verdict.detected_object,
                "confidence": verdict.confidence,
                "hazard_note": verdict.hazard_note,
            }

            if verdict.is_vape:
                blended_conf = round(0.4 * det.confidence + 0.6 * verdict.confidence, 4)
                verified_detections.append(
                    replace(
                        det,
                        label="vape",
                        confidence=blended_conf,
                        metadata={
                            **det.metadata,
                            "raw_yolo_label": det.label,
                            "vlm": vlm_meta,
                        },
                    )
                )
            else:
                if self.reject_non_vapes:
                    continue
                verified_detections.append(
                    replace(
                        det,
                        label=f"non_vape:{verdict.detected_object}",
                        confidence=round(verdict.confidence, 4),
                        metadata={
                            **det.metadata,
                            "raw_yolo_label": det.label,
                            "vlm": vlm_meta,
                        },
                    )
                )

        return verified_detections
