"""Measured radial/tangential lens correction in the camera's raw orientation."""

import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np


class LensCorrection:
    def __init__(self, data):
        if data.get("version") != 1:
            raise ValueError("Unsupported lens calibration version")
        self.size = tuple(data["image_size"])
        self.camera_matrix = np.asarray(data["camera_matrix"], dtype=np.float64)
        self.distortion = np.asarray(data["dist_coeffs"], dtype=np.float64).reshape(-1)
        if (len(self.size) != 2 or min(self.size) <= 0
                or self.camera_matrix.shape != (3, 3) or len(self.distortion) != 5
                or not np.isfinite(self.camera_matrix).all()
                or not np.isfinite(self.distortion).all()
                or self.camera_matrix[0, 0] <= 0 or self.camera_matrix[1, 1] <= 0
                or not np.allclose(self.camera_matrix[2], [0, 0, 1])):
            raise ValueError("Invalid camera matrix or lens distortion coefficients")
        parameters = {"image_size": self.size, "camera_matrix": self.camera_matrix.tolist(),
                      "dist_coeffs": self.distortion.tolist()}
        self.id = hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest()
        self.maps = cv2.initUndistortRectifyMap(
            self.camera_matrix, self.distortion, None, self.camera_matrix,
            self.size, cv2.CV_32FC1,
        )

    def apply(self, frame):
        if (frame.shape[1], frame.shape[0]) != self.size:
            raise ValueError("Camera resolution changed: redo lens calibration")
        # Keep the full frame, original intrinsics, and pixel dimensions. No hidden crop.
        return cv2.remap(frame, *self.maps, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT)


def load_lens(path):
    if path is None or not Path(path).is_file():
        return None
    return LensCorrection(json.loads(Path(path).read_text(encoding="utf-8")))


def check_workspace_lens(workspace, lens):
    if workspace is not None and workspace.get("lens_id") != (lens.id if lens else None):
        raise ValueError("Lens correction changed: redo the five-point paper calibration first")


def fit_lens(views, image_size, board=(9, 6), square_mm=20):
    if len(views) < 12:
        raise ValueError("Capture at least 12 different checkerboard views")
    template = np.zeros((board[0] * board[1], 3), np.float32)
    template[:, :2] = np.mgrid[:board[0], :board[1]].T.reshape(-1, 2) * square_mm
    views = [np.asarray(view, np.float32).reshape(-1, 1, 2) for view in views]
    if any(len(view) != len(template) or not np.isfinite(view).all() for view in views):
        raise ValueError("Invalid checkerboard corner observations")
    rms, matrix, distortion, rotations, _ = cv2.calibrateCamera(
        [template.copy() for _ in views], views, image_size, None, None,
    )
    if not np.isfinite(rms) or rms > 1.0:
        raise ValueError(f"Reprojection error {rms:.2f}px is too high; use sharper, flatter views")
    normals = np.array([cv2.Rodrigues(rotation)[0][:, 2] for rotation in rotations])
    maximum_tilt = np.degrees(np.arccos(np.clip((normals @ normals.T).min(), -1, 1)))
    if maximum_tilt < 12:
        raise ValueError("Tilt the board more; front-facing views alone are insufficient")
    data = {"version": 1, "image_size": list(image_size),
            "camera_matrix": matrix.tolist(), "dist_coeffs": distortion.reshape(-1).tolist(),
            "rms_px": float(rms), "samples": len(views), "board_inner_corners": list(board)}
    LensCorrection(data)  # reject invalid estimates before saving them
    return data


def calibrate_lens(camera_index, output, backend):
    from leukerbad.workspace import write_json

    camera = cv2.VideoCapture(camera_index, backend)
    board, views, size = (9, 6), [], None
    title = "Lens: Space capture | Enter fit/save (12+ views) | r reset | q cancel"
    deadline = time.monotonic() + 15
    try:
        if not camera.isOpened():
            raise RuntimeError("Cannot open camera for lens calibration")
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        while True:
            ok, frame = camera.read()
            if not ok:
                if time.monotonic() > deadline:
                    raise RuntimeError("Camera stopped providing lens calibration images")
                if cv2.waitKey(30) & 0xFF in (ord("q"), 27):
                    return
                continue
            deadline = time.monotonic() + 15
            current_size = (frame.shape[1], frame.shape[0])
            if size is not None and current_size != size:
                raise ValueError("Camera resolution changed during lens calibration")
            size = current_size
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCornersSB(
                gray, board, flags=cv2.CALIB_CB_NORMALIZE_IMAGE,
            )
            preview = frame.copy()
            if found:
                cv2.drawChessboardCorners(preview, board, corners, True)
            cv2.putText(preview, f"{len(views)} views | move AND tilt the printed board | Space: save view",
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow(title, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                return
            if key == ord("r"):
                views.clear()
            if key == ord(" "):
                if not found:
                    print("Show the full checkerboard until coloured corners appear.", flush=True)
                elif any(np.sqrt(np.mean((view - corners) ** 2)) < 0.02 * np.hypot(*size)
                         for view in views):
                    print("Too similar to an existing view: move/tilt the board.", flush=True)
                else:
                    views.append(corners.copy())
                    print(f"Captured checkerboard view {len(views)}", flush=True)
            if key in (10, 13):
                try:
                    data = fit_lens(views, size, board)
                except ValueError as error:
                    print(f"{error}. Capture more views, or r to reset.", flush=True)
                else:
                    write_json(output, data)
                    print(f"Saved {Path(output).resolve()} | RMS {data['rms_px']:.3f}px. "
                          "Now redo the paper calibration.", flush=True)
                    return
    finally:
        camera.release()
        cv2.destroyAllWindows()
