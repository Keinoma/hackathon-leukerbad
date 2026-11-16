"""Check grasp geometry, snapshot safety, and camera cleanup without hardware."""

from argparse import Namespace
from threading import Event

import cv2
import numpy as np
import pytest

from leukerbad.cli import next_snapshot, run_camera
from leukerbad.detection.camera import LatestFrames
from leukerbad.detection.yolo import grasp_line


class DirectFrames:
    """Make camera-loop tests deterministic; acquisition is tested separately."""

    def __init__(self, camera):
        self.camera = camera

    def read(self):
        return self.camera.read()

    def close(self):
        pass


@pytest.mark.parametrize("angle", [0, 30, 90, 135])
def test_grasp_spans_short_axis_of_rotated_mask(angle):
    polygon = cv2.boxPoints(((100, 120), (20, 80), angle))
    center, first, second = grasp_line(polygon)
    np.testing.assert_allclose(center, [100, 120], atol=0.001)
    assert np.linalg.norm(second - first) == pytest.approx(20, abs=0.001)
    direction = polygon[1] - polygon[0]
    assert abs(np.dot(second - first, direction)) < 0.01


def test_degenerate_mask_has_no_grasp():
    assert grasp_line(np.zeros((0, 2))) is None
    assert grasp_line(np.zeros((4, 2))) is None


def test_snapshot_preserves_previous_sessions(tmp_path):
    (tmp_path / "snapshot_0001.png").touch()
    assert next_snapshot(tmp_path) == tmp_path / "snapshot_0002.png"


def test_failed_camera_is_released(monkeypatch):
    class Camera:
        released = False

        def isOpened(self):
            return False

        def release(self):
            self.released = True

    camera = Camera()
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args: camera)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    with pytest.raises(RuntimeError, match="Cannot open camera 0"):
        run_camera(Namespace(camera=0), None)
    assert camera.released


def test_camera_keeps_native_format_and_recovers_from_a_dropped_frame(monkeypatch):
    frame = np.zeros((24, 32, 3), dtype=np.uint8)

    class Camera:
        released = False
        frames = iter([(False, None), (True, frame)])

        def isOpened(self):
            return True

        def read(self):
            return next(self.frames)

        def release(self):
            self.released = True

    class Detector:
        def annotate(self, image):
            assert image is frame
            return image.copy(), []

    camera = Camera()
    monkeypatch.setattr("leukerbad.cli.LatestFrames", DirectFrames)
    keys = iter([-1, ord("q")])
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args: camera)
    monkeypatch.setattr(cv2, "waitKey", lambda delay: next(keys))
    monkeypatch.setattr(cv2, "imshow", lambda *args: None)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    run_camera(Namespace(camera=0), Detector())
    assert camera.released


def test_camera_exits_after_sustained_frame_loss(monkeypatch):
    class Camera:
        released = False

        def isOpened(self):
            return True

        def read(self):
            return False, None

        def release(self):
            self.released = True

    camera = Camera()
    monkeypatch.setattr("leukerbad.cli.LatestFrames", DirectFrames)
    clock = iter([0.0, 3.1])
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args: camera)
    monkeypatch.setattr(cv2, "waitKey", lambda delay: -1)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr("leukerbad.cli.time.monotonic", lambda: next(clock))
    with pytest.raises(RuntimeError, match="3 seconds"):
        run_camera(Namespace(camera=0), None)
    assert camera.released


def test_continuous_capture_discards_old_frames_and_never_replays_them():
    exhausted = Event()

    class Camera:
        index = 0

        def read(self):
            if self.index < 100:
                self.index += 1
                return True, self.index
            exhausted.set()
            return False, None

    frames = LatestFrames(Camera())
    try:
        assert exhausted.wait(timeout=2)
        assert frames.read() == (True, 100)
        assert frames.read(timeout=0.01) == (False, None)
    finally:
        frames.close()
    assert not frames.thread.is_alive()
