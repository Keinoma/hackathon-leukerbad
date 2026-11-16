"""Continuously acquire webcam frames while inference processes the latest one."""

from queue import Empty, Full, Queue
from threading import Event, Thread

import cv2


class LatestFrames:
    """Keep one fresh frame, discarding older frames instead of accumulating lag."""

    def __init__(self, camera):
        self.camera = camera
        self.frames = Queue(maxsize=1)
        self.stopped = Event()
        self.error = None
        self.thread = Thread(target=self._capture, name="webcam-capture", daemon=True)
        self.thread.start()

    def _capture(self):
        try:
            while not self.stopped.is_set():
                ok, frame = self.camera.read()
                if not ok:
                    self.stopped.wait(0.03)
                    continue
                try:
                    self.frames.get_nowait()
                except Empty:
                    pass
                try:
                    self.frames.put_nowait(frame)
                except Full:
                    pass
        except (cv2.error, OSError, RuntimeError) as error:
            self.error = error

    def read(self, timeout=0.1):
        if self.error is not None:
            raise RuntimeError(f"Camera capture failed: {self.error}") from self.error
        try:
            return True, self.frames.get(timeout=timeout)
        except Empty:
            return False, None

    def close(self):
        self.stopped.set()
        self.thread.join()
