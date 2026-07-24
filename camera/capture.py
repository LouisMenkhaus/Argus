from __future__ import annotations

import contextlib
import platform
import threading
import time
from typing import Any, Callable, Optional

import cv2

from camera.health import CameraHealth
from core.config import HealthConfig


class _FrameGrabber(threading.Thread):
    """Background thread that continuously reads the newest frame.

    Latency rationale: with synchronous capture, every pipeline cycle pays
    (camera frame interval + decode) BEFORE inference even starts, and frames
    queue in the driver buffer while inference runs — so the frame being
    processed is already 1-3 frames old. Reading continuously in a thread and
    always handing the pipeline the FRESHEST frame removes camera I/O from the
    critical path and drops stale frames instead of processing them.
    """

    def __init__(self, cap: "cv2.VideoCapture", health: CameraHealth) -> None:
        super().__init__(daemon=True)
        self._cap = cap
        self._health = health
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._frame: Optional[Any] = None
        self._frame_time: float = 0.0
        self._ok: bool = False
        self.first_frame = threading.Event()

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                ret, frame = self._cap.read()
            except Exception:
                ret, frame = False, None
            if ret and frame is not None:
                self._health.tick()
                with self._lock:
                    self._frame = frame
                    self._frame_time = time.monotonic()
                    self._ok = True
                self.first_frame.set()
            else:
                with self._lock:
                    self._ok = False
                time.sleep(0.01)

    def latest(self, max_age_s: float = 1.0) -> tuple[bool, Optional[Any]]:
        with self._lock:
            if not self._ok or self._frame is None:
                return False, None
            if (time.monotonic() - self._frame_time) > max_age_s:
                return False, None
            return True, self._frame

    def stop(self) -> None:
        self._stop_event.set()


class CaptureWrapper:
    def __init__(
        self,
        source: str,
        width: int,
        height: int,
        fps: int,
        prefer_dshow: bool = True,
        health_cfg: HealthConfig | None = None,
        threaded: bool = True,
        cap_factory: Optional[Callable[[str], Any]] = None,
    ) -> None:
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.prefer_dshow = prefer_dshow
        self.threaded = threaded
        self._cap_factory = cap_factory  # injectable for tests
        hc = health_cfg
        self.health = CameraHealth(
            bad_checks_required=hc.bad_checks_required if hc else 3,
            good_checks_required=hc.good_checks_required if hc else 5,
            degraded_mean_interval_s=hc.degraded_mean_interval_s if hc else 0.12,
            dropping_frame_gap_s=hc.dropping_frame_gap_s if hc else 0.50,
            cooldown_sec=hc.cooldown_sec if hc else 2.0,
        )
        self.cap = self._open(source)
        self._grabber: Optional[_FrameGrabber] = None
        if self.threaded:
            self._start_grabber()
            # Warm-up: block briefly for the first frame. Webcam (DSHOW) and
            # RTSP handshakes can take 1-3 s; returning "no frame" during that
            # window makes callers reconnect, which reopens the device and
            # restarts the wait — an endless reconnect storm that presents as
            # "camera can't open".
            if self._grabber is not None:
                self._grabber.first_frame.wait(timeout=8.0)

    def _open(self, source: str) -> Any:
        if self._cap_factory is not None:
            return self._cap_factory(source)
        if source.startswith("rtsp://"):
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        else:
            if source.isdigit():
                idx = int(source)
                if platform.system().lower().startswith("win") and self.prefer_dshow:
                    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                else:
                    cap = cv2.VideoCapture(idx)
            else:
                cap = cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.height))
        cap.set(cv2.CAP_PROP_FPS, int(self.fps))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _start_grabber(self) -> None:
        self._grabber = _FrameGrabber(self.cap, self.health)
        self._grabber.start()

    def read_latest(self, buffer_drops: int = 2) -> tuple[bool, Any]:
        if self.threaded and self._grabber is not None:
            # A frame arrives roughly every 1/fps seconds; poll briefly rather
            # than declaring failure in the gap between two grabs.
            wait_s = max(0.15, 4.0 / max(1, int(self.fps)))
            deadline = time.monotonic() + wait_s
            while True:
                ok, frame = self._grabber.latest()
                if ok:
                    return True, frame
                if time.monotonic() >= deadline:
                    return False, None
                time.sleep(0.005)

        # Synchronous fallback: drain the driver buffer, then read.
        for _ in range(max(0, int(buffer_drops))):
            try:
                self.cap.grab()
            except Exception:
                break
        ret, frame = self.cap.read()
        if ret and frame is not None:
            self.health.tick()
        return ret, frame

    def reconnect(self, sleep_s: float = 0.2) -> None:
        # The grabber owns the capture while running — stop it before touching cap.
        if self._grabber is not None:
            self._grabber.stop()
            self._grabber.join(timeout=1.0)
            self._grabber = None
        # Releasing an already-dead device raises on some backends; the goal
        # is simply that the handle is gone.
        with contextlib.suppress(Exception):
            self.cap.release()
        time.sleep(sleep_s)
        self.cap = self._open(self.source)
        if self.threaded:
            self._start_grabber()

    def release(self) -> None:
        if self._grabber is not None:
            self._grabber.stop()
            self._grabber.join(timeout=1.0)
            self._grabber = None
        # Releasing an already-dead device raises on some backends; the goal
        # is simply that the handle is gone.
        with contextlib.suppress(Exception):
            self.cap.release()
