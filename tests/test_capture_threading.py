"""Threaded capture: freshest-frame delivery and clean lifecycle.

Uses an injected fake capture device — no camera or OpenCV backend needed.
"""
import time

import numpy as np

from camera.capture import CaptureWrapper


class FakeCap:
    """Produces numbered frames; frame[0,0,0] encodes the sequence number."""

    def __init__(self, interval_s: float = 0.005) -> None:
        self.n = 0
        self.interval = interval_s
        self.released = False

    def read(self):
        time.sleep(self.interval)
        self.n += 1
        frame = np.zeros((4, 4, 3), dtype=np.uint16)
        frame[0, 0, 0] = self.n
        return True, frame

    def grab(self):
        return True

    def release(self):
        self.released = True


def _make(threaded: bool) -> tuple[CaptureWrapper, FakeCap]:
    fake = FakeCap()
    cw = CaptureWrapper("0", 4, 4, 30, threaded=threaded, cap_factory=lambda src: fake)
    return cw, fake


def test_threaded_returns_fresh_frames():
    cw, fake = _make(threaded=True)
    try:
        time.sleep(0.05)
        ok1, f1 = cw.read_latest()
        time.sleep(0.05)
        ok2, f2 = cw.read_latest()
        assert ok1 and ok2
        assert int(f2[0, 0, 0]) > int(f1[0, 0, 0]), "second read must be a newer frame"
    finally:
        cw.release()


def test_threaded_health_ticks():
    cw, _ = _make(threaded=True)
    try:
        time.sleep(0.05)
        assert len(cw.health.expected_frame_times) > 0
    finally:
        cw.release()


def test_release_stops_grabber_and_releases_device():
    cw, fake = _make(threaded=True)
    time.sleep(0.02)
    grabber = cw._grabber
    cw.release()
    assert fake.released
    assert grabber is not None and not grabber.is_alive()


def test_synchronous_mode_still_works():
    cw, _ = _make(threaded=False)
    try:
        ok, frame = cw.read_latest(buffer_drops=1)
        assert ok and frame is not None
    finally:
        cw.release()


class SlowStartCap(FakeCap):
    """Simulates a DSHOW/RTSP device that takes a while to deliver frame 1."""

    def __init__(self) -> None:
        super().__init__(interval_s=0.005)
        self._first = True

    def read(self):
        if self._first:
            self._first = False
            time.sleep(0.3)   # slow handshake before the first frame
        return super().read()


def test_immediate_read_after_construction_succeeds():
    """Regression: the constructor must warm up until the first frame exists,
    so callers never see a spurious failure (which triggers reconnect storms)."""
    fake = SlowStartCap()
    cw = CaptureWrapper("0", 4, 4, 30, threaded=True, cap_factory=lambda src: fake)
    try:
        ok, frame = cw.read_latest()   # no sleep — the very first poll
        assert ok and frame is not None, "startup race: no frame on first read"
    finally:
        cw.release()


def test_read_survives_gap_between_grabs():
    """A single missed instant between frames must not report failure."""
    fake = FakeCap(interval_s=0.03)
    cw = CaptureWrapper("0", 4, 4, 30, threaded=True, cap_factory=lambda src: fake)
    try:
        for _ in range(10):
            ok, _f = cw.read_latest()
            assert ok
    finally:
        cw.release()
