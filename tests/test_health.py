"""Camera health monitor: hysteresis prevents healthy/degraded flapping."""
import time

from camera.health import CameraHealth


def _feed(h: CameraHealth, interval_s: float, n: int, start: float | None = None) -> float:
    t = start if start is not None else time.time()
    for _ in range(n):
        t += interval_s
        h.expected_frame_times.append(t)
    return t


def test_fast_frames_stay_healthy():
    h = CameraHealth(cooldown_sec=0.0)
    _feed(h, 0.016, 30)  # ~60 fps
    for _ in range(10):
        assert h.check_health() == "healthy"


def test_slow_frames_escalate_to_degraded_after_required_bad_checks():
    h = CameraHealth(bad_checks_required=3, cooldown_sec=0.0,
                     degraded_mean_interval_s=0.05, dropping_frame_gap_s=5.0)
    _feed(h, 0.2, 30)  # 5 fps — clearly degraded
    states = [h.check_health() for _ in range(5)]
    assert states[0] == "healthy" and states[1] == "healthy", "must not flip on first bad checks"
    assert states[2] == "degraded", f"expected escalation on 3rd bad check, got {states}"


def test_recovery_requires_sustained_good_checks():
    h = CameraHealth(bad_checks_required=2, good_checks_required=4, cooldown_sec=0.0,
                     degraded_mean_interval_s=0.05, dropping_frame_gap_s=5.0)
    t = _feed(h, 0.2, 30)
    for _ in range(3):
        h.check_health()
    assert h.state == "degraded"
    # Now feed healthy frames; must NOT recover until 4 consecutive good checks
    h.expected_frame_times.clear()
    _feed(h, 0.016, 30, start=t)
    states = [h.check_health() for _ in range(6)]
    assert states[2] == "degraded", "recovered too eagerly"
    assert states[3] == "healthy" or states[4] == "healthy"


def test_single_large_gap_flags_dropping_frames():
    h = CameraHealth(bad_checks_required=1, cooldown_sec=0.0, dropping_frame_gap_s=0.5)
    t = _feed(h, 0.016, 15)
    h.expected_frame_times.append(t + 2.0)  # 2 s stall
    _feed(h, 0.016, 5, start=t + 2.0)
    assert h.check_health() == "dropping_frames"
