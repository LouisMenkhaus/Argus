"""Behavior events: fall posture, raised hands, fast motion."""
import numpy as np

from core.behavior import BehaviorAnalyzer
from core.config import AppConfig


def _kp() -> np.ndarray:
    return np.zeros((17, 3), dtype=np.float32)


def test_upright_person_no_fall_event():
    b = BehaviorAnalyzer(AppConfig())
    events = b.analyze(1, np.array([100, 100, 180, 400], dtype=np.float32), _kp())
    assert "fall_like_posture" not in events


def test_wide_flat_box_flags_fall_posture():
    b = BehaviorAnalyzer(AppConfig())
    events = b.analyze(1, np.array([100, 300, 500, 420], dtype=np.float32), _kp())
    assert "fall_like_posture" in events


def test_raised_hands_detected_when_wrists_above_head():
    b = BehaviorAnalyzer(AppConfig())
    kp = _kp()
    kp[0] = [200, 150, 0.9]    # nose
    kp[9] = [170, 90, 0.9]     # left wrist well above head
    kp[10] = [230, 92, 0.9]    # right wrist well above head
    events = b.analyze(2, np.array([150, 80, 250, 400], dtype=np.float32), kp)
    assert "raised_hands" in events


def test_hands_at_sides_not_flagged():
    b = BehaviorAnalyzer(AppConfig())
    kp = _kp()
    kp[0] = [200, 150, 0.9]
    kp[9] = [170, 300, 0.9]
    kp[10] = [230, 300, 0.9]
    events = b.analyze(3, np.array([150, 80, 250, 400], dtype=np.float32), kp)
    assert "raised_hands" not in events


def test_fast_motion_triggers_on_large_center_jump():
    cfg = AppConfig()
    b = BehaviorAnalyzer(cfg)
    b.analyze(4, np.array([0, 0, 100, 200], dtype=np.float32), _kp())
    events = b.analyze(4, np.array([200, 300, 300, 500], dtype=np.float32), _kp())
    assert "fast_motion" in events


def test_slow_motion_does_not_trigger():
    b = BehaviorAnalyzer(AppConfig())
    b.analyze(5, np.array([100, 100, 200, 300], dtype=np.float32), _kp())
    events = b.analyze(5, np.array([102, 101, 202, 301], dtype=np.float32), _kp())
    assert "fast_motion" not in events
