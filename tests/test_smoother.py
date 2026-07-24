"""MotionSmoother: box/keypoint smoothing and track lifecycle.

These import from core.smoothing, which carries no inference dependency, so
they execute everywhere rather than silently skipping when torch is absent.
"""
import numpy as np

from core.config import AppConfig
from core.smoothing import MotionSmoother


def test_box_state_roundtrip():
    box = np.array([100, 50, 200, 250], dtype=np.float32)
    cx, cy, w, h = MotionSmoother._box_to_state(box)
    back = MotionSmoother._state_to_box(cx, cy, w, h)
    assert np.allclose(box, back, atol=1e-4)


def test_kalman_mode_smooths_noisy_boxes():
    cfg = AppConfig()
    cfg.tracking.filter = "kalman"
    sm = MotionSmoother(cfg)
    rng = np.random.default_rng(3)
    kp = np.zeros((17, 3), dtype=np.float32)
    raw_x, smooth_x = [], []
    for _ in range(40):
        noise = rng.normal(0, 6.0)
        box = np.array([100 + noise, 100, 200 + noise, 300], dtype=np.float32)
        sb, _ = sm.update(1, box, kp, 0.9)
        raw_x.append(float(box[0]))
        smooth_x.append(float(sb[0]))
    assert np.std(smooth_x[10:]) < np.std(raw_x[10:])


def test_lost_tracks_expire():
    cfg = AppConfig()
    cfg.tracking.filter = "kalman"
    sm = MotionSmoother(cfg)
    kp = np.zeros((17, 3), dtype=np.float32)
    sm.update(1, np.array([100, 100, 200, 300], dtype=np.float32), kp, 0.9)
    for _ in range(sm.max_lost_frames + 5):
        sm.predict_lost()
    assert 1 not in sm.kalman, "track must be dropped after max_lost_frames"


def test_one_euro_is_the_default_keypoint_filter():
    """Latency-critical default: keypoints use One Euro unless told otherwise."""
    cfg = AppConfig()
    assert cfg.tracking.keypoint_filter == "one_euro"
    sm = MotionSmoother(cfg)
    kp = np.zeros((17, 3), dtype=np.float32)
    kp[0] = [100.0, 100.0, 0.9]
    sm.update(5, np.array([50, 50, 150, 350], dtype=np.float32), kp, 0.9)
    assert 5 in sm.oe_filters, "One Euro filter should be instantiated per track"


def test_drop_clears_all_state_for_a_track():
    cfg = AppConfig()
    sm = MotionSmoother(cfg)
    kp = np.zeros((17, 3), dtype=np.float32)
    sm.update(11, np.array([10, 10, 60, 200], dtype=np.float32), kp, 0.9)
    sm.drop(11)
    assert 11 not in sm.kalman and 11 not in sm.oe_filters and 11 not in sm.kp_state
