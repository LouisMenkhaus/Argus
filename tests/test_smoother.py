"""MotionSmoother box/keypoint smoothing.

core.tracker imports torch/ultralytics at module level, so these tests are
skipped in minimal environments and run fully in CI (where requirements.txt
is installed).
"""
try:
    from core.tracker import MotionSmoother  # noqa: F401
    HAS_DEPS = True
except Exception:
    HAS_DEPS = False

import numpy as np

from core.config import AppConfig


def test_box_state_roundtrip():
    if not HAS_DEPS:
        return  # skipped: torch/ultralytics not installed
    from core.tracker import MotionSmoother
    box = np.array([100, 50, 200, 250], dtype=np.float32)
    cx, cy, w, h = MotionSmoother._box_to_state(box)
    back = MotionSmoother._state_to_box(cx, cy, w, h)
    assert np.allclose(box, back, atol=1e-4)


def test_kalman_mode_smooths_noisy_boxes():
    if not HAS_DEPS:
        return
    from core.tracker import MotionSmoother
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
    if not HAS_DEPS:
        return
    from core.tracker import MotionSmoother
    cfg = AppConfig()
    cfg.tracking.filter = "kalman"
    sm = MotionSmoother(cfg)
    kp = np.zeros((17, 3), dtype=np.float32)
    sm.update(1, np.array([100, 100, 200, 300], dtype=np.float32), kp, 0.9)
    for _ in range(sm.max_lost_frames + 5):
        sm.predict_lost()
    assert 1 not in sm.kalman, "track must be dropped after max_lost_frames"
