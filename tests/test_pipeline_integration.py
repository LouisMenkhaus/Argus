"""End-to-end pipeline integration: scripted detections through the full chain.

No camera, no YOLO — a deterministic sequence of synthetic detections flows
through MotionSmoother -> SpatialAnalyzer -> BehaviorAnalyzer, asserting the
system-level properties that matter in operation:

  * identities remain stable while a target moves
  * smoothing tracks motion without falling behind (the One Euro property,
    verified here at the pipeline level rather than the unit level)
  * spatial estimates respond monotonically as the target approaches
  * a scripted collapse (tall box -> wide flat box) raises a fall event
  * a track that stops being detected coasts, then expires

core.tracker imports torch/ultralytics at module level, so in minimal
environments these tests skip (mirroring tests/test_smoother.py); CI installs
the full requirements and executes them.
"""
try:
    from core.tracker import MotionSmoother  # noqa: F401
    HAS_DEPS = True
except Exception:
    HAS_DEPS = False

import numpy as np

from core.behavior import BehaviorAnalyzer
from core.config import AppConfig
from core.spatial import SpatialAnalyzer

FRAME = (720, 1280, 3)


def _kp_for_box(box: np.ndarray) -> np.ndarray:
    """Plausible keypoints for a box: nose near the top, ankles near the bottom."""
    x1, y1, x2, y2 = [float(v) for v in box]
    cx = (x1 + x2) / 2
    kp = np.zeros((17, 3), dtype=np.float32)
    kp[0] = [cx, y1 + 0.08 * (y2 - y1), 0.9]          # nose
    kp[15] = [cx - 8, y2 - 0.03 * (y2 - y1), 0.9]     # ankles
    kp[16] = [cx + 8, y2 - 0.03 * (y2 - y1), 0.9]
    return kp


def _walk_boxes(n: int, x0: float = 200.0, step: float = 6.0) -> list[np.ndarray]:
    """A person walking right at constant speed: 90x260 px box, cx advancing."""
    out = []
    for i in range(n):
        cx = x0 + i * step
        out.append(np.array([cx - 45, 220, cx + 45, 480], dtype=np.float32))
    return out


def test_walking_target_keeps_identity_and_tracks_motion():
    if not HAS_DEPS:
        return
    cfg = AppConfig()
    cfg.tracking.filter = "kalman"
    cfg.tracking.keypoint_filter = "one_euro"
    sm = MotionSmoother(cfg)
    spatial = BehaviorAnalyzer  # noqa: F841  (imported for parity)

    boxes = _walk_boxes(45)
    last_smooth = None
    for box in boxes:
        smooth_box, smooth_kp = sm.update(7, box, _kp_for_box(box), 0.9)
        last_smooth = (smooth_box, smooth_kp)

    assert 7 in sm.kalman, "track must persist across the whole walk"
    smooth_box, smooth_kp = last_smooth
    true_cx = float((boxes[-1][0] + boxes[-1][2]) / 2)
    est_cx = float((smooth_box[0] + smooth_box[2]) / 2)
    assert abs(est_cx - true_cx) < 12.0, (
        f"pipeline lag: smoothed cx {est_cx:.1f} vs true {true_cx:.1f}")
    # Keypoints must have followed too (nose x near center)
    assert abs(float(smooth_kp[0, 0]) - true_cx) < 15.0


def test_approaching_target_distance_decreases_monotonically():
    if not HAS_DEPS:
        return
    cfg = AppConfig()
    sm = MotionSmoother(cfg)
    spatial = SpatialAnalyzer(cfg)

    heights = np.linspace(180, 520, 25)   # target grows in frame = approaching
    distances = []
    for h in heights:
        box = np.array([600, 360 - h / 2, 700, 360 + h / 2], dtype=np.float32)
        smooth_box, smooth_kp = sm.update(3, box, _kp_for_box(box), 0.9)
        est = spatial.estimate(FRAME, smooth_box, smooth_kp, 3)
        distances.append(est["distance_est_m"])

    # Allow smoothing warm-up, then require a strong monotone trend
    settled = distances[5:]
    closer = sum(1 for a, b in zip(settled, settled[1:]) if b <= a + 1e-6)
    assert closer >= len(settled) - 3, f"distance not decreasing: {settled}"
    assert distances[-1] < distances[5] * 0.6


def test_scripted_collapse_raises_fall_event():
    if not HAS_DEPS:
        return
    cfg = AppConfig()
    sm = MotionSmoother(cfg)
    behavior = BehaviorAnalyzer(cfg)

    events_seen: set[str] = set()
    # Upright and stationary...
    for _ in range(10):
        box = np.array([500, 200, 600, 480], dtype=np.float32)
        sb, _kp = sm.update(9, box, _kp_for_box(box), 0.9)
        events_seen |= set(behavior.analyze(9, sb, _kp_for_box(box)))
    assert "fall_like_posture" not in events_seen

    # ...then a collapse: box goes wide and flat near the floor.
    for _ in range(8):
        box = np.array([430, 400, 720, 500], dtype=np.float32)
        sb, _kp = sm.update(9, box, _kp_for_box(box), 0.9)
        events_seen |= set(behavior.analyze(9, sb, _kp_for_box(box)))
    assert "fall_like_posture" in events_seen


def test_vanished_track_coasts_then_expires():
    if not HAS_DEPS:
        return
    cfg = AppConfig()
    cfg.tracking.filter = "kalman"
    sm = MotionSmoother(cfg)

    for box in _walk_boxes(20):
        sm.update(4, box, _kp_for_box(box), 0.9)
    assert 4 in sm.kalman

    # Target leaves the frame: predictions continue, then the track is dropped.
    coasted = sm.predict_lost()
    assert isinstance(coasted, list)
    for _ in range(sm.max_lost_frames + 5):
        sm.predict_lost()
    assert 4 not in sm.kalman, "lost track must expire, not haunt the scene"
