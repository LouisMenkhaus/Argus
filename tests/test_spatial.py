"""Monocular spatial estimates: distance monotonicity, azimuth sign, smoothing."""
import numpy as np

from core.config import AppConfig
from core.spatial import SpatialAnalyzer

FRAME = (720, 1280, 3)


def _box(cx: float, h_px: float, w_px: float = 80.0) -> np.ndarray:
    return np.array([cx - w_px / 2, 360 - h_px / 2, cx + w_px / 2, 360 + h_px / 2],
                    dtype=np.float32)


def test_distance_in_plausible_range():
    a = SpatialAnalyzer(AppConfig())
    r = a.estimate(FRAME, np.array([100, 100, 200, 300], dtype=np.float32), [], 1)
    assert 1.0 < r["distance_est_m"] < 20.0


def test_taller_in_frame_means_closer():
    """A person occupying more vertical pixels must be estimated nearer."""
    a = SpatialAnalyzer(AppConfig())
    far = a.estimate(FRAME, _box(640, 150), [], None)
    near = a.estimate(FRAME, _box(640, 500), [], None)
    assert near["distance_est_m"] < far["distance_est_m"]


def test_azimuth_sign_matches_frame_side():
    a = SpatialAnalyzer(AppConfig())
    left = a.estimate(FRAME, _box(200, 300), [], None)
    center = a.estimate(FRAME, _box(640, 300), [], None)
    right = a.estimate(FRAME, _box(1100, 300), [], None)
    assert left["azimuth_deg"] < 0 < right["azimuth_deg"]
    assert abs(center["azimuth_deg"]) < 2.0


def test_per_id_smoothing_damps_jitter():
    """Same track id: a sudden jump in box height should be partially absorbed."""
    a = SpatialAnalyzer(AppConfig())
    for _ in range(4):
        a.estimate(FRAME, _box(640, 300), [], track_id=7)
    steady = a.estimate(FRAME, _box(640, 300), [], track_id=7)["distance_est_m"]
    jumped = a.estimate(FRAME, _box(640, 360), [], track_id=7)["distance_est_m"]
    fresh = SpatialAnalyzer(AppConfig())
    raw_unsmoothed = fresh.estimate(FRAME, _box(640, 360), [], None)["distance_est_m"]
    assert abs(jumped - steady) < abs(raw_unsmoothed - steady)


def test_keypoints_refine_visible_height():
    """Confident head+ankle keypoints should override a loose detection box."""
    a = SpatialAnalyzer(AppConfig())
    box = _box(640, 400)
    kp = np.zeros((17, 3), dtype=np.float32)
    kp[0] = [640, 200, 0.9]    # nose near box top
    kp[15] = [630, 520, 0.9]   # ankles
    kp[16] = [650, 520, 0.9]
    r = a.estimate(FRAME, box, kp, None)
    assert r["box_h_px"] != 400.0  # keypoint-derived height was used (clipped to box range)
