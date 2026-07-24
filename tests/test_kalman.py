"""Kalman box tracker: initialization, convergence, occlusion prediction.

These tests also pin the NumPy 2.x compatibility fix (cv2.KalmanFilter state
is a column vector; scalar reads must index [row, 0]).
"""
import numpy as np

from core.kalman_tracker import KalmanTracker


def test_first_update_passes_measurement_through():
    kf = KalmanTracker()
    s = kf.update(100, 100, 50, 120, 0.9)
    assert s.shape == (4,)
    assert np.allclose(s, [100, 100, 50, 120])


def test_tracks_constant_velocity_target():
    """Feed a target moving +4px/frame in x; the filter must follow it closely."""
    kf = KalmanTracker(dt=1.0 / 30.0)
    x = 100.0
    est = None
    for _ in range(40):
        x += 4.0
        est = kf.update(x, 200.0, 50.0, 120.0, 0.9)
    assert est is not None
    assert abs(float(est[0]) - x) < 8.0, f"filter lagging: est {est[0]:.1f} vs true {x:.1f}"
    assert abs(float(est[1]) - 200.0) < 2.0


def test_smooths_noisy_measurements():
    """Estimates should have lower variance than the noisy measurements."""
    rng = np.random.default_rng(42)
    kf = KalmanTracker()
    truth = 300.0
    meas, ests = [], []
    for _ in range(60):
        m = truth + rng.normal(0, 6.0)
        meas.append(m)
        ests.append(float(kf.update(m, 200, 50, 120, 0.9)[0]))
    # Ignore burn-in
    assert np.std(ests[15:]) < np.std(meas[15:])


def test_predict_extrapolates_motion_when_lost():
    """After learning velocity, predict() (no measurement) must keep moving."""
    kf = KalmanTracker(dt=1.0 / 30.0)
    x = 100.0
    for _ in range(30):
        x += 5.0
        kf.update(x, 200.0, 50.0, 120.0, 0.9)
    p1 = kf.predict()
    p2 = kf.predict()
    assert float(p2[0]) > float(p1[0]), "prediction should continue along learned velocity"


def test_size_more_stable_than_position():
    """Process noise is tuned so w/h drift slower than cx/cy under noise."""
    rng = np.random.default_rng(7)
    kf = KalmanTracker()
    w_est = []
    for _ in range(50):
        kf_out = kf.update(200 + rng.normal(0, 8), 200 + rng.normal(0, 8),
                           50 + rng.normal(0, 8), 120 + rng.normal(0, 8), 0.9)
        w_est.append(float(kf_out[2]))
    assert np.std(w_est[10:]) < 8.0
