"""One Euro filter: the low-latency property that motivated adopting it.

The whole point: during motion it must track far closer to the raw signal
than the old fixed EMA, while still suppressing jitter at rest.
"""
import numpy as np

from core.one_euro import OneEuroFilter

DT = 1.0 / 30.0


def test_first_sample_passthrough():
    f = OneEuroFilter()
    out = f(0.0, np.array([[100.0, 200.0]]))
    assert np.allclose(out, [[100.0, 200.0]])


def test_far_less_lag_than_legacy_ema_on_step():
    """Signal at 0, then steps to 100 px and moves fast: One Euro must close
    the gap dramatically faster than the old alpha_slow=0.88 EMA."""
    oe = OneEuroFilter(min_cutoff=1.0, beta=0.05)
    t = 0.0
    ema = 0.0
    oe_out = 0.0
    for _ in range(30):        # settle at rest
        t += DT
        oe_out = float(oe(t, np.array([0.0]))[0])
        ema = 0.88 * ema + 0.12 * 0.0

    x = 0.0
    for _ in range(5):         # 5 frames of fast motion (600 px/s)
        t += DT
        x += 20.0
        oe_out = float(oe(t, np.array([x]))[0])
        ema = 0.88 * ema + 0.12 * x

    oe_lag = abs(x - oe_out)
    ema_lag = abs(x - ema)
    assert oe_lag < ema_lag * 0.25, f"One Euro lag {oe_lag:.1f}px vs EMA {ema_lag:.1f}px"
    assert oe_lag < 10.0, f"absolute lag too high: {oe_lag:.1f}px"


def test_still_smooths_jitter_at_rest():
    rng = np.random.default_rng(11)
    f = OneEuroFilter(min_cutoff=1.0, beta=0.05)
    t = 0.0
    raw, filt = [], []
    for _ in range(90):
        t += DT
        m = 300.0 + rng.normal(0, 2.0)   # detector jitter on a still person
        raw.append(m)
        filt.append(float(f(t, np.array([m]))[0]))
    assert np.std(filt[20:]) < np.std(raw[20:]) * 0.6


def test_vectorized_over_keypoint_array():
    f = OneEuroFilter()
    kp = np.random.default_rng(0).uniform(0, 640, size=(17, 2)).astype(np.float32)
    out1 = f(0.0, kp)
    out2 = f(DT, kp + 1.0)
    assert out1.shape == (17, 2) and out2.shape == (17, 2)


def test_non_advancing_time_returns_previous():
    f = OneEuroFilter()
    f(1.0, np.array([5.0]))
    out = f(1.0, np.array([999.0]))   # same timestamp — must not divide by zero
    assert float(out[0]) == 5.0


def test_survives_faster_than_realtime_frame_delivery():
    """Regression: benchmark mode and video-file playback deliver frames as
    fast as the CPU allows. With unclamped dt the smoothing factor collapses
    toward zero and the filter stops following its input entirely — the
    skeleton freezes while the subject moves. dt is clamped to a plausible
    frame interval to keep the filter responsive in that regime."""
    f = OneEuroFilter(min_cutoff=1.0, beta=0.05)
    t = 0.0
    x = 0.0
    out = 0.0
    for _ in range(45):
        t += 1e-5           # ~100,000 fps: what a tight processing loop looks like
        x += 6.0
        out = float(f(t, np.array([x]))[0])
    lag = abs(x - out)
    assert lag < 25.0, f"filter froze under fast delivery: {lag:.1f}px behind"


def test_dt_clamp_does_not_harm_normal_framerate():
    """The clamp must not change behavior at ordinary camera framerates."""
    f = OneEuroFilter(min_cutoff=1.0, beta=0.05)
    t = 0.0
    x = 0.0
    out = 0.0
    for _ in range(45):
        t += DT             # 30 fps
        x += 6.0
        out = float(f(t, np.array([x]))[0])
    assert abs(x - out) < 12.0
