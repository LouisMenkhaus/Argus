from __future__ import annotations

"""
One Euro filter — speed-adaptive low-latency smoothing.

The classic pose-smoothing tradeoff: a fixed low-pass filter that removes
detector jitter also adds visible lag when the person moves. The One Euro
filter (Casiez, Roussel & Vogel, CHI 2012) resolves it by adapting the cutoff
frequency to the signal's speed:

  - at rest      -> low cutoff  -> strong smoothing, jitter suppressed
  - fast motion  -> high cutoff -> filter tracks the raw signal, minimal lag

This is the standard choice for interactive tracking (VR/AR controllers,
MediaPipe-style pose pipelines) precisely because perceived latency matters
more than perfect smoothness during motion — nobody notices jitter on a
fast-moving hand, everybody notices a skeleton trailing behind one.

Tuning:
  min_cutoff (Hz): jitter suppression at rest. Lower = smoother but laggier
                   when nearly still. 1.0 is a good default for webcam pose.
  beta:            speed responsiveness. Higher = less lag during motion.
                   With pixel-per-second speeds, 0.02-0.1 is typical.
  d_cutoff (Hz):   cutoff for the internal speed estimate. 1.0 is standard.

Implementation is vectorized: one filter instance smooths an entire (N, 2)
keypoint array with per-coordinate adaptive cutoffs.
"""

import math

import numpy as np


# The filter's response is a function of the time between samples. That is
# correct for a live camera, where dt is the frame interval — but the same code
# also runs when frames arrive as fast as the CPU allows (benchmark mode, video
# files, batch replay). There dt collapses toward zero, the smoothing factor
# goes with it, and the filter effectively freezes: output stops following
# input. Clamping dt to a plausible frame interval keeps behavior sane in both
# regimes, and after a long stall it prevents a single huge dt from snapping
# the estimate.
MIN_DT_S = 1.0 / 240.0   # treat anything faster than 240 fps as 240 fps
MAX_DT_S = 0.5           # after a stall, resume as if half a second passed


def _smoothing_factor(dt: float, cutoff: "np.ndarray | float") -> "np.ndarray | float":
    """Exponential smoothing factor for a given timestep and cutoff frequency."""
    tau = 1.0 / (2.0 * math.pi * np.maximum(cutoff, 1e-6))
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """Vectorized One Euro filter over an ndarray signal (e.g. (17, 2) keypoints)."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.05, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._t_prev: float | None = None
        self._x_prev: np.ndarray | None = None
        self._dx_prev: np.ndarray | None = None

    def reset(self) -> None:
        self._t_prev = None
        self._x_prev = None
        self._dx_prev = None

    def __call__(self, t: float, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)

        if self._t_prev is None or self._x_prev is None:
            self._t_prev = float(t)
            self._x_prev = x.copy()
            self._dx_prev = np.zeros_like(x)
            return x.copy()

        dt = float(t) - self._t_prev
        if dt <= 0.0:
            return self._x_prev.copy()
        self._t_prev = float(t)
        # See MIN_DT_S / MAX_DT_S: keeps the filter responsive when frames are
        # processed faster than real time, and stable after a stall.
        dt = min(MAX_DT_S, max(MIN_DT_S, dt))

        # Local bindings: the None-check above guarantees these are set, but
        # mypy cannot narrow Optional *instance attributes* across statements —
        # locals it can.
        x_prev = self._x_prev
        dx_prev = self._dx_prev
        assert x_prev is not None and dx_prev is not None

        # Smoothed derivative (speed estimate per coordinate)
        dx = (x - x_prev) / dt
        a_d = _smoothing_factor(dt, self.d_cutoff)
        dx_hat = a_d * dx + (1.0 - a_d) * dx_prev
        self._dx_prev = dx_hat

        # Speed-adaptive cutoff, then smooth the signal itself
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = _smoothing_factor(dt, cutoff)
        x_hat = a * x + (1.0 - a) * x_prev
        self._x_prev = x_hat
        return x_hat.copy()
