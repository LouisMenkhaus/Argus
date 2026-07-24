from __future__ import annotations

import cv2
import numpy as np


class KalmanTracker:
    """
    8-state box tracker:
      state = [cx, cy, w, h, vx, vy, vw, vh]
      meas  = [cx, cy, w, h]
    Tuned for smoother motion, size stability, and better forward/back response.
    """
    def __init__(self, dt: float = 1.0 / 30.0) -> None:
        self.dt = float(dt)
        self.kf = cv2.KalmanFilter(8, 4)

        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ], np.float32)

        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, self.dt, 0, 0, 0],
            [0, 1, 0, 0, 0, self.dt, 0, 0],
            [0, 0, 1, 0, 0, 0, self.dt, 0],
            [0, 0, 0, 1, 0, 0, 0, self.dt],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ], np.float32)

        # Tuned process noise: allow realistic movement while keeping size steadier.
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 0.1
        self.kf.processNoiseCov[2:4, 2:4] *= 0.3   # w/h change slower than center
        self.kf.processNoiseCov[4:6, 4:6] *= 5.0   # x/y velocity should be responsive
        self.kf.processNoiseCov[6:8, 6:8] *= 3.0   # size velocity also moves, but slower

        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.5
        self.kf.errorCovPost = np.eye(8, dtype=np.float32) * 0.1
        self.initialized = False

    def update(
        self, cx: float, cy: float, w: float, h: float, confidence: float = 1.0
    ) -> np.ndarray:
        measurement = np.array([[cx], [cy], [w], [h]], dtype=np.float32)

        if not self.initialized:
            state = np.array([[cx], [cy], [w], [h], [0], [0], [0], [0]], dtype=np.float32)
            self.kf.statePre = state.copy()
            self.kf.statePost = state.copy()
            self.initialized = True
            return np.array([cx, cy, w, h], dtype=np.float32)

        # Confidence-aware measurement noise; bounded so we never over-trust detections.
        noise = float(np.clip(0.5 / (confidence + 0.1), 0.2, 1.5))
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * noise

        # Speed-adaptive position velocity process noise.
        # Note: cv2.KalmanFilter state vectors are column vectors (shape (8,1)),
        # so scalar reads must index [row, 0] — float() on a 1-element array is
        # an error on NumPy >= 2.0.
        vx = float(self.kf.statePost[4, 0])
        vy = float(self.kf.statePost[5, 0])
        speed = float(np.hypot(vx, vy))
        if speed > 10.0:
            self.kf.processNoiseCov[4:6, 4:6] = np.eye(2, dtype=np.float32) * 2.0
        else:
            self.kf.processNoiseCov[4:6, 4:6] = np.eye(2, dtype=np.float32) * 0.5

        self.kf.predict()
        estimated = self.kf.correct(measurement)
        return np.array(
            [
                float(estimated[0, 0]),
                float(estimated[1, 0]),
                float(estimated[2, 0]),
                float(estimated[3, 0]),
            ],
            dtype=np.float32,
        )

    def predict(self) -> np.ndarray:
        """Advance the filter one step with no measurement (occlusion/lost frames).
        Returns the predicted [cx, cy, w, h] as a flat float32 array."""
        pred = self.kf.predict()
        return np.array(
            [float(pred[0, 0]), float(pred[1, 0]), float(pred[2, 0]), float(pred[3, 0])],
            dtype=np.float32,
        )
