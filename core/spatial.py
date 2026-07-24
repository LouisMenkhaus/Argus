from __future__ import annotations

import math
from collections import deque

import numpy as np

from core.config import AppConfig


class SpatialAnalyzer:
    """
    Single-camera spatial estimates with per-ID smoothing.
    Uses visible body height and a vertical focal-length approximation for better
    distance behavior from a standard webcam feed.
    """

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self._distance_by_id: dict[int, deque[float]] = {}

    def _visible_body_height_px(self, box: np.ndarray, kp: np.ndarray) -> float:
        x1, y1, x2, y2 = [float(v) for v in box]
        box_h = max(1.0, y2 - y1)

        if kp is None or len(kp) == 0:
            return box_h

        conf_t = float(self.cfg.spatial.keypoint_conf_threshold)
        top_idxs = [0, 1, 2, 3, 4, 5, 6]
        bottom_idxs = [15, 16, 13, 14, 11, 12]

        top_y = None
        bottom_y = None

        for idx in top_idxs:
            if idx < len(kp) and float(kp[idx][2]) >= conf_t:
                y = float(kp[idx][1])
                top_y = y if top_y is None else min(top_y, y)

        for idx in bottom_idxs:
            if idx < len(kp) and float(kp[idx][2]) >= conf_t:
                y = float(kp[idx][1])
                bottom_y = y if bottom_y is None else max(bottom_y, y)

        if top_y is not None and bottom_y is not None and bottom_y > top_y:
            visible_h = bottom_y - top_y
            return float(np.clip(visible_h, box_h * 0.55, box_h * 1.08))

        return box_h

    def estimate(
        self,
        frame_shape: tuple[int, int, int],
        box: np.ndarray,
        kp: np.ndarray,
        track_id: int | None = None,
    ) -> dict[str, float]:
        h, w = frame_shape[0], frame_shape[1]
        x1, y1, x2, y2 = [float(v) for v in box]
        cx = (x1 + x2) * 0.5

        visible_h = max(1.0, self._visible_body_height_px(box, kp))

        # Prefer configured vertical FOV; otherwise approximate from horizontal FOV for 16:9.
        vert_fov_deg = float(
            self.cfg.spatial.vert_fov_deg
            if self.cfg.spatial.vert_fov_deg > 0
            else float(self.cfg.spatial.fov_deg) * 0.5625
        )
        focal_px = (h * 0.5) / math.tan(math.radians(vert_fov_deg * 0.5))
        raw_distance = float(self.cfg.spatial.person_height_m * focal_px / visible_h)

        if track_id is None:
            smoothed = raw_distance
        else:
            tid = int(track_id)
            if tid not in self._distance_by_id:
                self._distance_by_id[tid] = deque(maxlen=5)
            hist = self._distance_by_id[tid]
            hist.append(raw_distance)
            if len(hist) == 1:
                smoothed = hist[-1]
            else:
                weights = np.linspace(1.0, 2.0, num=len(hist))
                smoothed = float(np.average(np.array(hist, dtype=np.float32), weights=weights))

        norm_x = (cx - (w * 0.5)) / max(1.0, (w * 0.5))
        azimuth_deg = norm_x * (float(self.cfg.spatial.fov_deg) * 0.5)

        heading_deg = 0.0
        if kp is not None and len(kp) >= 7 and float(kp[5][2]) > 0.1 and float(kp[6][2]) > 0.1:
            shoulder_dx = float(kp[6][0] - kp[5][0])
            heading_deg = max(-90.0, min(90.0, shoulder_dx / max(1.0, x2 - x1) * 45.0))

        return {
            "distance_est_m": float(smoothed),
            "azimuth_deg": float(azimuth_deg),
            "heading_deg": float(heading_deg),
            "box_h_px": float(visible_h),
        }
