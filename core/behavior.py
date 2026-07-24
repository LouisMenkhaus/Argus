from __future__ import annotations
import numpy as np
from core.config import AppConfig

class BehaviorAnalyzer:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.prev_centers: dict[int, tuple[float, float]] = {}

    def analyze(self, global_id: int, box: np.ndarray, kp: np.ndarray) -> list[str]:
        events: list[str] = []
        x1, y1, x2, y2 = [float(v) for v in box]
        w = max(1.0, x2 - x1); h = max(1.0, y2 - y1)
        cx = (x1 + x2) * 0.5; cy = (y1 + y2) * 0.5
        prev = self.prev_centers.get(global_id)
        if prev is not None:
            speed = (((cx - prev[0]) / w) ** 2 + ((cy - prev[1]) / h) ** 2) ** 0.5
            if speed > self.cfg.behaviors.fast_motion_threshold:
                events.append("fast_motion")
        self.prev_centers[global_id] = (cx, cy)
        aspect = w / h
        if aspect > self.cfg.behaviors.fall_aspect_ratio:
            events.append("fall_like_posture")
        if kp is not None and len(kp) >= 11:
            if kp[9][2] > 0.1 and kp[10][2] > 0.1 and kp[0][2] > 0.1:
                margin = self.cfg.behaviors.raised_hands_margin_px
                if kp[9][1] < kp[0][1] - margin and kp[10][1] < kp[0][1] - margin:
                    events.append("raised_hands")
        return events
