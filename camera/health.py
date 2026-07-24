from __future__ import annotations
from collections import deque
import time
import numpy as np

class CameraHealth:
    """
    Hysteresis-based camera health to avoid flapping between healthy/degraded.
    """
    def __init__(
        self,
        bad_checks_required: int = 3,
        good_checks_required: int = 5,
        degraded_mean_interval_s: float = 0.12,
        dropping_frame_gap_s: float = 0.50,
        cooldown_sec: float = 2.0,
    ) -> None:
        self.expected_frame_times: deque[float] = deque(maxlen=30)
        self.bad_checks_required = int(bad_checks_required)
        self.good_checks_required = int(good_checks_required)
        self.degraded_mean_interval_s = float(degraded_mean_interval_s)
        self.dropping_frame_gap_s = float(dropping_frame_gap_s)
        self.cooldown_sec = float(cooldown_sec)

        self.state = "healthy"
        self.bad_count = 0
        self.good_count = 0
        self.last_change_time = 0.0

    def tick(self) -> None:
        self.expected_frame_times.append(time.time())

    def _raw_status(self) -> str:
        if len(self.expected_frame_times) < 10:
            return "healthy"
        intervals = np.diff(list(self.expected_frame_times))
        if len(intervals) == 0:
            return "healthy"
        mean_interval = float(np.mean(intervals))
        max_gap = float(np.max(intervals))
        if max_gap > self.dropping_frame_gap_s:
            return "dropping_frames"
        if mean_interval > self.degraded_mean_interval_s:
            return "degraded"
        return "healthy"

    def check_health(self) -> str:
        raw = self._raw_status()
        now = time.time()

        # Harder state wins immediately after enough bad checks.
        if raw in ("degraded", "dropping_frames"):
            self.bad_count += 1
            self.good_count = 0
        else:
            self.good_count += 1
            self.bad_count = 0

        if self.state == "healthy":
            if (self.bad_count >= self.bad_checks_required
                    and (now - self.last_change_time) >= self.cooldown_sec):
                self.state = raw
                self.last_change_time = now
        else:
            # if raw gets worse, allow quicker escalation
            if (raw == "dropping_frames" and self.state != "dropping_frames"
                    and self.bad_count >= self.bad_checks_required):
                self.state = "dropping_frames"
                self.last_change_time = now
            elif (raw == "healthy" and self.good_count >= self.good_checks_required
                    and (now - self.last_change_time) >= self.cooldown_sec):
                self.state = "healthy"
                self.last_change_time = now

        return self.state
