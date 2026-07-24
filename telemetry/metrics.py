from __future__ import annotations
METRICS_AVAILABLE = True
try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server
except Exception:
    METRICS_AVAILABLE = False

class MetricsCollector:
    def __init__(self, port: int) -> None:
        if not METRICS_AVAILABLE:
            raise RuntimeError("prometheus_client not installed")
        self.fps = Gauge("tracker_fps_avg", "Average FPS")
        self.tracks = Gauge("tracker_tracks", "Active tracks")
        self.infer_ms = Gauge("tracker_infer_ms", "Inference time ms")
        self.total_ms = Gauge("tracker_total_ms", "Total frame time ms")
        self.frames = Counter("tracker_frames_total", "Total frames processed")
        self.failures = Counter("tracker_failures_total", "Total failures")
        self.latency = Histogram("tracker_total_ms_hist", "Frame time histogram",
                                 buckets=(5, 10, 15, 20, 30, 50, 100, 200))
        start_http_server(port)

    def update(self, fps_avg: float, tracks: int, infer_ms: float, total_ms: float) -> None:
        self.fps.set(float(fps_avg))
        self.tracks.set(float(tracks))
        self.infer_ms.set(float(infer_ms))
        self.total_ms.set(float(total_ms))
        self.frames.inc()
        self.latency.observe(float(total_ms))

    def failure(self) -> None:
        self.failures.inc()
