from __future__ import annotations
import json
from pathlib import Path
import time
import numpy as np
import psutil
from camera.capture import CaptureWrapper
from core.config import AppConfig
from core.tracker import MultiCameraTracker


def run_benchmark(cfg: AppConfig, session_dir: Path, num_frames: int = 1000) -> dict[str, float]:
    tracker = MultiCameraTracker(cfg)
    source = cfg.cameras.sources[0]
    cap = CaptureWrapper(source, cfg.cameras.width, cfg.cameras.height,
                         cfg.cameras.fps, cfg.cameras.prefer_dshow)
    process = psutil.Process()
    start_mem = process.memory_info().rss / 1024 / 1024
    times = []
    infer_times = []
    for _ in range(int(num_frames)):
        ret, frame = cap.read_latest(cfg.cameras.buffer_drops)
        if not ret or frame is None:
            break
        start = time.perf_counter()
        result = tracker.process(0, frame)
        times.append(time.perf_counter() - start)
        infer_times.append(result["infer_ms"])
    cap.release()
    if not times:
        report = {"frames": 0, "avg_fps": 0.0, "p95_latency_ms": 0.0, "memory_delta_mb": 0.0}
    else:
        report = {
            "frames": len(times),
            "avg_fps": float(1.0 / np.mean(times)),
            "p95_latency_ms": float(np.percentile(times, 95) * 1000.0),
            "avg_infer_ms": float(np.mean(infer_times)),
            "memory_delta_mb": float(process.memory_info().rss / 1024 / 1024 - start_mem),
        }
    (session_dir / "benchmark_results.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    return report
