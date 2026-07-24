#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime
import os
from pathlib import Path
import time

import cv2
import numpy as np

from api.server import ControlState, start_api
from benchmark.benchmark import run_benchmark
from camera.capture import CaptureWrapper
from core.behavior import BehaviorAnalyzer
from core.config import AppConfig
from core.spatial import SpatialAnalyzer
from core.tracker import MultiCameraTracker, draw_pose_skeleton, id_color
from recording.recorder import SessionRecorder
from telemetry.audit import AuditLogger
from telemetry.logger import SessionLogger
from telemetry.metrics import MetricsCollector, METRICS_AVAILABLE


def print_banner() -> None:
    print(r'''
╔══════════════════════════════════════════════════════════════╗
║   ARGUS — Multi-Camera Pose Tracking & Behavior Analysis    ║
║  YOLO Pose • ByteTrack • Kalman • ReID • RTSP • Telemetry   ║
╚══════════════════════════════════════════════════════════════╝
''')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="argus", description="Argus — multi-camera pose tracking and behavior analysis")
    p.add_argument("--config", type=str, default="config/default.yaml", help="Path to YAML config")
    p.add_argument("--sources", type=str, default="", help="Comma-separated camera/video/RTSP sources")
    p.add_argument("--prefer-dshow", action="store_true", help="Prefer DirectShow for webcams on Windows")
    p.add_argument("--benchmark", action="store_true", help="Run benchmark and exit")
    p.add_argument("--benchmark-frames", type=int, default=500, help="Number of frames for benchmark mode")
    p.add_argument("--dashboard", action="store_true", help="Enable API/dashboard status server")
    p.add_argument("--dashboard-port", type=int, default=8000, help="Dashboard/API port")
    p.add_argument("--metrics", action="store_true", help="Enable Prometheus metrics endpoint")
    p.add_argument("--metrics-port", type=int, default=9090, help="Metrics endpoint port")
    p.add_argument("--jwt-secret", type=str, default="", help="Optional JWT secret for API")
    p.add_argument("--rbac-config", type=str, default="rbac.json", help="RBAC JSON path")
    p.add_argument("--reid", action="store_true", help="Enable simple ReID")
    p.add_argument("--tracking-filter", choices=["adaptive", "kalman"], default="", help="Tracking filter mode")
    p.add_argument("--keypoint-filter", choices=["one_euro", "ema"], default="",
                   help="Keypoint smoothing: one_euro (low latency, default) or legacy ema")
    p.add_argument("--model", type=str, default="",
                   help="YOLO pose weights, e.g. yolo11n-pose.pt (fastest), "
                        "yolo11s-pose.pt or yolo11m-pose.pt (more accurate, slower). "
                        "Downloaded automatically on first use.")
    p.add_argument("--record-video", action="store_true", help="Record video output")
    p.add_argument("--replay-json", action="store_true", help="Always write replay json")
    return p.parse_args()


def load_cfg(args: argparse.Namespace) -> AppConfig:
    cfg = AppConfig.from_yaml(args.config)
    if args.sources:
        cfg.cameras.sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    if args.prefer_dshow:
        cfg.cameras.prefer_dshow = True
    if args.dashboard:
        cfg.telemetry.dashboard = True
        cfg.telemetry.dashboard_port = args.dashboard_port
    if args.metrics:
        cfg.telemetry.metrics = True
        cfg.telemetry.metrics_port = args.metrics_port
    if args.reid:
        cfg.tracking.reid = True
    if args.tracking_filter:
        cfg.tracking.filter = args.tracking_filter
    if args.keypoint_filter:
        cfg.tracking.keypoint_filter = args.keypoint_filter
    if args.model:
        cfg.model.path = args.model
    if args.replay_json:
        cfg.playback.autosave_replay_json = True
    return cfg


def apply_ir_enhancement(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    gamma = 1.5
    gamma_corrected = np.power(enhanced / 255.0, gamma) * 255.0
    gamma_corrected = gamma_corrected.astype(np.uint8)
    return cv2.cvtColor(gamma_corrected, cv2.COLOR_GRAY2BGR)


def main() -> None:
    print_banner()
    args = parse_args()
    cfg = load_cfg(args)

    session_dir = Path(cfg.telemetry.out_dir) / f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session_dir.mkdir(parents=True, exist_ok=True)

    audit = AuditLogger(session_dir, webhook_url=cfg.alerts.webhook_url)
    logger = SessionLogger(session_dir, cfg.telemetry.write_csv, cfg.telemetry.write_jsonl)
    recorder = SessionRecorder(session_dir)

    if args.benchmark:
        report = run_benchmark(cfg, session_dir, args.benchmark_frames)
        print(report)
        return

    metrics = None
    if cfg.telemetry.metrics and METRICS_AVAILABLE:
        try:
            metrics = MetricsCollector(cfg.telemetry.metrics_port)
            audit.event("metrics_enabled", extra={"port": cfg.telemetry.metrics_port})
        except Exception as e:
            audit.event("metrics_init_failed", "ERROR", {"error": str(e)})

    tracker = MultiCameraTracker(cfg)
    if tracker.device == "cpu":
        print("[PERF] Inference device: CPU. If this machine has an NVIDIA GPU, "
              "install CUDA PyTorch for a large latency drop:")
        print("[PERF]   pip install torch --index-url https://download.pytorch.org/whl/cu121")
    else:
        print(f"[PERF] Inference device: {tracker.device.upper()}")
    behavior = BehaviorAnalyzer(cfg)
    spatial = SpatialAnalyzer(cfg)

    captures = [
        CaptureWrapper(src, cfg.cameras.width, cfg.cameras.height, cfg.cameras.fps,
                       cfg.cameras.prefer_dshow, cfg.health,
                       threaded=cfg.cameras.threaded_capture)
        for src in cfg.cameras.sources
    ]

    show_pose = True
    show_boxes = True
    show_ids = True
    night_view = False

    status_ref = ControlState(dashboard_status={"ok": True, "tracks": []}, toggles={"pose": True, "boxes": True, "ids": True})
    if cfg.telemetry.dashboard:
        # Secret resolution order: --jwt-secret flag, then the env var named in
        # config (default JWT_SECRET). Never hardcode secrets in config files.
        jwt_secret = args.jwt_secret or os.environ.get(cfg.security.jwt_secret_env, "")
        if not jwt_secret:
            print("[SECURITY] API starting WITHOUT auth (no JWT secret provided) — "
                  "all requests treated as role 'viewer'. Set JWT_SECRET to enable auth.")
        start_api("127.0.0.1", cfg.telemetry.dashboard_port, status_ref, jwt_secret, args.rbac_config, cfg.security.rate_limit, cfg.security.rate_burst)
        audit.event("dashboard_enabled", extra={"port": cfg.telemetry.dashboard_port})

    if args.record_video:
        recorder.start_recording("session.mp4", max(15, cfg.cameras.fps), (cfg.cameras.width, cfg.cameras.height))

    cv2.namedWindow("Argus", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Argus", cfg.cameras.width, cfg.cameras.height)

    frame_idx = 0
    try:
        while True:
            mosaics = []
            status_tracks = []
            # API can override toggles
            show_pose = status_ref.toggles.get("pose", show_pose)
            show_boxes = status_ref.toggles.get("boxes", show_boxes)
            show_ids = status_ref.toggles.get("ids", show_ids)

            for camera_idx, cap in enumerate(captures):
                ret, frame = cap.read_latest(cfg.cameras.buffer_drops)
                if not ret or frame is None:
                    cap.reconnect()
                    audit.event("capture_reconnect", "ERROR", {"camera": camera_idx, "source": cap.source})
                    if metrics:
                        metrics.failure()
                    continue

                if night_view:
                    frame = apply_ir_enhancement(frame)

                result = tracker.process(camera_idx, frame)
                annotated = frame.copy()
                frame_tracks = []

                for tr in result["tracks"]:
                    box = tr["box"]
                    kp = tr["keypoints"]
                    gid = tr["global_id"]
                    lid = tr["local_id"]
                    s = spatial.estimate(frame.shape, box, kp, gid)
                    events = behavior.analyze(gid, box, kp)

                    x1, y1, x2, y2 = [int(v) for v in box]
                    color = id_color(gid)
                    if show_boxes:
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    if show_pose:
                        annotated = draw_pose_skeleton(annotated, kp, color)

                    label = f"GID{gid} LID{lid} {s['distance_est_m']:.1f}m {s['azimuth_deg']:.0f}°"
                    if show_ids:
                        cv2.putText(annotated, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    if events:
                        cv2.putText(annotated, " | ".join(events), (x1, min(annotated.shape[0] - 10, y2 + 18)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    row = {
                        "camera": camera_idx,
                        "global_id": gid,
                        "local_id": lid,
                        "fps_avg": result["fps_avg"],
                        "infer_ms": result["infer_ms"],
                        "total_ms": result["total_ms"],
                        "distance_est_m": s["distance_est_m"],
                        "azimuth_deg": s["azimuth_deg"],
                        "heading_deg": s["heading_deg"],
                        "events": events,
                    }
                    logger.log_track(row)
                    recorder.add_record(row)

                    status_entry = dict(row)
                    status_entry["camera_health"] = cap.health.check_health()
                    status_tracks.append(status_entry)
                    frame_tracks.append(status_entry)

                cv2.putText(annotated,
                            f"CAM {camera_idx} | {result['fps_avg']:.1f} FPS | "
                            f"infer {result['infer_ms']:.0f} ms | {tracker.device.upper()}",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(annotated, f"Health: {cap.health.check_health()}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,200,0), 2)
                mosaics.append(annotated)

                if metrics:
                    metrics.update(result["fps_avg"], len(frame_tracks), result["infer_ms"], result["total_ms"])

            if not mosaics:
                continue

            display = mosaics[0] if len(mosaics) == 1 else cv2.hconcat(mosaics)
            frame_idx += 1

            status = {"ok": True, "frame": frame_idx, "tracks": status_tracks, "runtime": tracker.runtime}
            status_ref.dashboard_status = status
            logger.write_status(status)

            if args.record_video:
                recorder.write_frame(display, time.time())

            cv2.putText(display, "Q quit | N night | 1 pose | 2 boxes | 3 ids | E export", (10, display.shape[0]-12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
            cv2.imshow("Argus", display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break
            elif key in (ord('n'), ord('N')):
                night_view = not night_view
            elif key == ord('1'):
                show_pose = not show_pose; status_ref.toggles["pose"] = show_pose
            elif key == ord('2'):
                show_boxes = not show_boxes; status_ref.toggles["boxes"] = show_boxes
            elif key == ord('3'):
                show_ids = not show_ids; status_ref.toggles["ids"] = show_ids
            elif key in (ord('e'), ord('E')):
                recorder.save_json("replay_manual_export.json")
                audit.event("manual_replay_export")
    finally:
        if cfg.playback.autosave_replay_json:
            recorder.save_json("replay.json")
        recorder.stop()
        logger.close()
        for cap in captures:
            cap.release()
        cv2.destroyAllWindows()
        audit.event("session_closed")
        print(f"[APP] Session folder: {session_dir}")


if __name__ == "__main__":
    main()
