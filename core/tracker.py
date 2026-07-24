from __future__ import annotations

import contextlib

"""
Multi-camera inference layer: YOLO pose estimation + ByteTrack association.

The smoothing, identity, and rendering logic lives in `core/smoothing.py`,
which carries no inference dependency. This module composes that layer with
the detector. Names used elsewhere are re-exported below so existing imports
(`from core.tracker import MotionSmoother`) keep working.
"""

from collections import deque
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import supervision as sv

from core.config import AppConfig
from core.smoothing import (  # noqa: F401  (re-exported for compatibility)
    COCO_CONNECTIONS,
    GlobalIDManager,
    MotionSmoother,
    TrackState,
    draw_pose_skeleton,
    id_color,
)


class AdaptiveProcessor:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(cfg.model.path)
        self.model.to(self.device)
        if hasattr(self.model, "fuse"):
            # Layer fusion is a speed optimization; models that do not support
            # it simply run unfused.
            with contextlib.suppress(Exception):
                self.model.fuse()

    def auto_configure(self) -> dict[str, Any]:
        if not self.cfg.processor.auto_fallback:
            return {"device": self.device, "fallback_used": False}
        test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        start = cv2.getTickCount()
        _ = self.model(
            test_frame,
            verbose=False,
            conf=self.cfg.model.conf,
            iou=self.cfg.model.iou,
            max_det=1,
        )[0]
        elapsed_ms = (cv2.getTickCount() - start) * 1000.0 / cv2.getTickFrequency()
        fallback_used = False
        if (
            elapsed_ms > self.cfg.processor.slow_infer_threshold_ms
            and self.cfg.model.path != self.cfg.processor.fallback_model
        ):
            self.model = YOLO(self.cfg.processor.fallback_model)
            self.model.to(self.device)
            fallback_used = True
        return {
            "device": self.device,
            "fallback_used": fallback_used,
            "test_infer_ms": float(elapsed_ms),
        }


class MultiCameraTracker:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.processor = AdaptiveProcessor(cfg)
        self.runtime = self.processor.auto_configure()
        self.device = self.processor.device

        try:
            self.byte_track = sv.ByteTrack()
        except Exception:
            self.byte_track = None

        self.smoother = MotionSmoother(cfg)
        self.global_ids = GlobalIDManager(
            use_reid=cfg.tracking.reid,
            reid_threshold=float(cfg.tracking.reid_threshold),
        )
        self.t_infer: deque[float] = deque(maxlen=120)
        self.t_total: deque[float] = deque(maxlen=120)

    def process(self, camera_idx: int, frame: np.ndarray) -> dict[str, Any]:
        t0 = cv2.getTickCount()

        with torch.no_grad():
            t1 = cv2.getTickCount()
            results = self.processor.model(
                frame,
                conf=self.cfg.model.conf,
                iou=self.cfg.model.iou,
                max_det=self.cfg.model.max_det,
                verbose=False,
                device=self.device,
            )[0]
            t2 = cv2.getTickCount()

        infer_ms = (t2 - t1) * 1000.0 / cv2.getTickFrequency()
        self.t_infer.append(infer_ms)

        det = (
            sv.Detections.from_ultralytics(results)
            if results.boxes is not None
            else sv.Detections.empty()
        )

        if self.byte_track is not None and len(det) > 0:
            # A tracker hiccup on one frame must not drop the frame entirely;
            # fall back to raw detections for this cycle.
            with contextlib.suppress(Exception):
                det = self.byte_track.update_with_detections(det)

        tracks: list[dict[str, Any]] = []

        if results.keypoints is not None and results.keypoints.data is not None and len(det) > 0:
            kp_data = results.keypoints.data.detach().cpu().numpy()
            seen_gids: set[int] = set()

            for i, local_id in enumerate(det.tracker_id):
                if local_id is None or i >= len(kp_data):
                    continue

                box = det.xyxy[i]
                kp = kp_data[i]
                conf = float(det.confidence[i]) if det.confidence is not None else 1.0

                gid = self.global_ids.assign(camera_idx, int(local_id), frame, box)
                seen_gids.add(int(gid))

                sbox, skp = self.smoother.update(gid, box, kp, conf)
                tracks.append(
                    {
                        "camera": camera_idx,
                        "local_id": int(local_id),
                        "global_id": int(gid),
                        "box": sbox,
                        "keypoints": skp,
                        "confidence": conf,
                    }
                )

            for sid, state in list(self.smoother.kalman.items()):
                if sid not in seen_gids:
                    state.lost += 1
                    if state.lost > self.smoother.max_lost_frames:
                        self.smoother.drop(sid)

        elif self.cfg.tracking.filter == "kalman":
            tracks.extend(self.smoother.predict_lost())

        total_ms = (cv2.getTickCount() - t0) * 1000.0 / cv2.getTickFrequency()
        self.t_total.append(total_ms)
        fps_avg = (1000.0 / np.mean(self.t_total)) if self.t_total else 0.0

        return {
            "tracks": tracks,
            "infer_ms": float(infer_ms),
            "total_ms": float(total_ms),
            "fps_avg": float(fps_avg),
            "runtime": self.runtime,
        }
