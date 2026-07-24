from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import supervision as sv

from core.config import AppConfig
from core.kalman_tracker import KalmanTracker
from core.one_euro import OneEuroFilter
from core.reid import SimpleReID


COCO_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (5, 11), (6, 12),
]


def draw_pose_skeleton(
    frame: np.ndarray, keypoints: np.ndarray, color: tuple[int, int, int]
) -> np.ndarray:
    if keypoints is None or len(keypoints) == 0:
        return frame

    out = frame
    for a, b in COCO_CONNECTIONS:
        if a < len(keypoints) and b < len(keypoints):
            if float(keypoints[a][2]) > 0.1 and float(keypoints[b][2]) > 0.1:
                p1 = (int(keypoints[a][0]), int(keypoints[a][1]))
                p2 = (int(keypoints[b][0]), int(keypoints[b][1]))
                cv2.line(out, p1, p2, (0, 0, 0), 3)

    for kp in keypoints:
        conf = float(kp[2])
        if conf > 0.1:
            c = (int(kp[0]), int(kp[1]))
            cv2.circle(out, c, 5, (255, 255, 255), 1)
            cv2.circle(out, c, 4, (0, 0, 0), -1)
    return out


# Deterministic, high-contrast palette so each identity keeps a stable color
# across frames and cameras (BGR order for OpenCV).
_ID_PALETTE: list[tuple[int, int, int]] = [
    (66, 133, 244),   # blue
    (52, 168, 83),    # green
    (244, 180, 0),    # yellow
    (234, 67, 53),    # red
    (171, 71, 188),   # purple
    (0, 172, 193),    # cyan
    (255, 112, 67),   # orange
    (158, 157, 36),   # olive
    (240, 98, 146),   # pink
    (0, 137, 123),    # teal
]


def id_color(track_id: int) -> tuple[int, int, int]:
    """Stable per-identity color: same global ID always renders the same color."""
    return _ID_PALETTE[int(track_id) % len(_ID_PALETTE)]


class AdaptiveProcessor:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(cfg.model.path)
        self.model.to(self.device)
        if hasattr(self.model, "fuse"):
            try:
                self.model.fuse()
            except Exception:
                pass

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


@dataclass
class TrackState:
    kalman: KalmanTracker
    last_keypoints: np.ndarray
    last_box: np.ndarray
    confidence: float
    age: int = 1
    lost: int = 0


class MotionSmoother:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.kalman: dict[int, TrackState] = {}
        self.kp_state: dict[int, np.ndarray] = {}
        self.oe_filters: dict[int, OneEuroFilter] = {}
        self.max_lost_frames = max(10, int(cfg.cameras.fps * 0.5))

    def drop(self, sid: int) -> None:
        """Forget all smoothing state for a track id."""
        self.kalman.pop(sid, None)
        self.kp_state.pop(sid, None)
        self.oe_filters.pop(sid, None)

    def _one_euro_keypoints(self, sid: int, keypoints: np.ndarray) -> np.ndarray:
        """Speed-adaptive keypoint smoothing (see core/one_euro.py).
        Filters x/y only; the confidence column passes through untouched."""
        import time as _time
        f = self.oe_filters.get(sid)
        if f is None:
            f = OneEuroFilter(
                min_cutoff=float(self.cfg.tracking.one_euro_min_cutoff),
                beta=float(self.cfg.tracking.one_euro_beta),
            )
            self.oe_filters[sid] = f
        out = keypoints.astype(np.float32, copy=True)
        out[:, :2] = f(_time.monotonic(), keypoints[:, :2])
        return out

    @staticmethod
    def _box_to_state(box: np.ndarray) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = [float(v) for v in box]
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        return cx, cy, w, h

    @staticmethod
    def _state_to_box(cx: float, cy: float, w: float, h: float) -> np.ndarray:
        return np.array(
            [cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5],
            dtype=np.float32,
        )

    def _smooth_keypoints(self, sid: int, keypoints: np.ndarray, ref_box: np.ndarray) -> np.ndarray:
        keypoints = keypoints.astype(np.float32, copy=False)
        if sid not in self.kp_state:
            self.kp_state[sid] = keypoints.copy()
            return keypoints

        prev_kp = self.kp_state[sid]
        x1, y1, x2, y2 = [float(v) for v in ref_box]
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)

        dx = (keypoints[:, 0] - prev_kp[:, 0]) / w
        dy = (keypoints[:, 1] - prev_kp[:, 1]) / h
        motion_kp = float(np.sqrt(dx * dx + dy * dy).mean())

        t = min(1.0, motion_kp / max(1e-6, float(self.cfg.tracking.motion_thresh)))
        alpha_kp = (
            (1.0 - t) * float(self.cfg.tracking.alpha_slow)
            + t * float(self.cfg.tracking.alpha_fast)
        )
        alpha_kp = min(0.85, alpha_kp)
        new_kp = alpha_kp * prev_kp + (1.0 - alpha_kp) * keypoints
        self.kp_state[sid] = new_kp
        return new_kp

    def update(
        self,
        stable_id: int,
        box: np.ndarray,
        keypoints: np.ndarray,
        confidence: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        sid = int(stable_id)
        box = box.astype(np.float32, copy=False)
        keypoints = keypoints.astype(np.float32, copy=False)

        if self.cfg.tracking.filter != "kalman":
            if self.cfg.tracking.keypoint_filter == "one_euro":
                return box, self._one_euro_keypoints(sid, keypoints)
            if sid not in self.kp_state:
                self.kp_state[sid] = keypoints.copy()
                return box, keypoints
            x1, y1, x2, y2 = [float(v) for v in box]
            w = max(1.0, x2 - x1)
            h = max(1.0, y2 - y1)
            prev_kp = self.kp_state[sid]
            dx = (keypoints[:, 0] - prev_kp[:, 0]) / w
            dy = (keypoints[:, 1] - prev_kp[:, 1]) / h
            motion_kp = float(np.sqrt(dx * dx + dy * dy).mean())
            t = min(1.0, motion_kp / max(1e-6, float(self.cfg.tracking.motion_thresh)))
            alpha = (
                (1.0 - t) * float(self.cfg.tracking.alpha_slow)
                + t * float(self.cfg.tracking.alpha_fast)
            )
            alpha_kp = min(0.85, alpha)
            new_kp = alpha_kp * prev_kp + (1.0 - alpha_kp) * keypoints
            self.kp_state[sid] = new_kp
            return box, new_kp

        cx, cy, bw, bh = self._box_to_state(box)

        if sid not in self.kalman:
            fps = max(1.0, float(self.cfg.cameras.fps))
            kf = KalmanTracker(dt=1.0 / fps)
            pred = kf.update(cx, cy, bw, bh, float(confidence))
            smooth_box = self._state_to_box(float(pred[0]), float(pred[1]), float(pred[2]), float(pred[3]))
            self.kalman[sid] = TrackState(
                kalman=kf,
                last_keypoints=keypoints.copy(),
                last_box=smooth_box.copy(),
                confidence=float(confidence),
                age=1,
                lost=0,
            )
            self.kp_state[sid] = keypoints.copy()
            return smooth_box, keypoints

        state = self.kalman[sid]
        state.lost = 0
        state.age += 1
        state.confidence = state.confidence * 0.7 + float(confidence) * 0.3

        pred = state.kalman.update(cx, cy, bw, bh, float(confidence))
        smooth_box = self._state_to_box(float(pred[0]), float(pred[1]), float(pred[2]), float(pred[3]))

        conf_t = 0.20
        top_idxs = [0, 1, 2, 3, 4, 5, 6]
        top_y = None
        for idx in top_idxs:
            if idx < len(keypoints) and float(keypoints[idx][2]) >= conf_t:
                y = float(keypoints[idx][1])
                top_y = y if top_y is None else min(top_y, y)
        if top_y is not None and top_y < smooth_box[3]:
            smooth_box[1] = min(smooth_box[1], top_y)

        if self.cfg.tracking.keypoint_filter == "one_euro":
            smooth_kp = self._one_euro_keypoints(sid, keypoints)
        else:
            smooth_kp = self._smooth_keypoints(sid, keypoints, smooth_box)

        state.last_keypoints = smooth_kp.copy()
        state.last_box = smooth_box.copy()
        return smooth_box, smooth_kp

    def predict_lost(self, min_confidence: float = 0.3) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for sid, state in list(self.kalman.items()):
            state.lost += 1
            state.confidence *= 0.7
            if state.lost > self.max_lost_frames or state.confidence < min_confidence:
                self.drop(sid)
                continue
            pred = state.kalman.predict()
            smooth_box = self._state_to_box(
                float(pred[0]), float(pred[1]), float(pred[2]), float(pred[3])
            )
            state.last_box = smooth_box.copy()
            out.append(
                {
                    "camera": 0,
                    "local_id": -1,
                    "global_id": sid,
                    "box": smooth_box,
                    "keypoints": state.last_keypoints.copy(),
                    "confidence": float(state.confidence),
                }
            )
        return out


class GlobalIDManager:
    def __init__(self, use_reid: bool = False, reid_threshold: float = 0.65) -> None:
        self.next_gid = 1
        self.cam_local_to_global: dict[tuple[int, int], int] = {}
        self.reid_enabled = use_reid
        self.reid_threshold = float(reid_threshold)
        self.features: dict[int, np.ndarray] = {}
        self.reid = SimpleReID() if use_reid else None
        self.frame_counter = 0

    def assign(self, camera_idx: int, local_id: int, frame: np.ndarray, box: np.ndarray) -> int:
        key = (camera_idx, int(local_id))
        self.frame_counter += 1

        if key in self.cam_local_to_global:
            gid = self.cam_local_to_global[key]
            if self.reid_enabled and self.reid is not None and self.frame_counter % 5 == 0:
                feat = self.reid.extract_features(frame, box)
                if feat is not None:
                    self.features[gid] = feat
            return gid

        gid: Optional[int] = None
        if self.reid_enabled and self.reid is not None:
            feat = self.reid.extract_features(frame, box)
            if feat is not None:
                best_gid = None
                best_score = -1.0
                for existing_gid, existing_feat in self.features.items():
                    score = self.reid.cosine_similarity(feat, existing_feat)
                    if score > best_score:
                        best_score = score
                        best_gid = existing_gid
                if best_gid is not None and best_score > self.reid_threshold:
                    gid = int(best_gid)
                    self.features[gid] = feat

        if gid is None:
            gid = self.next_gid
            self.next_gid += 1
            if self.reid_enabled and self.reid is not None:
                feat = self.reid.extract_features(frame, box)
                if feat is not None:
                    self.features[gid] = feat

        self.cam_local_to_global[key] = gid
        return gid


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
        self.t_infer = deque(maxlen=120)
        self.t_total = deque(maxlen=120)

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
            try:
                det = self.byte_track.update_with_detections(det)
            except Exception:
                pass

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
