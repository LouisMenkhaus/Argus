from __future__ import annotations
from pathlib import Path
from typing import Any
import json
import cv2
import numpy as np

class SessionRecorder:
    def __init__(self, session_dir: Path) -> None:
        self.session_dir = session_dir
        self.video_writer = None
        self.timestamps: list[float] = []
        self.records: list[dict[str, Any]] = []

    def start_recording(self, filename: str, fps: int, size: tuple[int, int]) -> None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(str(self.session_dir / filename), fourcc, fps, size)

    def write_frame(self, frame: np.ndarray, timestamp: float) -> None:
        if self.video_writer is not None:
            self.video_writer.write(frame)
        self.timestamps.append(float(timestamp))

    def add_record(self, payload: dict[str, Any]) -> None:
        self.records.append(payload)

    def save_json(self, filename: str = "replay.json") -> None:
        (self.session_dir / filename).write_text(
            json.dumps(self.records, indent=2), encoding="utf-8")

    def stop(self) -> None:
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
