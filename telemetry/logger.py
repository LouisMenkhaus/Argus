from __future__ import annotations
import datetime
import json
from pathlib import Path
from typing import Any


class SessionLogger:
    def __init__(self, session_dir: Path, write_csv: bool = True, write_jsonl: bool = True) -> None:
        self.session_dir = session_dir
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.csv_fp = None
        self.jsonl_fp = None
        self.last_status_path = self.session_dir / "last_status.json"
        if write_csv:
            self.csv_fp = open(self.session_dir / "telemetry.csv", "w", encoding="utf-8")
            self.csv_fp.write(
                "t_iso,camera,global_id,local_id,fps_avg,infer_ms,total_ms,"
                "distance_est_m,azimuth_deg,heading_deg,events\n")
        if write_jsonl:
            self.jsonl_fp = open(self.session_dir / "telemetry.jsonl", "w", encoding="utf-8")

    def log_track(self, row: dict[str, Any]) -> None:
        row = dict(row)
        row.setdefault("t_iso", datetime.datetime.now().isoformat())
        if self.csv_fp:
            events = "|".join(row.get("events", []))
            self.csv_fp.write(
                f"{row.get('t_iso','')},{row.get('camera','')},"
                f"{row.get('global_id','')},{row.get('local_id','')},"
                f"{row.get('fps_avg',0):.2f},{row.get('infer_ms',0):.3f},"
                f"{row.get('total_ms',0):.3f},"
                f"{row.get('distance_est_m',0):.2f},{row.get('azimuth_deg',0):.2f},"
                f"{row.get('heading_deg',0):.2f},{events}\n"
            )
        if self.jsonl_fp:
            self.jsonl_fp.write(json.dumps(row) + "\n")

    def write_status(self, status: dict[str, Any]) -> None:
        self.last_status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

    def close(self) -> None:
        if self.csv_fp:
            self.csv_fp.flush()
            self.csv_fp.close()
        if self.jsonl_fp:
            self.jsonl_fp.flush()
            self.jsonl_fp.close()
