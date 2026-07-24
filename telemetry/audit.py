from __future__ import annotations
import datetime
from pathlib import Path
from typing import Any
import json
import urllib.request

class AuditLogger:
    def __init__(self, session_dir: Path, webhook_url: str = "") -> None:
        self.session_dir = session_dir
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.audit_path = self.session_dir / "audit.log"
        self.global_path = Path("audit_global.log")
        self.webhook_url = webhook_url
        self.event("session_started")

    def event(self, message: str, level: str = "INFO", extra: dict[str, Any] | None = None) -> None:
        ts = datetime.datetime.now().isoformat()
        payload = {"ts": ts, "level": level, "message": message, "extra": extra or {}}
        line = json.dumps(payload, ensure_ascii=False)
        for p in (self.audit_path, self.global_path):
            try:
                with open(p, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception:
                pass
        if level in ("ERROR", "FATAL") and self.webhook_url:
            try:
                req = urllib.request.Request(
                    self.webhook_url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                urllib.request.urlopen(req, timeout=2.0).read()
            except Exception:
                pass
