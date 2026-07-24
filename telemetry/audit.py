from __future__ import annotations

"""Audit logging with an optional alert webhook.

Two deliberate properties:

* Logging is best-effort. A full disk or a locked file must never take down
  the tracker that is being audited.
* Webhook delivery only ever speaks HTTP(S). `urllib.request.urlopen` will
  happily open `file://` (and other) schemes, so a webhook URL that reached
  the config from an untrusted source could otherwise be turned into a local
  file read. The scheme is validated before the request is built.
"""

import contextlib
import datetime
import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

ALLOWED_WEBHOOK_SCHEMES = ("http", "https")


def _is_safe_webhook(url: str) -> bool:
    """True only for well-formed http/https URLs with a host."""
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        return False
    return parsed.scheme in ALLOWED_WEBHOOK_SCHEMES and bool(parsed.netloc)


class AuditLogger:
    def __init__(self, session_dir: Path, webhook_url: str = "") -> None:
        self.session_dir = session_dir
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.audit_path = self.session_dir / "audit.log"
        self.global_path = Path("audit_global.log")

        self.webhook_url = ""
        if webhook_url:
            if _is_safe_webhook(webhook_url):
                self.webhook_url = webhook_url
            else:
                print(f"[audit] Ignoring webhook URL with unsupported scheme: "
                      f"{webhook_url[:40]!r}. Only http/https are permitted.")

        self.event("session_started")

    def event(self, message: str, level: str = "INFO",
              extra: dict[str, Any] | None = None) -> None:
        ts = datetime.datetime.now().isoformat()
        payload = {"ts": ts, "level": level, "message": message, "extra": extra or {}}
        line = json.dumps(payload, ensure_ascii=False)

        for p in (self.audit_path, self.global_path):
            # Best-effort: never let a logging failure reach the capture loop.
            with contextlib.suppress(Exception):
                with open(p, "a", encoding="utf-8") as f:
                    f.write(line + "\n")

        if level in ("ERROR", "FATAL") and self.webhook_url:
            self._post(payload)

    def _post(self, payload: dict[str, Any]) -> None:
        """Fire-and-forget alert delivery. Scheme was validated in __init__."""
        if not _is_safe_webhook(self.webhook_url):   # defensive re-check
            return
        req = urllib.request.Request(
            self.webhook_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        # A dead endpoint must not stall the pipeline. Scheme is validated
        # above and at construction, so B310 does not apply here.
        with contextlib.suppress(Exception):
            urllib.request.urlopen(req, timeout=2.0).read()  # nosec B310
