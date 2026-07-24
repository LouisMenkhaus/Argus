"""Audit logger: best-effort logging and webhook scheme validation."""
import json

from telemetry.audit import AuditLogger, _is_safe_webhook


def test_only_http_schemes_are_accepted():
    assert _is_safe_webhook("https://hooks.example.com/abc")
    assert _is_safe_webhook("http://10.0.0.5:9000/alert")


def test_dangerous_schemes_are_rejected():
    """urlopen will read local files given a file:// URL — never allow it."""
    assert not _is_safe_webhook("file:///etc/passwd")
    assert not _is_safe_webhook("ftp://example.com/x")
    assert not _is_safe_webhook("not a url")
    assert not _is_safe_webhook("https://")          # no host
    assert not _is_safe_webhook("")


def test_unsafe_webhook_is_dropped_at_construction(tmp_path):
    log = AuditLogger(tmp_path, webhook_url="file:///etc/passwd")
    assert log.webhook_url == "", "unsafe webhook must not be retained"


def test_safe_webhook_is_retained(tmp_path):
    log = AuditLogger(tmp_path, webhook_url="https://hooks.example.com/abc")
    assert log.webhook_url == "https://hooks.example.com/abc"


def test_events_are_written_as_jsonl(tmp_path):
    log = AuditLogger(tmp_path)
    log.event("camera_lost", "ERROR", {"camera": 1})
    lines = (tmp_path / "audit.log").read_text().strip().splitlines()
    assert len(lines) >= 2                      # session_started + the event
    parsed = json.loads(lines[-1])
    assert parsed["message"] == "camera_lost"
    assert parsed["level"] == "ERROR"
    assert parsed["extra"]["camera"] == 1
