"""API auth primitives — pure-Python, no FastAPI required.

TokenBucket / RateLimiter / RBAC are deliberately importable without the
optional web stack so their security behavior is unit-testable everywhere.
"""
import time

from api.server import RBAC, RateLimiter, TokenBucket


def test_token_bucket_allows_burst_then_blocks():
    b = TokenBucket(rate_per_sec=1.0, burst=3)
    assert b.allow() and b.allow() and b.allow()
    assert not b.allow(), "burst exhausted — must block"


def test_token_bucket_refills_over_time():
    b = TokenBucket(rate_per_sec=50.0, burst=1)
    assert b.allow()
    assert not b.allow()
    time.sleep(0.05)  # 50/sec -> ~2.5 tokens refilled, capped at burst
    assert b.allow()


def test_rate_limiter_isolates_clients():
    rl = RateLimiter(rate_per_sec=0.001, burst=1)
    assert rl.check("10.0.0.1")
    assert not rl.check("10.0.0.1"), "same client must be limited"
    assert rl.check("10.0.0.2"), "different client must have its own bucket"


def test_rbac_default_roles():
    rbac = RBAC(config_path=None)
    assert rbac.allowed("viewer", "status")
    assert rbac.allowed("viewer", "health")
    assert not rbac.allowed("viewer", "toggle"), "viewer must not control the tracker"
    assert rbac.allowed("operator", "toggle")
    assert rbac.allowed("admin", "anything_at_all"), "admin wildcard"


def test_rbac_unknown_role_denied():
    rbac = RBAC(config_path=None)
    assert not rbac.allowed("intruder", "status")


def test_rbac_loads_custom_config(tmp_path=None):
    import json, tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "rbac.json"
        p.write_text(json.dumps({"roles": {"auditor": ["status"]}}))
        rbac = RBAC(p)
    assert rbac.allowed("auditor", "status")
    assert not rbac.allowed("auditor", "toggle")
    assert not rbac.allowed("admin", "toggle"), "custom config replaces defaults entirely"
