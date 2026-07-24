from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import threading
from typing import Any, Optional

# The API stack (FastAPI/uvicorn/pydantic) is an optional extra
# (requirements-api.txt). The core tracker must run without it, and the
# auth primitives below (TokenBucket, RateLimiter, RBAC) stay importable
# and unit-testable with no web dependencies installed.
API_AVAILABLE = True
try:
    from fastapi import FastAPI, Depends, HTTPException, Request
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from pydantic import BaseModel
    import uvicorn
except Exception:
    API_AVAILABLE = False

JWT_AVAILABLE = True
try:
    import jwt
except Exception:
    JWT_AVAILABLE = False

if API_AVAILABLE:
    class ToggleRequest(BaseModel):
        value: bool


class TokenBucket:
    def __init__(self, rate_per_sec: float, burst: int) -> None:
        import time
        self.rate = float(rate_per_sec)
        self.burst = int(burst)
        self.tokens = float(burst)
        self.last = time.time()

    def allow(self) -> bool:
        import time
        now = time.time()
        elapsed = now - self.last
        self.last = now
        self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False

class RateLimiter:
    def __init__(self, rate_per_sec: float = 5.0, burst: int = 10) -> None:
        self.rate = rate_per_sec
        self.burst = burst
        self.buckets: dict[str, TokenBucket] = {}
        self.lock = threading.Lock()

    def check(self, key: str) -> bool:
        with self.lock:
            bucket = self.buckets.get(key)
            if bucket is None:
                bucket = TokenBucket(self.rate, self.burst)
                self.buckets[key] = bucket
            return bucket.allow()

class RBAC:
    def __init__(self, config_path: Path | None) -> None:
        self.roles = {
            "viewer": ["status", "health"],
            "operator": ["status", "health", "toggle"],
            "admin": ["*"],
        }
        self.users: dict[str, str] = {}
        if config_path and config_path.exists():
            data = json.loads(config_path.read_text(encoding="utf-8"))
            self.roles = data.get("roles", self.roles)
            self.users = data.get("users", self.users)

    def allowed(self, role: str, capability: str) -> bool:
        caps = self.roles.get(role, [])
        return "*" in caps or capability in caps

@dataclass
class ControlState:
    dashboard_status: dict[str, Any]
    toggles: dict[str, bool]

def start_api(host: str, port: int, status_ref: ControlState, jwt_secret: str,
              rbac_path: str, rate_limit: float,
              rate_burst: int) -> Optional[threading.Thread]:
    if not API_AVAILABLE:
        return None
    app = FastAPI(title="Argus Control API")
    security = HTTPBearer(auto_error=False)
    limiter = RateLimiter(rate_limit, rate_burst)
    rbac = RBAC(Path(rbac_path) if rbac_path else None)

    def require(capability: str):
        async def _dep(request: Request,
                       credentials: HTTPAuthorizationCredentials = Depends(security)):
            key = request.client.host if request.client else "unknown"
            if not limiter.check(key):
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
            if not jwt_secret:
                return {"user": "anonymous", "role": "viewer"}
            if credentials is None:
                raise HTTPException(status_code=401, detail="Missing bearer token")
            if not JWT_AVAILABLE:
                raise HTTPException(status_code=503, detail="PyJWT not installed")
            try:
                payload = jwt.decode(credentials.credentials, jwt_secret, algorithms=["HS256"])
            except Exception:
                raise HTTPException(status_code=401, detail="Invalid token")
            role = str(payload.get("role", "viewer"))
            if not rbac.allowed(role, capability):
                raise HTTPException(status_code=403, detail="Forbidden")
            return payload
        return _dep

    @app.get("/health")
    async def health(_ctx: dict[str, Any] = Depends(require("health"))):
        return {"ok": True}

    @app.get("/status")
    async def status(_ctx: dict[str, Any] = Depends(require("status"))):
        return status_ref.dashboard_status

    @app.post("/toggle/pose")
    async def toggle_pose(req: ToggleRequest, _ctx: dict[str, Any] = Depends(require("toggle"))):
        status_ref.toggles["pose"] = bool(req.value)
        return {"ok": True, "pose": status_ref.toggles["pose"]}

    @app.post("/toggle/boxes")
    async def toggle_boxes(req: ToggleRequest, _ctx: dict[str, Any] = Depends(require("toggle"))):
        status_ref.toggles["boxes"] = bool(req.value)
        return {"ok": True, "boxes": status_ref.toggles["boxes"]}

    @app.post("/toggle/ids")
    async def toggle_ids(req: ToggleRequest, _ctx: dict[str, Any] = Depends(require("toggle"))):
        status_ref.toggles["ids"] = bool(req.value)
        return {"ok": True, "ids": status_ref.toggles["ids"]}

    def _run() -> None:
        uvicorn.run(app, host=host, port=port, log_level="warning")

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t
