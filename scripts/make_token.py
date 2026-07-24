#!/usr/bin/env python3
"""Mint a short-lived JWT for the Argus control API.

Usage:
    export JWT_SECRET="a-long-random-secret"          # PowerShell: $env:JWT_SECRET="..."
    python scripts/make_token.py --role operator --minutes 60

Then call the API:
    curl -H "Authorization: Bearer <token>" http://127.0.0.1:8000/status
"""
from __future__ import annotations

import argparse
import datetime
import os
import sys

try:
    import jwt
except ImportError:
    sys.exit("PyJWT not installed — pip install -r requirements-api.txt")


def main() -> None:
    p = argparse.ArgumentParser(description="Mint a JWT for the Argus API")
    p.add_argument("--role", default="viewer", help="Role claim (viewer/operator/admin or custom)")
    p.add_argument("--minutes", type=int, default=60, help="Token lifetime in minutes")
    p.add_argument("--secret", default="", help="Override JWT secret (default: JWT_SECRET env var)")
    args = p.parse_args()

    secret = args.secret or os.environ.get("JWT_SECRET", "")
    if not secret:
        sys.exit("No secret: set JWT_SECRET or pass --secret")

    now = datetime.datetime.now(datetime.timezone.utc)
    payload = {
        "role": args.role,
        "iat": now,
        "exp": now + datetime.timedelta(minutes=args.minutes),
    }
    token = jwt.encode(payload, secret, algorithm="HS256")
    print(token)


if __name__ == "__main__":
    main()
