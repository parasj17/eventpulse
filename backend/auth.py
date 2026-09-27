"""
Phase 15 — Role-based auth: organizer vs. attendee.

Minimal, dependency-free signed tokens (HMAC-SHA256 over a base64 JSON
payload) rather than pulling in a JWT library — this project has stayed at
a handful of dependencies on purpose, and a token here only needs to prove
"this request is the organizer" or "this request is attendee X," not
interoperate with anything external.

Accounts (attendee and organiser, kept separate) live in engine/accounts.py.
This module only signs and checks the tokens, and keeps one built-in demo
organiser so a fresh checkout works before anyone has created an account.
"""

import base64
import hashlib
import hmac
import json
import os
import time

# A real deployment reads this from the environment. A fixed dev secret is
# fine here since nothing behind it is worth protecting beyond demo gate
# controls, and the tokens are not meant to survive a redeploy anyway.
SECRET_KEY = os.environ.get("EVENTPULSE_SECRET", "eventpulse-dev-secret-do-not-use-in-prod")

# Sign in once and stay signed in: a token lasts 30 days (EVENTPULSE_TOKEN_DAYS to change).
TOKEN_TTL_SECONDS = int(float(os.environ.get("EVENTPULSE_TOKEN_DAYS", "30")) * 86400)

ORGANIZER_EMAIL = os.environ.get("EVENTPULSE_ORGANIZER_EMAIL", "organizer@eventpulse.demo")
ORGANIZER_PASSWORD = os.environ.get("EVENTPULSE_ORGANIZER_PASSWORD", "eventpulse")


def _sign(payload_b64: str) -> str:
    sig = hmac.new(SECRET_KEY.encode(), payload_b64.encode(), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(sig).decode().rstrip("=")


def create_token(role: str, subject: str, **claims) -> str:
    payload = {"role": role, "sub": subject, "iat": int(time.time()), **claims}
    payload_b64 = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return f"{payload_b64}.{_sign(payload_b64)}"


def verify_token(token: str):
    """Returns the payload dict if the token is well-formed, correctly
    signed and unexpired; otherwise None."""
    if not token or "." not in token:
        return None

    payload_b64, sig = token.rsplit(".", 1)
    if not hmac.compare_digest(_sign(payload_b64), sig):
        return None

    try:
        padded = payload_b64 + "=" * (-len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded))
    except Exception:
        return None

    if time.time() - payload.get("iat", 0) > TOKEN_TTL_SECONDS:
        return None

    return payload


def demo_organizer_matches(email: str, password: str) -> bool:
    """The built-in demo organiser (env-overridable). Real organisers create their own
    accounts; this stays so the demo keeps working on a fresh checkout."""
    return hmac.compare_digest(email or "", ORGANIZER_EMAIL) and hmac.compare_digest(password or "", ORGANIZER_PASSWORD)
