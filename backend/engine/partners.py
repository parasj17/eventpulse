"""
Partner API keys: how a ticketing or events platform plugs EventPulse into its own app.

The business model is B2B. An organiser (someone at the platform that sells the tickets)
creates a key in the organiser console and gives it to their developers. With it, the
platform can list its events, read live conditions for each one, and register a ticket
buyer to get a personal live link: the page that shows the quietest gate on the way in
and the fastest way out afterwards. The platform shares that link in its own ticket
email or app, so the attendee needs no new account.

How keys are handled:

* **Shown once.** A key is `ep_live_` plus 32 random hex characters. Only its SHA-256
  hash is stored, so a leaked data file does not leak working keys.
* **Scoped to the organiser.** A key can see and register for exactly the events at
  venues its organiser can edit, nothing else.
* **Revocable.** A revoked key stops working at once; it stays listed as revoked.
* **Rate limited.** Each key gets `RATE_PER_MINUTE` requests a minute.

Run:  python -m engine.partners
"""

import hashlib
import hmac
import os
import re
import secrets
import time
from collections import defaultdict, deque

import pandas as pd

from engine.forecast import DATA_DIR

KEYS_CSV = "partner_keys.csv"
COLUMNS = ["key_id", "owner", "label", "prefix", "key_hash", "created_at", "revoked_at", "last_used_at"]
PREFIX = "ep_live_"
MAX_ACTIVE_KEYS = 10
RATE_PER_MINUTE = int(os.environ.get("EVENTPULSE_PARTNER_RATE", "120"))
LAST_USED_WRITE_SECONDS = 60           # how stale "last used" may get before it is written back

_KEY = re.compile(r"^ep_live_[0-9a-f]{32}$")
_calls = defaultdict(deque)
_last_used = {}                         # key_id -> time, kept in memory between writes


class PartnerError(ValueError):
    pass


class NotFound(PartnerError):
    pass


def _path(data_dir):
    return os.path.join(data_dir or DATA_DIR, KEYS_CSV)


def _load(data_dir=None):
    p = _path(data_dir)
    if not os.path.exists(p):
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(p, dtype=str).fillna("")


def _save(df, data_dir=None):
    df.to_csv(_path(data_dir), index=False)


def _hash(key):
    return hashlib.sha256(key.encode()).hexdigest()


def _public(row):
    kid = row["key_id"]
    used = _last_used.get(kid) or (float(row["last_used_at"]) if row["last_used_at"] else None)
    return {"key_id": kid, "label": row["label"], "prefix": row["prefix"],
            "created_at": float(row["created_at"]), "revoked": bool(row["revoked_at"]),
            "revoked_at": float(row["revoked_at"]) if row["revoked_at"] else None,
            "last_used_at": used}


def create_key(owner, label, data_dir=None):
    """Make a key for `owner`. The full key is in the result and nowhere else, ever."""
    owner = (owner or "").strip().lower()
    label = re.sub(r"\s+", " ", str(label or "").strip())
    if not 1 <= len(label) <= 60:
        raise PartnerError("Name the key (up to 60 characters), e.g. the app that will use it.")
    df = _load(data_dir)
    active = df[(df["owner"] == owner) & (df["revoked_at"] == "")]
    if len(active) >= MAX_ACTIVE_KEYS:
        raise PartnerError(f"You already have {MAX_ACTIVE_KEYS} active keys. Revoke one you no longer use.")

    key = PREFIX + secrets.token_hex(16)
    row = {"key_id": "k_" + secrets.token_hex(6), "owner": owner, "label": label,
           "prefix": key[:len(PREFIX) + 6], "key_hash": _hash(key), "created_at": str(time.time()),
           "revoked_at": "", "last_used_at": ""}
    _save(pd.concat([df, pd.DataFrame([row])], ignore_index=True), data_dir)
    return {**_public(row), "key": key}


def list_keys(owner, data_dir=None):
    owner = (owner or "").strip().lower()
    df = _load(data_dir)
    rows = [_public(r) for _, r in df[df["owner"] == owner].iterrows()]
    rows.sort(key=lambda r: (r["revoked"], -r["created_at"]))
    return rows


def revoke_key(owner, key_id, data_dir=None):
    owner = (owner or "").strip().lower()
    df = _load(data_dir)
    idx = df.index[(df["key_id"] == key_id) & (df["owner"] == owner)]
    if not len(idx):
        raise NotFound("No such key.")            # the same answer whether it is missing or someone else's
    if not df.loc[idx[0], "revoked_at"]:
        df.loc[idx[0], "revoked_at"] = str(time.time())
        _save(df, data_dir)
    return _public(df.loc[idx[0]])


def authenticate(key, data_dir=None, now=None):
    """
    The organiser a key acts for, or None. Also enforces the per-key rate limit
    (raises PartnerError with `rate_limited = True`).
    """
    if not isinstance(key, str) or not _KEY.match(key.strip()):
        return None
    key = key.strip()
    h = _hash(key)
    df = _load(data_dir)
    match = df[df["revoked_at"] == ""]
    row = None
    for _, r in match.iterrows():
        if hmac.compare_digest(r["key_hash"], h):
            row = r
            break
    if row is None:
        return None

    now = now or time.time()
    q = _calls[row["key_id"]]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= RATE_PER_MINUTE:
        e = PartnerError(f"Rate limit: {RATE_PER_MINUTE} requests a minute per key.")
        e.rate_limited = True
        raise e
    q.append(now)

    kid = row["key_id"]
    _last_used[kid] = now
    stored = float(row["last_used_at"]) if row["last_used_at"] else 0
    if now - stored > LAST_USED_WRITE_SECONDS:
        df.loc[df["key_id"] == kid, "last_used_at"] = str(now)
        _save(df, data_dir)
    return {"owner": row["owner"], "key_id": kid, "label": row["label"]}


if __name__ == "__main__":
    import tempfile
    d = tempfile.mkdtemp()
    k = create_key("ana@example.com", "Ticketing app", d)
    print("created", k["prefix"], "...", "auth ->", authenticate(k["key"], d))
    revoke_key("ana@example.com", k["key_id"], d)
    print("after revoke ->", authenticate(k["key"], d))
