"""
User accounts for the two roles: attendees and organisers.

The roles are separate on purpose. An attendee account cannot sign in to the
organiser console and vice versa, and the same email may hold one of each
without the two ever being linked.

Passwords are stored as salted PBKDF2-SHA256 hashes and never in the clear.
Stored in Supabase (Postgres, table ep_accounts) when it is configured (engine/db.py), so
accounts survive restarts and redeploys; otherwise in data/accounts.csv.

Run:  python -m engine.accounts
"""

import hashlib
import hmac
import os
import re
import secrets
import time

import pandas as pd

from engine import db
from engine.forecast import DATA_DIR

ACCOUNTS_CSV = "accounts.csv"
COLUMNS = ["role", "email", "name", "salt", "hash", "created_at"]
ROLES = ("attendee", "organizer")

PBKDF2_ITERATIONS = 200_000
MIN_PASSWORD = 8
MAX_PASSWORD = 128

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]{2,}$")


class AccountExists(ValueError):
    pass


def normalise_email(raw):
    """Lower-cased, trimmed, and shaped like an email. Raises ValueError otherwise."""
    email = str(raw or "").strip().lower()
    if len(email) > 254 or not _EMAIL.match(email):
        raise ValueError("Enter a valid email address")
    return email


def _path(data_dir):
    return os.path.join(data_dir or DATA_DIR, ACCOUNTS_CSV)


_migrated = set()


def _csv(data_dir):
    path = _path(data_dir)
    if os.path.exists(path):
        return pd.read_csv(path, dtype=str).fillna("")
    return pd.DataFrame(columns=COLUMNS)


def _migrate(data_dir):
    """Accounts made before Supabase was connected are copied in once, so nobody has to sign up again."""
    if data_dir in _migrated:
        return
    old = _csv(data_dir)
    if len(old):
        db.executemany("insert into ep_accounts (role, email, name, salt, hash, created_at) values (%s,%s,%s,%s,%s,%s) "
                       "on conflict (role, email) do nothing",
                       [(r["role"], r["email"], r["name"], r["salt"], r["hash"], int(r["created_at"] or 0)) for _, r in old.iterrows()])
    _migrated.add(data_dir)


def _find(role, email, data_dir):
    """The stored row for this role and email, or None. Supabase when connected, else the CSV."""
    if db.enabled():
        _migrate(data_dir)
        rows = db.execute("select name, salt, hash from ep_accounts where role=%s and email=%s", (role, email), fetch=True)
        return {"name": rows[0][0], "salt": rows[0][1], "hash": rows[0][2]} if rows else None
    a = _csv(data_dir)
    hit = a[(a["role"] == role) & (a["email"] == email)]
    return None if hit.empty else hit.iloc[0].to_dict()


def _insert(row, data_dir):
    if db.enabled():
        _migrate(data_dir)
        rows = db.execute("insert into ep_accounts (role, email, name, salt, hash, created_at) values (%s,%s,%s,%s,%s,%s) "
                          "on conflict (role, email) do nothing returning email",
                          (row["role"], row["email"], row["name"], row["salt"], row["hash"], int(row["created_at"])), fetch=True)
        return bool(rows)
    accounts = _csv(data_dir)
    if ((accounts["role"] == row["role"]) & (accounts["email"] == row["email"])).any():
        return False
    pd.concat([accounts, pd.DataFrame([row])], ignore_index=True).to_csv(_path(data_dir), index=False)
    return True


def _hash(password, salt_hex):
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode(), bytes.fromhex(salt_hex), PBKDF2_ITERATIONS
    ).hex()


def create_account(role, email, password, name, data_dir=None):
    """Create an account and return it (without the hash). Raises ValueError with a
    message fit to show the user."""
    if role not in ROLES:
        raise ValueError("Unknown account type")
    email = normalise_email(email)

    name = str(name or "").strip()
    if not 1 <= len(name) <= 80:
        raise ValueError("Enter your name")
    if not isinstance(password, str) or not MIN_PASSWORD <= len(password) <= MAX_PASSWORD:
        raise ValueError(f"Choose a password of at least {MIN_PASSWORD} characters")

    if _find(role, email, data_dir):
        raise AccountExists("An account with that email already exists. Sign in instead.")

    salt = secrets.token_hex(16)
    row = {"role": role, "email": email, "name": name, "salt": salt,
           "hash": _hash(password, salt), "created_at": str(int(time.time()))}
    if not _insert(row, data_dir):                    # someone took it in the meantime
        raise AccountExists("An account with that email already exists. Sign in instead.")
    return {"role": role, "email": email, "name": name}


def verify_login(role, email, password, data_dir=None):
    """The account (without its hash) if the credentials are right, else None."""
    try:
        email = normalise_email(email)
    except ValueError:
        return None
    if role not in ROLES or not isinstance(password, str) or len(password) > MAX_PASSWORD:
        return None

    row = _find(role, email, data_dir)
    if row is None:
        # Do the same amount of work as a real check, so response time does not
        # reveal which emails have accounts.
        _hash(password, "00" * 16)
        return None

    if not hmac.compare_digest(_hash(password, row["salt"]), row["hash"]):
        return None
    return {"role": role, "email": email, "name": row["name"]}


if __name__ == "__main__":
    print(create_account("attendee", "demo@example.com", "correct horse", "Demo", data_dir="/tmp"))
    print(verify_login("attendee", "DEMO@example.com", "correct horse", data_dir="/tmp"))
    print(verify_login("organizer", "demo@example.com", "correct horse", data_dir="/tmp"))
    os.remove("/tmp/accounts.csv")
