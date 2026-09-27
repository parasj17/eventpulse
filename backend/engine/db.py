"""
Supabase (Postgres) storage for what must survive restarts and redeploys: accounts and
attendee registrations.

Settings come from the environment or backend/.supabase (a private file, KEY=value):

    SUPABASE_DB_PASSWORD   the database password
    SUPABASE_PROJECT_REF   e.g. ozostnxlxllotzmrnold
    SUPABASE_DB_HOST       the session pooler, e.g. aws-0-ap-southeast-1.pooler.supabase.com
                           (IPv4; the direct db.<ref>.supabase.co address is IPv6-only)
    SUPABASE_DB_PORT       5432
    DATABASE_URL           or a full postgres:// URL instead of the above

The server talks to Postgres directly with the database password, so the tables need no
public access and the browser never sees them. Without settings (or with EVENTPULSE_DB=off)
the app falls back to the CSV files in data/.
"""

import os
import threading

import psycopg

CFG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".supabase")

SCHEMA = """
create table if not exists ep_accounts (
    role text not null, email text not null, name text not null,
    salt text not null, hash text not null, created_at bigint not null,
    primary key (role, email));
create table if not exists ep_attendees (
    attendee_token text primary key, email text, event_id text, venue_zone_id text,
    registered_hour text, partner text, legacy_phone text);
alter table ep_accounts enable row level security;
alter table ep_attendees enable row level security;
"""

_lock = threading.Lock()
_conn = None
_ready = False


def _cfg():
    c = {}
    try:
        with open(CFG_FILE) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    c[k.strip()] = v.strip()
    except OSError:
        pass
    for k in ("DATABASE_URL", "SUPABASE_DB_PASSWORD", "SUPABASE_PROJECT_REF", "SUPABASE_DB_HOST", "SUPABASE_DB_PORT"):
        if os.environ.get(k):
            c[k] = os.environ[k]
    return c


def enabled():
    if os.environ.get("EVENTPULSE_DB", "").lower() == "off":
        return False
    c = _cfg()
    return bool(c.get("DATABASE_URL") or (c.get("SUPABASE_DB_PASSWORD") and c.get("SUPABASE_PROJECT_REF")))


def _connect():
    c = _cfg()
    if c.get("DATABASE_URL"):
        return psycopg.connect(c["DATABASE_URL"], connect_timeout=10, autocommit=True)
    ref = c["SUPABASE_PROJECT_REF"]
    host = c.get("SUPABASE_DB_HOST") or f"db.{ref}.supabase.co"
    user = f"postgres.{ref}" if "pooler" in host else "postgres"
    return psycopg.connect(host=host, port=int(c.get("SUPABASE_DB_PORT") or 5432), user=user, dbname="postgres",
                           password=c["SUPABASE_DB_PASSWORD"], sslmode="require", connect_timeout=10, autocommit=True)


def execute(sql, params=None, fetch=False):
    """Run one statement on a shared connection, reconnecting once if it has dropped."""
    global _conn, _ready
    with _lock:
        for attempt in (1, 2):
            try:
                if _conn is None or _conn.closed:
                    _conn = _connect()
                if not _ready:
                    _conn.execute(SCHEMA)
                    _ready = True
                cur = _conn.execute(sql, params or ())
                return cur.fetchall() if fetch else None
            except psycopg.OperationalError:
                try:
                    if _conn is not None:
                        _conn.close()
                except Exception:
                    pass
                _conn = None
                if attempt == 2:
                    raise


def executemany(sql, rows):
    global _conn, _ready
    with _lock:
        if _conn is None or _conn.closed:
            _conn = _connect()
        if not _ready:
            _conn.execute(SCHEMA)
            _ready = True
        with _conn.cursor() as cur:
            cur.executemany(sql, rows)


if __name__ == "__main__":
    print("enabled:", enabled())
    print(execute("select count(*) from ep_accounts", fetch=True), execute("select count(*) from ep_attendees", fetch=True))
