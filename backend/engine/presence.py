"""
Crowd-sourced presence: where attendees actually are right now.

The city model knows about the places organisers have described (a stadium, an
arena) and about the zones and transport links in its data. It does not know that
a metro station just filled up because a match ended. Attendees who choose to
share their location anonymously fill that gap: enough of them in one spot is a
crowd, whether or not anyone mapped that spot.

Privacy is the design constraint, so it shapes everything here:

* **Opt-in, per person.** The browser only sends anything after the visitor turns
  sharing on. Turning it off (or leaving the page) removes them at once.
* **No raw coordinates are kept.** A position is reduced to a grid cell about
  150 m across the moment it arrives, and only the cell is stored.
* **No identity.** A random per-tab id lets one person count once rather than
  many times. It is never tied to an account, an email or a registration.
* **Short memory.** Everything is forgotten after `TTL_SECONDS`. It lives in
  memory only, so a restart forgets it too.
* **Only groups are published.** A cell is reported only when at least
  `MIN_GROUP` people are in it, so no output can point at an individual.
* **Hard to fake.** One address can only stand for a handful of people, and
  updates faster than every few seconds are ignored.

Run:  python -m engine.presence
"""

import math
import os
import re
import time
import uuid
from collections import defaultdict

from engine.recommend import haversine_km

TTL_SECONDS = int(os.environ.get("EVENTPULSE_PRESENCE_TTL", "300"))
MIN_GROUP = int(os.environ.get("EVENTPULSE_PRESENCE_MIN_GROUP", "3"))
CELL_DEG = 0.0015                 # about 165 m north-south, a little less east-west
MIN_INTERVAL_SECONDS = 4          # faster updates from one tab are ignored
MAX_PER_ADDRESS = 20              # people one network address may plausibly stand for
MAX_SESSIONS = 50_000
GEOFENCE_M = 500                  # how close counts as "at" a venue
COVERED_KM = 25                   # positions further than this from every zone are ignored

_SID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")

# sid -> {"cell": (row, col), "ts": float, "addr": str}
_sessions = {}


class PresenceError(ValueError):
    pass


def _now():
    return time.time()


def _prune(now=None):
    now = now or _now()
    dead = [sid for sid, s in _sessions.items() if now - s["ts"] > TTL_SECONDS]
    for sid in dead:
        del _sessions[sid]


def _cell(lat, lng):
    return (int(math.floor(lat / CELL_DEG)), int(math.floor(lng / CELL_DEG)))


def _centre(cell):
    return ((cell[0] + 0.5) * CELL_DEG, (cell[1] + 0.5) * CELL_DEG)


def _nearest(zones, lat, lng):
    best = None
    for z in zones:
        d = haversine_km(lat, lng, float(z["lat"]), float(z["lng"]))
        if best is None or d < best[0]:
            best = (d, z)
    return best


def report(sid, lat, lng, addr, zones, now=None):
    """
    Record one person's position, as a cell. Returns {"accepted": bool, ...}.
    Raises PresenceError for input that is not acceptable at all.
    """
    now = now or _now()
    if not isinstance(sid, str) or not _SID.match(sid):
        raise PresenceError("Invalid session id.")
    try:
        lat, lng = float(lat), float(lng)
    except (TypeError, ValueError):
        raise PresenceError("Invalid position.")
    if not (math.isfinite(lat) and math.isfinite(lng) and -90 <= lat <= 90 and -180 <= lng <= 180):
        raise PresenceError("Invalid position.")

    _prune(now)
    near = _nearest(zones, lat, lng)
    if near is None or near[0] > COVERED_KM:
        return {"accepted": False, "reason": "outside the area this deployment covers"}

    existing = _sessions.get(sid)
    if existing and now - existing["ts"] < MIN_INTERVAL_SECONDS:
        return {"accepted": True, "throttled": True}

    if not existing:
        if len(_sessions) >= MAX_SESSIONS:
            raise PresenceError("Too many people are sharing right now.")
        if sum(1 for s in _sessions.values() if s["addr"] == addr) >= MAX_PER_ADDRESS:
            raise PresenceError("Too many sessions from this address.")

    _sessions[sid] = {"cell": _cell(lat, lng), "ts": now, "addr": addr}
    return {"accepted": True, "throttled": False}


def leave(sid):
    return _sessions.pop(sid, None) is not None


def clear_demo():
    dead = [sid for sid in _sessions if sid.startswith("demo-")]
    for sid in dead:
        del _sessions[sid]
    return len(dead)


def seed_demo(lat, lng, people, zones, spread_m=110, now=None):
    """Synthetic attendees clustered around a point, for demos with nobody else about.
    Sessions are marked so they can be cleared, and expire like any other."""
    import random
    now = now or _now()
    rng = random.Random(int(now) // 30)         # stable within a demo, varied between them
    made = 0
    dlat = spread_m / 110570
    dlng = spread_m / (111320 * math.cos(math.radians(lat)))
    batch = uuid.uuid4().hex[:8]                 # unique per call, so two seedings never overwrite each other
    for i in range(int(people)):
        sid = f"demo-{batch}-{i:04d}"
        _sessions[sid] = {"cell": _cell(lat + rng.gauss(0, dlat), lng + rng.gauss(0, dlng)), "ts": now, "addr": "demo"}
        made += 1
    return made


def summary(zones, sites, now=None):
    """
    What may be published: crowded cells (at least MIN_GROUP people), how many
    people are at each venue, and the crowds that are NOT at any known venue.
    """
    now = now or _now()
    _prune(now)

    counts = defaultdict(int)
    for s in _sessions.values():
        counts[s["cell"]] += 1

    cells, hotspots = [], []
    site_rows = [(s, float(s["lat"]), float(s["lng"])) for s in sites]
    venue_people = defaultdict(int)

    for cell, n in counts.items():
        lat, lng = _centre(cell)
        # Which venue (if any) is this cell part of?
        at = None
        for s, slat, slng in site_rows:
            if haversine_km(lat, lng, slat, slng) * 1000 <= GEOFENCE_M:
                at = s
                break
        if at is not None:
            venue_people[at["venue_zone_id"]] += n          # counted even when the cell is too small to show on its own

        if n < MIN_GROUP:
            continue
        near = _nearest(zones, lat, lng)
        row = {"lat": round(lat, 5), "lng": round(lng, 5), "count": n,
               "zone_id": near[1]["zone_id"] if near else None,
               "kind": "venue" if at is not None else "unmapped"}
        cells.append(row)
        if at is None:
            hotspots.append({**row, "near": (near[1].get("name") if near else None)})

    venues = []
    for s, _, _ in site_rows:
        n = venue_people.get(s["venue_zone_id"], 0)
        # Below the group size the number could point at a person, so it is withheld.
        venues.append({"venue_zone_id": s["venue_zone_id"], "venue_name": s["venue_name"],
                       "people": n if n >= MIN_GROUP else None, "few": 0 < n < MIN_GROUP})

    cells.sort(key=lambda c: -c["count"])
    hotspots.sort(key=lambda c: -c["count"])
    return {"total_people": len(_sessions), "min_group": MIN_GROUP, "ttl_seconds": TTL_SECONDS,
            "cells": cells, "venues": venues, "hotspots": hotspots[:10]}


if __name__ == "__main__":
    zones = [{"zone_id": "Z04", "name": "Metro Interchange", "lat": 19.0812, "lng": 72.8501}]
    seed_demo(19.0812, 72.8501, 30, zones)
    s = summary(zones, [])
    print(s["total_people"], "people;", len(s["cells"]), "crowded cells;", s["hotspots"][:1])
