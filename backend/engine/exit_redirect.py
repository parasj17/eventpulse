"""
Phase 13 — Exit redirect engine.

Given where someone is on a venue's floor plan, ranks its gates by how good
an exit each one is right now: closer is better, less loaded is better. A
gate the crowd engine (engine/venues.py) has marked "closed" is excluded
outright; "restricted" stays in the running but is scored down rather than
removed, since a slightly-loaded gate can still beat a much farther clear
one.

This is what turns the auto-computed gate status (Phase 9) and floor-zone
density (Phase 12) into an actual instruction — "go this way" — rather than
just a readout. It is deliberately user-optimal for now (nearest, least
loaded, for the one person asking): Phase 25 revisits this with a
network-optimal objective, holding some people back onto a slightly slower
gate so recommendations do not all pile onto the same "best" one.

Run:  python -m engine.exit_redirect
"""

import math
import os

import pandas as pd

from engine.forecast import DATA_DIR
from engine.venues import GATE_LOAD_CRITICAL, current_event_for_zone

WEIGHTS = {"distance": 0.4, "congestion": 0.6}


def _distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _floor_zone_position(venue_zone_id, floor_zone_id, data_dir):
    """
    Where to measure distance from. An explicit floor zone if given and
    found, otherwise the bowl/hall — most of the crowd is there, so it is
    the reasonable default starting point.
    """
    path = os.path.join(data_dir, "floor_zones.csv")
    if not os.path.exists(path):
        return None

    zones = pd.read_csv(path)
    zones = zones[zones["venue_zone_id"] == venue_zone_id]

    if floor_zone_id:
        row = zones[zones["floor_zone_id"] == floor_zone_id]
        if not row.empty:
            r = row.iloc[0]
            return (float(r["x"]), float(r["y"]))

    bowl = zones[zones["kind"] == "bowl"]
    if not bowl.empty:
        r = bowl.iloc[0]
        return (float(r["x"]), float(r["y"]))
    return None


def _reason(gate, distance):
    bits = []
    if gate["live_status"] == "open":
        bits.append("clear")
    else:
        bits.append(f"busy, at {round(gate.get('load_pct', gate['load_ratio'] * 100))}% of capacity")
    bits.append("nearby" if distance <= 20 else "a walk across the floor plan")
    return ", ".join(bits).capitalize()


def recommend_exit(venue_zone_id, current_hour, from_floor_zone_id=None,
                    top_n=5, data_dir=None):
    """
    Rank this venue's gates as exits from wherever `from_floor_zone_id` is,
    given each gate's live status right now. Returns the same
    rank/score/score_breakdown/reason shape as recommend.py, for the same
    reason: the dashboard should be able to say why a gate was picked, not
    just assert it.
    """
    data_dir = data_dir or DATA_DIR

    venue = current_event_for_zone(venue_zone_id, current_hour, data_dir)
    if venue is None or not venue["gates"]:
        return {"venue_zone_id": venue_zone_id, "options": []}

    from_pos = _floor_zone_position(venue_zone_id, from_floor_zone_id, data_dir) or (50, 50)

    candidates = [g for g in venue["gates"] if g["live_status"] != "closed"]
    if not candidates:
        # Nothing open — surface the least-bad option rather than an empty
        # list, so the dashboard can still say "hold position" with a reason
        # instead of going silent.
        candidates = venue["gates"]

    scored = []
    for g in candidates:
        dist = _distance(from_pos, (g["x"], g["y"]))
        congestion = min(1.0, g["load_ratio"])                  # share of capacity in use, 0-1
        scored.append({"gate": g, "distance": dist, "congestion": congestion})

    max_dist = max(s["distance"] for s in scored) or 1.0

    options = []
    for s in scored:
        g = s["gate"]
        distance_score = 1.0 - (s["distance"] / max_dist)
        congestion_score = 1.0 - s["congestion"]
        score = (
            distance_score * WEIGHTS["distance"]
            + congestion_score * WEIGHTS["congestion"]
        )

        options.append({
            "gate_id": g["gate_id"],
            "name": g["name"],
            "x": g["x"],
            "y": g["y"],
            "live_status": g["live_status"],
            "load_ratio": g["load_ratio"],
            "distance": round(s["distance"], 1),
            "score": round(score, 3),
            "score_breakdown": {
                "distance_score": round(distance_score, 3),
                "congestion_score": round(congestion_score, 3),
            },
            "reason": _reason(g, s["distance"]),
        })

    # Best score first — the blend of "close" and "not overloaded" the
    # weights above define, not distance alone.
    options.sort(key=lambda o: -o["score"])
    options = options[:top_n]
    for i, o in enumerate(options, start=1):
        o["rank"] = i

    return {
        "venue_zone_id": venue_zone_id,
        "from_floor_zone_id": from_floor_zone_id,
        "hour": current_hour,
        "options": options,
    }


if __name__ == "__main__":
    for h in (100, 104, 106, 108):
        rec = recommend_exit("Z01", h)
        print(f"\n--- hour {h} ---")
        for o in rec["options"]:
            print(f"  {o['rank']}. {o['name']:<20} score {o['score']:.2f}  "
                  f"[{o['live_status']}]  {o['reason']}")
