"""
Phase 21 — Restaurant inventory, biased toward quiet zones.

Mirrors hotels.py exactly: two properties per zone, live availability and
badge derived from the zone's own occupancy, so "where should I eat while
I wait for the crowd to clear" reads off the same numbers as everything
else on the dashboard rather than a separate, disconnected list.

The one addition hotels.py does not need: a `quiet_pick` flag, since the
whole point of surfacing restaurants (per the incentive engine in
incentives.py) is steering people toward zones that are *not* crowded —
callers that want that bias filter on it rather than re-deriving it from
occupancy themselves.
"""

import os
import random

import pandas as pd

from engine.forecast import DATA_DIR

# Two restaurants per zone, named to read as a plausible city.
RESTAURANTS = {
    "Z01": [("Precinct Grill", 4.2, 1.15), ("Matchday Kitchen", 3.9, 0.95)],
    "Z02": [("Riverside Table", 4.5, 0.88), ("The Waterline Cafe", 4.1, 0.72)],
    "Z03": [("Old Town Thali", 4.6, 0.68), ("Fort & Field Bistro", 4.2, 0.90)],
    "Z04": [("Interchange Diner", 3.8, 0.62), ("Platform Nine Eats", 4.0, 0.75)],
    "Z05": [("Northgate Kitchen", 4.3, 0.80), ("The Gatehouse Grill", 4.1, 0.71)],
    "Z06": [("Skyway Bistro", 4.4, 1.05), ("Terminal Tandoor", 4.0, 0.89)],
    "Z07": [("Lakeside Kitchen", 4.7, 0.66), ("The Boathouse Cafe", 4.3, 0.58)],
    "Z08": [("Meridian Kitchen", 4.5, 1.20), ("The Exchange Bistro", 4.2, 1.02)],
    "Z09": [("Spice Quarter Table", 4.6, 0.55), ("Market Lodge Diner", 3.9, 0.48)],
    "Z10": [("West End Kitchen", 4.1, 0.60), ("The Long Barn Grill", 4.0, 0.52)],
}

# Occupancy at which a restaurant's zone is described as filling fast.
HOT_PCT = 78.0

# A zone quiet enough to be worth actively steering people toward.
QUIET_PCT = 55.0


def build_restaurants(current_hour=12, data_dir=None):
    """
    Every restaurant with its live availability and a `quiet_pick` flag for
    zones worth steering demand toward — the inventory incentives.py's
    offers point at.
    """
    data_dir = data_dir or DATA_DIR

    zones = pd.read_csv(os.path.join(data_dir, "zones.csv"))
    occ = pd.read_csv(os.path.join(data_dir, "occupancy_timeline.csv"))

    snap = occ[occ["hour_offset"] == current_hour]
    if snap.empty:
        return []

    pct_by_zone = dict(zip(snap["zone_id"], snap["occupancy_pct"]))

    rng = random.Random(11)
    out = []

    for _, z in zones.iterrows():
        zid = z["zone_id"]
        zone_pct = float(pct_by_zone.get(zid, 0.0))

        for name, rating, price_mult in RESTAURANTS.get(zid, []):
            price_level = max(1, min(4, round(price_mult * 2)))
            stock = int(z["hotel_capacity"] * rng.uniform(0.03, 0.06))
            tables_left = max(0, int(stock * (100 - zone_pct) / 100))

            if zone_pct >= HOT_PCT:
                badge, badge_note = "BUSY", (
                    f"{tables_left} tables left" if tables_left else "Fully booked"
                )
            elif zone_pct <= QUIET_PCT:
                badge, badge_note = "QUIET", "Walk-ins welcome"
            else:
                badge, badge_note = "OPEN", f"{tables_left} tables available"

            out.append({
                "restaurant_id": f"{zid}-{name[:3].upper()}",
                "name": name,
                "zone_id": zid,
                "zone_name": z["name"],
                "rating": rating,
                "price_level": price_level,
                "tables_left": tables_left,
                "tables_total": stock,
                "zone_occupancy_pct": round(zone_pct, 1),
                "distance_to_venue_km": float(z["distance_to_venue_km"]),
                "quiet_pick": zone_pct <= QUIET_PCT,
                "badge": badge,
                "badge_note": badge_note,
            })

    # Quiet zones first when picks are otherwise close, then nearest —
    # matches the incentive engine's intent of actively surfacing the
    # underused option rather than just the closest one.
    out.sort(key=lambda r: (not r["quiet_pick"], r["distance_to_venue_km"], -r["rating"]))
    return out


if __name__ == "__main__":
    for r in build_restaurants(current_hour=110)[:10]:
        print(f"{r['badge']:<6} {r['name']:<22} {r['zone_name']:<18} "
              f"{r['rating']}*  {r['tables_left']:>3} left  {r['badge_note']}")
