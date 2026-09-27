"""
Phase 6 — Hotel inventory.

Everything up to here works at zone level, which is the right grain for a
control room but useless to a visitor: nobody books a zone. This layer puts two
properties in every zone and derives their live state from the zone's occupancy,
so the operator view and the attendee view are reading the same numbers.

A property's badge is earned, not assigned:

  HOT   its zone is filling fast and rooms are going
  GEM   well rated, still quiet, and nobody has found it yet
  SAVE  materially cheaper than staying at the venue

The saving is real: it is measured against the venue zone's average rate, which
is what a visitor pays for the convenience of not moving.
"""

import hashlib
import math
import os
import random

import pandas as pd

from engine.forecast import DATA_DIR

# Two properties per zone, named to read as a plausible city.
HOTELS = {
    "Z01": [("The Arena Grand", 4.3, 1.28), ("Precinct House", 4.0, 1.12)],
    "Z02": [("Riverside Rooms", 4.6, 0.82), ("The Waterline", 4.2, 0.74)],
    "Z03": [("Old Town Haveli", 4.7, 0.95), ("Fort & Field", 4.1, 0.86)],
    "Z04": [("Interchange Inn", 3.9, 0.79), ("Platform Nine", 4.2, 0.88)],
    "Z05": [("Northgate Stay", 4.4, 0.91), ("The Gatehouse", 4.0, 0.83)],
    "Z06": [("Skyway Residency", 4.5, 1.18), ("Terminal Suites", 4.1, 1.05)],
    "Z07": [("Lakeside Retreat", 4.8, 0.71), ("The Boathouse", 4.3, 0.64)],
    "Z08": [("Meridian Business", 4.4, 1.34), ("The Exchange", 4.6, 1.22)],
    "Z09": [("Market Lodge", 3.8, 0.58), ("Spice Quarter", 4.5, 0.66)],
    "Z10": [("West End Rooms", 4.2, 0.69), ("The Long Barn", 4.0, 0.61)],
}

# Occupancy at which a property is described as filling fast.
HOT_PCT = 78.0

# A property is a find if it is this well rated and its zone is still this quiet.
GEM_RATING = 4.4
GEM_MAX_PCT = 60.0

# Minimum rupees below the venue rate before a stay is sold on price.
SAVE_THRESHOLD = 900


def _position(hotel_id, zone_lat, zone_lng):
    """
    A stable position for a hotel: its zone's centre plus an offset of 150 to 650
    metres in a fixed direction. Zones only have a centre, but a visitor's distance
    to a hotel is only useful if hotels in the same zone are not all in the same
    spot. Derived from the id, so it never moves between requests.
    """
    h = hashlib.md5(hotel_id.encode()).digest()
    angle = h[0] / 255 * 2 * math.pi
    metres = 150 + h[1] / 255 * 500
    dlat = metres * math.sin(angle) / 110570
    dlng = metres * math.cos(angle) / (111320 * math.cos(math.radians(zone_lat)))
    return round(zone_lat + dlat, 5), round(zone_lng + dlng, 5)


def _travel_minutes(distance_km):
    """
    Door-to-venue time on the shuttle network, plus a few minutes either end.

    Quoting a walking figure would be misleading: six kilometres is a seventy
    minute walk and nobody is doing that, so this reflects how visitors would
    actually make the trip.
    """
    return max(4, int(round(4 + distance_km * 3.5)))


def build_hotels(current_hour=12, data_dir=None):
    """
    Every property with its live availability, badge and saving.

    Availability comes from the zone's occupancy, so a hotel empties and fills
    with the zone it sits in rather than drifting off on its own.
    """
    data_dir = data_dir or DATA_DIR

    zones = pd.read_csv(os.path.join(data_dir, "zones.csv"))
    occ = pd.read_csv(os.path.join(data_dir, "occupancy_timeline.csv"))

    snap = occ[occ["hour_offset"] == current_hour]
    if snap.empty:
        return []

    pct_by_zone = dict(zip(snap["zone_id"], snap["occupancy_pct"]))

    # Trend over the last three hours, to tell filling from settled.
    prev = occ[occ["hour_offset"] == max(0, current_hour - 3)]
    prev_pct = dict(zip(prev["zone_id"], prev["occupancy_pct"]))

    venue = zones[zones["is_venue_zone"]].iloc[0]
    venue_rate = float(venue["avg_nightly_rate"])

    rng = random.Random(7)
    out = []

    for _, z in zones.iterrows():
        zid = z["zone_id"]
        zone_pct = float(pct_by_zone.get(zid, 0.0))
        delta = zone_pct - float(prev_pct.get(zid, zone_pct))

        for name, rating, rate_mult in HOTELS.get(zid, []):
            rate = int(round(float(z["avg_nightly_rate"]) * rate_mult / 50) * 50)

            # Rooms scale with the zone's own inventory, then empty as it fills.
            stock = int(z["hotel_capacity"] * rng.uniform(0.055, 0.095))
            rooms_left = max(0, int(stock * (100 - zone_pct) / 100))

            saving = int(venue_rate - rate)

            if zone_pct >= HOT_PCT or (delta > 2.5 and zone_pct > 55):
                badge, badge_note = "HOT", (
                    f"Filling fast, {rooms_left} left" if rooms_left
                    else "Sold out"
                )
            elif rating >= GEM_RATING and zone_pct <= GEM_MAX_PCT:
                badge, badge_note = "GEM", "Untapped, no queues"
            elif saving >= SAVE_THRESHOLD:
                badge, badge_note = "SAVE", f"Save Rs {saving:,} a night"
            else:
                badge, badge_note = "OPEN", f"{rooms_left} rooms available"

            hotel_id = f"{zid}-{name[:3].upper()}"
            lat, lng = _position(hotel_id, float(z["lat"]), float(z["lng"]))
            out.append({
                "hotel_id": hotel_id,
                "lat": lat,
                "lng": lng,
                "name": name,
                "zone_id": zid,
                "zone_name": z["name"],
                "rating": rating,
                "nightly_rate": rate,
                "saving_vs_venue": max(0, saving),
                "rooms_left": rooms_left,
                "rooms_total": stock,
                "zone_occupancy_pct": round(zone_pct, 1),
                "zone_trend_pct_3h": round(delta, 2),
                "travel_minutes": _travel_minutes(float(z["distance_to_venue_km"])),
                "distance_to_venue_km": float(z["distance_to_venue_km"]),
                "is_venue_zone": bool(z["is_venue_zone"]),
                "badge": badge,
                "badge_note": badge_note,
            })

    # Nearest first, matching how the platform routes people: a visitor turned
    # away from the venue wants the closest room that still exists. Sold-out
    # properties sink regardless of how close they are.
    out.sort(key=lambda h: (h["rooms_left"] == 0, h["distance_to_venue_km"], -h["rating"]))
    return out


if __name__ == "__main__":
    for h in build_hotels(current_hour=100)[:10]:
        print(f"{h['badge']:<5} {h['name']:<20} {h['zone_name']:<18} "
              f"Rs{h['nightly_rate']:>6,}  {h['rating']}*  "
              f"{h['rooms_left']:>3} left  {h['travel_minutes']:>2}min  "
              f"{h['badge_note']}")
