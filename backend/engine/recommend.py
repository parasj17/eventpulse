"""
Phase 4 — Recommendation engine.

Given a zone under pressure, rank alternative zones that could absorb the
overflow.

Every option carries its score breakdown. That transparency matters: on screen
it turns "the system says Riverside East" into "Riverside East, because it has
1,860 spare beds, is 2km away, and costs less" — which is what makes the
recommendation credible rather than magic.
"""

import math
import os

import pandas as pd

from engine.forecast import DATA_DIR, forecast_all

# How the three factors are weighted in the final score. Must sum to 1.0.
WEIGHTS = {
    "capacity": 0.45,   # spare beds matter most — no point sending people somewhere full
    "distance": 0.35,   # but not so far that attendees refuse
    "cost": 0.20,       # cheaper is a nice-to-have, not the driver
}

# A candidate only drops out once it is genuinely full. Visitors will accept a
# nearby zone that is three-quarters full over an empty one across the city, so
# filtering on occupancy would rule out exactly the places they actually want.
MAX_CANDIDATE_PCT = 99.5

# Occupancy at which a zone starts being given onward directions. Routing early
# is the point: by the time a zone is full, redirecting is damage control.
ROUTING_THRESHOLD_PCT = 45.0

# Beyond this distance a zone is treated as impractical regardless of capacity.
MAX_USEFUL_KM = 8.0


def haversine_km(lat1, lng1, lat2, lng2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _reason(spare, distance_km, rate, source_rate):
    bits = []

    if spare >= 1000:
        bits.append(f"{spare:,} beds free")
    else:
        bits.append(f"{spare} beds free")

    if distance_km <= 2.5:
        bits.append(f"only {distance_km}km away")
    else:
        bits.append(f"{distance_km}km away")

    if rate < source_rate * 0.9:
        bits.append("at a lower nightly rate")
    elif rate > source_rate * 1.1:
        bits.append("though pricier")

    return ", ".join(bits).capitalize()


def recommend_for_zone(zone_id, current_hour=12, data_dir=None, top_n=8):
    """
    Rank alternative zones for a zone under pressure.

    Returns a dict matching the "recommendations" shape in
    contracts/mock-data.json.
    """
    data_dir = data_dir or DATA_DIR
    zones = pd.read_csv(os.path.join(data_dir, "zones.csv"))
    occupancy = pd.read_csv(os.path.join(data_dir, "occupancy_timeline.csv"))

    forecasts = {f["zone_id"]: f for f in forecast_all(occupancy, current_hour=current_hour)}

    if zone_id not in set(zones["zone_id"]):
        return {"for_zone_id": zone_id, "options": []}

    source = zones[zones["zone_id"] == zone_id].iloc[0]

    # Build the candidate set.
    candidates = []
    for _, z in zones.iterrows():
        if z["zone_id"] == zone_id:
            continue

        fc = forecasts.get(z["zone_id"])
        if not fc:
            continue

        occ_pct = fc["current_occupancy_pct"]
        if occ_pct >= MAX_CANDIDATE_PCT:
            continue                                   # already too full to help

        distance = round(haversine_km(source["lat"], source["lng"], z["lat"], z["lng"]), 2)
        if distance > MAX_USEFUL_KM:
            continue                                   # impractically far

        spare = int(z["hotel_capacity"] * (100.0 - occ_pct) / 100.0)
        if spare <= 0:
            continue

        candidates.append({
            "zone": z,
            "occ_pct": occ_pct,
            "spare": spare,
            "distance": distance,
            "trend": fc["trend_pct_per_hour"],
        })

    if not candidates:
        return {"for_zone_id": zone_id, "options": []}

    # Normalise each factor to 0..1 across the candidate set, so the score
    # reflects relative merit rather than raw units.
    max_spare = max(c["spare"] for c in candidates)
    max_dist = max(c["distance"] for c in candidates) or 1.0
    rates = [c["zone"]["avg_nightly_rate"] for c in candidates]
    min_rate, max_rate = min(rates), max(rates)
    rate_span = (max_rate - min_rate) or 1.0

    options = []
    for c in candidates:
        capacity_score = c["spare"] / max_spare
        distance_score = 1.0 - (c["distance"] / max_dist)       # nearer is better
        cost_score = 1.0 - ((c["zone"]["avg_nightly_rate"] - min_rate) / rate_span)

        score = (
            capacity_score * WEIGHTS["capacity"]
            + distance_score * WEIGHTS["distance"]
            + cost_score * WEIGHTS["cost"]
        )

        options.append({
            "zone_id": c["zone"]["zone_id"],
            "zone_name": c["zone"]["name"],
            "spare_capacity": c["spare"],
            "current_occupancy_pct": c["occ_pct"],
            "distance_km": c["distance"],
            "avg_nightly_rate": int(c["zone"]["avg_nightly_rate"]),
            "score": round(score, 3),
            "score_breakdown": {
                "capacity_score": round(capacity_score, 3),
                "distance_score": round(distance_score, 3),
                "cost_score": round(cost_score, 3),
            },
            "reason": _reason(
                c["spare"], c["distance"],
                c["zone"]["avg_nightly_rate"], source["avg_nightly_rate"],
            ),
        })

    # Nearest first. Someone turned away from the venue wants the closest room
    # that still exists, not the emptiest one on the far side of the city.
    # Spare capacity breaks ties between zones at a similar distance.
    options.sort(key=lambda o: (o["distance_km"], -o["spare_capacity"]))
    options = options[:top_n]
    for i, o in enumerate(options, start=1):
        o["rank"] = i

    return {
        "for_zone_id": zone_id,
        "for_zone_name": source["name"],
        "options": options,
    }


if __name__ == "__main__":
    from engine.risk import detect_all

    alerts = detect_all(current_hour=12)
    accom = [a for a in alerts if a["metric"] == "accommodation"]

    if not accom:
        print("No accommodation alerts to recommend against.")
    else:
        target = accom[0]
        rec = recommend_for_zone(target["zone_id"], current_hour=12)

        print(f"Pressure at: {rec['for_zone_name']} ({target['current_pct']:.0f}% full)")
        print(f"Redistribute to:\n")

        for o in rec["options"]:
            print(f"  {o['rank']}. {o['zone_name']:<20} score {o['score']:.2f}")
            print(f"     {o['reason']}")
            b = o["score_breakdown"]
            print(f"     capacity {b['capacity_score']:.2f} | "
                  f"distance {b['distance_score']:.2f} | cost {b['cost_score']:.2f}\n")
