"""
Phase 26 — Dynamic incentive engine.

Every engine before this one *informs or routes* people away from a crowded
place: an alert, a recommended zone, a ranked exit. This is the one that
changes demand itself — when a zone is trending toward overcrowded, generate
an offer that actively pulls people toward a quieter one, sized by how much
relief is actually needed. It is the inverse of ride-hailing surge pricing:
discount the under-demanded option rather than upcharge the over-demanded
one, since raising prices during a crowd-safety-adjacent event is the wrong
lever.

No merchant registry exists yet anywhere in the project, so this reuses the
zone-level economics recommend.py and hotels.py already carry (nightly rate,
distance, spare capacity) rather than inventing a synthetic list of named
businesses with no other consumer. A real deployment would replace the
implicit "zone has spare capacity" signal used here with an actual merchant
sign-up (who is participating, what capacity they can serve) behind the same
interface, and would track redemptions to close the feedback loop this
version does not have real data for.

Run:  python -m engine.incentives
"""

from engine.forecast import DATA_DIR
from engine.recommend import recommend_for_zone
from engine.risk import CRITICAL_PCT, WARNING_PCT, detect_all

# A zone just past the warning threshold gets a token nudge; one at or past
# critical gets the strongest offer this scheme makes. Scaling within this
# band by how close the source zone is to critical is what "sized by how
# much redirection is needed" means in practice — not a blanket discount
# regardless of severity.
MIN_DISCOUNT_PCT = 5
MAX_DISCOUNT_PCT = 25


def _discount_for(occupancy_pct):
    """Linear ramp from MIN at the warning threshold to MAX at critical."""
    span = CRITICAL_PCT - WARNING_PCT
    frac = (occupancy_pct - WARNING_PCT) / span if span else 1.0
    frac = max(0.0, min(1.0, frac))
    return round(MIN_DISCOUNT_PCT + frac * (MAX_DISCOUNT_PCT - MIN_DISCOUNT_PCT))


def generate_incentives(current_hour=12, data_dir=None, top_n_per_zone=2):
    """
    One offer per (crowded zone, quiet destination) pair worth surfacing:
    "N% off at participating restaurants in <destination>, because <source>
    is filling up." Built directly on the same alerts and rankings the rest
    of the dashboard already shows, so an offer here always corresponds to a
    real, visible pressure reading rather than a separate judgement call.
    """
    data_dir = data_dir or DATA_DIR
    alerts = detect_all(current_hour=current_hour, data_dir=data_dir)
    crowd_alerts = [a for a in alerts if a["metric"] == "accommodation"]

    offers = []
    for a in crowd_alerts:
        rec = recommend_for_zone(a["zone_id"], current_hour=current_hour,
                                  data_dir=data_dir, top_n=top_n_per_zone)
        discount = _discount_for(a["current_pct"])

        for opt in rec["options"]:
            offers.append({
                "source_zone_id": a["zone_id"],
                "source_zone_name": a["zone_name"],
                "source_occupancy_pct": a["current_pct"],
                "destination_zone_id": opt["zone_id"],
                "destination_zone_name": opt["zone_name"],
                "discount_pct": discount,
                "message": (
                    f"{a['zone_name']} is filling up — get {discount}% off "
                    f"at participating restaurants in {opt['zone_name']} instead."
                ),
                "distance_km": opt["distance_km"],
                "spare_capacity": opt["spare_capacity"],
            })

    # Strongest incentive first — that is the zone under the most pressure.
    offers.sort(key=lambda o: -o["discount_pct"])
    return offers


if __name__ == "__main__":
    offers = generate_incentives(current_hour=110)
    if not offers:
        print("No zones crowded enough to trigger an incentive at this hour.")
    for o in offers:
        print(f"[{o['discount_pct']:>2}%] {o['message']}  "
              f"({o['distance_km']}km, {o['spare_capacity']:,} spare)")
