"""
Phase 4b — Redistribution simulator.

Answers the question a recommendation raises but does not settle: if we actually
redirect these people, what happens?

Given a source zone under pressure and a number of visitors to move, this works
out where they would go, what each zone's occupancy becomes, and whether the
alert clears. It is the difference between a dashboard that advises and one that
lets an organiser test the advice before acting on it.

Two things it models honestly rather than hand-waving:

* **Capacity is finite.** Visitors fill the best-ranked zone until it reaches a
  practical ceiling, then spill into the next. You cannot move 3,000 people into
  a zone with 900 free beds.

* **Not everyone complies.** A share of redirected visitors ignore the guidance
  and stay put. The default assumes roughly three in four follow it, which is
  optimistic for a nudge and pessimistic for a hard booking block, so it is
  adjustable.
"""

import os

import pandas as pd

from engine.forecast import DATA_DIR, forecast_all
from engine.recommend import recommend_for_zone
from engine.risk import CRITICAL_PCT, WARNING_PCT

# Occupancy a receiving zone is not pushed past during redistribution. Filling a
# zone to exactly 100% just moves the problem, so we stop short.
RECEIVING_CEILING_PCT = 92.0

# Share of redirected visitors who actually go where they are directed.
DEFAULT_COMPLIANCE = 0.75


def simulate_redistribution(
    zone_id,
    move_count,
    current_hour=12,
    compliance=DEFAULT_COMPLIANCE,
    data_dir=None,
    frames=None,
    recs=None,
):
    """
    Move `move_count` visitors out of `zone_id` and into the ranked alternatives.

    Returns the before/after picture for every zone touched, plus what it means
    for the alert on the source zone.
    """
    data_dir = data_dir or DATA_DIR

    # `frames` lets a caller hand in dataframes it already holds, and `recs` the
    # ranking for this zone and hour. Both are optional; without them this reads
    # from disk and ranks as before.
    if frames is None:
        zones = pd.read_csv(os.path.join(data_dir, "zones.csv"))
        occupancy = pd.read_csv(os.path.join(data_dir, "occupancy_timeline.csv"))
    else:
        zones, occupancy = frames["zones"], frames["occupancy"]

    snap = occupancy[occupancy["hour_offset"] == current_hour]
    if snap.empty or zone_id not in set(zones["zone_id"]):
        return {"error": "No data for that zone or hour"}

    occ_now = dict(zip(snap["zone_id"], snap["occupancy"]))
    cap = dict(zip(zones["zone_id"], zones["hotel_capacity"]))
    names = dict(zip(zones["zone_id"], zones["name"]))

    forecasts = {f["zone_id"]: f for f in forecast_all(occupancy, current_hour=current_hour)}

    source_before = occ_now.get(zone_id, 0)
    source_cap = cap[zone_id]

    # You cannot move more people than are actually there.
    requested = int(move_count)
    movable = min(requested, source_before)

    # Only the compliant share actually relocates.
    actually_moved = int(movable * compliance)

    # Rank destinations with the same engine the dashboard already shows, so the
    # simulation tests the recommendation rather than some separate logic.
    if recs is None:
        recs = recommend_for_zone(zone_id, current_hour=current_hour,
                                  data_dir=data_dir, top_n=6)

    placements = []
    remaining = actually_moved

    for opt in recs["options"]:
        if remaining <= 0:
            break

        zid = opt["zone_id"]
        headroom = int(cap[zid] * RECEIVING_CEILING_PCT / 100) - occ_now.get(zid, 0)
        if headroom <= 0:
            continue

        placed = min(remaining, headroom)
        before_pct = occ_now.get(zid, 0) / cap[zid] * 100
        after_occ = occ_now.get(zid, 0) + placed
        after_pct = after_occ / cap[zid] * 100

        placements.append({
            "zone_id": zid,
            "zone_name": names[zid],
            "received": placed,
            "occupancy_before_pct": round(before_pct, 1),
            "occupancy_after_pct": round(after_pct, 1),
            "spare_after": max(0, cap[zid] - after_occ),
            "distance_km": opt["distance_km"],
        })
        remaining -= placed

    # Anyone we could not place stays where they were.
    unplaced = remaining
    relocated = actually_moved - unplaced

    source_after = source_before - relocated
    before_pct = source_before / source_cap * 100
    after_pct = source_after / source_cap * 100

    fc = forecasts.get(zone_id)
    hours_before = fc["hours_to_saturation"] if fc else None

    # Re-derive time-to-saturation at the new level using the same trend.
    trend = fc["trend_pct_per_hour"] if fc else 0.0
    if after_pct >= CRITICAL_PCT:
        hours_after = 0.0
    elif trend > 0.01:
        hours_after = round((CRITICAL_PCT - after_pct) / trend, 1)
    else:
        hours_after = None

    def band(pct):
        if pct >= CRITICAL_PCT:
            return "critical"
        if pct >= WARNING_PCT:
            return "warning"
        return "ok"

    return {
        "zone_id": zone_id,
        "zone_name": names[zone_id],
        "hour": current_hour,
        "requested_move": requested,
        "compliance": compliance,
        "relocated": relocated,
        "stayed_put": movable - relocated,
        "unplaceable": unplaced,
        "source": {
            "occupancy_before": source_before,
            "occupancy_after": source_after,
            "occupancy_before_pct": round(before_pct, 1),
            "occupancy_after_pct": round(after_pct, 1),
            "hours_to_saturation_before": hours_before,
            "hours_to_saturation_after": hours_after,
            "status_before": band(before_pct),
            "status_after": band(after_pct),
        },
        "placements": placements,
        "alert_clears": band(before_pct) != "ok" and band(after_pct) == "ok",
    }


if __name__ == "__main__":
    from engine.risk import detect_all

    HOUR = 110
    alerts = [a for a in detect_all(current_hour=HOUR) if a["metric"] == "accommodation"]

    if not alerts:
        print(f"No accommodation alerts at hour {HOUR}.")
    else:
        target = alerts[0]["zone_id"]

        for n in (300, 800, 1500):
            r = simulate_redistribution(target, n, current_hour=HOUR)
            src = r["source"]

            print(f"\nMove {n} from {r['zone_name']}")
            print(f"  {r['relocated']} relocated, {r['stayed_put']} ignored the guidance, "
                  f"{r['unplaceable']} had nowhere to go")
            print(f"  {r['zone_name']}: {src['occupancy_before_pct']}% "
                  f"-> {src['occupancy_after_pct']}%  "
                  f"[{src['status_before']} -> {src['status_after']}]")

            for p in r["placements"]:
                print(f"    {p['zone_name']:<20} +{p['received']:>4}  "
                      f"{p['occupancy_before_pct']:>5.1f}% -> {p['occupancy_after_pct']:>5.1f}%")

            if r["alert_clears"]:
                print("  Alert clears.")
