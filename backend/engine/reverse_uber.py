"""
Phase 25 — Reverse-Uber: system-optimal egress allocation.

exit_redirect.py (Phase 13) is user-optimal: it tells one person the best
gate for them, right now. At crowd scale that is exactly the problem —
everyone gets told the same "best" gate, and it stops being the best gate
the moment they all arrive. Real behaviour makes this worse before any
recommendation even runs: engine/venues.py's `default_exit_share` already
models people gravitating to the gate they entered through regardless of its
capacity, which is why one gate overloads while others sit under capacity
(visible in engine/density.py's floor-zone numbers).

This module asks a different question: not "what's best for this person"
but "how should the whole crowd split across gates so the venue clears
fastest overall". The answer is the textbook fix for exactly this kind of
imbalance — balance flow proportionally to capacity rather than by habit —
and it is worth building because engine/venues.py's aggregate
`hours_to_clear_venue` figure already *assumes* that balance is achieved.
This module quantifies the gap between that assumption and what naive,
habit-driven behaviour actually produces, and says how many people would
need to be actively redirected off the over-drawn gate to close it.

Run:  python -m engine.reverse_uber
"""

from engine.forecast import DATA_DIR
from engine.venues import EGRESS_CLEARANCE_TARGET_HOURS, current_event_for_zone


def optimal_exit_distribution(venue_zone_id, hour, data_dir=None):
    """
    For a venue currently in its egress surge, compare the naive per-gate
    load (driven by where people habitually head, from engine/venues.py's
    gate breakdown) against a capacity-proportional balanced allocation, and
    say how many people need to be redirected off the over-drawn gate(s).
    """
    data_dir = data_dir or DATA_DIR
    venue = current_event_for_zone(venue_zone_id, hour, data_dir)
    if venue is None or not venue["gates"]:
        return {"venue_zone_id": venue_zone_id, "hour": hour, "in_egress": False, "gates": []}

    gates = [g for g in venue["gates"] if g["phase"] == "egress"]
    if not gates:
        return {"venue_zone_id": venue_zone_id, "hour": hour, "in_egress": False, "gates": []}

    total_leaving = venue["admitted"]
    total_capacity = sum(g["exit_capacity_per_hour"] for g in gates)
    if total_capacity <= 0:
        return {"venue_zone_id": venue_zone_id, "hour": hour, "in_egress": True, "gates": []}

    # The load every gate would sit at if the crowd split exactly by
    # capacity share instead of by habit — the system optimum for this
    # simple case, and the number engine/venues.py's aggregate egress
    # clearance time already assumes is happening everywhere.
    balanced_ratio = round((total_leaving / total_capacity) / EGRESS_CLEARANCE_TARGET_HOURS, 2)

    out = []
    for g in gates:
        naive_leaving = g.get("demand_people", g.get("estimated_leaving", 0))     # where people head by habit
        balanced_leaving = total_leaving * (g["exit_capacity_per_hour"] / total_capacity)

        out.append({
            "gate_id": g["gate_id"],
            "name": g["name"],
            "exit_capacity_per_hour": g["exit_capacity_per_hour"],
            "naive_leaving": round(naive_leaving),
            "naive_load_ratio": g.get("demand_ratio", g["load_ratio"]),
            "balanced_leaving": round(balanced_leaving),
            "balanced_load_ratio": balanced_ratio,
            "redirect_away": round(max(0, naive_leaving - balanced_leaving)),
            "redirect_toward": round(max(0, balanced_leaving - naive_leaving)),
        })

    naive_bottleneck_gate = max(out, key=lambda g: g["naive_load_ratio"])

    return {
        "venue_zone_id": venue_zone_id,
        "hour": hour,
        "in_egress": True,
        "total_leaving": total_leaving,
        "naive_bottleneck_gate": naive_bottleneck_gate["name"],
        "naive_bottleneck_load_ratio": naive_bottleneck_gate["naive_load_ratio"],
        "balanced_load_ratio": balanced_ratio,
        "gates": out,
    }


if __name__ == "__main__":
    for h in (105, 106, 107, 108, 110):
        r = optimal_exit_distribution("Z01", h)
        if not r["in_egress"]:
            print(f"hour {h}: no egress under way")
            continue

        print(f"\n--- hour {h} ---")
        print(f"naive bottleneck: {r['naive_bottleneck_gate']} at {r['naive_bottleneck_load_ratio']}x")
        print(f"balanced target:  every gate at {r['balanced_load_ratio']}x")
        for g in r["gates"]:
            if g["redirect_away"]:
                tag = f"shed {g['redirect_away']:,}"
            elif g["redirect_toward"]:
                tag = f"absorb {g['redirect_toward']:,}"
            else:
                tag = "on target"
            print(f"  {g['name']:<20} naive {g['naive_leaving']:>6,} ({g['naive_load_ratio']}x)  "
                  f"balanced {g['balanced_leaving']:>6,} ({g['balanced_load_ratio']}x)  [{tag}]")
