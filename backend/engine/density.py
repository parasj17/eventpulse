"""
Phase 12 — Crowd density per floor zone.

Built directly on the per-gate live load already computed in engine/venues.py
rather than a separate synthetic device-ping stream: queue size during
ingress and exit demand during egress are exactly what a real wifi/BLE
localization feed would need to reconstruct anyway, so this module is written
against that same shape (headcount per named place, right now) and can be
swapped for a real ping-aggregation pipeline later without any caller of
`zone_density_for_venue` changing.

Run:  python -m engine.density
"""

import os

import pandas as pd

from engine.forecast import DATA_DIR
from engine.venues import EGRESS_CLEARANCE_TARGET_HOURS, EGRESS_SURGE_WINDOW_HOURS, current_event_for_zone

# Same scheme as the gates (venues.py): busy from 75% of capacity, critical from 95%.
# Nothing is shown past 100%: at 95% a gate closes and its overflow is redirected, so
# an area's count is capped at its capacity.
DENSITY_WARNING = 0.75
DENSITY_CRITICAL = 0.95
# How long someone spends on a concourse while leaving: its crowd is the flow through
# its gate over this time, not everyone who will leave by it.
EGRESS_DWELL_HOURS = 0.75


def _severity(pct):
    ratio = pct / 100
    if ratio >= DENSITY_CRITICAL:
        return "critical"
    if ratio >= DENSITY_WARNING:
        return "warning"
    return "ok"


def _bowl_occupancy(venue, hour):
    """
    How many people are in the seating bowl/hall right now: however many
    have been let in by this hour while ingress is still running, the full
    admitted total once seated, and a linear drain to zero over the egress
    surge window once the event ends.

    The drain is a simplification of the real curve — egress rarely empties
    perfectly evenly — but it is directional and matches the one number
    (`admitted`) the rest of the platform already treats as ground truth.
    """
    end = venue["end_hour"]
    admitted_total = venue["admitted"]

    if hour <= end:
        so_far = sum(t["admitted"] for t in venue["ingress"]["timeline"]
                     if t["hour_offset"] <= hour)
        return min(admitted_total, so_far)

    hours_since_end = hour - end
    if hours_since_end >= EGRESS_SURGE_WINDOW_HOURS:
        return 0
    return admitted_total * (1 - hours_since_end / EGRESS_SURGE_WINDOW_HOURS)


def _concourse_occupancy(gate):
    """
    A concourse's headcount mirrors whatever is driving its gate: the queue
    backed up behind it during ingress, or the demand trying to leave
    through it during egress. Both numbers already exist on the gate's live
    breakdown — this just reads them under their floor-zone name.
    """
    if gate["phase"] == "ingress":
        return gate.get("queue", 0)
    if gate["phase"] == "egress":
        per_hour = gate.get("estimated_leaving", 0) / EGRESS_CLEARANCE_TARGET_HOURS
        return per_hour * EGRESS_DWELL_HOURS
    return 0


def zone_density_for_venue(venue_zone_id, hour, data_dir=None):
    """
    Live occupancy for every floor zone (bowl + concourses) inside one venue
    at `hour`: headcount, capacity, percentage and severity — the data a 2D
    or 3D floor heatmap renders directly.
    """
    data_dir = data_dir or DATA_DIR
    floor_path = os.path.join(data_dir, "floor_zones.csv")
    if not os.path.exists(floor_path):
        return []

    floor_zones = pd.read_csv(floor_path)
    floor_zones = floor_zones[floor_zones["venue_zone_id"] == venue_zone_id]
    if floor_zones.empty:
        return []

    venue = current_event_for_zone(venue_zone_id, hour, data_dir)

    gate_by_id = {g["gate_id"]: g for g in (venue["gates"] or [])} if venue else {}

    out = []
    for _, z in floor_zones.iterrows():
        if z["kind"] == "bowl":
            count = _bowl_occupancy(venue, hour) if venue else 0
        else:
            gate = gate_by_id.get(z["adjacent_gate_id"])
            count = _concourse_occupancy(gate) if gate else 0

        capacity = int(z["capacity"])
        count = min(count, capacity)
        pct = round(count / capacity * 100, 1) if capacity else 0.0

        out.append({
            "floor_zone_id": z["floor_zone_id"],
            "name": z["name"],
            "kind": z["kind"],
            "x": float(z["x"]),
            "y": float(z["y"]),
            "capacity": capacity,
            "people_count": round(count),
            "occupancy_pct": pct,
            "severity": _severity(pct),
        })

    return out


if __name__ == "__main__":
    for h in (100, 102, 104, 106, 108, 110):
        print(f"\n--- hour {h} ---")
        for z in zone_density_for_venue("Z01", h):
            print(f"  {z['name']:<22} {z['people_count']:>6}/{z['capacity']:<6} "
                  f"{z['occupancy_pct']:>6.1f}%  [{z['severity']}]")
