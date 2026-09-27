"""
Phase 22 — What-if simulator: named disruption scenarios.

simulate.py already answers "move N people out of a zone, what happens" by
running the existing forecast/recommend engines against modified inputs.
This extends the same idea to three scenarios worth war-gaming before they
happen for real, each implemented the same way: copy the real dataset into a
scratch directory, apply exactly the one change the scenario describes, and
re-run the unmodified engines (venues.analyse_all, routing.find_routes,
risk.detect_all) against it. Nothing about venues.py, routing.py or risk.py
changes for this — the scenario is entirely in which numbers they are handed.

Run:  python -m engine.scenarios
"""

import contextlib
import os
import shutil
import tempfile

import pandas as pd

from engine.forecast import DATA_DIR
from engine.risk import detect_all
from engine.routing import find_routes
from engine.venues import analyse_all


@contextlib.contextmanager
def _scenario_copy(data_dir):
    """A throwaway copy of the dataset to mutate for one scenario run."""
    tmp = tempfile.mkdtemp(prefix="eventpulse_scenario_")
    try:
        shutil.copytree(data_dir, tmp, dirs_exist_ok=True)
        yield tmp
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _venue_for_event(event_id, hour, data_dir):
    return next((v for v in analyse_all(current_hour=hour, data_dir=data_dir)
                 if v["event_id"] == event_id), None)


def simulate_event_cancellation(event_id, current_hour, data_dir=None):
    """
    Cancel an event at `current_hour`.

    If ingress has not started yet, this is simple: nobody arrives. If
    people are already inside, whoever the ingress curve says has been
    admitted by this hour has to be evacuated now, and nobody still outside
    gets in. Modelled by truncating the event to end at this hour with only
    the already-admitted crowd — the existing egress model then plays out
    exactly as it would for a real end-of-event surge, just starting
    immediately instead of on schedule, with no separate "evacuation" logic
    needed.
    """
    data_dir = data_dir or DATA_DIR
    events = pd.read_csv(os.path.join(data_dir, "events.csv"))
    if event_id not in set(events["event_id"]):
        return {"error": f"No event {event_id}"}

    before = _venue_for_event(event_id, current_hour, data_dir)
    before_alerts = detect_all(current_hour=current_hour, data_dir=data_dir)

    start = before["start_hour"]
    already_in = sum(
        t["admitted"] for t in before["ingress"]["timeline"]
        if t["hour_offset"] <= current_hour
    )

    with _scenario_copy(data_dir) as tmp:
        events2 = events.copy()
        mask = events2["event_id"] == event_id

        if current_hour < start - 4:
            # Doors have not opened yet — nobody has arrived, so cancelling
            # is a clean zero-out with nothing to evacuate.
            events2.loc[mask, "expected_attendance"] = 0
        else:
            events2.loc[mask, "expected_attendance"] = already_in
            events2.loc[mask, "end_hour"] = current_hour
        events2.to_csv(os.path.join(tmp, "events.csv"), index=False)

        after = _venue_for_event(event_id, current_hour, tmp)
        after_alerts = detect_all(current_hour=current_hour, data_dir=tmp)

    return {
        "scenario": "event_cancellation",
        "event_id": event_id,
        "event_name": before["name"],
        "zone_id": before["zone_id"],
        "hour": current_hour,
        "already_admitted": already_in,
        "before": {"admitted": before["admitted"], "egress": before["egress"],
                   "gates": before["gates"]},
        "after": {"admitted": after["admitted"], "egress": after["egress"],
                  "gates": after["gates"]},
        "alerts_before": len(before_alerts),
        "alerts_after": len(after_alerts),
    }


def simulate_demand_surge(event_id, extra_people, current_hour, data_dir=None):
    """
    `extra_people` unexpected arrivals on top of what was already expected.

    Simplification, stated plainly: this scales the whole arrival curve up
    rather than concentrating the extra people into a single hour, since the
    arrival profile is a shared shape across every event rather than a
    per-event input. It still exercises the thing that actually matters —
    over-subscription and peak queue at higher demand — just without
    pinpointing which single hour the extra people show up in.
    """
    data_dir = data_dir or DATA_DIR
    events = pd.read_csv(os.path.join(data_dir, "events.csv"))
    if event_id not in set(events["event_id"]):
        return {"error": f"No event {event_id}"}

    before = _venue_for_event(event_id, current_hour, data_dir)
    before_alerts = detect_all(current_hour=current_hour, data_dir=data_dir)

    with _scenario_copy(data_dir) as tmp:
        events2 = events.copy()
        mask = events2["event_id"] == event_id
        events2.loc[mask, "expected_attendance"] += int(extra_people)
        events2.to_csv(os.path.join(tmp, "events.csv"), index=False)

        after = _venue_for_event(event_id, current_hour, tmp)
        after_alerts = detect_all(current_hour=current_hour, data_dir=tmp)

    return {
        "scenario": "demand_surge",
        "event_id": event_id,
        "event_name": before["name"],
        "zone_id": before["zone_id"],
        "hour": current_hour,
        "extra_people": extra_people,
        "before": {
            "expected_attendance": before["expected_attendance"],
            "turned_away": before["turned_away"],
            "ingress_severity": before["ingress"]["severity"],
            "peak_queue": before["ingress"]["peak_queue"],
        },
        "after": {
            "expected_attendance": after["expected_attendance"],
            "turned_away": after["turned_away"],
            "ingress_severity": after["ingress"]["severity"],
            "peak_queue": after["ingress"]["peak_queue"],
        },
        "alerts_before": len(before_alerts),
        "alerts_after": len(after_alerts),
    }


def simulate_transport_closure(link_id, current_hour, data_dir=None):
    """
    Close one transport link entirely (e.g. a metro station shutting), and
    see whether an alternative route exists and whether any venue's egress
    clearance time gets worse as a result — both already computed by
    routing.py and venues.py, just handed a network with that link missing.
    """
    data_dir = data_dir or DATA_DIR
    transport = pd.read_csv(os.path.join(data_dir, "transport.csv"))
    row = transport[transport["link_id"] == link_id]
    if row.empty:
        return {"error": f"No link {link_id}"}

    from_zone, to_zone = row.iloc[0]["from_zone"], row.iloc[0]["to_zone"]

    before_route = find_routes(from_zone, to_zone, current_hour, data_dir=data_dir)
    before_venues = {
        v["zone_id"]: v["egress"]
        for v in analyse_all(current_hour=current_hour, data_dir=data_dir)
        if v["zone_id"] in (from_zone, to_zone)
    }

    with _scenario_copy(data_dir) as tmp:
        transport2 = transport[transport["link_id"] != link_id]
        transport2.to_csv(os.path.join(tmp, "transport.csv"), index=False)

        after_route = find_routes(from_zone, to_zone, current_hour, data_dir=tmp)
        after_venues = {
            v["zone_id"]: v["egress"]
            for v in analyse_all(current_hour=current_hour, data_dir=tmp)
            if v["zone_id"] in (from_zone, to_zone)
        }

    return {
        "scenario": "transport_closure",
        "link_id": link_id,
        "from_zone": from_zone,
        "to_zone": to_zone,
        "hour": current_hour,
        "before": {"route": before_route, "venue_egress": before_venues},
        "after": {"route": after_route, "venue_egress": after_venues},
    }


if __name__ == "__main__":
    print("=== Event cancellation: E5 mid-ingress ===")
    r = simulate_event_cancellation("E5", 101)
    print(f"{r['already_admitted']:,} already inside when cancelled")
    print(f"egress before: {r['before']['egress']['hours_to_clear_zone']}h "
          f"[{r['before']['egress']['severity']}]")
    print(f"egress after:  {r['after']['egress']['hours_to_clear_zone']}h "
          f"[{r['after']['egress']['severity']}]")
    print(f"alerts {r['alerts_before']} -> {r['alerts_after']}")

    print("\n=== Demand surge: +10,000 on E5 ===")
    r = simulate_demand_surge("E5", 10000, 100)
    print(f"expected {r['before']['expected_attendance']:,} -> {r['after']['expected_attendance']:,}")
    print(f"turned away {r['before']['turned_away']:,} -> {r['after']['turned_away']:,}")
    print(f"peak queue {r['before']['peak_queue']:,} -> {r['after']['peak_queue']:,}")
    print(f"ingress severity {r['before']['ingress_severity']} -> {r['after']['ingress_severity']}")

    print("\n=== Transport closure: busiest link at hour 106 ===")
    load = pd.read_csv(os.path.join(DATA_DIR, "transport_load.csv"))
    snap = load[load["hour_offset"] == 106].sort_values("load_pct", ascending=False)
    worst_link = snap.iloc[0]["link_id"]
    r = simulate_transport_closure(worst_link, 106)
    print(f"closing {worst_link}: {r['from_zone']} <-> {r['to_zone']}")
    print(f"verdict before removal: {r['before']['route']['verdict']}")
    print(f"verdict after removal:  {r['after']['route']['verdict']}")
