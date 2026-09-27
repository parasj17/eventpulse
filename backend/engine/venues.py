"""
Phase 7 — Venue capacity.

Everything else in the platform treats an event as a source of demand. This
treats it as a constrained building, which is what the problem statement means
by a venue capacity limit, and it turns out to be three separate problems that
peak at different times:

* **Over-subscription.** More tickets sold than seats. The shortfall does not
  disappear; it stands outside, and the city has to absorb it.

* **Ingress.** A crowd does not arrive evenly. Most turn up in the two hours
  before the start, and gates can only process so many per hour, so a queue
  builds whenever arrival rate outruns throughput. Peak queue and clearance
  time are what an operator can actually act on.

* **Egress.** The sharper problem, and the one that connects venues to
  transport: everyone leaves within about an hour. Fifty thousand people hit
  the network at once, and whether that clears depends on the links out of the
  zone rather than on anything the venue controls.

The arrival curve is the usual one for ticketed events: a slow build from about
four hours out, rising steeply and peaking shortly before the start, with a
late tail of people who cut it fine.
"""

import os

import pandas as pd

from engine.forecast import DATA_DIR

# Share of the crowd arriving in each hour, keyed by hours before the start.
# Negative keys are latecomers, after kick-off. Deliberately front-loaded onto
# the final hour: that concentration is the whole reason queues form, since
# gates are sized against the average rather than the peak.
ARRIVAL_PROFILE = {
    4: 0.03,
    3: 0.06,
    2: 0.13,
    1: 0.25,
    0: 0.40,
    -1: 0.10,
    -2: 0.03,
}

# A queue longer than this is a crowd-safety problem, not an inconvenience.
QUEUE_CRITICAL = 8000
QUEUE_WARNING = 3000

# Hours to clear the crowd out of the zone after the event before it counts as
# a problem. Beyond this people are stranded rather than travelling.
EGRESS_CLEARANCE_TARGET_HOURS = 2.0

# Share of a zone's transport capacity left for ordinary travellers during an
# egress surge. The network does not empty itself for the stadium.
NON_EVENT_TRAFFIC_SHARE = 0.25

# A gate's *live* status is automatic, not something an organizer clicks —
# it is derived from how loaded that gate actually is right now, as a ratio
# against its own hourly capacity. 1.0 means the gate is handling exactly
# one hour's worth of backlog; past that it is falling behind.
# A gate's load is the share of its capacity it is handling right now, and it never goes
# past 100%. From 75% it is busy (warning, amber). At 95% it closes to new people
# (critical, red) and everyone above that is redirected to gates with room: first to
# gates under 75% (filling them no further than 75%), then to any gate still under 95%.
# If every gate is full, the rest are held back and told to wait.
GATE_WARN_PCT = 75.0
GATE_CLOSE_PCT = 95.0
# Kept for callers that still think in ratios (1.0 = full).
GATE_LOAD_WARNING = GATE_WARN_PCT / 100
GATE_LOAD_CRITICAL = GATE_CLOSE_PCT / 100

# How long after an event ends the egress surge is considered "live" for
# gate status purposes. Beyond this the crowd has dispersed and every gate
# reverts to open.
EGRESS_SURGE_WINDOW_HOURS = EGRESS_CLEARANCE_TARGET_HOURS * 2


def _gate_live_status(load_ratio):
    """open below 75%, restricted (busy) from 75%, closed from 95%."""
    pct = load_ratio * 100
    if pct >= GATE_CLOSE_PCT:
        return "closed"
    if pct >= GATE_WARN_PCT:
        return "restricted"
    return "open"


def rebalance(gates):
    """
    Shift demand off gates at or past 95% and off gates out of service.

    `gates`: [{"id", "name", "demand", "cap", "oos"}] where demand and cap are people over
    the same period. Returns ({id: {"load", "out", "in", "to": {id: people}, "from": {id: people}}}, held),
    with every load at or below 95% of capacity unless nothing could take the excess.
    """
    res = {g["id"]: {"load": 0.0, "out": 0.0, "in": 0.0, "to": {}, "from": {}} for g in gates}
    shed = {}
    for g in gates:
        limit = 0 if g["oos"] else GATE_CLOSE_PCT / 100 * g["cap"]
        out = max(0.0, g["demand"] - limit)
        res[g["id"]]["load"] = g["demand"] - out
        if out > 0:
            shed[g["id"]] = out
    total = sum(shed.values())
    if total <= 0:
        return res, 0.0
    # Gates that shed are closed to new people; everyone else can take some.
    receivers = [g for g in gates if not g["oos"] and g["id"] not in shed]
    left = total
    for ceiling in (GATE_WARN_PCT, GATE_CLOSE_PCT):
        room = {g["id"]: max(0.0, ceiling / 100 * g["cap"] - res[g["id"]]["load"]) for g in receivers}
        space = sum(room.values())
        if space <= 0 or left <= 0:
            continue
        take = min(left, space)
        for rid, r in room.items():
            got = take * r / space
            if got <= 0:
                continue
            res[rid]["load"] += got
            res[rid]["in"] += got
            for sid, amount in shed.items():           # each source contributes by its share of the overflow
                part = got * amount / total
                res[sid]["to"][rid] = res[sid]["to"].get(rid, 0) + part
                res[rid]["from"][sid] = res[rid]["from"].get(sid, 0) + part
        left -= take
    for sid, amount in shed.items():
        res[sid]["out"] = amount * (total - left) / total
    return res, max(0.0, left)


def _gate_breakdown(gates, timeline, gate_total_entry, admitted_total,
                     exit_rate_total, start, end, current_hour):
    """
    Per-gate live state at `current_hour`: which phase the gate is in
    (ingress queue building, egress surge, or idle) and the automatic
    open/restricted/closed status that phase implies.

    A gate an organizer has manually taken out of service (`status` !=
    "open" in gates.csv) always reports closed — that override wins over
    whatever the live load would otherwise say.
    """
    ingress_hours = [t["hour_offset"] for t in timeline]
    ingress_lo, ingress_hi = min(ingress_hours), max(ingress_hours)
    by_hour = {t["hour_offset"]: t for t in timeline}

    rows = []
    for _, g in gates.iterrows():
        configured = g["status"]
        entry_cap = int(g["entry_capacity_per_hour"])
        exit_cap = int(g["exit_capacity_per_hour"])
        exit_share = float(g.get("default_exit_share", exit_cap / exit_rate_total if exit_rate_total else 0))
        row = {"gate_id": g["gate_id"], "name": g["name"], "status": configured,
               "entry_capacity_per_hour": entry_cap, "exit_capacity_per_hour": exit_cap,
               "x": float(g["x"]), "y": float(g["y"]), "phase": "idle", "demand": 0.0, "cap": 1.0,
               "oos": configured != "open"}
        if current_hour < ingress_lo:
            pass
        elif ingress_lo <= current_hour <= ingress_hi:
            # Everyone trying to get through this gate this hour: those admitted plus those
            # still queueing, by the gate's share of entry capacity.
            t = by_hour.get(current_hour, timeline[-1])
            share = entry_cap / gate_total_entry if gate_total_entry else 0
            row.update(phase="ingress", demand=(t["admitted"] + t["queue"]) * share, cap=float(entry_cap),
                       queue_share=t["queue"] * share)
        elif current_hour < end:
            row["phase"] = "running"
        elif current_hour - end <= EGRESS_SURGE_WINDOW_HOURS:
            # Everyone heading for this gate, against what it can let out in the clearance target.
            row.update(phase="egress", demand=admitted_total * exit_share, cap=exit_cap * EGRESS_CLEARANCE_TARGET_HOURS)
        rows.append(row)

    active = [r for r in rows if r["phase"] in ("ingress", "egress")]
    flow, held = rebalance([{"id": r["gate_id"], "name": r["name"], "demand": r["demand"], "cap": r["cap"],
                             "oos": r["oos"]} for r in active]) if active else ({}, 0.0)
    names = {r["gate_id"]: r["name"] for r in rows}

    out = []
    for r in rows:
        f = flow.get(r["gate_id"])
        detail = {}
        if r["phase"] in ("ingress", "egress") and f:
            load = f["load"]
            load_pct = min(100.0, load / r["cap"] * 100) if r["cap"] else 0.0
            demand_pct = r["demand"] / r["cap"] * 100 if r["cap"] else 0.0
            if r["oos"]:
                live = "closed"
            elif f["out"] > 0 or load_pct >= GATE_CLOSE_PCT:
                live = "closed"                          # at 95%: closed to new people, the rest redirected
            else:
                live = _gate_live_status(load_pct / 100)
            detail = {
                "load_pct": round(load_pct, 1), "demand_pct": round(demand_pct, 1),
                "demand_ratio": round(demand_pct / 100, 2),
                "demand_people": round(r["demand"]),
                "redirected_out": round(f["out"]), "redirected_in": round(f["in"]),
                "redirect_to": [{"gate_id": k, "name": names[k], "people": round(v)} for k, v in
                                sorted(f["to"].items(), key=lambda kv: -kv[1]) if v >= 1],
                "redirect_from": [{"gate_id": k, "name": names[k], "people": round(v)} for k, v in
                                  sorted(f["from"].items(), key=lambda kv: -kv[1]) if v >= 1],
            }
            if r["phase"] == "ingress":
                scale = load / r["demand"] if r["demand"] else 0
                detail["queue"] = round(r.get("queue_share", 0) * scale)
            else:
                detail["estimated_leaving"] = round(load)
                detail["hours_to_clear"] = round(load / r["exit_capacity_per_hour"], 1) if r["exit_capacity_per_hour"] else None
            ratio = load_pct / 100
        else:
            live = "closed" if r["oos"] or r["phase"] == "running" else "open"
            ratio = 0.0
            detail = {"load_pct": 0.0, "demand_pct": 0.0, "demand_ratio": 0.0, "redirected_out": 0, "redirected_in": 0,
                      "redirect_to": [], "redirect_from": []}
        out.append({
            "gate_id": r["gate_id"], "name": r["name"], "status": r["status"],
            "entry_capacity_per_hour": r["entry_capacity_per_hour"], "exit_capacity_per_hour": r["exit_capacity_per_hour"],
            "x": r["x"], "y": r["y"],
            "phase": "idle" if r["phase"] == "running" else r["phase"],
            "load_ratio": round(ratio, 2),
            "live_status": live,
            **detail,
        })
    return out, round(held)


def analyse_event(event, transport_out_capacity=None, gates=None, current_hour=None):
    """
    Capacity picture for one event: who gets in, who queues, who is turned
    away, and how long the crowd takes to clear afterwards.

    `gates`, if given, is this venue's rows from gates.csv. Ingress/egress
    throughput then comes from summing capacity across gates whose
    `status` is "open" rather than the event's flat gate/exit fields — a
    gate an organizer has closed stops contributing capacity, which is the
    whole point of being able to close one during an incident.
    """
    expected = int(event["expected_attendance"])
    capacity = int(event["venue_capacity"])
    start = int(event["start_hour"])
    end = int(event["end_hour"])

    if gates is not None and len(gates):
        open_gates = gates[gates["status"] == "open"]
        gate = int(open_gates["entry_capacity_per_hour"].sum())
        exit_rate = int(open_gates["exit_capacity_per_hour"].sum())
    else:
        gate = int(event["gate_throughput_per_hour"])
        exit_rate = int(event["exit_throughput_per_hour"])

    # --- Over-subscription ------------------------------------------------
    admitted_total = min(expected, capacity)
    turned_away = max(0, expected - capacity)

    # --- Ingress queue ----------------------------------------------------
    # Walk the arrival curve hour by hour. Everyone who bought a ticket turns
    # up, including the people who will not get in, so the curve runs against
    # `expected` rather than against seats. Each hour admits whichever is
    # smaller: the gate rate, or the seats still free.
    queue = 0
    admitted = 0
    timeline = []
    peak_queue = 0
    peak_queue_hour = start

    for hours_before in sorted(ARRIVAL_PROFILE.keys(), reverse=True):
        hour = start - hours_before
        arriving = int(expected * ARRIVAL_PROFILE[hours_before])

        waiting = queue + arriving
        through = min(waiting, gate, capacity - admitted)
        queue = waiting - through
        admitted += through

        if queue > peak_queue:
            peak_queue, peak_queue_hour = queue, hour

        timeline.append({
            "hour_offset": hour,
            "arriving": arriving,
            "admitted": through,
            "queue": queue,
        })

    # Once the building is full the queue stops moving, so a clearance time is
    # meaningless — those people are not getting in at any rate.
    queue_clear_hours = (
        None if admitted >= capacity and queue > 0
        else round(queue / gate, 1) if gate else None
    )

    if peak_queue >= QUEUE_CRITICAL:
        ingress_severity = "critical"
    elif peak_queue >= QUEUE_WARNING:
        ingress_severity = "warning"
    else:
        ingress_severity = "ok"

    # --- Egress -----------------------------------------------------------
    # Everyone leaves at once. The venue can push them out at exit_rate, but
    # they then have to actually get out of the zone, which is the transport
    # network's problem.
    egress_hours_at_gate = round(admitted_total / exit_rate, 1) if exit_rate else None

    egress = {
        "leaving": admitted_total,
        "exit_throughput_per_hour": exit_rate,
        "hours_to_clear_venue": egress_hours_at_gate,
        "transport_capacity_per_hour": transport_out_capacity,
        "hours_to_clear_zone": None,
        "severity": "ok",
    }

    if transport_out_capacity and transport_out_capacity > 0:
        # The binding constraint is whichever is slower: the doors or the
        # network outside them. In practice it is always the network.
        effective = min(exit_rate, transport_out_capacity)
        hours = round(admitted_total / effective, 1)
        egress["hours_to_clear_zone"] = hours
        egress["severity"] = (
            "critical" if hours > EGRESS_CLEARANCE_TARGET_HOURS * 2
            else "warning" if hours > EGRESS_CLEARANCE_TARGET_HOURS
            else "ok"
        )

    gate_breakdown = None
    held = 0
    if gates is not None and len(gates):
        effective_hour = start if current_hour is None else current_hour
        gate_breakdown, held = _gate_breakdown(
            gates, timeline, gate, admitted_total, exit_rate,
            start, end, effective_hour,
        )

    return {
        "event_id": event["event_id"],
        "name": event["name"],
        "zone_id": event["zone_id"],
        "venue_name": event["venue_name"],
        "start_hour": start,
        "end_hour": end,
        "expected_attendance": expected,
        "venue_capacity": capacity,
        "utilisation_pct": round(expected / capacity * 100, 1) if capacity else None,
        "admitted": admitted_total,
        "turned_away": turned_away,
        "oversubscribed": turned_away > 0,
        "ingress": {
            "gate_throughput_per_hour": gate,
            "peak_queue": peak_queue,
            "peak_queue_hour": peak_queue_hour,
            "queue_at_start": queue,
            "hours_to_clear_queue": queue_clear_hours,
            "severity": ingress_severity,
            "timeline": timeline,
        },
        "egress": egress,
        "gates": gate_breakdown,
        # People no gate could take once every gate reached 95%: held back, told to wait.
        "gates_held": held,
    }


def current_event_for_zone(zone_id, hour, data_dir=None):
    """
    Whichever of a venue's events is actually relevant at `hour` — ingress
    through the egress surge window — falling back to the first scheduled
    event so a venue between events still returns something (its gates then
    correctly report idle/open rather than any live load). Shared by
    density.py and exit_redirect.py so both agree on which event is
    "current" for the same venue at the same hour.
    """
    venues = [v for v in analyse_all(current_hour=hour, data_dir=data_dir)
              if v["zone_id"] == zone_id]
    if not venues:
        return None
    return next(
        (v for v in venues
         if v["start_hour"] - 4 <= hour <= v["end_hour"] + EGRESS_SURGE_WINDOW_HOURS),
        venues[0],
    )


def _transport_capacity_out(zone_id, transport):
    """
    How many people an hour the network can physically move out of a zone.

    This is nameplate capacity less a share held back for ordinary travellers,
    not the spare capacity showing at that hour. Spare capacity would be
    circular: the egress crowd is itself most of the load on those links at
    that moment, so measuring the headroom left after them and then asking how
    long they take to clear divides the crowd by a number the crowd created.
    """
    links = transport[
        (transport["from_zone"] == zone_id) | (transport["to_zone"] == zone_id)
    ]
    if links.empty:
        return None

    total = int(links["capacity_per_hour"].sum())
    return int(total * (1 - NON_EVENT_TRAFFIC_SHARE))


def analyse_all(current_hour=None, data_dir=None):
    """Capacity picture for every event, with the current one flagged."""
    data_dir = data_dir or DATA_DIR

    events = pd.read_csv(os.path.join(data_dir, "events.csv"))
    transport = pd.read_csv(os.path.join(data_dir, "transport.csv"))

    gates_path = os.path.join(data_dir, "gates.csv")
    gates = pd.read_csv(gates_path) if os.path.exists(gates_path) else None

    out = []
    for _, ev in events.iterrows():
        out_cap = _transport_capacity_out(ev["zone_id"], transport)
        venue_gates = (
            gates[gates["venue_zone_id"] == ev["zone_id"]]
            if gates is not None else None
        )
        row = analyse_event(ev, transport_out_capacity=out_cap, gates=venue_gates,
                             current_hour=current_hour)

        if current_hour is not None:
            if current_hour < row["start_hour"]:
                row["state"] = "upcoming"
                row["hours_away"] = row["start_hour"] - current_hour
            elif current_hour <= row["end_hour"]:
                row["state"], row["hours_away"] = "running", 0
            else:
                row["state"] = "finished"
                row["hours_away"] = current_hour - row["end_hour"]

        out.append(row)

    return out


def detect_venue_alerts(current_hour=12, data_dir=None, lookahead_hours=8):
    """
    Venue problems worth raising now.

    Scoped to events close enough to act on: an over-subscribed concert three
    days out is a planning matter, not a control-room alert.
    """
    alerts = []

    for v in analyse_all(current_hour=current_hour, data_dir=data_dir):
        hours_to_start = v["start_hour"] - current_hour
        hours_to_end = v["end_hour"] - current_hour

        # Ignore anything finished, or too far out to act on.
        if hours_to_end < 0 or hours_to_start > lookahead_hours:
            continue

        name, venue = v["name"], v["venue_name"]

        if v["oversubscribed"]:
            alerts.append({
                "zone_id": v["zone_id"],
                "zone_name": venue,
                "title": venue,
                "severity": "critical",
                "metric": "venue",
                "message": (
                    f"{name} is oversubscribed: {v['expected_attendance']:,} expected "
                    f"against {v['venue_capacity']:,} seats, "
                    f"{v['turned_away']:,} will not get in"
                ),
                "current_pct": v["utilisation_pct"],
                "predicted_breach_in_hours": max(0, hours_to_start),
                "trend_pct_per_hour": None,
            })

        ing = v["ingress"]
        if ing["severity"] != "ok" and hours_to_start >= 0:
            alerts.append({
                "zone_id": v["zone_id"],
                "zone_name": venue,
                "title": f"{venue} gates",
                "severity": ing["severity"],
                "metric": "venue",
                "message": (
                    f"{name} ingress peaks at {ing['peak_queue']:,} queuing, "
                    f"{ing['gate_throughput_per_hour']:,} per hour through the gates"
                ),
                "current_pct": min(100.0, round(
                    ing["peak_queue"] / max(1, v["admitted"]) * 100, 1)),
                "predicted_breach_in_hours": max(0, ing["peak_queue_hour"] - current_hour),
                "trend_pct_per_hour": None,
            })

        eg = v["egress"]
        if eg["severity"] != "ok":
            alerts.append({
                "zone_id": v["zone_id"],
                "zone_name": venue,
                "title": f"{venue} egress",
                "severity": eg["severity"],
                "metric": "venue",
                "message": (
                    f"{name} releases {eg['leaving']:,} people; the network out of "
                    f"this zone clears them in {eg['hours_to_clear_zone']}h"
                ),
                "current_pct": min(100.0, round(
                    eg["leaving"] / max(1, eg["transport_capacity_per_hour"] or 1) * 100, 1)),
                "predicted_breach_in_hours": max(0, hours_to_end),
                "trend_pct_per_hour": None,
            })

    return alerts


if __name__ == "__main__":
    for v in analyse_all(current_hour=100):
        flag = "OVER" if v["oversubscribed"] else "ok"
        print(f"{v['name']:<18} {v['venue_name']:<16} "
              f"{v['expected_attendance']:>6,} / {v['venue_capacity']:>6,} "
              f"({v['utilisation_pct']:>5.1f}%) {flag}")
        ing, eg = v["ingress"], v["egress"]
        print(f"   ingress  peak queue {ing['peak_queue']:>6,} at h{ing['peak_queue_hour']}"
              f"  [{ing['severity']}]")
        print(f"   egress   {eg['leaving']:,} out, network clears in "
              f"{eg['hours_to_clear_zone']}h  [{eg['severity']}]")
        print()

    print("Alerts at hour 100:")
    for a in detect_venue_alerts(current_hour=100):
        print(f"  [{a['severity']:>8}] {a['message']}")
