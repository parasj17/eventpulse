"""
Phase 8 — Alternative routing.

The dashboard could already tell an operator that a metro link was at 96%. It
could not tell anyone what to do about it, which is the gap the problem
statement names when it asks for alternative transportation routes.

This treats the transport network as a graph and answers a narrower, more
useful question: given that someone wants to get from A to B at this hour, and
the obvious way is jammed, what else works?

Two things make the answer non-obvious, and both are modelled here:

* **A congested link is slower, not just fuller.** Dwell times stretch, people
  wait for the next service. A link at capacity runs roughly half again as
  long as a quiet one, so a longer route on paper can be quicker in practice.

* **Changing is expensive.** A two-leg route pays an interchange penalty, so it
  only wins when the direct route is genuinely bad. Without that the model
  would recommend detours to save a percentage point of load.

Paths are limited to one interchange. On a ten-zone network that covers
everything a visitor would actually accept, and it keeps the output short
enough for an operator to read at a glance.
"""

import math
import os

import pandas as pd

from engine.forecast import DATA_DIR

# Average speed by mode, km/h, door to door on an uncongested network.
MODE_SPEED_KMH = {
    "metro": 45.0,
    "shuttle": 30.0,
    "bus": 22.0,
}
DEFAULT_SPEED_KMH = 28.0

# Fixed cost of changing service: walking between platforms, waiting for the
# next departure.
INTERCHANGE_MINUTES = 8.0

# Waiting to board at all, before any travel. Rises steeply once a link is
# beyond comfortable loading.
BOARDING_MINUTES = 4.0

# Load above which a link starts slowing down, and how much slower it gets at
# full capacity.
CONGESTION_FREE_FLOW = 0.70
CONGESTION_MAX_PENALTY = 0.55

# A link at or above this is treated as one to route around.
CONGESTED_PCT = 85.0

# An alternative has to beat the direct route's bottleneck by this much to be
# worth offering. Below it, the advice is noise.
MIN_RELIEF_PCT = 10.0


def haversine_km(lat1, lng1, lat2, lng2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _congestion_multiplier(load_pct):
    """
    How much longer a leg takes at a given load.

    Flat until the link is busy, then rising: the first two thirds of capacity
    cost nothing, and the last third is where journey times stretch.
    """
    frac = max(0.0, min(load_pct, 120.0) / 100.0)
    if frac <= CONGESTION_FREE_FLOW:
        return 1.0
    over = (frac - CONGESTION_FREE_FLOW) / (1.0 - CONGESTION_FREE_FLOW)
    return 1.0 + min(1.0, over) * CONGESTION_MAX_PENALTY


def build_graph(hour, data_dir=None):
    """
    The network at one hour: every link with its load, distance and the time
    it takes to ride given that load.
    """
    data_dir = data_dir or DATA_DIR

    zones = pd.read_csv(os.path.join(data_dir, "zones.csv"))
    transport = pd.read_csv(os.path.join(data_dir, "transport.csv"))

    load_path = os.path.join(data_dir, "transport_load.csv")
    if os.path.exists(load_path):
        load = pd.read_csv(load_path)
        snap = load[load["hour_offset"] == hour]
        load_by_link = dict(zip(snap["link_id"], snap["load_pct"]))
        used_by_link = dict(zip(snap["link_id"], snap["current_load"]))
    else:
        load_by_link, used_by_link = {}, {}

    pos = {r["zone_id"]: (r["lat"], r["lng"]) for _, r in zones.iterrows()}
    names = dict(zip(zones["zone_id"], zones["name"]))

    edges = {}
    for _, l in transport.iterrows():
        a, b = l["from_zone"], l["to_zone"]
        if a not in pos or b not in pos:
            continue

        km = haversine_km(*pos[a], *pos[b])
        speed = MODE_SPEED_KMH.get(l["mode"], DEFAULT_SPEED_KMH)
        load_pct = float(load_by_link.get(l["link_id"], 0.0))

        ride = (km / speed) * 60.0 * _congestion_multiplier(load_pct)
        minutes = round(ride + BOARDING_MINUTES, 1)

        edge = {
            "link_id": l["link_id"],
            "mode": l["mode"],
            "distance_km": round(km, 2),
            "load_pct": round(load_pct, 1),
            "capacity_per_hour": int(l["capacity_per_hour"]),
            "current_load": int(used_by_link.get(l["link_id"], 0)),
            "spare_per_hour": max(0, int(l["capacity_per_hour"]) - int(used_by_link.get(l["link_id"], 0))),
            "minutes": minutes,
        }
        # Links are two-way.
        edges[(a, b)] = edge
        edges[(b, a)] = dict(edge)

    return {"edges": edges, "names": names, "zones": list(pos.keys())}


def _describe(graph, legs):
    """Turn a list of legs into a route record."""
    total_minutes = sum(l["minutes"] for l in legs)
    if len(legs) > 1:
        total_minutes += INTERCHANGE_MINUTES * (len(legs) - 1)

    # A route is only as good as its worst leg: that is where people stack up.
    bottleneck = max(legs, key=lambda l: l["load_pct"])

    names = graph["names"]
    if len(legs) == 1:
        via, label = None, "Direct"
    else:
        via = legs[0]["to_zone"]
        label = f"via {names.get(via, via)}"

    return {
        "legs": [
            {
                "from_zone": l["from_zone"],
                "to_zone": l["to_zone"],
                "from_name": names.get(l["from_zone"], l["from_zone"]),
                "to_name": names.get(l["to_zone"], l["to_zone"]),
                "mode": l["mode"],
                "load_pct": l["load_pct"],
                "spare_per_hour": l["spare_per_hour"],
                "minutes": l["minutes"],
            }
            for l in legs
        ],
        "via_zone_id": via,
        "label": label,
        "interchanges": len(legs) - 1,
        "total_minutes": round(total_minutes),
        "bottleneck_pct": bottleneck["load_pct"],
        "bottleneck_link": bottleneck["link_id"],
        # Throughput of the whole route is set by its tightest leg.
        "spare_per_hour": min(l["spare_per_hour"] for l in legs),
    }


def find_routes(from_zone, to_zone, hour, data_dir=None, max_results=4):
    """
    Ways to get from one zone to another at this hour, least congested first.

    Returns the direct route (if one exists) plus any one-interchange route
    that meaningfully beats it, so the operator sees both what people will do
    by default and what they should be told to do instead.
    """
    graph = build_graph(hour, data_dir=data_dir)
    edges, names = graph["edges"], graph["names"]

    if from_zone == to_zone:
        return {"from_zone": from_zone, "to_zone": to_zone, "hour": hour, "routes": []}

    def leg(a, b):
        e = edges.get((a, b))
        if not e:
            return None
        return {**e, "from_zone": a, "to_zone": b}

    candidates = []

    direct = leg(from_zone, to_zone)
    if direct:
        candidates.append(_describe(graph, [direct]))

    # One interchange, through any other zone.
    for mid in graph["zones"]:
        if mid in (from_zone, to_zone):
            continue
        first, second = leg(from_zone, mid), leg(mid, to_zone)
        if first and second:
            candidates.append(_describe(graph, [first, second]))

    if not candidates:
        return {"from_zone": from_zone, "to_zone": to_zone, "hour": hour, "routes": []}

    # Least congested first, then quickest. Congestion leads because the point
    # of the exercise is relieving a jam, not shaving minutes.
    candidates.sort(key=lambda r: (r["bottleneck_pct"], r["total_minutes"]))

    direct_route = next((c for c in candidates if c["interchanges"] == 0), None)
    baseline = direct_route["bottleneck_pct"] if direct_route else None

    routes = []
    for c in candidates:
        if c["interchanges"] == 0:
            c["is_direct"] = True
            c["relief_pct"] = 0.0
            c["recommended"] = False
            routes.append(c)
            continue

        c["is_direct"] = False
        if baseline is None:
            c["relief_pct"] = None
            c["recommended"] = False
        else:
            relief = round(baseline - c["bottleneck_pct"], 1)
            c["relief_pct"] = relief
            c["extra_minutes"] = c["total_minutes"] - direct_route["total_minutes"]
            # Worth saying out loud only if it takes real load off the jam.
            c["recommended"] = relief >= MIN_RELIEF_PCT
        routes.append(c)

    # Keep the direct route regardless, plus the best alternatives.
    alts = [r for r in routes if not r["is_direct"]][: max_results - (1 if direct_route else 0)]
    ordered = ([direct_route] if direct_route else []) + alts

    best = next((r for r in alts if r["recommended"]), None)
    # With no direct route at all (the link that provided it is gone), there
    # is no baseline to compare an alternative against, so nothing gets
    # marked "recommended" above — but the best remaining alternative is
    # still the answer, not "no relief". Candidates are already sorted
    # least-congested-first, so it is simply the first one left.
    if direct_route is None and alts:
        best = alts[0]
    congested = bool(direct_route and direct_route["bottleneck_pct"] >= CONGESTED_PCT)

    # Say plainly what the answer is. "No alternative" is a real finding during
    # an egress surge, when every link out of a zone is saturated at once: the
    # problem is capacity, not routing, and the operator needs to hear that
    # rather than scan four equally jammed options.
    if direct_route is None:
        if best:
            verdict = "reroute"
            summary = (
                f"No direct route; go {best['label']} instead — "
                f"{best['total_minutes']} min, busiest leg at {best['bottleneck_pct']:.0f}%."
            )
        else:
            verdict = "no_relief"
            summary = "No route at all between these zones right now."
    elif not congested:
        verdict = "clear"
        summary = f"Direct route is running at {direct_route['bottleneck_pct']:.0f}%. No diversion needed."
    elif best:
        verdict = "reroute"
        summary = (
            f"Send traffic {best['label']}: busiest leg drops from "
            f"{direct_route['bottleneck_pct']:.0f}% to {best['bottleneck_pct']:.0f}% "
            f"for {best['extra_minutes']} minutes more."
        )
    else:
        verdict = "no_relief"
        summary = (
            "Every route out of this corridor is saturated. Rerouting will not "
            "help; stagger departures or add capacity."
        )

    return {
        "from_zone": from_zone,
        "to_zone": to_zone,
        "from_name": names.get(from_zone, from_zone),
        "to_name": names.get(to_zone, to_zone),
        "hour": hour,
        "direct_congested": congested,
        "verdict": verdict,
        "summary": summary,
        "best_alternative": best,
        "routes": ordered,
    }


def reroute_for_link(link_id, hour, data_dir=None):
    """
    Alternatives for a specific congested link, which is how the dashboard
    reaches this: an operator clicks a transport alert and wants the answer
    for that corridor.
    """
    data_dir = data_dir or DATA_DIR
    transport = pd.read_csv(os.path.join(data_dir, "transport.csv"))

    row = transport[transport["link_id"] == link_id]
    if row.empty:
        return None

    l = row.iloc[0]
    return find_routes(l["from_zone"], l["to_zone"], hour, data_dir=data_dir)


if __name__ == "__main__":
    import pandas as _pd

    HOUR = 102
    load = _pd.read_csv(os.path.join(DATA_DIR, "transport_load.csv"))
    snap = load[load["hour_offset"] == HOUR].sort_values("load_pct", ascending=False)

    worst = snap.iloc[0]["link_id"]
    out = reroute_for_link(worst, HOUR)

    print(f"Hour {HOUR}: {out['from_name']} to {out['to_name']}")
    print(f"Direct route congested: {out['direct_congested']}\n")

    for r in out["routes"]:
        tag = "RECOMMEND" if r.get("recommended") else ("direct" if r["is_direct"] else "")
        print(f"  {r['label']:<26} {r['total_minutes']:>3} min   "
              f"busiest leg {r['bottleneck_pct']:>5.1f}%   "
              f"{r['spare_per_hour']:>5,}/hr spare   {tag}")
        for lg in r["legs"]:
            print(f"      {lg['from_name']} to {lg['to_name']} "
                  f"by {lg['mode']}, {lg['load_pct']}%")
