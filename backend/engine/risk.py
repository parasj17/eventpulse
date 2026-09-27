"""
Phase 3 — Risk detection and alerting.

A thin layer over the forecast that applies thresholds and emits structured
alerts. Kept separate from forecasting so the thresholds can be tuned without
touching the prediction logic, and so the dashboard has a simple list to render.
"""

from engine.forecast import SATURATION_PCT, forecast_all, load_data

# Occupancy thresholds, in percent. Redirection starts at 75: waiting until a
# zone is nearly full leaves no time to move anyone.
WARNING_PCT = 75.0
CRITICAL_PCT = SATURATION_PCT

# A zone also warrants attention if it will breach soon, even at a lower
# current occupancy — this is the "early warning" the problem statement asks for.
IMMINENT_HOURS = 4.0

# Transport link load thresholds.
TRANSPORT_WARNING_PCT = 80.0

# Displayed occupancy rounds to whole numbers, so anything from here up reads as
# 100% on screen and should be treated as full.
FULL_PCT = 99.5
TRANSPORT_CRITICAL_PCT = 92.0


def _severity(current_pct, hours_to_sat):
    if current_pct >= CRITICAL_PCT:
        return "critical"
    if hours_to_sat is not None and hours_to_sat <= IMMINENT_HOURS:
        return "critical"
    if current_pct >= WARNING_PCT:
        return "warning"
    if hours_to_sat is not None and hours_to_sat <= IMMINENT_HOURS * 2:
        return "warning"
    return "ok"


def _message(name, current_pct, hours_to_sat, severity, redirect_to=None):
    """
    The alert text. Where a destination is known it is named, so the operator
    reads what to do rather than only what is wrong. Full zones report where
    arrivals are already going; zones still filling get it as a prompt, since
    the whole point of warning at 75% is to act while there is time.
    """
    if current_pct >= FULL_PCT:
        if redirect_to:
            return f"{name} is full, arrivals are being turned away towards {redirect_to}"
        return f"{name} is full, arrivals are being turned away"

    if current_pct >= CRITICAL_PCT:
        if redirect_to:
            return (f"{name} is at capacity ({current_pct:.0f}%), "
                    f"send arrivals to {redirect_to}")
        return f"{name} is at capacity ({current_pct:.0f}%)"

    if hours_to_sat is not None and hours_to_sat <= IMMINENT_HOURS:
        base = f"{name} projected to exceed capacity within {hours_to_sat:.1f} hours"
        return f"{base}, start directing arrivals to {redirect_to}" if redirect_to else base

    if severity == "warning" and hours_to_sat is not None:
        base = f"{name} approaching capacity, saturation in ~{hours_to_sat:.0f} hours"
        return f"{base}, consider directing arrivals to {redirect_to}" if redirect_to else base

    base = f"{name} occupancy elevated at {current_pct:.0f}%"
    return f"{base}, {redirect_to} has room" if redirect_to else base


def detect_accommodation_alerts(forecasts, zones, redirect_targets=None):
    """Alerts driven by hotel occupancy forecasts."""
    names = dict(zip(zones["zone_id"], zones["name"]))
    redirect_targets = redirect_targets or {}
    alerts = []

    for fc in forecasts:
        current = fc["current_occupancy_pct"]
        hours = fc["hours_to_saturation"]
        sev = _severity(current, hours)

        if sev == "ok":
            continue

        zid = fc["zone_id"]
        alerts.append({
            "zone_id": zid,
            "zone_name": names.get(zid, zid),
            "title": names.get(zid, zid),
            "severity": sev,
            "metric": "accommodation",
            "message": _message(names.get(zid, zid), current, hours, sev,
                                redirect_targets.get(zid)),
            "current_pct": current,
            "predicted_breach_in_hours": hours,
            "trend_pct_per_hour": fc["trend_pct_per_hour"],
        })

    return alerts


def detect_transport_alerts(transport, zones, current_hour=None, load_df=None):
    """Alerts driven by transport link load."""
    names = dict(zip(zones["zone_id"], zones["name"]))
    alerts = []

    # Load varies hour to hour, so join the snapshot for the hour being viewed.
    if load_df is not None and current_hour is not None:
        snap = load_df[load_df["hour_offset"] == current_hour]
        load_by_link = dict(zip(snap["link_id"], snap["load_pct"]))
        used_by_link = dict(zip(snap["link_id"], snap["current_load"]))
    else:
        load_by_link, used_by_link = {}, {}

    for _, link in transport.iterrows():
        load = float(load_by_link.get(link["link_id"], 0.0))
        carrying = int(used_by_link.get(link["link_id"], 0))

        if load >= TRANSPORT_CRITICAL_PCT:
            sev = "critical"
        elif load >= TRANSPORT_WARNING_PCT:
            sev = "warning"
        else:
            continue

        a, b = link["from_zone"], link["to_zone"]
        route = f"{names.get(a, a)} \u2192 {names.get(b, b)}"

        alerts.append({
            "zone_id": a,
            "zone_name": names.get(a, a),
            "title": route,
            "severity": sev,
            "metric": "transport",
            "message": f"{link['mode'].title()} link carrying "
                       f"{carrying:,} of {int(link['capacity_per_hour']):,} per hour",
            "current_pct": load,
            "predicted_breach_in_hours": None,
            "trend_pct_per_hour": None,
        })

    return alerts


def detect_all(current_hour=12, data_dir=None):
    """
    Run every detector and return a single ranked alert list.

    Returns dicts matching the "alerts" shape in contracts/mock-data.json.
    """
    import os
    import pandas as pd

    from engine.forecast import DATA_DIR

    data_dir = data_dir or DATA_DIR
    zones = pd.read_csv(os.path.join(data_dir, "zones.csv"))
    occupancy = pd.read_csv(os.path.join(data_dir, "occupancy_timeline.csv"))
    transport = pd.read_csv(os.path.join(data_dir, "transport.csv"))

    load_path = os.path.join(data_dir, "transport_load.csv")
    load_df = pd.read_csv(load_path) if os.path.exists(load_path) else None

    forecasts = forecast_all(occupancy, current_hour=current_hour)

    # Work out where each alerting zone's arrivals should go, so the alert can
    # say it rather than leaving the operator to look it up. Done for anything
    # at or above the warning line, not only zones already full: warning early
    # is only useful if it comes with somewhere to send people.
    from engine.recommend import recommend_for_zone

    redirect_targets = {}
    for fc in forecasts:
        pct = fc["current_occupancy_pct"]
        hours = fc["hours_to_saturation"]
        if pct < WARNING_PCT and not (hours is not None and hours <= IMMINENT_HOURS * 2):
            continue

        rec = recommend_for_zone(fc["zone_id"], current_hour=current_hour,
                                 data_dir=data_dir)
        if not rec["options"]:
            continue

        # Take the nearest zone that is meaningfully emptier than this one.
        #
        # Pure proximity can pair two struggling neighbours and advise each to
        # send arrivals to the other. Requiring a real gap breaks that, while
        # still keeping the nearest sensible option rather than pushing people
        # to the far side of the city: a zone at 80% is a perfectly good
        # destination for one that is full.
        RELIEF_MARGIN = 5.0
        better = [
            o for o in rec["options"]
            if o["current_occupancy_pct"] <= pct - RELIEF_MARGIN
        ]
        pick = better[0] if better else rec["options"][0]
        redirect_targets[fc["zone_id"]] = pick["zone_name"]

    # Venue problems are a third category alongside beds and links: a building
    # can be oversubscribed, or its gates too slow, while the zone around it
    # still has room.
    from engine.venues import detect_venue_alerts

    alerts = (
        detect_accommodation_alerts(forecasts, zones, redirect_targets)
        + detect_transport_alerts(transport, zones, current_hour, load_df)
        + detect_venue_alerts(current_hour=current_hour, data_dir=data_dir)
    )

    # Critical first, then by how soon the breach lands, then by severity of load.
    order = {"critical": 0, "warning": 1}
    alerts.sort(key=lambda a: (
        order.get(a["severity"], 2),
        a["predicted_breach_in_hours"] if a["predicted_breach_in_hours"] is not None else 999,
        -a["current_pct"],
    ))

    for i, a in enumerate(alerts, start=1):
        a["id"] = f"A{i:03d}"
        a["raised_at_hour"] = current_hour

    return alerts


if __name__ == "__main__":
    alerts = detect_all(current_hour=12)

    print(f"{len(alerts)} alerts at hour 12\n")
    for a in alerts:
        tag = "CRIT" if a["severity"] == "critical" else "WARN"
        print(f"[{tag}] {a['id']}  {a['message']}")
