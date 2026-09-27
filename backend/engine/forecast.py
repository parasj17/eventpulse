"""
Phase 2 — Forecasting engine.

Projects zone occupancy forward and works out how long until each zone hits
saturation.

Deliberately simple and explainable: a weighted linear trend over recent
history, with the recent hours weighted more heavily than older ones. This is
defensible in a live demo in a way a black-box model is not — when a judge asks
"why does it predict that?", you can answer in one sentence.
"""

import os

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# On Vercel the deployed files are read-only: work on a copy in /tmp. Accounts and
# registrations live in Supabase, so only demo state (organiser edits, caches) is per-instance.
if os.environ.get("VERCEL") and not os.access(DATA_DIR, os.W_OK):
    import shutil as _shutil
    _tmp = "/tmp/eventpulse-data"
    if not os.path.isdir(_tmp):
        _shutil.copytree(DATA_DIR, _tmp)
    DATA_DIR = _tmp

# Occupancy percentage at which a zone counts as saturated. Everything that
# needs this line reads it from here: the countdown, the alert bands and the
# dashboard colours all have to agree, and they drifted apart when each kept
# its own copy.
SATURATION_PCT = 90.0

# How many hours of history to fit the trend on.
TREND_WINDOW = 6

# How many hours forward to project.
FORECAST_HORIZON = 12


def load_data(data_dir=DATA_DIR):
    zones = pd.read_csv(os.path.join(data_dir, "zones.csv"))
    occupancy = pd.read_csv(os.path.join(data_dir, "occupancy_timeline.csv"))
    return zones, occupancy


def weighted_trend(series):
    """
    Percentage-points-per-hour change, weighting recent observations higher.

    Uses linearly increasing weights over the window so a zone that has just
    started filling fast is not masked by a flat earlier period.
    """
    values = list(series)
    if len(values) < 2:
        return 0.0

    deltas = [values[i] - values[i - 1] for i in range(1, len(values))]
    weights = list(range(1, len(deltas) + 1))          # oldest .. newest
    total_weight = sum(weights)

    return sum(d * w for d, w in zip(deltas, weights)) / total_weight


def forecast_zone(zone_history, current_hour, horizon=FORECAST_HORIZON):
    """
    Project one zone forward from current_hour.

    Returns a dict matching the "forecast" shape in contracts/mock-data.json.
    """
    history = zone_history[zone_history["hour_offset"] <= current_hour]
    window = history.tail(TREND_WINDOW)

    if window.empty:
        return None

    current_pct = float(window["occupancy_pct"].iloc[-1])
    trend = weighted_trend(window["occupancy_pct"])

    timeline = []
    for step in range(horizon + 1):
        projected = current_pct + (trend * step)
        timeline.append({
            "hour_offset": current_hour + step,
            "predicted_occupancy_pct": round(max(0.0, projected), 2),
        })

    # Hours until saturation, given the current trend.
    if current_pct >= SATURATION_PCT:
        hours_to_sat = 0.0
    elif trend <= 0.01:
        hours_to_sat = None                            # not trending toward saturation
    else:
        hours_to_sat = round((SATURATION_PCT - current_pct) / trend, 1)
        if hours_to_sat > horizon * 3:
            hours_to_sat = None                        # too far out to be meaningful

    return {
        "zone_id": zone_history["zone_id"].iloc[0],
        "timeline": timeline,
        "hours_to_saturation": hours_to_sat,
        "trend_pct_per_hour": round(trend, 3),
        "current_occupancy_pct": round(current_pct, 2),
    }


def forecast_all(occupancy=None, current_hour=12, horizon=FORECAST_HORIZON):
    """Forecast every zone. Returns a list of forecast dicts."""
    if occupancy is None:
        _, occupancy = load_data()

    results = []
    for zone_id, group in occupancy.groupby("zone_id"):
        group = group.sort_values("hour_offset")
        fc = forecast_zone(group, current_hour, horizon)
        if fc:
            results.append(fc)

    return sorted(results, key=lambda f: f["current_occupancy_pct"], reverse=True)


if __name__ == "__main__":
    zones, occupancy = load_data()
    names = dict(zip(zones["zone_id"], zones["name"]))

    print(f"Forecasting from hour 12, horizon {FORECAST_HORIZON}h, "
          f"saturation at {SATURATION_PCT}%\n")

    for fc in forecast_all(occupancy, current_hour=12):
        sat = fc["hours_to_saturation"]
        sat_txt = f"saturates in {sat}h" if sat is not None else "not trending to saturation"
        if sat == 0.0:
            sat_txt = "ALREADY SATURATED"

        print(f"{fc['zone_id']}  {names[fc['zone_id']]:<20} "
              f"{fc['current_occupancy_pct']:>6.1f}%  "
              f"{fc['trend_pct_per_hour']:>+5.2f}%/h   {sat_txt}")
