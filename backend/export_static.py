"""
Static export.

The dataset is synthetic and deterministic, so every response the dashboard can
ask for is knowable ahead of time. This writes them all out as JSON, which lets
the whole thing be hosted as a static site: no server, no cold start, and
nothing to keep running.

One file per hour holds everything that hour needs, including the recommendation
list for every zone and the simulator's results across the slider's range. The
simulator is precomputed rather than reimplemented in the browser deliberately:
a second copy of that logic would drift from this one, and the thresholds in
this project have drifted twice already.

Run:  python export_static.py
Out:  ../frontend/data/hour-<h>.json, ../frontend/data/meta.json
"""

import json
import os
import shutil

import pandas as pd

from engine.density import zone_density_for_venue
from engine.exit_redirect import recommend_exit
from engine.forecast import DATA_DIR, forecast_all
from engine.hotels import build_hotels
from engine.incentives import generate_incentives
from engine.model import load_model, predict_all, train
from engine.recommend import recommend_for_zone
from engine.restaurants import build_restaurants
from engine.reverse_uber import optimal_exit_distribution
from engine.risk import detect_all
from engine.simulate import simulate_redistribution
from engine.venues import analyse_all as analyse_venues
from engine.routing import find_routes, CONGESTED_PCT

OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "data"
)

# Must match the slider in the dashboard.
FIRST_HOUR = 24
MOVE_MIN, MOVE_MAX, MOVE_STEP = 100, 2500, 100


def _zones_at(zones, occ, hour):
    snap = occ[occ["hour_offset"] == hour]
    current = dict(zip(snap["zone_id"], snap["occupancy_pct"]))
    occupied = dict(zip(snap["zone_id"], snap["occupancy"]))

    out = []
    for _, z in zones.iterrows():
        pct = float(current.get(z["zone_id"], 0.0))
        out.append({
            "id": z["zone_id"],
            "name": z["name"],
            "lat": float(z["lat"]),
            "lng": float(z["lng"]),
            "hotel_capacity": int(z["hotel_capacity"]),
            "current_occupancy": int(occupied.get(z["zone_id"], 0)),
            "occupancy_pct": round(pct, 2),
            "spare_capacity": max(0, int(z["hotel_capacity"] * (100 - pct) / 100)),
            "avg_nightly_rate": int(z["avg_nightly_rate"]),
            "is_venue_zone": bool(z["is_venue_zone"]),
            "distance_to_venue_km": float(z["distance_to_venue_km"]),
        })
    return out


def _summary(zone_rows, alerts, hour):
    total_cap = sum(z["hotel_capacity"] for z in zone_rows)
    total_occ = sum(z["current_occupancy"] for z in zone_rows)
    return {
        "hour": hour,
        "total_capacity": total_cap,
        "total_occupied": total_occ,
        "citywide_occupancy_pct": round(total_occ / total_cap * 100, 1) if total_cap else 0,
        "spare_capacity": max(0, total_cap - total_occ),
        "zones_saturated": len([z for z in zone_rows if z["occupancy_pct"] >= 95]),
        "zones_total": len(zone_rows),
        "alerts_critical": len([a for a in alerts if a["severity"] == "critical"]),
        "alerts_warning": len([a for a in alerts if a["severity"] == "warning"]),
    }


def _transport(transport, load_df, hour):
    t = transport.copy()
    if load_df is not None:
        snap = load_df[load_df["hour_offset"] == hour]
        t = t.merge(snap[["link_id", "current_load", "load_pct"]], on="link_id", how="left")
        t["load_pct"] = t["load_pct"].fillna(0.0)
        t["current_load"] = t["current_load"].fillna(0).astype(int)
    return json.loads(t.to_json(orient="records"))


def _history(occ, zones, hour, window=24, top=5):
    names = dict(zip(zones["zone_id"], zones["name"]))
    lo = max(0, hour - window)
    span = occ[(occ["hour_offset"] >= lo) & (occ["hour_offset"] <= hour)]
    now = span[span["hour_offset"] == hour].sort_values("occupancy_pct", ascending=False)

    series = []
    for zid in list(now["zone_id"].head(top)):
        rows = span[span["zone_id"] == zid].sort_values("hour_offset")
        series.append({
            "zone_id": zid,
            "zone_name": names.get(zid, zid),
            "points": [
                {"hour": int(r["hour_offset"]), "pct": float(r["occupancy_pct"])}
                for _, r in rows.iterrows()
            ],
        })
    return {"from_hour": lo, "to_hour": hour, "series": series}


def _reroutes(transport, load_df, hour):
    """Alternative routes for every link busy enough to warrant one."""
    if load_df is None:
        return {}

    snap = load_df[load_df["hour_offset"] == hour]
    busy = snap[snap["load_pct"] >= 80.0]["link_id"]

    out = {}
    for link_id in busy:
        row = transport[transport["link_id"] == link_id]
        if row.empty:
            continue
        l = row.iloc[0]
        out[link_id] = find_routes(l["from_zone"], l["to_zone"], hour)
    return out


def _events(ev, zones, hour):
    names = dict(zip(zones["zone_id"], zones["name"]))
    out = []
    for _, e in ev.iterrows():
        start, end = int(e["start_hour"]), int(e["end_hour"])
        if hour < start:
            state, away = "upcoming", start - hour
        elif hour <= end:
            state, away = "running", 0
        else:
            state, away = "finished", hour - end

        out.append({
            "event_id": e["event_id"], "name": e["name"],
            "zone_id": e["zone_id"], "zone_name": names.get(e["zone_id"], e["zone_id"]),
            "start_hour": start, "end_hour": end,
            "expected_attendance": int(e["expected_attendance"]),
            "state": state, "hours_away": int(away),
        })

    live = [x for x in out if x["state"] == "running"]
    ahead = sorted([x for x in out if x["state"] == "upcoming"], key=lambda x: x["hours_away"])
    return {"hour": hour, "schedule": live + ahead, "all": out}


def _venue_scoped(venue_zone_ids, hour):
    """
    The per-venue engines (density, exit redirect, reverse-uber) keyed by
    venue zone id, for the venue-floor panels that would otherwise fetch
    these live — on a static deploy there is no live API to fetch from, so
    they need to arrive with everything else in the hour bundle.
    """
    return {
        "density": {zid: zone_density_for_venue(zid, hour) for zid in venue_zone_ids},
        "exit_redirect": {zid: recommend_exit(zid, hour) for zid in venue_zone_ids},
        "reverse_uber": {zid: optimal_exit_distribution(zid, hour) for zid in venue_zone_ids},
    }


def main():
    zones = pd.read_csv(os.path.join(DATA_DIR, "zones.csv"))
    occ = pd.read_csv(os.path.join(DATA_DIR, "occupancy_timeline.csv"))
    transport = pd.read_csv(os.path.join(DATA_DIR, "transport.csv"))
    ev = pd.read_csv(os.path.join(DATA_DIR, "events.csv"))

    load_path = os.path.join(DATA_DIR, "transport_load.csv")
    load_df = pd.read_csv(load_path) if os.path.exists(load_path) else None

    gates_path = os.path.join(DATA_DIR, "gates.csv")
    venue_zone_ids = (
        list(pd.read_csv(gates_path)["venue_zone_id"].unique())
        if os.path.exists(gates_path) else []
    )

    last_hour = int(occ["hour_offset"].max())
    zone_ids = list(zones["zone_id"])
    moves = list(range(MOVE_MIN, MOVE_MAX + 1, MOVE_STEP))

    if os.path.isdir(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)

    # Model scorecard is the same at every hour, so it lives in meta.
    if load_model() is None:
        train(verbose=False)
    _m, _df, metrics = train(verbose=False)

    meta = {
        "first_hour": FIRST_HOUR,
        "last_hour": last_hour,
        "moves": moves,
        "benchmark": {
            "horizon_hours": metrics["horizon_hours"],
            "train_rows": metrics["train_rows"],
            "test_rows": metrics["test_rows"],
            "split_at_hour": metrics["split_at_hour"],
            "baseline": {"name": "Trend projection",
                         "mae": round(metrics["baseline_mae"], 2),
                         "r2": round(metrics["baseline_r2"], 3)},
            "model": {"name": "Gradient boosting",
                      "mae": round(metrics["model_mae"], 2),
                      "r2": round(metrics["model_r2"], 3)},
            "mae_improvement_pct": metrics["improvement_pct"],
        },
    }
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, separators=(",", ":"))

    frames = {"zones": zones, "occupancy": occ}

    written = 0
    for hour in range(FIRST_HOUR, last_hour + 1):
        zone_rows = _zones_at(zones, occ, hour)
        forecasts = forecast_all(occ, current_hour=hour)
        alerts = detect_all(current_hour=hour)

        # Rank once per zone, then reuse it across the whole slider range.
        # Recomputing the ranking for all 25 move values was the bottleneck.
        recs = {z: recommend_for_zone(z, current_hour=hour) for z in zone_ids}

        sims = {}
        for z in zone_ids:
            sims[z] = {
                str(mv): simulate_redistribution(
                    z, mv, current_hour=hour, frames=frames, recs=recs[z]
                )
                for mv in moves
            }

        payload = {
            "hour": hour,
            "zones": zone_rows,
            "forecast": forecasts,
            "alerts": alerts,
            "summary": _summary(zone_rows, alerts, hour),
            "transport": _transport(transport, load_df, hour),
            "history": _history(occ, zones, hour),
            "events": _events(ev, zones, hour),
            "hotels": build_hotels(current_hour=hour),
            "restaurants": build_restaurants(current_hour=hour),
            "incentives": generate_incentives(current_hour=hour),
            "recommendations": recs,
            "venues": analyse_venues(current_hour=hour),
            # Reroutes only for corridors actually under strain. Precomputing
            # every zone pair would be 90 routes an hour, almost all of them
            # for links nobody is looking at.
            "reroutes": _reroutes(transport, load_df, hour),
            **_venue_scoped(venue_zone_ids, hour),
        }

        with open(os.path.join(OUT_DIR, f"hour-{hour}.json"), "w") as f:
            json.dump(payload, f, separators=(",", ":"), default=str)

        # Simulations are four fifths of the payload but only ever shown for the
        # one selected zone, so they go in their own files and are fetched on
        # demand. Keeping them inline made every hour tick download work the
        # viewer would never look at.
        for z in zone_ids:
            with open(os.path.join(OUT_DIR, f"sim-{hour}-{z}.json"), "w") as f:
                json.dump(sims[z], f, separators=(",", ":"), default=str)

        written += 1
        if written % 20 == 0:
            print(f"  {written} hours written")

    total_mb = sum(
        os.path.getsize(os.path.join(OUT_DIR, n)) for n in os.listdir(OUT_DIR)
    ) / 1_048_576

    print(f"\n{written} hour files + meta.json")
    print(f"Hours {FIRST_HOUR} to {last_hour}")
    print(f"Total {total_mb:.1f} MB in {OUT_DIR}")


if __name__ == "__main__":
    main()
