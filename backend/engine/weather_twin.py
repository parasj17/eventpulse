"""
The weather layer of the digital twin.

It answers one question for the rest of EventPulse: given this weather (live, forecast,
or a what-if), how will the city's zones, transport, venue and visitors behave, and how
sure are we?

**What it learns, and from what.** The city's history (data/*.csv) records hour by hour
whether it rained, how loaded every transport link was, and how occupancy moved between
zones. A Bayesian linear regression on each (with the time of day and the event schedule
held fixed) learns how rain changes transport demand and how it slows movement between
zones. Bayesian because the result is a distribution, not a number: every prediction
below carries the model's own uncertainty. The model keeps learning: an organiser's
field report (a measured link load, or a street flooding or not) updates it in place and
is saved, so it survives a restart.

**What it cannot learn from this history, and so assumes.** Flooding, heat, wind, staff
getting to work, how many ticket holders stay home. The history has no examples of
these, so each starts from a stated prior with a wide spread (PRIORS, below). Social
reports and field observations move the flooding estimates for the zones they name.

**How effects cascade.** Weather reaches places first (rain, heat, flood risk per zone:
1st order). Places change the network (links lose capacity as streets flood, demand for
them rises as people stop walking; gates slow on wet ground; staff can't get in;
fewer people come: 2nd order). The network changes people's outcomes (visitors who
can't get home need rooms, hotels near the venue fill, restaurants see different
demand, new alerts: 3rd order).

**Uncertainty.** Every run draws 400 samples from the learned posteriors and the priors
(and jitters the rain itself), pushes each through the cascade, and reports the 10th,
50th and 90th percentiles. The baseline shares the same draws, so a difference between
scenario and baseline is the weather's effect, not sampling noise.

Nothing here writes to the live system. It reads the current state and works on copies.

Run:  python -m engine.weather_twin
"""

import csv
import math
import os
import threading
import time

import numpy as np
import pandas as pd

from engine.forecast import DATA_DIR
from engine.recommend import haversine_km

N_DRAWS = 400
OBS_CSV = "weather_observations.csv"

# Rain intensity. The history only records *whether* it rained; call that a typical
# 7.5 mm/h shower. Heavier rain has a bigger effect that levels off (a cloudburst does
# not stop twice as many people as a heavy shower). This curve is an assumption.
REF_RAIN_MM_H = 7.5
RAIN_K = 12.0


def rain_scale(mm):
    mm = np.maximum(0.0, np.asarray(mm, dtype=float))
    return (1 - np.exp(-mm / RAIN_K)) / (1 - math.exp(-REF_RAIN_MM_H / RAIN_K))


# Priors for what the history can't teach: (mean, sd, low, high, what it means).
PRIORS = {
    "flood_threshold_mm": (55, 10, 20, 120, "rain over the period at which a zone at sea level starts to flood"),
    "flood_mm_per_metre": (6, 2, 0, 15, "extra rain needed per metre of elevation"),
    "flood_scale_mm": (15, 3, 6, 30, "how sharply flood risk rises past the threshold"),
    "closure_metro": (0.45, 0.10, 0.1, 0.9, "share of a metro link's capacity lost when an end floods"),
    "closure_shuttle": (0.80, 0.10, 0.3, 1.0, "share of a road shuttle's capacity lost when an end floods"),
    "closure_bus": (0.85, 0.10, 0.3, 1.0, "share of a bus link's capacity lost when an end floods"),
    "wind_road_loss": (0.30, 0.10, 0.0, 0.7, "road capacity lost at 90 km/h gusts (from 50 km/h up)"),
    "wet_gate_factor": (0.85, 0.05, 0.6, 1.0, "gate throughput in heavy rain, as a share of dry"),
    "staff_loss": (0.35, 0.10, 0.0, 0.8, "share of staff who can't get in when access routes flood"),
    "attendance_drop_rain": (0.12, 0.05, 0.0, 0.4, "share of ticket holders who stay home in heavy rain"),
    "attendance_drop_heat_per_c": (0.012, 0.005, 0.0, 0.05, "share who stay home per degree above 35 C"),
    "strand_share": (0.35, 0.10, 0.05, 0.7, "share of delayed leavers who end up needing a room"),
    "dine_in_drop": (0.30, 0.10, 0.0, 0.7, "drop in walk-in restaurant demand in heavy rain"),
    "heat_incidents_per_10k_per_c": (0.8, 0.3, 0.0, 3.0, "heat-related medical calls per 10,000 people per degree above 35 C"),
}
HEAT_C = 35.0


# ---------------------------------------------------------------- Bayesian regression
class Channel:
    """Bayesian linear regression with a Gaussian prior and noise variance fixed from the
    fit. Closed form, and an observation can be added at any time (online learning)."""

    def __init__(self, names, prior_sd=1.0):
        self.names = names
        self.m = np.zeros(len(names))
        self.P = np.eye(len(names)) / prior_sd ** 2        # precision
        self.s2 = 1.0
        self.n_history = 0
        self.n_live = 0
        self.r2 = None

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        self.s2 = float(max(1e-4, resid.var()))
        self.r2 = float(1 - resid.var() / max(1e-9, y.var()))
        P0 = self.P.copy()
        self.P = P0 + X.T @ X / self.s2
        self.m = np.linalg.solve(self.P, P0 @ self.m + X.T @ y / self.s2)
        self.n_history = len(y)

    def fit_ratio(self, X, y, wet):
        """For an effect that scales an outcome: fit the outcome on dry rows, then the
        wet rows' ratio to that prediction. Stored on the log scale as the 'rain' weight."""
        beta, *_ = np.linalg.lstsq(X[~wet], y[~wet], rcond=None)
        yhat = X @ beta
        yw, hw = y[wet], yhat[wet]
        ratio = float(yw @ hw / (hw @ hw))
        se = float(math.sqrt((yw - ratio * hw).var() / (hw @ hw)))
        i = self.names.index("rain")
        self.m = np.zeros(len(self.names)); self.m[i] = math.log(max(1e-3, ratio))
        self.P = np.eye(len(self.names)); self.P[i, i] = (ratio / se) ** 2       # delta method: sd(log r) = se / r
        self.s2 = float((yw - ratio * hw).var())
        dry_res = y[~wet] - yhat[~wet]
        self.r2 = float(1 - dry_res.var() / max(1e-9, y[~wet].var()))
        self.n_history = int(len(y))

    def update(self, x, y, weight=1.0):
        x = np.asarray(x, float)
        Pn = self.P + weight * np.outer(x, x) / self.s2
        self.m = np.linalg.solve(Pn, self.P @ self.m + weight * x * y / self.s2)
        self.P = Pn
        self.n_live += 1

    @property
    def cov(self):
        return np.linalg.inv(self.P)

    def coef(self, name):
        i = self.names.index(name)
        return float(self.m[i]), float(math.sqrt(self.cov[i, i]))

    def predict(self, x):
        x = np.asarray(x, float)
        return float(x @ self.m), float(math.sqrt(x @ self.cov @ x + self.s2))


HOURS_OF_DAY = [f"h{k:02d}" for k in range(1, 24)]     # hour 0 is the reference


def _row(rain, h, ei, eo):
    """Features: rain, event pressure, and the hour of day (as its own level: rain
    windows fall at particular times, and a smooth daily curve would blame the rain
    for what the clock did)."""
    hod = [1.0 if (h % 24) == k else 0.0 for k in range(1, 24)]
    return [float(rain), ei, eo, *hod]


def _event_terms(zone_ids, h, events):
    """Inbound and outbound event pressure touching these zones at hour h (the shape the
    city data was built with: a surge before the start, a sharper one after the end)."""
    ev_in = ev_out = 0.0
    for e in events:
        if e["zone_id"] not in zone_ids:
            continue
        scale = e["expected_attendance"] / 50000
        until, since = e["start_hour"] - h, h - e["end_hour"]
        if 0 <= until <= 6:
            ev_in += scale * ((6 - until) / 6) ** 2
        if 0 <= since <= 3:
            ev_out += scale * math.exp(-since / 1.4)
    return ev_in, ev_out


TRANSPORT_X = ["rain", "event_in", "event_out", *HOURS_OF_DAY]
MOVE_X = ["rain", "event_in", "event_out", *HOURS_OF_DAY]


class WeatherModel:
    def __init__(self, data_dir=None):
        self.data_dir = data_dir or DATA_DIR
        self.lock = threading.Lock()
        self.transport = Channel(TRANSPORT_X)
        self.move = Channel(MOVE_X)
        self.link_means = {}
        self.flood_reports = []           # [{zone_id, flooded, ts}]
        self._fit_history()
        self._replay()

    # ---- history
    def _fit_history(self):
        d = self.data_dir
        weather = pd.read_csv(os.path.join(d, "weather.csv"))
        rain = dict(zip(weather["hour_offset"], weather["is_raining"]))
        events = pd.read_csv(os.path.join(d, "events.csv")).to_dict("records")
        self.events = events
        links = pd.read_csv(os.path.join(d, "transport.csv"))
        self.links_meta = links.set_index("link_id").to_dict("index")

        tl = pd.read_csv(os.path.join(d, "transport_load.csv"))
        rows = []
        for r in tl.itertuples(index=False):
            meta = self.links_meta[r.link_id]
            ei, eo = _event_terms({meta["from_zone"], meta["to_zone"]}, r.hour_offset, events)
            rows.append((r.link_id, math.log(max(0.5, r.load_pct)), *_row(rain.get(r.hour_offset, 0), r.hour_offset, ei, eo)))
        df = pd.DataFrame(rows, columns=["link", "y", *TRANSPORT_X])
        # A fixed effect per link: each link is compared with itself.
        means = df.groupby("link")[["y", *TRANSPORT_X]].mean()
        self.link_means = means.to_dict("index")
        dm = df[["y", *TRANSPORT_X]] - means.loc[df["link"]].to_numpy()
        self.transport.fit(dm[TRANSPORT_X].to_numpy(), dm["y"].to_numpy())

        # Movement: learn the normal hour-to-hour change from dry hours (zone, hour of day,
        # event pull), then measure how much smaller the change is when it rains. Hours at
        # a zone's limits are left out: a full zone can't move whatever the weather.
        occ = pd.read_csv(os.path.join(d, "occupancy_timeline.csv")).sort_values(["zone_id", "hour_offset"])
        rows = []
        for zid, g in occ.groupby("zone_id"):
            pct = g["occupancy_pct"].to_numpy()
            hrs = g["hour_offset"].to_numpy()
            for i in range(1, len(g)):
                if not (5.5 < pct[i - 1] < 99 and 5.5 < pct[i] < 99):
                    continue
                h = int(hrs[i])
                pull = sum(e["expected_attendance"] / 50000 * math.exp(-((e["start_hour"] - h) / 4) ** 2) for e in events)
                rows.append((zid, pct[i] - pct[i - 1], float(rain.get(h, 0)), pull, *_row(0, h, 0, 0)[3:]))
        mf = pd.DataFrame(rows, columns=["zone", "y", "rain", "pull", *HOURS_OF_DAY])
        X = pd.concat([pd.get_dummies(mf["zone"], dtype=float), mf[["pull", *HOURS_OF_DAY]]], axis=1).to_numpy(float)
        self.move.fit_ratio(X, mf["y"].to_numpy(float), mf["rain"].to_numpy() > 0)

    # ---- live observations (continuous learning)
    def _obs_path(self):
        return os.path.join(self.data_dir, OBS_CSV)

    def _replay(self):
        p = self._obs_path()
        if not os.path.exists(p):
            return
        with open(p) as f:
            for row in csv.DictReader(f):
                self._apply(row, persist=False)

    def _transport_x(self, link_id, hour, rr):
        meta = self.links_meta[link_id]
        ei, eo = _event_terms({meta["from_zone"], meta["to_zone"]}, hour, self.events)
        mu = self.link_means[link_id]
        x = np.array(_row(rr, hour, ei, eo)) - np.array([mu[k] for k in TRANSPORT_X])
        return x, mu["y"]

    def _apply(self, row, persist=True):
        kind = row["kind"]
        if kind == "transport":
            x, ybar = self._transport_x(row["key"], int(row["hour"]), float(row["rain_scale"]))
            self.transport.update(x, math.log(max(0.5, float(row["value"]))) - ybar)
        elif kind == "flood":
            self.flood_reports.append({"zone_id": row["key"], "flooded": row["value"] in ("1", 1, True, "True"),
                                       "ts": float(row["ts"])})
        if persist:
            p = self._obs_path()
            new = not os.path.exists(p)
            with open(p, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["kind", "key", "hour", "rain_scale", "value", "ts"])
                if new:
                    w.writeheader()
                w.writerow(row)

    def observe_transport(self, link_id, hour, rain_mm_h, load_pct):
        if link_id not in self.links_meta:
            raise ValueError("No such transport link.")
        if not (0 < float(load_pct) <= 400):
            raise ValueError("Load must be between 0 and 400 percent.")
        with self.lock:
            before = self.rain_effects()["transport_load_multiplier"]
            self._apply({"kind": "transport", "key": link_id, "hour": int(hour),
                         "rain_scale": float(rain_scale(rain_mm_h)), "value": float(load_pct), "ts": time.time()})
            after = self.rain_effects()["transport_load_multiplier"]
        return {"before": before, "after": after}

    def observe_flood(self, zone_id, flooded, now=None):
        with self.lock:
            self._apply({"kind": "flood", "key": zone_id, "hour": 0, "rain_scale": 0,
                         "value": 1 if flooded else 0, "ts": now or time.time()})

    def flood_log_odds(self, zone_id, now=None):
        """Evidence from field reports, fading over six hours."""
        now = now or time.time()
        lo = 0.0
        for r in self.flood_reports:
            if r["zone_id"] != zone_id:
                continue
            w = 0.5 ** (max(0, now - r["ts"]) / 3600 / 6)
            lo += (2.5 if r["flooded"] else -2.5) * w
        return lo

    # ---- what it has learned
    def rain_effects(self):
        bt, st = self.transport.coef("rain")
        bm, sm = self.move.coef("rain")
        f = lambda b, s: {"mean": round(math.exp(b), 3), "low": round(math.exp(b - 1.645 * s), 3),
                          "high": round(math.exp(b + 1.645 * s), 3)}
        return {"transport_load_multiplier": f(bt, st), "movement_multiplier": f(bm, sm)}

    def summary(self):
        eff = self.rain_effects()
        return {
            "learned": {
                "transport": {**eff["transport_load_multiplier"], "rows": self.transport.n_history,
                              "live_updates": self.transport.n_live, "r2": round(self.transport.r2, 3),
                              "plain": "In a typical shower, transport demand is multiplied by about "
                                       f"{eff['transport_load_multiplier']['mean']:.2f} (90% range "
                                       f"{eff['transport_load_multiplier']['low']:.2f}-{eff['transport_load_multiplier']['high']:.2f}): "
                                       "people stop walking and take the metro or a shuttle."},
                "movement": {**eff["movement_multiplier"], "rows": self.move.n_history,
                             "live_updates": self.move.n_live, "r2": round(self.move.r2, 3),
                             "plain": "Rain slows movement between zones to about "
                                      f"{eff['movement_multiplier']['mean'] * 100:.0f}% of normal (90% range "
                                      f"{eff['movement_multiplier']['low'] * 100:.0f}-{eff['movement_multiplier']['high'] * 100:.0f}%): "
                                      "people stay where they are."},
            },
            "assumed": [{"name": k, "mean": v[0], "sd": v[1], "meaning": v[4]} for k, v in PRIORS.items()],
            "intensity_note": f"The history records whether it rained, not how hard. Effects are scaled from a "
                              f"typical {REF_RAIN_MM_H} mm/h shower along a curve that levels off; that curve is an assumption.",
            "flood_reports": len(self.flood_reports),
        }

    def draw(self, n, rng):
        """Joint samples of everything uncertain, one row per draw."""
        out = {}
        for name, ch in (("b_transport", self.transport), ("b_move", self.move)):
            i = ch.names.index("rain")
            out[name] = rng.normal(ch.m[i], math.sqrt(ch.cov[i, i]), n)
        for k, (mu, sd, lo, hi, _) in PRIORS.items():
            out[k] = np.clip(rng.normal(mu, sd, n), lo, hi)
        out["rain_jitter"] = np.exp(rng.normal(0, 0.15, n))
        return out


_model = None
_model_lock = threading.Lock()


def model(data_dir=None):
    global _model
    with _model_lock:
        if _model is None or (data_dir and _model.data_dir != data_dir):
            _model = WeatherModel(data_dir)
        return _model


# ---------------------------------------------------------------- scenarios
class ScenarioError(ValueError):
    pass


def _bounded(v, lo, hi, label):
    try:
        v = float(v)
    except (TypeError, ValueError):
        raise ScenarioError(f"{label} must be a number.")
    if not (lo <= v <= hi) or not math.isfinite(v):
        raise ScenarioError(f"{label} must be between {lo} and {hi}.")
    return v


def normal_weather(duration_h=3):
    return {"label": "Normal weather", "rain_mm_h": 0.0, "duration_h": duration_h, "temp_c": 30.0, "wind_kmh": 10.0}


def parse_scenario(s, zones):
    """Validate a what-if and turn it into per-zone rain."""
    s = dict(s or {})
    out = {
        "label": str(s.get("label") or "What-if")[:60],
        "rain_mm_h": _bounded(s.get("rain_mm_h", 0), 0, 150, "Rain intensity"),
        "duration_h": _bounded(s.get("duration_h", 3), 0.5, 48, "Duration"),
        "temp_c": _bounded(s.get("temp_c", 30), -5, 50, "Temperature"),
        "wind_kmh": _bounded(s.get("wind_kmh", 10), 0, 200, "Wind"),
    }
    radius = s.get("radius_km")
    center = s.get("center") or {}
    if radius not in (None, "", 0) and center:
        out["radius_km"] = _bounded(radius, 0.5, 30, "Storm radius")
        if center.get("zone_id"):
            z = next((z for z in zones if z["id"] == center["zone_id"]), None)
            if z is None:
                raise ScenarioError("Unknown zone for the storm centre.")
            out["center"] = {"lat": z["lat"], "lng": z["lng"], "zone_id": z["id"]}
        else:
            lat = _bounded(center.get("lat"), -90, 90, "Storm latitude")
            lng = _bounded(center.get("lng"), -180, 180, "Storm longitude")
            if min(haversine_km(lat, lng, z["lat"], z["lng"]) for z in zones) > 30:
                raise ScenarioError("The storm centre is outside the area this deployment covers.")
            out["center"] = {"lat": lat, "lng": lng}
    return out


def rain_by_zone(sc, zones):
    """mm/h per zone: uniform, a storm cell that fades with distance, or live per-zone readings."""
    if "rain_by_zone" in sc:
        return np.array([sc["rain_by_zone"].get(z["id"], 0.0) for z in zones])
    if "center" in sc:
        c, r = sc["center"], sc["radius_km"]
        d = np.array([haversine_km(c["lat"], c["lng"], z["lat"], z["lng"]) for z in zones])
        return sc["rain_mm_h"] * np.exp(-(d / r) ** 2)
    return np.full(len(zones), sc["rain_mm_h"])


# ---------------------------------------------------------------- the cascade
def _q(a):
    a = np.asarray(a, float)
    p10, p50, p90 = np.percentile(a, [10, 50, 90], axis=0)
    r = lambda v: np.round(v, 2).tolist() if np.ndim(v) else round(float(v), 2)
    return {"p10": r(p10), "p50": r(p50), "p90": r(p90)}


def _run(system, sc, D, flood_lo):
    """Push one weather scenario through the system for every draw at once."""
    zones, links, venue = system["zones"], system["links"], system.get("venue")
    n = len(D["b_transport"])
    zi = {z["id"]: i for i, z in enumerate(zones)}
    rain = rain_by_zone(sc, zones)[None, :] * D["rain_jitter"][:, None]           # (n, Z)
    rr = rain_scale(rain)
    dur = sc["duration_h"]
    accum = rain * dur + np.array([system["rain_prior_mm"].get(z["id"], 0.0) for z in zones])[None, :]
    elev = np.array([system["elevation"].get(z["id"], 10.0) for z in zones])[None, :]

    # 1st order: flood risk per zone (logistic in rain beyond an elevation-dependent threshold,
    # nudged by field and social evidence) and heat.
    thr = D["flood_threshold_mm"][:, None] + D["flood_mm_per_metre"][:, None] * elev
    logit = (accum - thr) / D["flood_scale_mm"][:, None] + flood_lo[None, :]
    pf = 1 / (1 + np.exp(-np.clip(logit, -30, 30)))
    heat = max(0.0, sc["temp_c"] - HEAT_C)
    wind_ramp = min(1.0, max(0.0, (sc["wind_kmh"] - 50) / 40))

    # 2nd order: the network.
    L = len(links)
    a = np.array([zi[l["from_zone"]] for l in links]); b = np.array([zi[l["to_zone"]] for l in links])
    rr_link = (rr[:, a] + rr[:, b]) / 2
    pf_link = np.maximum(pf[:, a], pf[:, b])
    sens = np.stack([D["closure_" + l["mode"]] if "closure_" + l["mode"] in D else D["closure_shuttle"] for l in links], 1)
    road = np.array([l["mode"] in ("shuttle", "bus") for l in links])[None, :]
    cap0 = np.array([l["capacity_per_hour"] for l in links], float)[None, :]
    load0 = np.array([l["current_load"] for l in links], float)[None, :]
    cap = cap0 * (1 - sens * pf_link) * (1 - road * D["wind_road_loss"][:, None] * wind_ramp)
    demand = np.exp(D["b_transport"][:, None] * rr_link)
    load = load0 * demand
    load_pct = load / np.maximum(1.0, cap) * 100

    # Zones: movement between zones slows in rain (learned), so the change the city was about
    # to see over this window happens more slowly.
    occ0 = np.array([z["current_occupancy"] for z in zones], float)[None, :]
    capz = np.array([z["hotel_capacity"] for z in zones], float)[None, :]
    ahead = np.array([system["occ_ahead"].get(z["id"], 0.0) for z in zones], float)[None, :]
    occ = np.clip(occ0 + ahead * np.exp(D["b_move"][:, None] * rr), 0, capz)

    out = {"flood": pf, "rain": rain, "link_cap": cap / cap0, "link_load_pct": load_pct, "occ_pct": None}
    if venue:
        v = zi[venue["zone_id"]]
        vl = [i for i, l in enumerate(links) if venue["zone_id"] in (l["from_zone"], l["to_zone"])]
        w = cap0[0, vl]
        access = (pf_link[:, vl] * w).sum(1) / w.sum()
        staff = np.clip(1 - D["staff_loss"] * access, 0.2, 1)
        gate = (1 - (1 - D["wet_gate_factor"]) * np.minimum(1.0, rr[:, v])) * staff
        att = np.clip(1 - D["attendance_drop_rain"] * np.minimum(1.5, rr[:, v])
                      - D["attendance_drop_heat_per_c"] * heat, 0.3, 1)
        people = np.minimum(venue["capacity"], venue["expected"] * att) if venue["phase"] != "leaving" \
            else np.full(n, float(venue["people"]))
        out_cap = cap[:, vl].sum(1) * 0.75 / np.maximum(1.0, demand[:, vl].mean(1))
        clear = np.maximum(people / np.maximum(1.0, venue["exit_per_hour"] * gate), people / np.maximum(1.0, out_cap))
        out.update(staff=staff, gate=gate, attendance=people, clear_h=clear,
                   heat_incidents=people / 10000 * D["heat_incidents_per_10k_per_c"] * heat)
        out["_people"] = people

    # Stranded visitors look for rooms: the venue zone first, then outward by distance.
    out["occ_after_move"] = occ
    return out


def _strand_and_house(system, base, scen, D):
    """3rd order: extra clearing time strands people, who then fill hotel rooms nearby."""
    zones, venue = system["zones"], system.get("venue")
    n = len(D["b_transport"])
    occ = scen["occ_after_move"].copy()
    capz = np.array([z["hotel_capacity"] for z in zones], float)[None, :].repeat(n, 0)
    stranded = np.zeros(n); unhoused = np.zeros(n)
    if venue:
        # Up to an hour's extra wait just means waiting. Past that, a growing share can't get
        # home tonight (last trains, closed roads), all of the share by four hours' delay.
        extra_h = scen["clear_h"] - base["clear_h"]
        stranded = scen["_people"] * D["strand_share"] * np.clip((extra_h - 1) / 3, 0, 1)
        order = sorted(range(len(zones)), key=lambda i: zones[i]["distance_km"])
        left = stranded.copy()
        for i in order:
            room = np.maximum(0, capz[:, i] - occ[:, i])
            take = np.minimum(room, left)
            occ[:, i] += take
            left -= take
        unhoused = left
    scen["occ_pct"] = occ / capz * 100
    scen["stranded"] = stranded
    scen["unhoused"] = unhoused
    rr = rain_scale(scen["rain"])
    scen["dine_in"] = 1 - D["dine_in_drop"][:, None] * np.minimum(1.3, rr)
    return scen


def simulate(system, scenario, baseline=None, seed=7):
    """Scenario vs baseline, both through the same draws. Returns distributions and the
    effects they add up to, ordered 1st -> 3rd."""
    m = model()
    zones = system["zones"]
    baseline = baseline or normal_weather(scenario["duration_h"])
    rng = np.random.default_rng(seed)
    with m.lock:
        D = m.draw(N_DRAWS, rng)
        now = time.time()
        flood_lo = np.array([m.flood_log_odds(z["id"], now) + system["social_flood_lo"].get(z["id"], 0.0) for z in zones])
    b = _run(system, baseline, D, flood_lo)
    s = _run(system, scenario, D, flood_lo)
    _strand_and_house(system, b, b, D)          # compared with itself: nobody is stranded by the baseline
    _strand_and_house(system, b, s, D)
    return _report(system, scenario, baseline, b, s)


def _report(system, sc, bl, b, s):
    zones, links, venue = system["zones"], system["links"], system.get("venue")
    Z = [{"zone_id": z["id"], "name": z["name"], "lat": z["lat"], "lng": z["lng"],
          "rain_mm_h": round(float(np.median(s["rain"][:, i])), 1),
          "flood_pct": _q(s["flood"][:, i] * 100), "flood_pct_baseline": round(float(np.median(b["flood"][:, i]) * 100), 1),
          "occupancy_pct": _q(s["occ_pct"][:, i]), "occupancy_pct_baseline": round(float(np.median(b["occ_pct"][:, i])), 1),
          "dine_in_demand_pct": round(float(np.median(s["dine_in"][:, i]) * 100)),
          } for i, z in enumerate(zones)]
    zl = {z["id"]: z for z in zones}
    Lk = [{"link_id": l["link_id"], "mode": l["mode"], "from_zone": l["from_zone"], "to_zone": l["to_zone"],
           "from": [zl[l["from_zone"]]["lng"], zl[l["from_zone"]]["lat"]], "to": [zl[l["to_zone"]]["lng"], zl[l["to_zone"]]["lat"]],
           "capacity_pct": _q(s["link_cap"][:, j] * 100), "load_pct": _q(s["link_load_pct"][:, j]),
           "capacity_pct_baseline": round(float(np.median(b["link_cap"][:, j]) * 100), 1),
           "load_pct_baseline": round(float(np.median(b["link_load_pct"][:, j])), 1)} for j, l in enumerate(links)]
    city = {
        "stranded": _q(s["stranded"]), "unhoused": _q(s["unhoused"]),
        "zones_full": _q((s["occ_pct"] >= 95).sum(1)), "zones_full_baseline": int(np.median((b["occ_pct"] >= 95).sum(1))),
        "links_over": _q((s["link_load_pct"] >= 100).sum(1)), "links_over_baseline": int(np.median((b["link_load_pct"] >= 100).sum(1))),
        "links_disrupted": _q((s["link_cap"] <= 0.6).sum(1)),
        "spare_beds": _q((np.array([z["hotel_capacity"] for z in zones])[None, :] * (1 - s["occ_pct"] / 100)).sum(1)),
        "spare_beds_baseline": int(np.median((np.array([z["hotel_capacity"] for z in zones])[None, :] * (1 - b["occ_pct"] / 100)).sum(1))),
        "max_flood_pct": _q(s["flood"].max(1) * 100),
    }
    V = None
    if venue:
        V = {"event_id": venue["event_id"], "event_name": venue["event_name"], "venue_name": venue["venue_name"],
             "zone_id": venue["zone_id"], "phase": venue["phase"],
             "people": _q(s["attendance"]), "people_baseline": round(float(np.median(b["attendance"]))),
             "gate_throughput_pct": _q(s["gate"] * 100), "staff_available_pct": _q(s["staff"] * 100),
             "staff_available_pct_baseline": round(float(np.median(b["staff"]) * 100), 1),
             "hours_to_clear": _q(s["clear_h"]), "hours_to_clear_baseline": round(float(np.median(b["clear_h"])), 2),
             "heat_incidents": _q(s["heat_incidents"])}
    return {"hour": system["hour"], "scenario": sc, "baseline": bl, "draws": N_DRAWS, "zones": Z, "links": Lk,
            "venue": V, "city": city, "effects": effects(system, sc, Z, Lk, V, city)}


def effects(system, sc, Z, Lk, V, city):
    """The cascade in words, strongest first within each order, with where it happens and
    what caused it (for drawing the propagation on the map)."""
    out = []
    zl = {z["zone_id"]: z for z in Z}
    rng = lambda q, unit="", k=0: f"{q['p50']:.{k}f}{unit} (likely {q['p10']:.{k}f}-{q['p90']:.{k}f}{unit})"

    wet = [z for z in Z if z["rain_mm_h"] >= 0.5]
    if wet:
        top = max(wet, key=lambda z: z["rain_mm_h"])
        out.append({"id": "rain", "order": 1, "kind": "weather", "lat": top["lat"], "lng": top["lng"], "cause": None, "severity": 1e6,
                    "text": f"Rain up to {top['rain_mm_h']:.0f} mm/h over {len(wet)} of {len(Z)} zones for {sc['duration_h']:g} h, "
                            f"heaviest at {top['name']}."})
    if sc["temp_c"] > HEAT_C:
        out.append({"id": "heat", "order": 1, "kind": "weather", "lat": None, "lng": None, "cause": None, "severity": 1e6,
                    "text": f"Heat: {sc['temp_c']:.0f} C, {sc['temp_c'] - HEAT_C:.0f} degrees past the point where crowds suffer."})
    if sc["wind_kmh"] >= 50:
        out.append({"id": "wind", "order": 1, "kind": "weather", "lat": None, "lng": None, "cause": None, "severity": 1e6,
                    "text": f"Gusts of {sc['wind_kmh']:.0f} km/h slow road shuttles and buses."})
    for z in sorted(Z, key=lambda z: -z["flood_pct"]["p50"]):
        if z["flood_pct"]["p50"] < 15 or z["flood_pct"]["p50"] - z["flood_pct_baseline"] < 5:
            continue
        out.append({"id": f"flood-{z['zone_id']}", "order": 1, "kind": "zone", "zone_id": z["zone_id"], "lat": z["lat"], "lng": z["lng"],
                    "cause": "rain", "severity": z["flood_pct"]["p50"],
                    "text": f"{z['name']}: chance of flooding {rng(z['flood_pct'], '%')}."})

    # Measured against the baseline: a street already flooded before the what-if isn't its effect.
    for l in sorted(Lk, key=lambda l: l["capacity_pct"]["p50"] - l["capacity_pct_baseline"]):
        lost = l["capacity_pct_baseline"] - l["capacity_pct"]["p50"]
        hot = l["load_pct"]["p50"] >= 100 and l["load_pct_baseline"] < 100
        if lost < 10 and not hot:
            continue
        cause = next((f"flood-{zid}" for zid in (l["from_zone"], l["to_zone"]) if any(e["id"] == f"flood-{zid}" for e in out)), "rain")
        a, b = zl[l["from_zone"]]["name"], zl[l["to_zone"]]["name"]
        out.append({"id": f"link-{l['link_id']}", "order": 2, "kind": "link", "link_id": l["link_id"],
                    "lat": (l["from"][1] + l["to"][1]) / 2, "lng": (l["from"][0] + l["to"][0]) / 2, "cause": cause,
                    "severity": lost,
                    "text": f"{l['mode'].capitalize()} {a} - {b}: runs at {rng(l['capacity_pct'], '%')} of its capacity "
                            f"(was {l['capacity_pct_baseline']:.0f}%); "
                            f"demand is {l['load_pct']['p50']:.0f}% of what it can now carry (was {l['load_pct_baseline']:.0f}%)."})
    link_effects = [e["id"] for e in out if e["kind"] == "link"]
    if V:
        vz = zl[V["zone_id"]]
        vcause = link_effects[0] if link_effects else ("rain" if wet else ("heat" if sc["temp_c"] > HEAT_C else None))
        if V["hours_to_clear"]["p50"] - V["hours_to_clear_baseline"] >= 0.1:
            out.append({"id": "venue-clear", "order": 2, "kind": "venue", "zone_id": V["zone_id"], "lat": vz["lat"], "lng": vz["lng"],
                        "cause": vcause, "severity": V["hours_to_clear"]["p50"],
                        "text": f"{V['venue_name']} takes {rng(V['hours_to_clear'], ' h', 1)} to empty, "
                                f"up from {V['hours_to_clear_baseline']:.1f} h: gates at {V['gate_throughput_pct']['p50']:.0f}% "
                                f"and fewer ways out."})
        if V["staff_available_pct_baseline"] - V["staff_available_pct"]["p50"] >= 3:
            out.append({"id": "staff", "order": 2, "kind": "venue", "zone_id": V["zone_id"], "lat": vz["lat"], "lng": vz["lng"],
                        "cause": vcause, "text": f"Only {rng(V['staff_available_pct'], '%')} of gate staff can get in."})
        drop = V["people_baseline"] - V["people"]["p50"]
        if V["phase"] != "leaving" and drop >= 200:
            out.append({"id": "attendance", "order": 2, "kind": "venue", "zone_id": V["zone_id"], "lat": vz["lat"], "lng": vz["lng"],
                        "cause": "rain" if wet else "heat",
                        "text": f"About {drop:,.0f} fewer people turn up to {V['event_name']} "
                                f"({V['people']['p10']:,.0f}-{V['people']['p90']:,.0f} expected)."})
        if V["heat_incidents"]["p50"] >= 1:
            out.append({"id": "medical", "order": 2, "kind": "venue", "zone_id": V["zone_id"], "lat": vz["lat"], "lng": vz["lng"],
                        "cause": "heat", "text": f"Expect {rng(V['heat_incidents'])} heat-related medical calls at the venue."})
    c = city
    if c["stranded"]["p50"] >= 50:
        out.append({"id": "stranded", "order": 3, "kind": "people", "zone_id": V["zone_id"] if V else None,
                    "lat": zl[V["zone_id"]]["lat"] if V else None, "lng": zl[V["zone_id"]]["lng"] if V else None,
                    "cause": "venue-clear", "severity": c["stranded"]["p50"],
                    "text": f"{rng(c['stranded'])} visitors can't get home in time and need a room tonight."})
    for z in sorted(Z, key=lambda z: -(z["occupancy_pct"]["p50"] - z["occupancy_pct_baseline"])):
        up = z["occupancy_pct"]["p50"] - z["occupancy_pct_baseline"]
        if up >= 3:
            out.append({"id": f"fill-{z['zone_id']}", "order": 3, "kind": "zone", "zone_id": z["zone_id"], "lat": z["lat"], "lng": z["lng"],
                        "cause": "stranded" if c["stranded"]["p50"] >= 50 else "rain", "severity": up,
                        "text": f"{z['name']} hotels fill to {rng(z['occupancy_pct'], '%')} (from {z['occupancy_pct_baseline']:.0f}%)."})
        elif up <= -3:
            out.append({"id": f"slow-{z['zone_id']}", "order": 3, "kind": "zone", "zone_id": z["zone_id"], "lat": z["lat"], "lng": z["lng"],
                        "cause": "rain", "severity": -up,
                        "text": f"{z['name']} fills more slowly, {z['occupancy_pct']['p50']:.0f}% instead of {z['occupancy_pct_baseline']:.0f}%: "
                                f"people wait out the rain where they are."})
    if c["unhoused"]["p50"] >= 50:
        out.append({"id": "unhoused", "order": 3, "kind": "people", "lat": None, "lng": None, "cause": "stranded",
                    "text": f"{rng(c['unhoused'])} of them find no bed anywhere in the city: shelters or late transport needed."})
    dine = [z for z in Z if z["dine_in_demand_pct"] <= 85 and z["rain_mm_h"] >= 0.5]
    if dine:
        out.append({"id": "restaurants", "order": 3, "kind": "business", "lat": None, "lng": None, "cause": "rain",
                    "text": f"Walk-in restaurant demand drops to about {min(z['dine_in_demand_pct'] for z in dine)}% in "
                            f"{len(dine)} zone{'s' if len(dine) != 1 else ''}; delivery and indoor venues pick up."})
    more_full = c["zones_full"]["p50"] - c["zones_full_baseline"]
    more_over = c["links_over"]["p50"] - c["links_over_baseline"]
    if more_full >= 1 or more_over >= 1:
        out.append({"id": "alerts", "order": 3, "kind": "system", "lat": None, "lng": None,
                    "cause": "stranded" if c["stranded"]["p50"] >= 50 else (link_effects[0] if link_effects else "rain"),
                    "text": f"New alerts: {max(0, more_full):.0f} more zone{'s' if more_full != 1 else ''} at capacity, "
                            f"{max(0, more_over):.0f} more transport link{'s' if more_over != 1 else ''} over capacity."})
    out.sort(key=lambda e: (e["order"], -(e.get("severity") or 0)))
    return _condense(out)


def _condense(effects, keep=4):
    """Keep every effect for the map, but in the list show the worst few of each kind and
    fold the rest into one line."""
    groups = {"link": ("more transport links lose capacity or overflow", "link"),
              "fill": ("more zones' hotels fill up", "zone"), "flood": ("more zones at risk of flooding", "zone")}
    out, seen = [], {}
    for e in effects:
        g = "link" if e["kind"] == "link" else ("fill" if e["id"].startswith("fill-") else
                                                 ("flood" if e["id"].startswith("flood-") else None))
        if g:
            seen[g] = seen.get(g, 0) + 1
            if seen[g] > keep:
                e = {**e, "folded": True}
        out.append(e)
    for g, n in seen.items():
        if n > keep:
            last = max(i for i, e in enumerate(out) if e.get("folded") and (
                (g == "link" and e["kind"] == "link") or e["id"].startswith(g + "-")))
            first = next(e for e in out if e.get("folded") and ((g == "link" and e["kind"] == "link") or e["id"].startswith(g + "-")))
            out.insert(last + 1, {"id": f"{g}-more", "order": first["order"], "kind": "summary", "lat": None, "lng": None,
                                  "cause": first["cause"], "text": f"...and {n - keep} {groups[g][0]}."})
    return out


# ---------------------------------------------------------------- live
def scenario_from_live(weather):
    """The weather right now, zone by zone, as a scenario."""
    zs = weather["zones"]
    return {
        "label": "Live weather",
        "rain_mm_h": max(z["rain_mm_h"] for z in zs),
        "rain_by_zone": {z["zone_id"]: z["rain_mm_h"] for z in zs},
        "duration_h": float(max(1, max(z["wet_hours_next_6h"] for z in zs))),
        "temp_c": max(z["apparent_c"] for z in zs),
        "wind_kmh": max(z["gust_kmh"] for z in zs),
    }


if __name__ == "__main__":
    import json
    m = model()
    print(json.dumps(m.summary()["learned"], indent=1))
