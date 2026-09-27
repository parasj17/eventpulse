"""
Phase 2b — Machine-learning occupancy forecaster.

Predicts a zone's occupancy percentage HORIZON hours ahead using gradient
boosting, and benchmarks it against the Phase 2 trend-line baseline so the
claim "the model helps" comes with a number attached.

Three design decisions worth defending in a demo:

* The split is chronological, not random. Shuffling time-series rows lets a
  model see the future through neighbouring hours and produces a flattering
  score that would not survive contact with reality.

* Every feature is knowable at prediction time — past occupancy, the published
  event schedule, the weather forecast, and static zone attributes. Nothing
  leaks backwards from the target.

* The horizon is 6 hours. That is long enough for an organiser to act on and
  short enough to stay accurate.

Run:  python -m engine.model
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")

HORIZON = 6
TRAIN_FRAC = 0.70
LAGS = [1, 2, 3, 6, 12, 24]
ROLL_WINDOWS = [3, 6, 12]


def haversine_km(lat1, lng1, lat2, lng2):
    import math
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def load_frames(data_dir=DATA_DIR):
    occ = pd.read_csv(os.path.join(data_dir, "occupancy_timeline.csv"))
    zones = pd.read_csv(os.path.join(data_dir, "zones.csv"))
    events = pd.read_csv(os.path.join(data_dir, "events.csv"))
    return occ, zones, events


def build_features(occ, zones, events, horizon=HORIZON):
    """One row per (zone, hour): features known now, target `horizon` hours on."""
    df = occ.merge(
        zones[["zone_id", "lat", "lng", "distance_to_venue_km",
               "avg_nightly_rate", "is_venue_zone", "demand_elasticity"]],
        on="zone_id", how="left",
    ).sort_values(["zone_id", "hour_offset"]).reset_index(drop=True)

    grp = df.groupby("zone_id")["occupancy_pct"]

    # --- History -----------------------------------------------------------
    for lag in LAGS:
        df[f"lag_{lag}"] = grp.shift(lag)

    for w in ROLL_WINDOWS:
        shifted = grp.shift(1)
        df[f"roll_mean_{w}"] = shifted.rolling(w, min_periods=2).mean().reset_index(level=0, drop=True)
        df[f"roll_std_{w}"] = shifted.rolling(w, min_periods=2).std().reset_index(level=0, drop=True)

    # Momentum at three timescales.
    df["delta_1"] = df["occupancy_pct"] - df["lag_1"]
    df["delta_3"] = df["occupancy_pct"] - df["lag_3"]
    df["delta_6"] = df["occupancy_pct"] - df["lag_6"]

    # How much room is left — bounds how far it can still climb.
    df["headroom_pct"] = 100.0 - df["occupancy_pct"]

    # --- Event schedule (published ahead of time, so safe to use) -----------
    zpos = zones.set_index("zone_id")[["lat", "lng"]].to_dict("index")
    ev_dist = {
        (z, e): haversine_km(zpos[z]["lat"], zpos[z]["lng"], zpos[e]["lat"], zpos[e]["lng"])
        for z in zones["zone_id"] for e in events["zone_id"].unique()
    }

    ev_sorted = events.sort_values("start_hour")
    starts = ev_sorted["start_hour"].to_numpy()

    hours_to = np.empty(len(df))
    attendance = np.zeros(len(df))
    ev_km = np.empty(len(df))

    for i, (zid, hour) in enumerate(zip(df["zone_id"].to_numpy(), df["hour_offset"].to_numpy())):
        nxt_idx = np.searchsorted(starts, hour, side="left")
        if nxt_idx < len(ev_sorted):
            ev = ev_sorted.iloc[nxt_idx]
            hours_to[i] = ev["start_hour"] - hour
            attendance[i] = ev["expected_attendance"]
            ev_km[i] = ev_dist[(zid, ev["zone_id"])]
        else:
            hours_to[i], attendance[i], ev_km[i] = 999.0, 0.0, 99.0

    df["hours_to_next_event"] = hours_to
    df["next_event_attendance"] = attendance
    df["next_event_distance_km"] = ev_km

    # Big, close and soon = most pressure. One combined signal.
    df["event_pressure"] = (
        (df["next_event_attendance"] / 50000)
        / (1 + df["next_event_distance_km"])
        / (1 + df["hours_to_next_event"] / 6)
    )

    # --- Time of day, encoded cyclically so hour 23 neighbours hour 0 ------
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)

    df["is_venue_zone"] = df["is_venue_zone"].astype(int)

    # --- Target ------------------------------------------------------------
    # Predict the CHANGE over the horizon, not the absolute level. Predicting
    # the level lets the model score well by echoing its own input, which is
    # why the first version lost to a trend line. Modelling the delta forces it
    # to learn what actually drives movement.
    future = df.groupby("zone_id")["occupancy_pct"].shift(-horizon)
    df["target_level"] = future
    df["target"] = future - df["occupancy_pct"]

    return df.dropna(subset=["target", "lag_24"]).reset_index(drop=True)


FEATURES = (
    ["occupancy_pct", "hour_of_day", "hour_sin", "hour_cos", "is_raining",
     "distance_to_venue_km", "avg_nightly_rate", "is_venue_zone",
     "demand_elasticity", "headroom_pct", "delta_1", "delta_3", "delta_6",
     "hours_to_next_event", "next_event_attendance", "next_event_distance_km",
     "event_pressure"]
    + [f"lag_{l}" for l in LAGS]
    + [f"roll_mean_{w}" for w in ROLL_WINDOWS]
    + [f"roll_std_{w}" for w in ROLL_WINDOWS]
)


def chronological_split(df, train_frac=TRAIN_FRAC):
    cutoff = df["hour_offset"].quantile(train_frac)
    return df[df["hour_offset"] <= cutoff], df[df["hour_offset"] > cutoff], cutoff


def baseline_predict(test, horizon=HORIZON):
    """
    The Phase 2 trend line, as the thing to beat: assume the last 3 hours' rate
    of change continues unchanged.
    """
    rate = (test["occupancy_pct"] - test["lag_3"]) / 3.0
    return (test["occupancy_pct"] + rate * horizon).clip(0, 115)


def train(data_dir=DATA_DIR, horizon=HORIZON, save=True, verbose=True):
    occ, zones, events = load_frames(data_dir)
    df = build_features(occ, zones, events, horizon)
    train_df, test_df, cutoff = chronological_split(df)

    model = HistGradientBoostingRegressor(
        max_iter=400,
        learning_rate=0.06,
        max_depth=6,
        min_samples_leaf=10,
        l2_regularization=1.0,
        random_state=42,
    )
    model.fit(train_df[FEATURES], train_df["target"])

    # Model predicts a delta; add it back to get a level we can compare
    # against the baseline on the same scale.
    pred_delta = model.predict(test_df[FEATURES])
    pred = np.clip(test_df["occupancy_pct"].to_numpy() + pred_delta, 0, 115)

    base = baseline_predict(test_df, horizon)
    truth = test_df["target_level"]

    metrics = {
        "horizon_hours": horizon,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "split_at_hour": float(cutoff),
        "model_mae": float(mean_absolute_error(truth, pred)),
        "model_r2": float(r2_score(truth, pred)),
        "baseline_mae": float(mean_absolute_error(truth, base)),
        "baseline_r2": float(r2_score(truth, base)),
    }
    metrics["improvement_pct"] = round(
        (metrics["baseline_mae"] - metrics["model_mae"]) / metrics["baseline_mae"] * 100, 1
    )

    if save:
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({"model": model, "features": FEATURES,
                         "horizon": horizon, "metrics": metrics}, f)

    if verbose:
        print(f"Target: occupancy {horizon} hours ahead")
        print(f"Train hours 0–{cutoff:.0f} ({metrics['train_rows']} rows) | "
              f"test hours {cutoff:.0f}+ ({metrics['test_rows']} rows)")
        print(f"Split is chronological — no future leakage\n")

        print(f"{'':<24}{'MAE':>10}{'R²':>10}")
        print(f"{'-' * 44}")
        print(f"{'Trend baseline':<24}{metrics['baseline_mae']:>10.3f}{metrics['baseline_r2']:>10.3f}")
        print(f"{'Gradient boosting':<24}{metrics['model_mae']:>10.3f}{metrics['model_r2']:>10.3f}")
        print(f"{'-' * 44}")
        print(f"\nMAE reduced by {metrics['improvement_pct']}% "
              f"({metrics['baseline_mae']:.2f} → {metrics['model_mae']:.2f} "
              f"percentage points of occupancy)")

    return model, df, metrics


def feature_importance(model, df, top_n=10):
    """Permutation importance on the held-out set."""
    from sklearn.inspection import permutation_importance

    _, test_df, _ = chronological_split(df)
    r = permutation_importance(
        model, test_df[FEATURES], test_df["target"],
        n_repeats=5, random_state=42, scoring="neg_mean_absolute_error",
    )
    order = np.argsort(r.importances_mean)[::-1][:top_n]
    return [(FEATURES[i], float(r.importances_mean[i])) for i in order]


def load_model(path=MODEL_PATH):
    """Load the trained model for use by the API. Returns None if not trained."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_all(data_dir=DATA_DIR, current_hour=None):
    """
    Predict every zone's occupancy HORIZON hours ahead from `current_hour`.

    Returns a list of dicts the API can serve directly.
    """
    bundle = load_model()
    if bundle is None:
        return []

    model, feats, horizon = bundle["model"], bundle["features"], bundle["horizon"]

    occ, zones, events = load_frames(data_dir)
    df = build_features(occ, zones, events, horizon)

    if current_hour is None:
        current_hour = int(df["hour_offset"].max())

    snap = df[df["hour_offset"] == current_hour]
    if snap.empty:
        return []

    pred_deltas = model.predict(snap[feats])
    preds = snap["occupancy_pct"].to_numpy() + pred_deltas
    names = dict(zip(zones["zone_id"], zones["name"]))

    out = []
    for (_, row), p in zip(snap.iterrows(), preds):
        out.append({
            "zone_id": row["zone_id"],
            "zone_name": names.get(row["zone_id"], row["zone_id"]),
            "current_occupancy_pct": round(float(row["occupancy_pct"]), 2),
            "predicted_occupancy_pct": round(float(np.clip(p, 0, 115)), 2),
            "horizon_hours": horizon,
            "predicted_change": round(float(p - row["occupancy_pct"]), 2),
        })

    return sorted(out, key=lambda r: r["predicted_occupancy_pct"], reverse=True)


if __name__ == "__main__":
    model, df, metrics = train()

    print("\nWhat the model is actually using (permutation importance):")
    imps = feature_importance(model, df)
    top = imps[0][1] or 1.0
    for name, score in imps:
        bar = "#" * max(1, int(score / top * 30))
        print(f"  {name:<26}{score:>7.3f}  {bar}")

    print(f"\nSaved to {MODEL_PATH}")
