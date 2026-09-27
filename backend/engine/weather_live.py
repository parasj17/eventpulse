"""
Live weather for every zone, from Open-Meteo (free, no key).

One request covers all zones: current conditions, the last six hours (rain that has
already fallen is what floods streets), and the next 24 hours of forecast. Each
location also comes back with its ground elevation, which the weather twin uses for
flood risk. Results are cached for ten minutes; if the service can't be reached the
caller gets `available: False` and the twin falls back to calm weather, saying so.

Run:  python -m engine.weather_live
"""

import os
import time

import httpx

API_URL = os.environ.get("EVENTPULSE_WEATHER_URL", "https://api.open-meteo.com/v1/forecast")
CACHE_SECONDS = int(os.environ.get("EVENTPULSE_WEATHER_CACHE", "600"))
TIMEOUT_S = 12

CURRENT = ["temperature_2m", "apparent_temperature", "relative_humidity_2m", "precipitation", "rain",
           "weather_code", "cloud_cover", "wind_speed_10m", "wind_gusts_10m"]
HOURLY = ["precipitation", "precipitation_probability", "temperature_2m", "apparent_temperature", "wind_gusts_10m",
          "weather_code"]

_cache = {"at": 0, "key": None, "data": None, "backoff_until": 0}
LAST_GOOD_FILE = "weather_last.json"
STALE_OK_S = 6 * 3600          # a saved reading this old can still stand in, marked stale
BACKOFF_S = 600                # after the service refuses (e.g. rate limit), wait this long


def _last_good_path():
    from engine.forecast import DATA_DIR
    return os.path.join(DATA_DIR, LAST_GOOD_FILE)


def _load_last_good(now):
    import json
    try:
        with open(_last_good_path()) as f:
            d = json.load(f)
        if now - d.get("fetched_at", 0) <= STALE_OK_S:
            return d
    except (OSError, ValueError):
        pass
    return None

WMO = {0: "Clear", 1: "Mostly clear", 2: "Partly cloudy", 3: "Overcast", 45: "Fog", 48: "Fog",
       51: "Light drizzle", 53: "Drizzle", 55: "Heavy drizzle", 56: "Freezing drizzle", 57: "Freezing drizzle",
       61: "Light rain", 63: "Rain", 65: "Heavy rain", 66: "Freezing rain", 67: "Freezing rain",
       71: "Snow", 73: "Snow", 75: "Heavy snow", 77: "Snow grains", 80: "Rain showers", 81: "Heavy showers",
       82: "Violent showers", 85: "Snow showers", 86: "Snow showers", 95: "Thunderstorm",
       96: "Thunderstorm with hail", 99: "Severe thunderstorm"}


def describe(code):
    return WMO.get(int(code), "Unknown") if code is not None else "Unknown"


def _num(v, default=0.0):
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def _zone_reading(z, loc):
    cur = loc.get("current", {}) or {}
    hourly = loc.get("hourly", {}) or {}
    times = hourly.get("time", []) or []
    now_key = (cur.get("time") or "")[:13]            # "YYYY-MM-DDTHH"
    idx = next((i for i, t in enumerate(times) if t[:13] == now_key), None)
    col = lambda name, i: _num((hourly.get(name) or [None] * len(times))[i]) if i is not None and i < len(times) else 0.0

    past = [col("precipitation", i) for i in range(max(0, (idx or 0) - 6), idx or 0)] if idx is not None else []
    ahead = []
    if idx is not None:
        for i in range(idx, min(len(times), idx + 24)):
            ahead.append({"time": times[i], "precip_mm": col("precipitation", i),
                          "prob_pct": col("precipitation_probability", i), "temp_c": col("temperature_2m", i),
                          "apparent_c": col("apparent_temperature", i), "gust_kmh": col("wind_gusts_10m", i),
                          "code": int(col("weather_code", i))})

    # "precipitation" in `current` is the last 15 minutes; as an hourly rate that is x4.
    interval_h = max(1, int(cur.get("interval") or 900)) / 3600
    rate_now = _num(cur.get("precipitation")) / interval_h
    next_hour = ahead[0]["precip_mm"] if ahead else 0.0
    return {
        "zone_id": z["zone_id"], "name": z["name"], "lat": float(z["lat"]), "lng": float(z["lng"]),
        "elevation_m": _num(loc.get("elevation"), 10.0),
        "time": cur.get("time"),
        "temp_c": _num(cur.get("temperature_2m")), "apparent_c": _num(cur.get("apparent_temperature")),
        "humidity_pct": _num(cur.get("relative_humidity_2m")), "cloud_pct": _num(cur.get("cloud_cover")),
        "wind_kmh": _num(cur.get("wind_speed_10m")), "gust_kmh": _num(cur.get("wind_gusts_10m")),
        "code": int(_num(cur.get("weather_code"))), "condition": describe(cur.get("weather_code")),
        # The intensity the twin works with: whichever is heavier, what is falling now or the coming hour.
        "rain_mm_h": round(max(rate_now, next_hour), 2),
        "rain_last_6h_mm": round(sum(past), 1),
        "rain_next_6h_mm": round(sum(a["precip_mm"] for a in ahead[:6]), 1),
        "max_prob_next_6h": max([a["prob_pct"] for a in ahead[:6]] or [0]),
        "wet_hours_next_6h": sum(1 for a in ahead[:6] if a["precip_mm"] >= 0.5),
        "next_24h": ahead,
    }


MET_URL = os.environ.get("EVENTPULSE_MET_URL", "https://api.met.no/weatherapi/locationforecast/2.0/compact")
MET_UA = "EventPulse/1.0 (hackathon demo; jadhavparas17@gmail.com)"
MET_CODES = {"clearsky": 0, "fair": 1, "partlycloudy": 2, "cloudy": 3, "fog": 45, "lightrain": 61, "rain": 63, "heavyrain": 65,
             "lightrainshowers": 80, "rainshowers": 80, "heavyrainshowers": 81, "rainandthunder": 95, "heavyrainandthunder": 95,
             "lightrainandthunder": 95, "rainshowersandthunder": 95, "heavyrainshowersandthunder": 99}


def _apparent(t, rh, wind_ms):
    """Feels-like temperature (Australian apparent temperature), for sources that don't give one."""
    import math
    e = rh / 100 * 6.105 * math.exp(17.27 * t / (237.7 + t))
    return t + 0.33 * e - 0.70 * wind_ms - 4.0


def _met(zones, now):
    """Backup source: MET Norway's free forecast, one request per zone, shaped like Open-Meteo's rows."""
    from datetime import datetime, timezone
    try:
        from engine.forecast import DATA_DIR
        import json
        with open(os.path.join(DATA_DIR, "zone_elevation.json")) as f:
            elev = json.load(f)
    except (OSError, ValueError):
        elev = {}
    rows = []
    for z in zones:
        r = httpx.get(MET_URL, params={"lat": round(float(z["lat"]), 4), "lon": round(float(z["lng"]), 4)},
                      headers={"User-Agent": MET_UA}, timeout=TIMEOUT_S)
        r.raise_for_status()
        ts = r.json()["properties"]["timeseries"]
        cur = ts[0]["data"]
        inst = cur["instant"]["details"]
        sym = ((cur.get("next_1_hours") or cur.get("next_6_hours") or {}).get("summary") or {}).get("symbol_code", "cloudy")
        code = MET_CODES.get(sym.split("_")[0], 3)
        ahead = []
        for t in ts[:24]:
            d = t["data"]
            n1 = (d.get("next_1_hours") or {})
            ahead.append({"time": t["time"][:16], "precip_mm": float((n1.get("details") or {}).get("precipitation_amount", 0.0)),
                          "prob_pct": 0.0, "temp_c": float(d["instant"]["details"].get("air_temperature", 0)),
                          "apparent_c": float(d["instant"]["details"].get("air_temperature", 0)),
                          "gust_kmh": float(d["instant"]["details"].get("wind_speed", 0)) * 3.6 * 1.4,
                          "code": MET_CODES.get(((n1.get("summary") or {}).get("symbol_code", "cloudy")).split("_")[0], 3)})
        for a in ahead:                                   # no probability from MET: rain in the hour counts as likely
            a["prob_pct"] = 80.0 if a["precip_mm"] >= 0.2 else 10.0
        t0, rh, ws = float(inst.get("air_temperature", 0)), float(inst.get("relative_humidity", 70)), float(inst.get("wind_speed", 0))
        rows.append({
            "zone_id": z["zone_id"], "name": z["name"], "lat": float(z["lat"]), "lng": float(z["lng"]),
            "elevation_m": float(elev.get(z["zone_id"], 10.0)), "time": ts[0]["time"][:16],
            "temp_c": t0, "apparent_c": round(_apparent(t0, rh, ws), 1), "humidity_pct": rh,
            "cloud_pct": float(inst.get("cloud_area_fraction", 0)), "wind_kmh": round(ws * 3.6, 1), "gust_kmh": round(ws * 3.6 * 1.4, 1),
            "code": code, "condition": describe(code),
            "rain_mm_h": round(ahead[0]["precip_mm"] if ahead else 0.0, 2),
            "rain_last_6h_mm": 0.0,                                   # MET gives forecasts only
            "rain_next_6h_mm": round(sum(a["precip_mm"] for a in ahead[:6]), 1),
            "max_prob_next_6h": max([a["prob_pct"] for a in ahead[:6]] or [0]),
            "wet_hours_next_6h": sum(1 for a in ahead[:6] if a["precip_mm"] >= 0.5),
            "next_24h": ahead,
        })
    return rows


def fetch(zones, now=None, force=False):
    """
    `zones`: dicts with zone_id, name, lat, lng. Returns
    {available, source, fetched_at, zones: [...], city: {...}} or {available: False, error}.
    """
    now = now or time.time()
    if not force and now < _cache["backoff_until"]:
        if _cache["data"] and _cache["data"].get("available") and now - _cache["at"] < CACHE_SECONDS:
            return _cache["data"]
        try:
            key = tuple((z["zone_id"], round(float(z["lat"]), 4), round(float(z["lng"]), 4)) for z in zones)
            return _publish(_met(zones, now), zones, now, key, "MET Norway (backup)")
        except Exception:
            pass
        last = _cache["data"] if _cache["data"] and _cache["data"].get("available") else _load_last_good(now)
        if last:
            return {**last, "stale": True}
        return {"available": False, "source": "Open-Meteo", "error": "Live weather is paused for a few minutes (rate limit)."}
    key = tuple((z["zone_id"], round(float(z["lat"]), 4), round(float(z["lng"]), 4)) for z in zones)
    if not force and _cache["data"] is not None and _cache["key"] == key and now - _cache["at"] < CACHE_SECONDS:
        return _cache["data"]

    params = {
        "latitude": ",".join(str(round(float(z["lat"]), 4)) for z in zones),
        "longitude": ",".join(str(round(float(z["lng"]), 4)) for z in zones),
        "current": ",".join(CURRENT), "hourly": ",".join(HOURLY),
        "past_hours": 6, "forecast_hours": 24, "timezone": "auto",
    }
    try:
        r = httpx.get(API_URL, params=params, timeout=TIMEOUT_S)
        r.raise_for_status()
        body = r.json()
    except Exception as e:                                   # network, HTTP or JSON: try the backup source
        try:
            rows = _met(zones, now)
            return _publish(rows, zones, now, key, "MET Norway (backup)")
        except Exception:
            pass
        _cache["backoff_until"] = now + BACKOFF_S
        out = {"available": False, "source": "Open-Meteo", "error": f"Live weather unavailable ({type(e).__name__})."}
        # A stale reading beats none: the last good one (in memory, or saved on disk across restarts).
        if _cache["data"] and _cache["data"].get("available") and now - _cache["at"] < STALE_OK_S:
            return {**_cache["data"], "stale": True}
        last = _load_last_good(now)
        return {**last, "stale": True} if last else out

    locs = body if isinstance(body, list) else [body]
    if len(locs) != len(zones):
        return {"available": False, "source": "Open-Meteo", "error": "Live weather came back incomplete."}
    return _publish([_zone_reading(z, loc) for z, loc in zip(zones, locs)], zones, now, key, "Open-Meteo")


def _publish(rows, zones, now, key, source):
    n = len(rows)
    city = {
        "temp_c": round(sum(r["temp_c"] for r in rows) / n, 1),
        "apparent_c": round(max(r["apparent_c"] for r in rows), 1),
        "humidity_pct": round(sum(r["humidity_pct"] for r in rows) / n),
        "rain_mm_h_max": max(r["rain_mm_h"] for r in rows),
        "rain_last_6h_mm_max": max(r["rain_last_6h_mm"] for r in rows),
        "rain_next_6h_mm_max": max(r["rain_next_6h_mm"] for r in rows),
        "max_prob_next_6h": max(r["max_prob_next_6h"] for r in rows),
        "gust_kmh_max": max(r["gust_kmh"] for r in rows),
        "condition": max(rows, key=lambda r: r["code"])["condition"],
        "code": max(r["code"] for r in rows),
    }
    data = {"available": True, "source": source, "fetched_at": now, "time": rows[0]["time"],
            "zones": rows, "city": city}
    # Ground elevation doesn't change: keep it, so flood risk stays right when offline.
    try:
        from engine.forecast import DATA_DIR
        import json
        with open(os.path.join(DATA_DIR, "zone_elevation.json"), "w") as f:
            json.dump({r["zone_id"]: r["elevation_m"] for r in rows}, f)
    except OSError:
        pass
    _cache.update(at=now, key=key, data=data)
    try:
        import json
        with open(_last_good_path() + ".tmp", "w") as f:
            json.dump(data, f)
        os.replace(_last_good_path() + ".tmp", _last_good_path())
    except OSError:
        pass
    return data


if __name__ == "__main__":
    import pandas as pd
    from engine.forecast import DATA_DIR
    zs = pd.read_csv(os.path.join(DATA_DIR, "zones.csv")).to_dict("records")
    w = fetch(zs)
    print(w["city"] if w.get("available") else w)
