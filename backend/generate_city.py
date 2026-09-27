"""
Phase 1 (revised) — Synthetic city generator with learnable structure.

The first version filled zones on a smooth linear rule. A model trained on that
would just recover the rule and report a near-perfect score that means nothing.

This version generates occupancy from several interacting effects, so there are
genuine patterns for a model to find and a real gap between a naive trend
projection and a fitted model:

  1. Baseline drift        slow underlying fill as the event weekend approaches
  2. Event pull            zones near a venue surge before an event starts and
                           release after it ends
  3. Daily rhythm          check-in peaks in the late afternoon, check-out in
                           the morning
  4. Transport coupling    zones on congested links fill more slowly, because
                           people cannot physically get there
  5. Weather               a rain window suppresses movement and holds people
                           in place
  6. Zone character        each zone has its own responsiveness to event demand
  7. Noise                 observation noise so nothing is perfectly predictable

Run:  python generate_city.py
Out:  data/zones.csv, data/occupancy_timeline.csv, data/transport.csv,
      data/events.csv, data/weather.csv
"""

import math
import os
import random

import numpy as np
import pandas as pd

RNG_SEED = 42
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Simulate several days at hourly resolution so there is enough history to
# train on and a daily cycle for the model to learn.
DAYS = 6
HOURS = DAYS * 24

# --------------------------------------------------------------------------
# Zones
# --------------------------------------------------------------------------
# id, name, lat, lng, capacity, start_occupancy, nightly_rate, is_venue
ZONES = [
    ("Z01", "Stadium Precinct",  19.0760, 72.8777, 4200, 0.38, 6800, True),
    ("Z02", "Riverside East",    19.0896, 72.8656, 3100, 0.22, 4200, False),
    ("Z03", "Old Town",          19.0650, 72.8890, 2400, 0.31, 5200, False),
    ("Z04", "Metro Interchange", 19.0812, 72.8501, 1800, 0.34, 4900, False),
    ("Z05", "North Gate",        19.1020, 72.8720, 2900, 0.18, 5100, False),
    ("Z06", "Airport Corridor",  19.0990, 72.8340, 3600, 0.26, 7400, False),
    ("Z07", "Lakeside",          19.0540, 72.8620, 1500, 0.14, 3800, False),
    ("Z08", "Business District", 19.0700, 72.8450, 2700, 0.36, 8100, False),
    ("Z09", "South Market",      19.0480, 72.8800, 2100, 0.24, 3600, False),
    ("Z10", "West End",          19.0870, 72.8210, 2000, 0.16, 4400, False),
]

# The two venues. Seated capacity is what the building holds; gate throughput is
# how fast people can physically be processed through the doors, which is the
# constraint that actually creates queues. Exit is faster than entry because
# there is no security check on the way out.
VENUES = {
    # zone: (name, seated_capacity, gate_throughput_per_hour)
    "Z01": ("Central Stadium", 55000, 19000),
    "Z03": ("Old Town Arena",  24000,  9000),
}

EXIT_THROUGHPUT_MULTIPLIER = 1.6

# Named gates per venue, on a normalised 0-100 floor-plan grid. A single
# entry/exit pair gives the redirect engine nowhere to redirect to, so every
# venue gets at least 5, split unevenly like a real building (a grand main
# gate plus smaller side gates) rather than N identical doors.
#
# `throughput_share` sizes the gate's physical capacity. `default_exit_share`
# is a separate number: the share of the crowd that instinctively heads for
# that gate on the way out, independent of how big it is. People leave through
# the door they remember arriving through, which is disproportionately the
# main gate — that mismatch between "how big" and "how popular" is exactly
# what overcrowds a main exit even when smaller side gates sit half-empty,
# and it is the reason redirect is worth anything: capacity alone would make
# every gate status identical.
#
# zone: [(gate_id, name, type, throughput_share, default_exit_share, x, y), ...]
GATES = {
    "Z01": [
        ("Z01-G1", "Main Gate",       "both", 0.34, 0.50, 50,  5),
        ("Z01-G2", "North Gate",      "both", 0.18, 0.14, 85, 30),
        ("Z01-G3", "South Gate",      "both", 0.18, 0.14, 15, 30),
        ("Z01-G4", "East Concourse",  "both", 0.15, 0.11, 90, 70),
        ("Z01-G5", "West Concourse",  "both", 0.15, 0.11, 10, 70),
    ],
    "Z03": [
        ("Z03-G1", "Main Entrance",   "both", 0.32, 0.48, 50,  5),
        ("Z03-G2", "North Annex",     "both", 0.19, 0.15, 80, 40),
        ("Z03-G3", "South Annex",     "both", 0.19, 0.15, 20, 40),
        ("Z03-G4", "Rear Exit",       "both", 0.15, 0.12, 50, 95),
        ("Z03-G5", "VIP Gate",        "both", 0.15, 0.10, 50, 50),
    ],
}

# Events across the window. Zones near these surge beforehand.
# Two are deliberately sold beyond the building's capacity, which is the
# situation the problem statement calls a venue capacity limit: the shortfall
# has to be absorbed by the city, not the venue.
EVENTS = [
    ("E1", "Opening Ceremony", "Z01", 30,  33, 48000),
    ("E2", "Group Match A",    "Z01", 54,  57, 41000),
    ("E3", "Cultural Night",   "Z03", 66,  70, 21000),
    ("E4", "Group Match B",    "Z01", 78,  81, 44000),
    ("E5", "Semi Final",       "Z01", 102, 105, 58000),
    ("E6", "Closing Concert",  "Z03", 114, 118, 33000),
]


def haversine_km(lat1, lng1, lat2, lng2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def build_zones():
    venue = next(z for z in ZONES if z[7])
    rows = []
    for zid, name, lat, lng, cap, start, rate, is_venue in ZONES:
        rows.append({
            "zone_id": zid,
            "name": name,
            "lat": lat,
            "lng": lng,
            "hotel_capacity": cap,
            "avg_nightly_rate": rate,
            "is_venue_zone": is_venue,
            "distance_to_venue_km": round(haversine_km(lat, lng, venue[2], venue[3]), 2),
            # How strongly this zone responds to event demand. Some areas soak
            # up overflow readily, others barely move — a per-zone trait the
            # model has to learn rather than a global constant.
            "demand_elasticity": round(random.uniform(0.55, 1.45), 3),
        })
    return pd.DataFrame(rows)


def build_events():
    """
    The schedule, carrying the venue constraints each event runs against.

    Capacity travels with the event rather than sitting in a separate table
    because every question about an event ("will everyone get in", "how long to
    clear the gates") needs both numbers at once.
    """
    rows = []
    for e, n, z, s, t, a in EVENTS:
        vname, vcap, vgate = VENUES[z]
        rows.append({
            "event_id": e,
            "name": n,
            "zone_id": z,
            "start_hour": s,
            "end_hour": t,
            "expected_attendance": a,
            "venue_name": vname,
            "venue_capacity": vcap,
            "gate_throughput_per_hour": vgate,
            "exit_throughput_per_hour": int(vgate * EXIT_THROUGHPUT_MULTIPLIER),
        })
    return pd.DataFrame(rows)


def build_gates():
    """
    Per-gate capacity, derived from each venue's total gate/exit throughput
    split by the share defined in GATES. Kept as a share rather than an
    absolute number so it stays correct if VENUES' throughput changes.

    `status` here is the organizer-configured setting (open by default; an
    organizer can take a gate fully out of service, e.g. for maintenance).
    It is distinct from the live, auto-computed open/restricted/closed state
    the crowd engine derives per hour in engine/venues.py — that one reacts
    to actual queue and exit load, this one is a standing on/off switch.
    """
    rows = []
    for zid, gate_list in GATES.items():
        _vname, _vcap, vgate = VENUES[zid]
        vexit = int(vgate * EXIT_THROUGHPUT_MULTIPLIER)
        for gid, name, gtype, share, exit_share, x, y in gate_list:
            rows.append({
                "gate_id": gid,
                "venue_zone_id": zid,
                "name": name,
                "type": gtype,
                "entry_capacity_per_hour": int(vgate * share),
                "exit_capacity_per_hour": int(vexit * share),
                "default_exit_share": exit_share,
                "x": x,
                "y": y,
                "status": "open",
            })
    return pd.DataFrame(rows)


def build_floor_zones():
    """
    Floor-plan zones inside each venue: one bowl/hall (where the crowd sits
    during the event) plus one concourse per gate (where the crowd queues
    before it and clusters after it). Concourse capacity is sized as a
    multiple of that gate's hourly throughput — a physical space that has to
    hold more than one hour's worth of people passing through it.

    Concourse coordinates sit a short distance in from their gate, toward the
    centre of the floor plan, so a concourse tile is visually "in front of"
    its gate rather than sitting on top of it.
    """
    CONCOURSE_CAPACITY_HOURS = 2.5
    rows = []

    for zid, gate_list in GATES.items():
        _vname, vcap, _vgate = VENUES[zid]
        rows.append({
            "floor_zone_id": f"{zid}-BOWL",
            "venue_zone_id": zid,
            "name": "Main Bowl" if zid == "Z01" else "Main Hall",
            "kind": "bowl",
            "capacity": vcap,
            "adjacent_gate_id": None,
            "x": 50,
            "y": 50,
        })

        for gid, name, _gtype, share, _exit_share, x, y in gate_list:
            entry_cap = int(VENUES[zid][2] * share)
            cx = x + (50 - x) * 0.25
            cy = y + (50 - y) * 0.25
            zone_name = name if "concourse" in name.lower() else f"{name} Concourse"
            rows.append({
                "floor_zone_id": f"{gid}-CONCOURSE",
                "venue_zone_id": zid,
                "name": zone_name,
                "kind": "concourse",
                "capacity": int(entry_cap * CONCOURSE_CAPACITY_HOURS),
                "adjacent_gate_id": gid,
                "x": round(cx),
                "y": round(cy),
            })

    return pd.DataFrame(rows)


def build_weather():
    """
    A few rain windows. Rain suppresses movement between zones, which slows
    how fast occupancy redistributes.
    """
    rows = []
    rain_windows = [(20, 26), (60, 64), (96, 103)]
    for h in range(HOURS + 1):
        raining = any(a <= h <= b for a, b in rain_windows)
        rows.append({
            "hour_offset": h,
            "is_raining": int(raining),
            "temp_c": round(28 - (4 if raining else 0) + 4 * math.sin(h / 24 * 2 * math.pi), 1),
        })
    return pd.DataFrame(rows)


def build_transport(zones_df):
    """
    The link network: which zones connect, by what mode, and at what capacity.

    Static topology only. Load varies by hour and lives in build_transport_load.
    """
    venue_id = zones_df.loc[zones_df["is_venue_zone"], "zone_id"].iloc[0]
    dist = dict(zip(zones_df["zone_id"], zones_df["distance_to_venue_km"]))
    max_dist = max(dist.values()) or 1.0

    zone_ids = list(zones_df["zone_id"])
    pos = {r["zone_id"]: (r["lat"], r["lng"]) for _, r in zones_df.iterrows()}

    # Metro Interchange is a hub by name, so make it one by topology too.
    HUB_ID = "Z04"

    def near(a, b):
        return haversine_km(*pos[a], *pos[b])

    # Build the edge set first, then cost it. Three kinds of connection, which
    # together give a mesh rather than a star:
    #
    #   spokes     every zone reaches the venue directly, the corridor most
    #              visitors actually travel
    #   hub        every zone reaches the interchange, the obvious way round
    #   orbital    each zone joins its two nearest neighbours, so there is a
    #              way between adjacent districts without going via the centre
    #
    # Without the last two the network is a star, every journey runs through
    # the stadium, and "take a different route" has no meaning.
    pairs = set()

    for z in zone_ids:
        if z != venue_id:
            pairs.add(tuple(sorted((venue_id, z))))
        if z != HUB_ID:
            pairs.add(tuple(sorted((HUB_ID, z))))

    for z in zone_ids:
        others = sorted((o for o in zone_ids if o != z), key=lambda o: near(z, o))
        for o in others[:2]:
            pairs.add(tuple(sorted((z, o))))

    # A few extra chords, so the mesh is not perfectly regular.
    for i, a in enumerate(zone_ids):
        for b in zone_ids[i + 1:]:
            if random.random() < 0.18:
                pairs.add((a, b))

    modes = ["metro", "bus", "shuttle"]
    rows = []
    for link_id, (a, b) in enumerate(sorted(pairs), start=1):
        proximity = 1.0 - (min(dist[a], dist[b]) / max_dist)
        serves_venue = venue_id in (a, b)

        # Trunk routes carry more than orbital ones.
        if serves_venue or HUB_ID in (a, b):
            capacity = random.choice([2400, 3000, 3600])
            mode = random.choice(["metro", "metro", "shuttle"])
        else:
            capacity = random.choice([1200, 1800, 2400])
            mode = random.choice(modes)

        rows.append({
            "link_id": f"L{link_id:03d}",
            "from_zone": a,
            "to_zone": b,
            "mode": mode,
            "capacity_per_hour": capacity,
            "serves_venue": int(serves_venue),
            # Quiet-hours baseline before any event or commuter effect.
            "base_load": round(random.uniform(0.16, 0.30) + proximity * 0.16, 3),
            "proximity": round(proximity, 3),
        })

    return pd.DataFrame(rows)


def build_transport_load(transport_df, zones_df, events_df, weather_df):
    """
    Load on every link, hour by hour.

    Transport is spikier than accommodation. A link fills and empties around an
    event rather than drifting, so the model is dominated by two surges: people
    travelling in beforehand, and a sharper one leaving afterwards. Links that
    serve the venue take the brunt of both.
    """
    rain = dict(zip(weather_df["hour_offset"], weather_df["is_raining"]))
    rows = []

    for _, link in transport_df.iterrows():
        for hour in range(HOURS + 1):
            load = link["base_load"]

            # Commuter shape: morning and evening peaks, dead overnight.
            h = hour % 24
            load += 0.16 * math.exp(-((h - 9) ** 2) / 8)
            load += 0.20 * math.exp(-((h - 18) ** 2) / 10)
            if h < 5:
                load *= 0.45

            # Event traffic.
            for _, ev in events_df.iterrows():
                venue_zone = ev["zone_id"]
                touches = venue_zone in (link["from_zone"], link["to_zone"])
                # Links not touching the venue still carry some spillover.
                weight = 1.0 if touches else 0.30 * link["proximity"]
                scale = (ev["expected_attendance"] / 50000) * weight

                hours_until = ev["start_hour"] - hour
                if 0 <= hours_until <= 6:
                    # Inbound surge, sharpest in the last two hours.
                    load += 0.42 * scale * ((6 - hours_until) / 6) ** 2
                elif 0 <= hour - ev["end_hour"] <= 3:
                    # Egress is sharper than arrival: everyone leaves at once.
                    load += 0.58 * scale * math.exp(-(hour - ev["end_hour"]) / 1.4)

            # Rain pushes people off foot and onto transport.
            if rain.get(hour, 0):
                load *= 1.18

            load += np.random.normal(0, 0.022)
            load = float(np.clip(load, 0.05, 1.00))

            rows.append({
                "link_id": link["link_id"],
                "hour_offset": hour,
                "current_load": int(link["capacity_per_hour"] * load),
                "capacity_per_hour": int(link["capacity_per_hour"]),
                "load_pct": round(load * 100, 2),
            })

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Occupancy — where the learnable structure lives
# --------------------------------------------------------------------------
def _event_pull(hour, zone_lat, zone_lng, zones_df, events_df):
    """
    Demand pressure from upcoming events.

    Builds in the hours before an event and decays after it ends. Falls off with
    distance from the event venue, so a concert in Old Town pulls differently
    from a match at the stadium. This is the main non-linear signal in the data.
    """
    pull = 0.0

    for _, ev in events_df.iterrows():
        venue = zones_df[zones_df["zone_id"] == ev["zone_id"]].iloc[0]
        d = haversine_km(zone_lat, zone_lng, venue["lat"], venue["lng"])

        # Distance decay — beyond ~6km an event barely registers.
        prox = math.exp(-d / 2.5)

        hours_until = ev["start_hour"] - hour

        if 0 <= hours_until <= 18:
            # Ramp up, steepest right before the event.
            ramp = (18 - hours_until) / 18
            intensity = ramp ** 1.7
        elif ev["start_hour"] <= hour <= ev["end_hour"]:
            intensity = 1.0
        elif 0 < hour - ev["end_hour"] <= 10:
            # Release after the event.
            intensity = -0.45 * math.exp(-(hour - ev["end_hour"]) / 4)
        else:
            intensity = 0.0

        scale = ev["expected_attendance"] / 50000
        pull += intensity * prox * scale

    return pull


def _daily_rhythm(hour):
    """Check-in peaks late afternoon, check-out drains in the morning."""
    h = hour % 24
    checkin = 0.9 * math.exp(-((h - 17) ** 2) / 18)
    checkout = -0.7 * math.exp(-((h - 9) ** 2) / 12)
    return checkin + checkout


def build_occupancy(zones_df, events_df, weather_df, transport_df):
    # Mean transport load touching each zone — congested zones fill slower.
    load_by_zone = {}
    for zid in zones_df["zone_id"]:
        touching = transport_df[
            (transport_df["from_zone"] == zid) | (transport_df["to_zone"] == zid)
        ]
        load_by_zone[zid] = touching["base_load"].mean() if len(touching) else 0.35

    rain = dict(zip(weather_df["hour_offset"], weather_df["is_raining"]))
    start_by_zone = dict(zip([x[0] for x in ZONES], [x[5] for x in ZONES]))
    max_dist = zones_df["distance_to_venue_km"].max() or 1.0

    rows = []

    for _, z in zones_df.iterrows():
        occ = start_by_zone[z["zone_id"]]
        proximity = 1.0 - (z["distance_to_venue_km"] / max_dist)

        for hour in range(HOURS + 1):
            # 1. Baseline drift toward the event weekend.
            baseline = 0.0022 + 0.0026 * proximity

            # 2. Event pull, scaled by this zone own responsiveness.
            pull = _event_pull(hour, z["lat"], z["lng"], zones_df, events_df)
            event_term = pull * 0.011 * z["demand_elasticity"]

            # 3. Daily check-in / check-out rhythm.
            rhythm = _daily_rhythm(hour) * 0.008

            # 4. Transport congestion damps inflow.
            congestion_damp = 1.0 - (load_by_zone[z["zone_id"]] * 0.35)

            # 5. Rain holds people where they are.
            rain_damp = 0.62 if rain.get(hour, 0) else 1.0

            delta = (baseline + event_term + rhythm) * congestion_damp * rain_damp

            # 6. Observation noise.
            delta += np.random.normal(0, 0.0032)

            occ = float(np.clip(occ + delta, 0.05, 1.00))

            rows.append({
                "zone_id": z["zone_id"],
                "hour_offset": hour,
                "occupancy": int(z["hotel_capacity"] * occ),
                "hotel_capacity": int(z["hotel_capacity"]),
                "occupancy_pct": round(occ * 100, 2),
                "is_raining": rain.get(hour, 0),
                "hour_of_day": hour % 24,
            })

    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    zones = build_zones()
    events = build_events()
    gates = build_gates()
    floor_zones = build_floor_zones()
    weather = build_weather()
    transport = build_transport(zones)
    transport_load = build_transport_load(transport, zones, events, weather)
    occupancy = build_occupancy(zones, events, weather, transport)

    zones.to_csv(os.path.join(OUT_DIR, "zones.csv"), index=False)
    events.to_csv(os.path.join(OUT_DIR, "events.csv"), index=False)
    gates.to_csv(os.path.join(OUT_DIR, "gates.csv"), index=False)
    floor_zones.to_csv(os.path.join(OUT_DIR, "floor_zones.csv"), index=False)
    weather.to_csv(os.path.join(OUT_DIR, "weather.csv"), index=False)
    transport.to_csv(os.path.join(OUT_DIR, "transport.csv"), index=False)
    transport_load.to_csv(os.path.join(OUT_DIR, "transport_load.csv"), index=False)
    occupancy.to_csv(os.path.join(OUT_DIR, "occupancy_timeline.csv"), index=False)

    print(f"{DAYS} days at hourly resolution\n")
    print(f"zones.csv               {len(zones):>6} rows")
    print(f"events.csv              {len(events):>6} rows")
    print(f"gates.csv               {len(gates):>6} rows")
    print(f"floor_zones.csv         {len(floor_zones):>6} rows")
    print(f"weather.csv             {len(weather):>6} rows")
    print(f"transport.csv           {len(transport):>6} rows")
    print(f"transport_load.csv      {len(transport_load):>6} rows")
    print(f"occupancy_timeline.csv  {len(occupancy):>6} rows")
    print(f"\nWritten to {OUT_DIR}")

    peak = occupancy.groupby("zone_id")["occupancy_pct"].max().sort_values(ascending=False)
    names = dict(zip(zones["zone_id"], zones["name"]))
    print("\nPeak occupancy reached:")
    for zid, pct in peak.items():
        bar = "#" * int(pct / 4)
        print(f"  {zid}  {names[zid]:<20} {pct:>6.1f}%  {bar}")


if __name__ == "__main__":
    main()
