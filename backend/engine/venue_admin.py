"""
Venue and event administration for organisers.

An organiser can create an event at a new venue (or add one to a venue they
already have), then lay out its gates on a floor plan. This module owns the
rules for that, so the API layer stays thin:

* **Ownership.** Every venue has an owner. The two seeded demo venues have no
  owner and are editable by any organiser; anything an organiser creates is
  theirs alone, so one business cannot change another's gates.
* **Where a venue lives.** A venue is attached to the nearest city zone that has
  no venue yet, because the city's occupancy and transport data is per zone.
  The exact pin the organiser chose is kept separately as the venue's site.
* **Gates stay consistent.** Adding, moving, resizing or deleting a gate keeps
  its concourse floor zone in step and rebalances how the crowd is assumed to
  split across exits, so the totals still add up to everyone leaving.

Data lives beside the rest, in CSVs: `venue_sites.csv` (new) and extra columns on
`events.csv`.
"""

import math
import os

import pandas as pd

from engine.forecast import DATA_DIR
from engine.recommend import haversine_km

SITES_CSV = "venue_sites.csv"
SITE_COLUMNS = ["venue_zone_id", "venue_name", "lat", "lng", "owner"]

MIN_GATES = 4
MAX_GATES = 24
MAX_DISTANCE_KM = 20.0            # a pin further than this from every zone is outside the covered area
FIRST_HOUR, LAST_HOUR = 24, 143   # the simulated run an event must fit inside
CONCOURSE_CAPACITY_HOURS = 2.5    # a concourse holds this many hours of its gate's throughput
ENTRY_SHARE_OF_CAPACITY = 0.35    # gate throughput per hour, as a share of venue capacity
EXIT_MULTIPLIER = 1.6             # nobody is being screened on the way out

# Gate layout for a brand-new venue: name, share of throughput, share of the crowd
# that instinctively heads for it on the way out, and its spot on the 0-100 plan.
# A big main gate plus smaller side gates, as in a real building.
DEFAULT_GATES = [
    ("Main Gate", 0.34, 0.50, 50, 5),
    ("North Gate", 0.18, 0.14, 85, 30),
    ("South Gate", 0.18, 0.14, 15, 30),
    ("East Gate", 0.15, 0.11, 90, 70),
    ("West Gate", 0.15, 0.11, 10, 70),
]


class VenueError(ValueError):
    """A problem worth showing to the organiser as-is."""


class Forbidden(VenueError):
    """The organiser is not allowed to change that venue."""


def _p(name, data_dir=None):
    return os.path.join(data_dir or DATA_DIR, name)


def _events(data_dir):
    df = pd.read_csv(_p("events.csv"))
    if "organizer" not in df.columns:
        df["organizer"] = ""
    df["organizer"] = df["organizer"].fillna("")
    return df


def _zones(data_dir):
    return pd.read_csv(_p("zones.csv"))


def _sites(data_dir):
    """The venue sites, seeded from the existing events on first use."""
    path = _p(SITES_CSV, data_dir)
    if os.path.exists(path):
        return pd.read_csv(path, dtype={"owner": str}).fillna({"owner": ""})

    events, zones = _events(data_dir), _zones(data_dir).set_index("zone_id")
    rows = []
    for zid, grp in events.groupby("zone_id"):
        rows.append({"venue_zone_id": zid, "venue_name": grp.iloc[0]["venue_name"],
                     "lat": float(zones.loc[zid, "lat"]), "lng": float(zones.loc[zid, "lng"]), "owner": ""})
    df = pd.DataFrame(rows, columns=SITE_COLUMNS)
    df.to_csv(path, index=False)
    return df


def list_sites(data_dir=None):
    """Every venue's site (name and pin), as plain records. Public: these are just
    where the venues are, which the map shows to everyone."""
    return _sites(data_dir).to_dict("records")


def can_edit(zone_id, email, data_dir=None):
    site = _sites(data_dir)
    row = site[site["venue_zone_id"] == zone_id]
    if row.empty:
        return False
    owner = row.iloc[0]["owner"]
    return owner == "" or owner == (email or "").strip().lower()


def require_edit(zone_id, email, data_dir=None):
    if not can_edit(zone_id, email, data_dir):
        raise Forbidden("That venue belongs to another organiser.")


def venue_zone_for_gate(gate_id, data_dir=None):
    gates = pd.read_csv(_p("gates.csv", data_dir))
    row = gates[gates["gate_id"] == gate_id]
    return None if row.empty else row.iloc[0]["venue_zone_id"]


def list_venues(email, data_dir=None):
    """The venues this organiser may edit, each with its events and gate count."""
    email = (email or "").strip().lower()
    sites = _sites(data_dir)
    mine = sites[(sites["owner"] == "") | (sites["owner"] == email)]
    events, gates = _events(data_dir), pd.read_csv(_p("gates.csv", data_dir))

    out = []
    for _, s in mine.iterrows():
        evs = events[events["zone_id"] == s["venue_zone_id"]].sort_values("start_hour")
        out.append({
            "venue_zone_id": s["venue_zone_id"], "venue_name": s["venue_name"],
            "lat": float(s["lat"]), "lng": float(s["lng"]),
            "owned": s["owner"] == email and email != "",
            "capacity": int(evs.iloc[0]["venue_capacity"]) if len(evs) else None,
            "gate_count": int((gates["venue_zone_id"] == s["venue_zone_id"]).sum()),
            "events": [{"event_id": e["event_id"], "name": e["name"], "start_hour": int(e["start_hour"]),
                        "end_hour": int(e["end_hour"]), "expected_attendance": int(e["expected_attendance"])}
                       for _, e in evs.iterrows()],
        })
    return out


def _nearest_free_zone(lat, lng, data_dir):
    zones, sites = _zones(data_dir), _sites(data_dir)
    taken = set(sites["venue_zone_id"])
    best = None
    for _, z in zones.iterrows():
        d = haversine_km(lat, lng, float(z["lat"]), float(z["lng"]))
        if best is None or d < best[1]:
            best = (z["zone_id"], d, z["zone_id"] in taken)
    if best is None or best[1] > MAX_DISTANCE_KM:
        raise VenueError("That location is outside the area this deployment covers.")

    free = [(haversine_km(lat, lng, float(z["lat"]), float(z["lng"])), z["zone_id"])
            for _, z in zones.iterrows() if z["zone_id"] not in taken]
    if not free:
        raise VenueError("Every zone already has a venue. Add the event to one of your existing venues instead.")
    d, zid = min(free)
    if d > MAX_DISTANCE_KM:
        raise VenueError("That location is outside the area this deployment covers.")
    return zid


def _next_event_id(events):
    nums = [int(str(e)[1:]) for e in events["event_id"] if str(e)[1:].isdigit()]
    return f"E{(max(nums) + 1) if nums else 1}"


def _default_gate_rows(zone_id, capacity):
    entry_total = int(capacity * ENTRY_SHARE_OF_CAPACITY)
    exit_total = int(entry_total * EXIT_MULTIPLIER)
    rows = []
    for i, (name, share, exit_share, x, y) in enumerate(DEFAULT_GATES, start=1):
        rows.append({"gate_id": f"{zone_id}-G{i}", "venue_zone_id": zone_id, "name": name, "type": "both",
                     "entry_capacity_per_hour": int(entry_total * share), "exit_capacity_per_hour": int(exit_total * share),
                     "default_exit_share": exit_share, "x": x, "y": y, "status": "open"})
    return rows


def _concourse_row(gate, zone_id):
    name = gate["name"] if "concourse" in gate["name"].lower() else f"{gate['name']} Concourse"
    return {"floor_zone_id": f"{gate['gate_id']}-CONCOURSE", "venue_zone_id": zone_id, "name": name, "kind": "concourse",
            "capacity": int(gate["entry_capacity_per_hour"] * CONCOURSE_CAPACITY_HOURS),
            "adjacent_gate_id": gate["gate_id"],
            "x": round(gate["x"] + (50 - gate["x"]) * 0.25, 1), "y": round(gate["y"] + (50 - gate["y"]) * 0.25, 1)}


def _read_gates(data_dir):
    return pd.read_csv(_p("gates.csv", data_dir))


def _read_floor(data_dir):
    path = _p("floor_zones.csv", data_dir)
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(
        columns=["floor_zone_id", "venue_zone_id", "name", "kind", "capacity", "adjacent_gate_id", "x", "y"])


def _normalise_exit_shares(gates, zone_id):
    """Rescale the exit shares of one venue so they sum to 1: everyone who leaves
    goes somewhere, no more and no fewer."""
    m = gates["venue_zone_id"] == zone_id
    total = float(gates.loc[m, "default_exit_share"].sum())
    if total <= 0:
        gates.loc[m, "default_exit_share"] = round(1.0 / max(1, int(m.sum())), 4)
    else:
        gates.loc[m, "default_exit_share"] = (gates.loc[m, "default_exit_share"] / total).round(4)
    return gates


def _save(gates, floor, data_dir):
    gates.to_csv(_p("gates.csv", data_dir), index=False)
    floor.to_csv(_p("floor_zones.csv", data_dir), index=False)


def _check_event_fields(name, venue_name, capacity, expected, start_hour, end_hour):
    name, venue_name = str(name or "").strip(), str(venue_name or "").strip()
    if not 1 <= len(name) <= 60:
        raise VenueError("Give the event a name (up to 60 characters).")
    if not 1 <= len(venue_name) <= 60:
        raise VenueError("Give the venue a name (up to 60 characters).")
    if not 500 <= capacity <= 200_000:
        raise VenueError("Venue capacity must be between 500 and 200,000.")
    if not 1 <= expected <= 400_000:
        raise VenueError("Expected attendance must be a positive number.")
    if not FIRST_HOUR <= start_hour < end_hour <= LAST_HOUR:
        raise VenueError("The event must start and end inside the simulated run (Day 2 00:00 to Day 6 23:00), with the end after the start.")
    return name, venue_name


def create_event(email, name, expected_attendance, start_hour, end_hour, venue_zone_id=None,
                 venue_name=None, lat=None, lng=None, venue_capacity=None, data_dir=None):
    """
    Create an event, at one of the organiser's existing venues (`venue_zone_id`) or
    at a new venue (`venue_name`, `lat`, `lng`, `venue_capacity`).
    Returns {event, venue, created_venue}.
    """
    email = (email or "").strip().lower()
    events = _events(data_dir)

    if venue_zone_id:
        require_edit(venue_zone_id, email, data_dir)
        site = _sites(data_dir).set_index("venue_zone_id").loc[venue_zone_id]
        existing = events[events["zone_id"] == venue_zone_id]
        capacity = int(venue_capacity or existing.iloc[0]["venue_capacity"])
        venue_name = site["venue_name"]
        zone_id, created = venue_zone_id, False
    else:
        try:
            lat, lng = float(lat), float(lng)
        except (TypeError, ValueError):
            raise VenueError("Choose the venue's location on the map.")
        if not (-90 <= lat <= 90 and -180 <= lng <= 180):
            raise VenueError("That location isn't valid.")
        capacity = int(venue_capacity or 0)
        zone_id, created = _nearest_free_zone(lat, lng, data_dir), True

    name, venue_name = _check_event_fields(name, venue_name, capacity, int(expected_attendance), int(start_hour), int(end_hour))

    entry_total = int(capacity * ENTRY_SHARE_OF_CAPACITY)
    row = {c: "" for c in events.columns}
    row.update({"event_id": _next_event_id(events), "name": name, "zone_id": zone_id,
                "start_hour": int(start_hour), "end_hour": int(end_hour),
                "expected_attendance": int(expected_attendance), "venue_name": venue_name,
                "venue_capacity": capacity, "gate_throughput_per_hour": entry_total,
                "exit_throughput_per_hour": int(entry_total * EXIT_MULTIPLIER), "organizer": email})
    events = pd.concat([events, pd.DataFrame([row])], ignore_index=True)
    events.to_csv(_p("events.csv", data_dir), index=False)

    if created:
        sites = _sites(data_dir)
        sites = pd.concat([sites, pd.DataFrame([{"venue_zone_id": zone_id, "venue_name": venue_name,
                                                 "lat": lat, "lng": lng, "owner": email}])], ignore_index=True)
        sites.to_csv(_p(SITES_CSV, data_dir), index=False)

        gates, floor = _read_gates(data_dir), _read_floor(data_dir)
        new_gates = _default_gate_rows(zone_id, capacity)
        floor_rows = [{"floor_zone_id": f"{zone_id}-BOWL", "venue_zone_id": zone_id,
                       "name": "Main Bowl", "kind": "bowl", "capacity": capacity,
                       "adjacent_gate_id": None, "x": 50, "y": 50}]
        floor_rows += [_concourse_row(g, zone_id) for g in new_gates]
        gates = pd.concat([gates, pd.DataFrame(new_gates)], ignore_index=True)
        floor = pd.concat([floor, pd.DataFrame(floor_rows)], ignore_index=True)
        _save(gates, floor, data_dir)

    return {"event": {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()},
            "venue_zone_id": zone_id, "created_venue": created}


def add_gate(email, zone_id, name, entry_capacity, exit_capacity, x, y, data_dir=None):
    require_edit(zone_id, email, data_dir)
    gates, floor = _read_gates(data_dir), _read_floor(data_dir)
    mine = gates[gates["venue_zone_id"] == zone_id]
    if len(mine) >= MAX_GATES:
        raise VenueError(f"A venue can have at most {MAX_GATES} gates.")
    name = str(name or "").strip()
    if not 1 <= len(name) <= 40:
        raise VenueError("Give the gate a name (up to 40 characters).")
    _check_gate_numbers(entry_capacity, exit_capacity, x, y)

    nums = [int(g.rsplit("-G", 1)[1]) for g in mine["gate_id"] if g.rsplit("-G", 1)[-1].isdigit()]
    gate_id = f"{zone_id}-G{(max(nums) + 1) if nums else 1}"
    gate = {"gate_id": gate_id, "venue_zone_id": zone_id, "name": name, "type": "both",
            "entry_capacity_per_hour": int(entry_capacity), "exit_capacity_per_hour": int(exit_capacity),
            "default_exit_share": 0.1, "x": round(float(x), 1), "y": round(float(y), 1), "status": "open"}
    gates = pd.concat([gates, pd.DataFrame([gate])], ignore_index=True)
    gates = _normalise_exit_shares(gates, zone_id)
    floor = pd.concat([floor, pd.DataFrame([_concourse_row(gate, zone_id)])], ignore_index=True)
    _save(gates, floor, data_dir)
    return gates[gates["gate_id"] == gate_id].iloc[0].to_dict()


def _check_gate_numbers(entry, exit_, x, y):
    for label, v in (("Entry capacity", entry), ("Exit capacity", exit_)):
        if not (isinstance(v, (int, float)) and math.isfinite(v) and 100 <= v <= 200_000):
            raise VenueError(f"{label} must be between 100 and 200,000 people an hour.")
    for label, v in (("x", x), ("y", y)):
        if not (isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 100):
            raise VenueError(f"Position {label} must be between 0 and 100.")


def update_gate(email, gate_id, changes, data_dir=None):
    """Apply any of name, x, y, entry_capacity_per_hour, exit_capacity_per_hour."""
    zone_id = venue_zone_for_gate(gate_id, data_dir)
    if zone_id is None:
        raise VenueError("No such gate.")
    require_edit(zone_id, email, data_dir)

    gates, floor = _read_gates(data_dir), _read_floor(data_dir)
    idx = gates.index[gates["gate_id"] == gate_id][0]
    cur = gates.loc[idx].to_dict()
    new = {**cur, **{k: v for k, v in changes.items() if v is not None}}
    if "name" in changes and changes["name"] is not None:
        new["name"] = str(new["name"]).strip()
        if not 1 <= len(new["name"]) <= 40:
            raise VenueError("Give the gate a name (up to 40 characters).")
    _check_gate_numbers(float(new["entry_capacity_per_hour"]), float(new["exit_capacity_per_hour"]), float(new["x"]), float(new["y"]))

    gates["x"] = gates["x"].astype(float)
    gates["y"] = gates["y"].astype(float)
    gates.loc[idx, "name"] = new["name"]
    gates.loc[idx, "x"] = round(float(new["x"]), 1)          # one decimal: fine enough to place a gate by eye
    gates.loc[idx, "y"] = round(float(new["y"]), 1)
    gates.loc[idx, "entry_capacity_per_hour"] = int(round(float(new["entry_capacity_per_hour"])))
    gates.loc[idx, "exit_capacity_per_hour"] = int(round(float(new["exit_capacity_per_hour"])))
    row = gates.loc[idx].to_dict()

    # Keep the gate's concourse in step with it.
    fz = _concourse_row({**row, "x": float(row["x"]), "y": float(row["y"])}, zone_id)
    m = floor["floor_zone_id"] == fz["floor_zone_id"]
    if m.any():
        for k, v in fz.items():
            floor.loc[m, k] = v
    else:
        floor = pd.concat([floor, pd.DataFrame([fz])], ignore_index=True)
    _save(gates, floor, data_dir)
    return gates.loc[idx].to_dict()


def delete_gate(email, gate_id, data_dir=None):
    zone_id = venue_zone_for_gate(gate_id, data_dir)
    if zone_id is None:
        raise VenueError("No such gate.")
    require_edit(zone_id, email, data_dir)

    gates, floor = _read_gates(data_dir), _read_floor(data_dir)
    rest = gates[(gates["venue_zone_id"] == zone_id) & (gates["gate_id"] != gate_id)]
    if len(rest) < MIN_GATES or int((rest["status"] == "open").sum()) < MIN_GATES:
        raise VenueError(f"A venue needs at least {MIN_GATES} working gates, so this one can't be removed.")

    gates = gates[gates["gate_id"] != gate_id].reset_index(drop=True)
    gates = _normalise_exit_shares(gates, zone_id)
    floor = floor[floor["adjacent_gate_id"] != gate_id].reset_index(drop=True)
    _save(gates, floor, data_dir)
    return {"deleted": gate_id, "remaining": int(len(rest))}
