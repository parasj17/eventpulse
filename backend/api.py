"""
Phase 5 — FastAPI wrapper.

Exposes the engine over HTTP in exactly the shapes agreed in
contracts/mock-data.json, so the frontend can swap the mock file for this API
without changing anything else.

Run:  uvicorn api:app --reload
Docs: http://127.0.0.1:8000/docs
"""

import hmac
import secrets
import json
import os
import time
from collections import defaultdict, deque

import pandas as pd
from fastapi import Body, Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from auth import create_token, demo_organizer_matches, verify_token
from engine.accounts import AccountExists, create_account, verify_login
from engine import ai, copilot, nugen, partners, playbook, presence, social, venue_admin, weather_live, weather_twin
from engine.attendees import attendee_status, event_status, get_attendee, list_registrations, register_attendee
from engine.forecast import DATA_DIR, forecast_all
from engine.recommend import haversine_km, recommend_for_zone
from engine.risk import detect_all
from engine.simulate import simulate_redistribution
from engine.hotels import build_hotels
from engine.venues import analyse_all as analyse_venues
from engine.routing import find_routes, reroute_for_link
from engine.density import zone_density_for_venue
from engine.exit_redirect import recommend_exit
from engine.reverse_uber import optimal_exit_distribution
from engine.incentives import generate_incentives
from engine.restaurants import build_restaurants
from engine.scenarios import (
    simulate_demand_surge,
    simulate_event_cancellation,
    simulate_transport_closure,
)

try:
    from engine.model import predict_all as model_predict
    from engine.model import train as model_train
    ML_AVAILABLE = True
except ImportError:                       # scikit-learn not installed
    ML_AVAILABLE = False

app = FastAPI(
    title="EventPulse API",
    description="Intelligent capacity and crowd management for mega-events.",
    version="0.1.0",
)

# The React dev server runs on a different port, so allow it through.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Default "now" within the simulated 24-hour window.
DEFAULT_HOUR = 110   # the Final — peak pressure, the moment worth demoing


def _require_data():
    zones_path = os.path.join(DATA_DIR, "zones.csv")
    if not os.path.exists(zones_path):
        raise HTTPException(
            status_code=503,
            detail="No data generated yet. Run: python generate_city.py",
        )


def _require_role(role, authorization):
    """Expects `Authorization: Bearer <token>` from a prior sign-in as `role`.
    Raises 401/403 rather than returning None, so a protected route never has to
    remember to check."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Sign in required")
    payload = verify_token(authorization[len("Bearer "):])
    if not payload or payload.get("role") != role:
        raise HTTPException(403, f"{role.capitalize()} access required")
    return payload


def require_organizer(authorization: str = Header(None)):
    return _require_role("organizer", authorization)


def require_attendee(authorization: str = Header(None)):
    return _require_role("attendee", authorization)


# --------------------------------------------------------------------------
# Shared demo clock. The simulated event runs from FIRST_HOUR to LAST_HOUR;
# this steps through it in real time so every attendee's status page sees the
# same "now", and it keeps moving without anyone touching a slider.
# --------------------------------------------------------------------------
FIRST_HOUR, LAST_HOUR = 24, 144
CLOCK_TICK_SECONDS = float(os.environ.get("EVENTPULSE_CLOCK_TICK_SECONDS", "6"))
# Every server instance must agree on "now": a fixed start (EVENTPULSE_CLOCK_EPOCH, unix
# seconds) when one is set, as on serverless hosts; otherwise when this process started.
_CLOCK_START = float(os.environ.get("EVENTPULSE_CLOCK_EPOCH") or time.time())
# Where in the simulated run the clock begins. Starting it a few hours before an
# event lets a demo watch that event's gates fill, run and empty in real time.
CLOCK_START_HOUR = int(os.environ.get("EVENTPULSE_CLOCK_START_HOUR", str(FIRST_HOUR)))


def _clock_hour():
    span = LAST_HOUR - FIRST_HOUR + 1
    start = max(0, min(span - 1, CLOCK_START_HOUR - FIRST_HOUR))
    return FIRST_HOUR + (start + int((time.time() - _CLOCK_START) / CLOCK_TICK_SECONDS)) % span


@app.get("/health/db")
def health_db():
    """Touches the database. Vercel calls this daily (vercel.json crons) so the free Supabase
    project never counts as inactive and is never paused."""
    from engine import db
    if not db.enabled():
        return {"db": "off"}
    try:
        n = db.execute("select count(*) from ep_accounts", fetch=True)[0][0]
        return {"db": "ok", "accounts": n, "at": int(time.time())}
    except Exception as e:
        raise HTTPException(503, f"Database unreachable ({type(e).__name__})")


@app.get("/clock")
def get_clock():
    """The current simulated hour, and how long until it next changes."""
    into = (time.time() - _CLOCK_START) % CLOCK_TICK_SECONDS
    return {"hour": _clock_hour(), "first": FIRST_HOUR, "last": LAST_HOUR,
            "tick_seconds": CLOCK_TICK_SECONDS, "next_in": round(CLOCK_TICK_SECONDS - into, 2)}


# --------------------------------------------------------------------------
# Crowd-sourced presence: anonymous, opt-in, coarse and short-lived. See
# engine/presence.py for what is (and is not) kept.
# --------------------------------------------------------------------------
DEMO_CROWD = os.environ.get("EVENTPULSE_DEMO_CROWD") == "1"


def _zone_records():
    return pd.read_csv(os.path.join(DATA_DIR, "zones.csv")).to_dict("records")


@app.get("/presence/config")
def presence_config():
    return {"min_group": presence.MIN_GROUP, "ttl_seconds": presence.TTL_SECONDS,
            "cell_metres": 150, "demo_enabled": DEMO_CROWD}


class PresenceReport(BaseModel):
    sid: str
    lat: float
    lng: float


@app.post("/presence")
def presence_report(body: PresenceReport, request: Request):
    """One person's position, reduced to a grid cell on arrival. Nothing else is stored."""
    _require_data()
    try:
        return presence.report(body.sid, body.lat, body.lng,
                               request.client.host if request.client else "?", _zone_records())
    except presence.PresenceError as e:
        code = 429 if "Too many" in str(e) else 400
        raise HTTPException(code, str(e))


class PresenceLeave(BaseModel):
    sid: str


@app.post("/presence/leave")
def presence_leave(body: PresenceLeave):
    """Stop counting this person straight away (sharing turned off, or the page closed)."""
    return {"removed": presence.leave(body.sid)}


@app.get("/presence/summary")
def presence_summary():
    """The crowds worth showing: busy cells, people at each venue, and crowds at
    places nobody has mapped. Groups smaller than the minimum are never reported."""
    _require_data()
    return presence.summary(_zone_records(), venue_admin.list_sites())


class DemoCrowd(BaseModel):
    zone_id: str = ""
    people: int = 40


@app.post("/presence/demo")
def presence_demo(body: DemoCrowd):
    """Demos only: add synthetic attendees around a zone. Off unless
    EVENTPULSE_DEMO_CROWD=1, and they are marked so they can be cleared."""
    if not DEMO_CROWD:
        raise HTTPException(404, "Not found")
    _require_data()
    zones = _zone_records()
    z = next((q for q in zones if q["zone_id"] == body.zone_id), None)
    if z is None:
        raise HTTPException(400, "Unknown zone")
    n = presence.seed_demo(float(z["lat"]), float(z["lng"]), max(1, min(200, body.people)), zones)
    return {"added": n, "zone": z["name"]}


@app.delete("/presence/demo")
def presence_demo_clear():
    if not DEMO_CROWD:
        raise HTTPException(404, "Not found")
    return {"removed": presence.clear_demo()}


# --------------------------------------------------------------------------
# Accounts. Attendees and organisers are separate roles with separate sign-in.
# --------------------------------------------------------------------------
LOGIN_WINDOW_SECONDS = 600
LOGIN_MAX_FAILURES = 8
_failures = defaultdict(deque)


def _throttle(role, email):
    """Refuse further attempts on one account after repeated failures, which
    blunts password guessing. In memory, so a restart clears it."""
    key = (role, (email or "").strip().lower())
    q = _failures[key]
    cutoff = time.time() - LOGIN_WINDOW_SECONDS
    while q and q[0] < cutoff:
        q.popleft()
    if len(q) >= LOGIN_MAX_FAILURES:
        raise HTTPException(429, "Too many failed attempts. Try again in a few minutes.")
    return q


def _account_token(acct):
    return create_token(acct["role"], acct["email"], name=acct.get("name", ""))


@app.get("/auth/config")
def auth_config():
    """Tells the sign-up form whether organiser accounts need an access code."""
    return {"organizer_access_code_required": bool(os.environ.get("EVENTPULSE_ORGANIZER_ACCESS_CODE"))}


class SignupRequest(BaseModel):
    role: str
    name: str
    email: str
    password: str
    access_code: str = ""


@app.post("/auth/signup")
def signup(body: SignupRequest):
    """Create an attendee or organiser account and sign it in."""
    if body.role == "organizer":
        code = os.environ.get("EVENTPULSE_ORGANIZER_ACCESS_CODE")
        if code and not hmac.compare_digest(body.access_code or "", code):
            raise HTTPException(403, "That organiser access code isn't right.")
    try:
        acct = create_account(body.role, body.email, body.password, body.name)
    except AccountExists as e:
        raise HTTPException(409, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"token": _account_token(acct), "role": acct["role"], "email": acct["email"], "name": acct["name"]}


class LoginRequest(BaseModel):
    email: str
    password: str
    role: str = "organizer"


@app.post("/auth/login")
def login(body: LoginRequest):
    """Sign in to an attendee or organiser account."""
    failures = _throttle(body.role, body.email)
    acct = verify_login(body.role, body.email, body.password)
    if not acct and body.role == "organizer" and demo_organizer_matches(body.email, body.password):
        acct = {"role": "organizer", "email": body.email.strip().lower(), "name": "Demo organiser"}
    if not acct:
        failures.append(time.time())
        raise HTTPException(401, "That email or password isn't right.")
    failures.clear()
    return {"token": _account_token(acct), "role": acct["role"], "email": acct["email"], "name": acct["name"]}


class RegisterRequest(BaseModel):
    event_id: str


def _lan_ip():
    """This machine's address on the local network (what a phone on the same Wi-Fi can reach)."""
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sk:
            sk.connect(("8.8.8.8", 80))
            return sk.getsockname()[0]
    except OSError:
        return None


def _public_url(request):
    """
    The address to put in emails, which must open on the attendee's own device:
    EVENTPULSE_PUBLIC_URL when set (a real deployment); otherwise, if the request came in
    on localhost, this machine's Wi-Fi address on the API's port (phones on the same network
    can open it; the API serves the web app too).
    """
    if os.environ.get("EVENTPULSE_PUBLIC_URL"):
        return os.environ["EVENTPULSE_PUBLIC_URL"].rstrip("/")
    base = str(request.base_url).rstrip("/")
    host = request.url.hostname or ""
    if host in ("127.0.0.1", "localhost", "::1"):
        ip = _lan_ip()
        if ip:
            return f"{request.url.scheme}://{ip}:{request.url.port or 8000}"
    return base


@app.post("/register")
def register(body: RegisterRequest, request: Request, hour: int = None,
             who=Depends(require_attendee)):
    """
    Register the signed-in attendee for an event and email them their status
    link (see engine/attendees.py and engine/mailer.py).
    """
    _require_data()
    try:
        # The link in the email has to work from the attendee's own device, so a
        # configured public address wins over whatever this request came in on.
        base_url = _public_url(request)
        result = register_attendee(who["sub"], body.event_id, _clock_hour() if hour is None else hour,
                                   base_url=base_url.rstrip("/"))
    except ValueError as e:
        raise HTTPException(400, str(e))
    if "error" in result:
        raise HTTPException(404, result["error"])
    return result


@app.get("/attendee/registrations")
def attendee_registrations(who=Depends(require_attendee)):
    """The signed-in attendee's registrations, each with its status link."""
    _require_data()
    return list_registrations(who["sub"])


@app.get("/status/{token}")
def get_status(token: str, hour: int = None):
    """
    The live page an attendee's emailed link opens. Follows the shared demo
    clock unless an hour is given explicitly.
    """
    _require_data()
    h = _clock_hour() if hour is None else hour
    result = attendee_status(token, h)
    if "error" in result:
        raise HTTPException(404, result["error"])
    result["clock"] = {"tick_seconds": CLOCK_TICK_SECONDS}
    result["weather"] = _venue_weather(result, h)
    return result


def _venue_weather(status, hour):
    """Weather at this attendee's venue, and what the weather twin expects it to do to their
    exit. Best effort: the status page works the same without it."""
    try:
        live = _live_weather()
        if not live.get("available"):
            return None
        z = next((z for z in live["zones"] if z["zone_id"] == status["venue_zone_id"]), None)
        if z is None:
            return None
        out = {"condition": z["condition"], "temp_c": z["temp_c"], "apparent_c": z["apparent_c"], "rain_mm_h": z["rain_mm_h"],
               "rain_next_6h_mm": z["rain_next_6h_mm"], "advice": [], "exit_hours": None, "exit_hours_normal": None}
        if z["rain_mm_h"] >= 2:
            out["advice"].append("It's raining at the venue: allow extra time and head for a covered exit.")
        elif z["rain_next_6h_mm"] >= 5:
            out["advice"].append("Rain is expected in the next few hours: bring something waterproof.")
        if z["apparent_c"] >= 35:
            out["advice"].append("It's very hot: carry water and use shaded queues.")
        # What people are reporting: near the venue first, then citywide flooding or disruption, last 24 h.
        soc = social._cache.get("data") or {}
        fresh = [i for i in soc.get("items", []) if i.get("age_h") is not None and i["age_h"] <= 24
                 and set(i["categories"]) & {"flooding", "transport", "alert", "heat"}]
        near = [i for i in fresh if any(p["zone_id"] == status["venue_zone_id"] for p in i["places"])]
        pick = (near + [i for i in fresh if i not in near])[:2]
        out["reports"] = [{"text": i["text"][:180], "source": i["source"], "channel": i["channel"], "age_h": round(i["age_h"], 1),
                           "url": i["url"], "near_venue": i in near} for i in pick]
        if status["event_state"] in ("arriving", "running", "exiting"):
            st = weather_state(hour)
            v = st.get("venue")
            if v and v["event_id"] == status["event_id"] and v["hours_to_clear"]["p50"] - v["hours_to_clear_baseline"] >= 0.25:
                out["exit_hours"] = v["hours_to_clear"]["p50"]
                out["exit_hours_normal"] = v["hours_to_clear_baseline"]
                out["advice"].append(f"Because of the weather, getting everyone out is expected to take about "
                                     f"{v['hours_to_clear']['p50']:.1f} h instead of {v['hours_to_clear_baseline']:.1f} h. "
                                     f"Leaving a little early or waiting inside for the first rush to pass will be quicker.")
        return out
    except Exception:
        return None


@app.get("/api")
def api_index():
    return {
        "service": "EventPulse API",
        "endpoints": ["/zones", "/forecast", "/alerts", "/recommendations/{zone_id}",
                      "/summary", "/transport", "/history", "/hotels", "/restaurants", "/events",
                      "/venues", "/gates", "/organizer/gates/{gate_id}/status",
                      "/organizer/venues/{venue_zone_id}/gates",
                      "/organizer/venues/{venue_zone_id}/map", "/clock",
                      "/presence", "/presence/summary", "/presence/config",
                      "/density/{venue_zone_id}", "/exit-redirect/{venue_zone_id}",
                      "/reverse-uber/{venue_zone_id}", "/incentives",
                      "/routes/{from}/{to}", "/reroute/{link_id}",
                      "/simulate/{zone_id}",
                      "/scenarios/cancel/{event_id}", "/scenarios/surge/{event_id}",
                      "/scenarios/close-link/{link_id}",
                      "/auth/signup", "/auth/login", "/auth/config", "/register", "/attendee/registrations", "/status/{token}",
                      "/ml/forecast", "/ml/benchmark",
                      "/organizer/keys", "/partner/v1/events", "/partner/v1/events/{event_id}",
                      "/partner/v1/attendees", "/partner/v1/attendees/{token}",
                      "/weather/live", "/weather/social", "/weather-twin/model", "/weather-twin/state",
                      "/weather-twin/simulate", "/organizer/weather/observe",
                      "/ai/nugen", "/ai/plan", "/weather-twin/plan"],
        "ml_available": ML_AVAILABLE,
    }


@app.get("/zones")
def get_zones(hour: int = DEFAULT_HOUR):
    """All zones with their current occupancy at the given hour."""
    _require_data()

    zones = pd.read_csv(os.path.join(DATA_DIR, "zones.csv"))
    occupancy = pd.read_csv(os.path.join(DATA_DIR, "occupancy_timeline.csv"))

    snapshot = occupancy[occupancy["hour_offset"] == hour]
    current = dict(zip(snapshot["zone_id"], snapshot["occupancy_pct"]))
    occupied = dict(zip(snapshot["zone_id"], snapshot["occupancy"]))

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


@app.get("/forecast")
def get_forecast(hour: int = DEFAULT_HOUR, horizon: int = 12):
    """Projected occupancy and time-to-saturation for every zone."""
    _require_data()
    occupancy = pd.read_csv(os.path.join(DATA_DIR, "occupancy_timeline.csv"))
    return forecast_all(occupancy, current_hour=hour, horizon=horizon)


@app.get("/alerts")
def get_alerts(hour: int = DEFAULT_HOUR):
    """Active alerts, most urgent first."""
    _require_data()
    return detect_all(current_hour=hour)


@app.get("/recommendations/{zone_id}")
def get_recommendations(zone_id: str, hour: int = DEFAULT_HOUR):
    """Ranked alternative zones for a zone under pressure."""
    _require_data()
    return recommend_for_zone(zone_id, current_hour=hour)


@app.get("/summary")
def get_summary(hour: int = DEFAULT_HOUR):
    """Headline numbers for the dashboard's top bar."""
    _require_data()

    zones = get_zones(hour)
    alerts = detect_all(current_hour=hour)

    total_capacity = sum(z["hotel_capacity"] for z in zones)
    total_occupied = sum(z["current_occupancy"] for z in zones)
    saturated = [z for z in zones if z["occupancy_pct"] >= 95]

    return {
        "hour": hour,
        "total_capacity": total_capacity,
        "total_occupied": total_occupied,
        "citywide_occupancy_pct": round(total_occupied / total_capacity * 100, 1),
        "spare_capacity": max(0, total_capacity - total_occupied),
        "zones_saturated": len(saturated),
        "zones_total": len(zones),
        "alerts_critical": len([a for a in alerts if a["severity"] == "critical"]),
        "alerts_warning": len([a for a in alerts if a["severity"] == "warning"]),
    }


@app.get("/simulate/{zone_id}")
def simulate(zone_id: str, move: int, hour: int = DEFAULT_HOUR,
             compliance: float = 0.75):
    """
    What-if: move `move` visitors out of a zone and see where they land.

    Returns the before/after occupancy for the source and every receiving zone,
    plus whether the alert clears.
    """
    _require_data()
    result = simulate_redistribution(
        zone_id, move, current_hour=hour, compliance=compliance
    )
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.get("/events")
def events(hour: int = DEFAULT_HOUR):
    """
    The event schedule around the hour being viewed.

    Occupancy rises because something is coming; without the schedule on screen
    the dashboard shows pressure with no explanation of why.
    """
    _require_data()

    ev = pd.read_csv(os.path.join(DATA_DIR, "events.csv"))
    zones = pd.read_csv(os.path.join(DATA_DIR, "zones.csv"))
    names = dict(zip(zones["zone_id"], zones["name"]))

    out = []
    for _, e in ev.iterrows():
        start, end = int(e["start_hour"]), int(e["end_hour"])

        if hour < start:
            state, hours_away = "upcoming", start - hour
        elif hour <= end:
            state, hours_away = "running", 0
        else:
            state, hours_away = "finished", hour - end

        out.append({
            "event_id": e["event_id"],
            "name": e["name"],
            "zone_id": e["zone_id"],
            "zone_name": names.get(e["zone_id"], e["zone_id"]),
            "start_hour": start,
            "end_hour": end,
            "expected_attendance": int(e["expected_attendance"]),
            "state": state,
            "hours_away": int(hours_away),
        })

    # Anything still to come, plus whatever is running now, soonest first.
    live = [x for x in out if x["state"] == "running"]
    ahead = sorted([x for x in out if x["state"] == "upcoming"],
                   key=lambda x: x["hours_away"])

    return {"hour": hour, "schedule": live + ahead, "all": out}


@app.get("/venues")
def venues(hour: int = DEFAULT_HOUR):
    """
    Venue capacity for every event: over-subscription, ingress queues and how
    long the crowd takes to clear the zone afterwards.
    """
    _require_data()
    return analyse_venues(current_hour=hour)


@app.get("/gates")
def get_gates(venue_zone_id: str = None, hour: int = DEFAULT_HOUR):
    """
    Named entry/exit gates for a venue, with live open/restricted/closed
    status computed for `hour` (queue-driven pre-event, exit-load-driven
    post-event) — not a status someone has to set by hand.

    Powers the organizer floor-map page and matches the per-gate breakdown
    already folded into /venues.
    """
    _require_data()
    path = os.path.join(DATA_DIR, "gates.csv")
    if not os.path.exists(path):
        return []

    gates = pd.read_csv(path)
    if venue_zone_id:
        gates = gates[gates["venue_zone_id"] == venue_zone_id]

    live_by_id = {}
    for v in analyse_venues(current_hour=hour):
        for g in (v["gates"] or []):
            live_by_id[g["gate_id"]] = g

    out = []
    for _, g in gates.iterrows():
        row = g.to_dict()
        live = live_by_id.get(g["gate_id"])
        if live:
            row["phase"] = live["phase"]
            for k in ("load_ratio", "live_status", "load_pct", "demand_pct", "redirected_out", "redirected_in",
                      "redirect_to", "redirect_from"):
                if k in live:
                    row[k] = live[k]
        out.append(row)
    return out


# A venue with fewer open gates than this has nowhere left to redirect to. Enforced
# here, not just when a venue is created, so an organiser cannot click their way
# below it during an incident either.
MIN_OPEN_GATES = venue_admin.MIN_GATES


def _venue_call(fn, *args, **kwargs):
    """Run a venue_admin operation, turning its errors into HTTP ones."""
    try:
        return fn(*args, **kwargs)
    except venue_admin.Forbidden as e:
        raise HTTPException(403, str(e))
    except venue_admin.VenueError as e:
        raise HTTPException(400, str(e))


@app.get("/organizer/venues")
def organizer_venues(who=Depends(require_organizer)):
    """The venues this organiser can edit, with their events: the shared demo
    venues plus any they created."""
    _require_data()
    return venue_admin.list_venues(who["sub"])


class NewEventRequest(BaseModel):
    name: str
    expected_attendance: int
    start_hour: int
    end_hour: int
    venue_zone_id: str = ""        # add to one of your existing venues...
    venue_name: str = ""           # ...or create a new one here
    lat: float = None
    lng: float = None
    venue_capacity: int = None


@app.post("/organizer/events")
def organizer_create_event(body: NewEventRequest, who=Depends(require_organizer)):
    """Create an event at a new or existing venue. A new venue starts with a sensible
    set of gates, ready to be arranged on the floor plan."""
    _require_data()
    return _venue_call(venue_admin.create_event, who["sub"], body.name, body.expected_attendance,
                       body.start_hour, body.end_hour, venue_zone_id=body.venue_zone_id or None,
                       venue_name=body.venue_name, lat=body.lat, lng=body.lng,
                       venue_capacity=body.venue_capacity)


@app.post("/organizer/gates/{gate_id}/status")
def set_gate_status(gate_id: str, status: str = Body(..., embed=True),
                     who=Depends(require_organizer)):
    """
    Take a named gate out of service, or bring it back.

    A closed gate stops contributing capacity in /venues and /gates immediately,
    which is the lever an organiser needs during a crowding incident. Everything
    downstream (queue maths, redirect targets) reads gates.csv fresh on every
    request rather than caching status. Refused if it would leave the venue with
    fewer than MIN_OPEN_GATES open.
    """
    if status not in ("open", "closed", "restricted"):
        raise HTTPException(400, "status must be one of: open, closed, restricted")

    _require_data()
    path = os.path.join(DATA_DIR, "gates.csv")
    gates = pd.read_csv(path)
    if gate_id not in set(gates["gate_id"]):
        raise HTTPException(404, f"No gate {gate_id}")

    venue_zone_id = gates.loc[gates["gate_id"] == gate_id, "venue_zone_id"].iloc[0]
    _venue_call(venue_admin.require_edit, venue_zone_id, who["sub"])

    if status != "open":
        remaining_open = gates[
            (gates["venue_zone_id"] == venue_zone_id)
            & (gates["gate_id"] != gate_id)
            & (gates["status"] == "open")
        ]
        if len(remaining_open) < MIN_OPEN_GATES:
            raise HTTPException(
                400,
                f"Cannot take this gate out of service: fewer than "
                f"{MIN_OPEN_GATES} open gates would remain for this venue.",
            )

    gates.loc[gates["gate_id"] == gate_id, "status"] = status
    gates.to_csv(path, index=False)
    return gates[gates["gate_id"] == gate_id].iloc[0].to_dict()


class NewGateRequest(BaseModel):
    name: str
    entry_capacity_per_hour: int
    exit_capacity_per_hour: int
    x: float
    y: float


@app.post("/organizer/venues/{venue_zone_id}/gates")
def add_gate(venue_zone_id: str, body: NewGateRequest, who=Depends(require_organizer)):
    """Add a gate at a point on the floor plan, with a matching concourse floor zone
    so the density engine picks it up immediately."""
    _require_data()
    return _venue_call(venue_admin.add_gate, who["sub"], venue_zone_id, body.name,
                       body.entry_capacity_per_hour, body.exit_capacity_per_hour, body.x, body.y)


class GateChanges(BaseModel):
    name: str = None
    x: float = None
    y: float = None
    entry_capacity_per_hour: int = None
    exit_capacity_per_hour: int = None


@app.patch("/organizer/gates/{gate_id}")
def update_gate(gate_id: str, body: GateChanges, who=Depends(require_organizer)):
    """Move, rename or resize a gate; its concourse follows."""
    _require_data()
    return _venue_call(venue_admin.update_gate, who["sub"], gate_id, body.model_dump(exclude_none=True))


@app.delete("/organizer/gates/{gate_id}")
def delete_gate(gate_id: str, who=Depends(require_organizer)):
    """Remove a gate, unless the venue would be left with too few."""
    _require_data()
    return _venue_call(venue_admin.delete_gate, who["sub"], gate_id)


MAPS_DIR = os.path.join(DATA_DIR, "maps")
MAP_EXTENSIONS = (".png", ".jpg", ".jpeg", ".svg", ".webp")
MAX_MAP_BYTES = 8 * 1024 * 1024


@app.post("/organizer/venues/{venue_zone_id}/map")
async def upload_map(venue_zone_id: str, file: UploadFile = File(...),
                      who=Depends(require_organizer)):
    """Upload a floor plan image for a venue the organiser owns."""
    _venue_call(venue_admin.require_edit, venue_zone_id, who["sub"])
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in MAP_EXTENSIONS:
        raise HTTPException(400, f"Unsupported image type: {ext or '(none)'}")

    content = await file.read(MAX_MAP_BYTES + 1)
    if len(content) > MAX_MAP_BYTES:
        raise HTTPException(400, "That image is over 8 MB. Use a smaller one.")

    os.makedirs(MAPS_DIR, exist_ok=True)
    dest = os.path.join(MAPS_DIR, f"{venue_zone_id}{ext}")
    # Clear any previous map for this venue under a different extension, so
    # re-uploading in a new format does not leave a stale file being served.
    for old_ext in MAP_EXTENSIONS:
        old_path = os.path.join(MAPS_DIR, f"{venue_zone_id}{old_ext}")
        if old_path != dest and os.path.exists(old_path):
            os.remove(old_path)
    with open(dest, "wb") as f:
        f.write(content)
    return {"venue_zone_id": venue_zone_id, "map_url": f"/organizer/venues/{venue_zone_id}/map"}


@app.delete("/organizer/venues/{venue_zone_id}/map")
def delete_map(venue_zone_id: str, who=Depends(require_organizer)):
    """Remove the uploaded floor plan, going back to the generic schematic."""
    _venue_call(venue_admin.require_edit, venue_zone_id, who["sub"])
    removed = False
    for ext in MAP_EXTENSIONS:
        path = os.path.join(MAPS_DIR, f"{venue_zone_id}{ext}")
        if os.path.exists(path):
            os.remove(path); removed = True
    return {"removed": removed}


@app.get("/organizer/venues/{venue_zone_id}/map")
def get_map(venue_zone_id: str):
    for ext in MAP_EXTENSIONS:
        path = os.path.join(MAPS_DIR, f"{venue_zone_id}{ext}")
        if os.path.exists(path):
            return FileResponse(path)
    raise HTTPException(404, "No map uploaded for this venue")


@app.get("/density/{venue_zone_id}")
def get_density(venue_zone_id: str, hour: int = DEFAULT_HOUR):
    """
    Live headcount per floor zone (bowl + concourses) inside one venue —
    the feed a floor heatmap renders directly, and the same numbers the
    per-gate live_status in /venues and /gates is derived from.
    """
    _require_data()
    return zone_density_for_venue(venue_zone_id, hour)


@app.get("/exit-redirect/{venue_zone_id}")
def get_exit_redirect(venue_zone_id: str, hour: int = DEFAULT_HOUR,
                       from_floor_zone_id: str = None):
    """
    Ranked gates to head for right now, from wherever `from_floor_zone_id`
    is (defaults to the bowl/hall) — the actual "go this way" instruction
    built on top of /gates' live status and /density's floor-zone load.
    """
    _require_data()
    return recommend_exit(venue_zone_id, hour, from_floor_zone_id=from_floor_zone_id)


# --------------------------------------------------------------------------- partner API
# A ticketing platform's developers call these with a key their organiser created in the
# console. See engine/partners.py for how keys are stored, scoped and limited, and
# frontend/partners.html for the developer guide.

class NewKeyRequest(BaseModel):
    label: str


@app.get("/organizer/keys")
def organizer_keys(who=Depends(require_organizer)):
    return partners.list_keys(who["sub"])


@app.post("/organizer/keys")
def organizer_create_key(body: NewKeyRequest, who=Depends(require_organizer)):
    """Create a partner key. The full key is returned this once only."""
    try:
        return partners.create_key(who["sub"], body.label)
    except partners.PartnerError as e:
        raise HTTPException(400, str(e))


@app.delete("/organizer/keys/{key_id}")
def organizer_revoke_key(key_id: str, who=Depends(require_organizer)):
    try:
        return partners.revoke_key(who["sub"], key_id)
    except partners.NotFound as e:
        raise HTTPException(404, str(e))


def require_partner(x_api_key: str = Header(None), authorization: str = Header(None)):
    key = x_api_key or (authorization[7:] if authorization and authorization.lower().startswith("bearer ") else None)
    if not key:
        raise HTTPException(401, "Send your partner key in the X-API-Key header.")
    try:
        who = partners.authenticate(key)
    except partners.PartnerError as e:
        raise HTTPException(429, str(e))
    if who is None:
        raise HTTPException(401, "That key is not valid, or it has been revoked.")
    return who


def _partner_events(owner):
    """The events a key may reach: those at venues its organiser can edit."""
    return {e["event_id"]: {**e, "venue_zone_id": v["venue_zone_id"], "venue_name": v["venue_name"],
                            "lat": v["lat"], "lng": v["lng"]}
            for v in venue_admin.list_venues(owner) for e in v["events"]}


def _partner_event(owner, event_id):
    ev = _partner_events(owner).get(event_id)
    if ev is None:
        raise HTTPException(404, "No such event for this key.")    # the same answer for someone else's event
    return ev


@app.get("/partner/v1/events")
def partner_events(who=Depends(require_partner)):
    """Your events, with where each one is in its life right now."""
    _require_data()
    hour = _clock_hour()
    out = []
    for ev in _partner_events(who["owner"]).values():
        st = "upcoming" if hour < ev["start_hour"] else "running" if hour <= ev["end_hour"] else "finished"
        out.append({k: ev[k] for k in ("event_id", "name", "venue_zone_id", "venue_name", "lat", "lng",
                                        "start_hour", "end_hour", "expected_attendance")} | {"state": st})
    out.sort(key=lambda e: e["start_hour"])
    return {"hour": hour, "events": out}


@app.get("/partner/v1/events/{event_id}")
def partner_event_live(event_id: str, who=Depends(require_partner)):
    """Live conditions for one event: gates, how full it is, the best exit once it ends,
    and one sentence of advice you can show your users as is."""
    _require_data()
    _partner_event(who["owner"], event_id)
    st = event_status(event_id, _clock_hour())
    if "error" in st:
        raise HTTPException(404, st["error"])
    return {**st, "advice": ai.guidance(st)["text"]}


class PartnerAttendee(BaseModel):
    event_id: str
    email: str
    send_email: bool = False


@app.post("/partner/v1/attendees")
def partner_register(body: PartnerAttendee, request: Request, who=Depends(require_partner)):
    """
    Register one of your ticket buyers and get their personal live link. Put the link in
    your own ticket email or app; set send_email to have EventPulse email it as well.
    Calling again for the same email and event returns the same link.
    """
    _require_data()
    _partner_event(who["owner"], body.event_id)
    base_url = _public_url(request)
    try:
        r = register_attendee(body.email, body.event_id, _clock_hour(), base_url=base_url.rstrip("/"),
                              send=body.send_email, partner=who["owner"])
    except ValueError as e:
        raise HTTPException(400, str(e))
    if "error" in r:
        raise HTTPException(404, r["error"])
    return {"attendee_token": r["attendee_token"], "event_id": r["event_id"], "event_name": r["event_name"],
            "live_link": r["status_link"], "already_registered": r["already_registered"],
            # sent | simulated (no mail server configured) | skipped (sent moments ago) | not_requested | failed
            "email": r["email_result"]["status"]}


@app.get("/partner/v1/attendees/{token}")
def partner_attendee(token: str, who=Depends(require_partner)):
    """What that attendee's live page shows right now, for showing inside your own app."""
    _require_data()
    a = get_attendee(token)
    # Only the platform that registered this buyer can read them.
    if a is None or a.get("partner") != who["owner"] or a["event_id"] not in _partner_events(who["owner"]):
        raise HTTPException(404, "No such attendee for this key.")
    st = attendee_status(token, _clock_hour())
    if "error" in st:
        raise HTTPException(404, st["error"])
    return {**st, "advice": ai.guidance(st)["text"]}


# --------------------------------------------------------------------------- weather twin
# Live weather and social signals feed a learned model of how weather moves through the
# city (engine/weather_twin.py). Everything here reads the current state and simulates
# on copies: nothing it does changes the live system.

def _zone_rows():
    return pd.read_csv(os.path.join(DATA_DIR, "zones.csv")).to_dict("records")


def _live_weather():
    return weather_live.fetch(_zone_rows())


def _social():
    return social.fetch(_zone_rows())


def _weather_system(hour, duration_h, live):
    """The part of the live system the weather twin works on, at `hour`."""
    zones = get_zones(hour)
    links = get_transport(hour)
    occ = pd.read_csv(os.path.join(DATA_DIR, "occupancy_timeline.csv"))
    ahead_h = int(min(6, max(1, round(duration_h))))
    now = occ[occ["hour_offset"] == hour].set_index("zone_id")["occupancy"]
    later = occ[occ["hour_offset"] == min(hour + ahead_h, int(occ["hour_offset"].max()))].set_index("zone_id")["occupancy"]
    occ_ahead = {z: float(later.get(z, now.get(z, 0)) - now.get(z, 0)) for z in now.index}

    # The event the weather could touch: one whose arrival-to-exit window overlaps this one.
    venue = None
    window = (hour, hour + max(1, duration_h))
    best = None
    for v in analyse_venues(current_hour=hour):
        span = (v["start_hour"] - 4, v["end_hour"] + 3)
        overlap = min(window[1], span[1]) - max(window[0], span[0])
        if overlap > 0 and (best is None or overlap > best[0]):
            best = (overlap, v)
    if best:
        v = best[1]
        leaving = hour > v["end_hour"]
        remaining = v["admitted"] * max(0.0, 1 - (hour - v["end_hour"]) / 3) if leaving else v["admitted"]
        exit_per_hour = (v.get("egress") or {}).get("exit_throughput_per_hour") or 1
        venue = {"event_id": v["event_id"], "event_name": v["name"], "venue_name": v["venue_name"],
                 "zone_id": v["zone_id"], "expected": v["expected_attendance"], "capacity": v["venue_capacity"],
                 "exit_per_hour": exit_per_hour, "people": remaining,
                 "phase": "leaving" if leaving else ("inside" if hour >= v["start_hour"] else "arriving")}
    vz = next((z for z in zones if z["id"] == (venue["zone_id"] if venue else "Z01")), zones[0])
    for z in zones:
        z["distance_km"] = haversine_km(z["lat"], z["lng"], vz["lat"], vz["lng"])

    elevation, prior = {}, {}
    try:                                    # last known real elevations (saved by weather_live)
        with open(os.path.join(DATA_DIR, "zone_elevation.json")) as f:
            elevation = json.load(f)
    except (OSError, ValueError):
        pass
    if live and live.get("available"):
        elevation = {z["zone_id"]: z["elevation_m"] for z in live["zones"]}
        prior = {z["zone_id"]: z["rain_last_6h_mm"] for z in live["zones"]}
    soc = social._cache.get("data") or {}
    social_lo = {z["zone_id"]: min(3.0, 1.2 * z["flood_evidence"] + 0.4 * z["disruption_evidence"])
                 for z in soc.get("zones", [])}
    return {"hour": hour, "zones": zones, "links": links, "venue": venue, "occ_ahead": occ_ahead,
            "elevation": elevation, "rain_prior_mm": prior, "social_flood_lo": social_lo}


@app.get("/weather/live")
def weather_now():
    """Current weather and the next 24 hours for every zone (Open-Meteo)."""
    return _live_weather()


@app.get("/weather/social")
def weather_social():
    """What people and the news are saying about the weather, classified and placed on the map."""
    return _social()


@app.get("/weather-twin/model")
def weather_model():
    """What the model has learned from the city's history and live reports, and what it assumes."""
    _require_data()
    return weather_twin.model(DATA_DIR).summary()


_state_cache = {}


@app.get("/weather-twin/state")
def weather_state(hour: int = None):
    """
    The twin under the weather as it is right now: live readings, social evidence and the
    current state of the city, pushed through the learned cascade, compared with a normal
    day. Refreshes as new weather, reports and hours arrive.
    """
    _require_data()
    h = _clock_hour() if hour is None else int(hour)
    live = _live_weather()
    soc = _social()
    key = (h, live.get("fetched_at"), soc.get("fetched_at"), len(weather_twin.model(DATA_DIR).flood_reports),
           weather_twin.model(DATA_DIR).transport.n_live)
    hit = _state_cache.get("v")
    if hit and hit[0] == key:
        return hit[1]
    if live.get("available"):
        sc = weather_twin.scenario_from_live(live)
    else:
        sc = {**weather_twin.normal_weather(3), "label": "Live weather unavailable: assuming calm"}
    out = weather_twin.simulate(_weather_system(h, sc["duration_h"], live), sc)
    out["live"] = {"available": live.get("available", False), "city": live.get("city"), "time": live.get("time"),
                   "source": live.get("source"), "error": live.get("error"), "stale": live.get("stale", False),
                   "zones": [{k: z[k] for k in ("zone_id", "name", "lat", "lng", "temp_c", "apparent_c", "rain_mm_h",
                                                "rain_last_6h_mm", "rain_next_6h_mm", "max_prob_next_6h", "gust_kmh",
                                                "condition", "elevation_m")} for z in live.get("zones", [])]}
    out["social"] = {"available": soc.get("available", False), "signal_level": soc.get("signal_level", 0),
                     "counts_24h": soc.get("counts_24h", {}), "zones": soc.get("zones", [])}
    _state_cache["v"] = (key, out)
    return out


class WeatherScenario(BaseModel):
    rain_mm_h: float = 0
    duration_h: float = 3
    temp_c: float = 30
    wind_kmh: float = 10
    radius_km: float = None
    center: dict = None
    label: str = None
    hour: int = None
    baseline: str = "live"          # "live" (what is happening now) or "normal" (a calm day)


_sim_calls = defaultdict(deque)


@app.post("/weather-twin/simulate")
def weather_simulate(body: WeatherScenario, request: Request):
    """
    What-if: this weather instead of the baseline. Read-only: it simulates on a copy of the
    current state, with uncertainty, and never touches the live system.
    """
    _require_data()
    ip = request.client.host if request.client else "?"
    q = _sim_calls[ip]
    now = time.time()
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= 40:
        raise HTTPException(429, "Too many simulations at once. Give it a moment.")
    q.append(now)

    h = _clock_hour() if body.hour is None else int(body.hour)
    zones = get_zones(h)
    try:
        sc = weather_twin.parse_scenario(body.model_dump(), zones)
    except weather_twin.ScenarioError as e:
        raise HTTPException(400, str(e))
    live = _live_weather()
    if body.baseline == "live" and live.get("available"):
        bl = {**weather_twin.scenario_from_live(live), "duration_h": sc["duration_h"]}
    else:
        bl = weather_twin.normal_weather(sc["duration_h"])
    return weather_twin.simulate(_weather_system(h, sc["duration_h"], live), sc, bl)


class WeatherObservation(BaseModel):
    kind: str                       # "transport" or "flood"
    link_id: str = None
    load_pct: float = None
    zone_id: str = None
    flooded: bool = None


@app.post("/organizer/weather/observe")
def weather_observe(body: WeatherObservation, who=Depends(require_organizer)):
    """
    A field report that the twin learns from: a measured transport load (updates the
    learned rain effect), or whether a zone is flooded (updates its flood estimate).
    """
    _require_data()
    m = weather_twin.model(DATA_DIR)
    live = _live_weather()
    if body.kind == "transport":
        if body.link_id is None or body.load_pct is None:
            raise HTTPException(400, "Give the link and its measured load.")
        rain = 0.0
        if live.get("available"):
            ends = [l for l in get_transport(_clock_hour()) if l["link_id"] == body.link_id]
            if ends:
                by = {z["zone_id"]: z["rain_mm_h"] for z in live["zones"]}
                rain = (by.get(ends[0]["from_zone"], 0) + by.get(ends[0]["to_zone"], 0)) / 2
        try:
            change = m.observe_transport(body.link_id, _clock_hour(), rain, body.load_pct)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"learned": True, "rain_mm_h": rain, "transport_load_multiplier": change}
    if body.kind == "flood":
        if body.zone_id not in {z["zone_id"] for z in _zone_rows()} or body.flooded is None:
            raise HTTPException(400, "Give a zone and whether it is flooded.")
        m.observe_flood(body.zone_id, body.flooded)
        return {"learned": True, "zone_id": body.zone_id, "flooded": body.flooded}
    raise HTTPException(400, "kind must be 'transport' or 'flood'.")


# --------------------------------------------------------------------------- domain model (Nugen)
# EventPulse's own model: a base model aligned on the EventPulse operations corpus through
# Nugen (nugen_align.py). It turns a situation into an operator's action plan and an
# attendee message. Every answer is checked against the situation it was given; if it
# isn't configured, can't be reached, or invents a number, the playbook answers instead.

def _plan_facts(hour, twin=None, duration_h=3):
    live = _live_weather() if twin is None else None
    system = _weather_system(hour, duration_h, live)
    zones = {z["id"]: z["name"] for z in system["zones"]}
    transport = [{**l, "from_name": zones[l["from_zone"]], "to_name": zones[l["to_zone"]]} for l in system["links"]]
    extras = {"alerts": get_alerts(hour), "transport": transport, "incentives": get_incentives(hour)}
    v = system.get("venue")
    if v:
        extras["gates"] = get_gates(v["zone_id"], hour)
        extras["reverse_uber"] = get_reverse_uber(v["zone_id"], hour)
        opts = get_exit_redirect(v["zone_id"], hour).get("options") or []
        if v["phase"] == "leaving" and opts:
            extras["best_exit"] = opts[0]
    return playbook.gather(system, twin, extras)


_plan_cache = {}


def _domain_plan(facts):
    """Plan from the aligned model when possible, the playbook otherwise. Cached per situation."""
    text = playbook.situation(facts)
    fallback = playbook.plan(facts)
    base = {"situation": text, "source": "playbook", "model": None, "confidence": None, "rejected": None, **fallback}
    if not nugen.configured(DATA_DIR):
        return base
    hit = _plan_cache.get(text)
    if hit and time.time() - hit[0] < 300:
        return hit[1]
    try:
        r = nugen.chat([{"role": "system", "content": playbook.SYSTEM_PROMPT},
                        {"role": "user", "content": text}], model=nugen.model_id(DATA_DIR))
        ok, why = playbook.check(r["text"], text)
        if ok:
            p = playbook.parse(r["text"])
            out = {**base, "source": "nugen", "model": r["model"], "confidence": r["confidence"],
                   "latency_ms": r["latency_ms"], "actions": p["actions"],
                   "attendee_message": p["attendee_message"] or fallback["attendee_message"]}
        else:
            out = {**base, "model": r["model"], "rejected": why}
    except nugen.NugenError as e:
        out = {**base, "rejected": str(e)}
    if len(_plan_cache) > 200:
        _plan_cache.clear()
    _plan_cache[text] = (time.time(), out)
    return out


@app.get("/ai/nugen/verify")
def nugen_verify():
    """Checked live with Nugen (at most every 5 minutes): is the key accepted, and where do
    EventPulse's documents, benchmark and alignment stand on the account. The key itself is
    never returned, only a masked form."""
    return nugen.verify(DATA_DIR)


@app.get("/ai/nugen")
def nugen_status():
    """Where EventPulse's domain model stands: base model, alignment, evaluation, deployment."""
    st = nugen.load_state(DATA_DIR)
    return {"configured": nugen.configured(DATA_DIR), "has_key": bool(nugen.api_key()),
            "model_id": nugen.model_id(DATA_DIR), **{k: st.get(k) for k in (
                "base_model_id", "base_model_name", "alignment_id", "alignment_name", "alignment_status",
                "document_ids", "benchmark_id", "benchmark_questions", "corpus", "deployment_status",
                "performance_metrics", "evaluation", "updated_at", "stages")}}


@app.get("/ai/plan")
def ai_plan(hour: int = None):
    """The operator's action plan for right now, under the live weather."""
    _require_data()
    h = _clock_hour() if hour is None else int(hour)
    return _domain_plan(_plan_facts(h, weather_state(h)))


@app.post("/weather-twin/plan")
def weather_plan(body: WeatherScenario, request: Request):
    """The action plan for a what-if: simulate it, then plan for the result."""
    twin = weather_simulate(body, request)
    return _domain_plan(_plan_facts(twin["hour"], twin, body.duration_h))


# --------------------------------------------------------------------------- AI layer
# The model asks these read-only tools for numbers and explains them; it can not change
# anything. Without ANTHROPIC_API_KEY every endpoint below still answers, more plainly.

def _copilot_tools(hour):
    obj = lambda props, req=(): {"type": "object", "properties": props, "required": list(req)}
    s, i = {"type": "string"}, {"type": "integer"}
    return {
        "city_summary": {"description": "Headline numbers for the whole city right now: occupancy, spare beds, alerts.",
                         "schema": obj({}), "fn": lambda: get_summary(hour)},
        "top_alerts": {"description": "Current critical and warning alerts, worst first.",
                       "schema": obj({}), "fn": lambda: get_alerts(hour)[:8]},
        "busiest_zones": {"description": "The most crowded zones with occupancy and spare capacity.",
                          "schema": obj({"limit": i}),
                          "fn": lambda limit=6: sorted(get_zones(hour), key=lambda z: -z["occupancy_pct"])[:max(1, min(int(limit), 12))]},
        "list_events": {"description": "Events and venues: ids, names, start and end hours, state.",
                        "schema": obj({}), "fn": lambda: events(hour)},
        "venue_gates": {"description": "Live gate status and load for one venue zone (e.g. Z01).",
                        "schema": obj({"venue_zone_id": s}, ["venue_zone_id"]),
                        "fn": lambda venue_zone_id: get_gates(venue_zone_id, hour)},
        "exit_plan": {"description": "Balanced exit split for a venue: which gate is over-drawn and how many people to redirect.",
                      "schema": obj({"venue_zone_id": s}, ["venue_zone_id"]),
                      "fn": lambda venue_zone_id: get_reverse_uber(venue_zone_id, hour)},
        "simulate_move": {"description": "What-if: move N visitors out of a zone; returns where they land and whether the alert clears. compliance is 0-1.",
                          "schema": obj({"zone_id": s, "move": i, "compliance": {"type": "number"}}, ["zone_id", "move"]),
                          "fn": lambda zone_id, move, compliance=0.75: simulate(zone_id, int(move), hour, float(compliance))},
        "cancel_event": {"description": "What-if: cancel an event now; who is inside and must be evacuated.",
                         "schema": obj({"event_id": s}, ["event_id"]), "fn": lambda event_id: scenario_cancel(event_id, hour)},
        "surge_event": {"description": "What-if: this many extra unexpected arrivals at an event.",
                        "schema": obj({"event_id": s, "extra": i}, ["event_id", "extra"]),
                        "fn": lambda event_id, extra: scenario_surge(event_id, int(extra), hour)},
        "live_weather": {"description": "Live weather now and the next 6 hours across the city, plus social reports of weather trouble.",
                         "schema": obj({}), "fn": lambda: {"weather": (_live_weather() or {}).get("city"),
                                                            "social": {k: _social().get(k) for k in ("counts_24h", "signal_level")}}},
        "weather_impact_now": {"description": "The weather twin's current prediction: how today's live weather is affecting zones, transport, the venue and visitors, with ranges.",
                               "schema": obj({}), "fn": lambda: {k: weather_state(hour).get(k) for k in ("scenario", "effects", "city", "venue")}},
        "weather_whatif": {"description": "What-if for weather: rain in mm/h, duration in hours, temperature C, wind km/h, optionally a storm centred on a zone id with a radius in km. Returns cascading effects with 10th-90th percentile ranges. Read-only.",
                           "schema": obj({"rain_mm_h": {"type": "number"}, "duration_h": {"type": "number"}, "temp_c": {"type": "number"},
                                          "wind_kmh": {"type": "number"}, "zone_id": s, "radius_km": {"type": "number"}}),
                           "fn": lambda rain_mm_h=0, duration_h=3, temp_c=30, wind_kmh=10, zone_id=None, radius_km=None: (
                               lambda r: {k: r[k] for k in ("effects", "city", "venue")})(weather_twin.simulate(
                                   _weather_system(hour, float(duration_h), _live_weather()),
                                   weather_twin.parse_scenario({"rain_mm_h": rain_mm_h, "duration_h": duration_h, "temp_c": temp_c,
                                                                "wind_kmh": wind_kmh, "radius_km": radius_km if zone_id else None,
                                                                "center": {"zone_id": zone_id} if zone_id else None}, get_zones(hour))))},
        "close_link": {"description": "What-if: a transport link closes entirely; is there still a route?",
                       "schema": obj({"link_id": s}, ["link_id"]), "fn": lambda link_id: scenario_close_link(link_id, hour)},
    }


def _copilot_router_tools(hour):
    """What the copilot's question router may read or simulate (all read-only)."""
    def whatif(p):
        zones = get_zones(hour)
        sc = weather_twin.parse_scenario(p, zones)
        return weather_twin.simulate(_weather_system(hour, sc["duration_h"], _live_weather()), sc,
                                     weather_twin.normal_weather(sc["duration_h"]))
    return {
        "zones": lambda: get_zones(hour), "alerts": lambda: get_alerts(hour),
        "gates": lambda zid: get_gates(zid, hour), "exits": lambda zid: get_exit_redirect(zid, hour),
        "transport": lambda: get_transport(hour), "hotels": lambda: hotels(hour), "events": lambda: events(hour),
        "forecast": lambda: get_forecast(hour), "recommend": lambda zid: get_recommendations(zid, hour),
        "simulate": lambda zid, move: simulate(zid, int(move), hour, 0.75),
        "surge": lambda eid, extra: scenario_surge(eid, int(extra), hour),
        "cancel": lambda eid: scenario_cancel(eid, hour), "close_link": lambda lid: scenario_close_link(lid, hour),
        "weather_now": lambda: weather_state(hour), "weather_whatif": whatif, "social": _social,
    }


_copilot_calls = defaultdict(deque)
_refines = {}


def _queue_refine(question, facts, hour, role):
    """Remember an answer the Nugen-aligned model may reword; the page asks for it separately."""
    if not (nugen.configured(DATA_DIR) and nugen.available()):
        return None
    now = time.time()
    for k in [k for k, v in _refines.items() if now - v["at"] > 300]:
        del _refines[k]
    rid = secrets.token_hex(8)
    _refines[rid] = {"at": now, "q": question, "facts": facts, "hour": hour, "role": role}
    return rid


class RefineRequest(BaseModel):
    id: str


@app.post("/copilot/refine")
def copilot_refine(body: RefineRequest, authorization: str = Header(None)):
    """
    The Nugen-aligned EventPulse model's wording of an answer the copilot already gave. Slow
    (the model runs on Nugen), so the page shows the fact-based answer first and asks for this
    after. Only returned if it agrees with the facts (engine/copilot.faithful).
    """
    if not authorization or not authorization.startswith("Bearer ") or not verify_token(authorization[7:]):
        raise HTTPException(401, "Sign in required")
    job = _refines.pop(body.id, None)
    if job is None:
        raise HTTPException(404, "Nothing to refine.")
    hour = job["hour"]
    try:
        r = nugen.chat([{"role": "system", "content": playbook.SYSTEM_PROMPT_COPILOT},
                        {"role": "user", "content": f"Facts:\n{job['facts']}\n\nQuestion: {job['q']}"}],
                       model=nugen.model_id(DATA_DIR), max_tokens=220, temperature=0.1, timeout=25)
    except nugen.NugenError as e:
        return {"ok": False, "reason": str(e)[:160]}
    evs = events(hour)["all"]
    names = [z["name"] for z in get_zones(hour)] + [e["name"] for e in evs]
    names += [g["name"] for v in analyse_venues(current_hour=hour) for g in (v["gates"] or [])]
    ok, why = copilot.faithful(r["text"], job["facts"], sorted(set(names)), job["q"], [e["name"] for e in evs])
    if not ok:
        return {"ok": False, "reason": f"the model's answer {why}", "model": r["model"]}
    return {"ok": True, "text": r["text"], "model": r["model"], "confidence": r["confidence"], "latency_ms": r["latency_ms"]}


class AttendeeCopilotRequest(BaseModel):
    question: str


@app.post("/attendee/copilot")
def attendee_copilot(body: AttendeeCopilotRequest, who=Depends(require_attendee)):
    """The attendee's copilot: about their own events, in their terms."""
    _require_data()
    q = _copilot_calls["att:" + who["sub"]]
    now = time.time()
    while q and now - q[0] > 300:
        q.popleft()
    if len(q) >= 30:
        raise HTTPException(429, "That's a lot of questions. Give it a few minutes.")
    q.append(now)
    question = (body.question or "").strip()
    if not question:
        raise HTTPException(400, "Ask a question first.")
    if len(question) > 400:
        raise HTTPException(400, "Keep the question under 400 characters.")
    hour = _clock_hour()
    regs = list_registrations(who["sub"])
    a = copilot.answer_attendee(question, hour, _copilot_router_tools(hour), regs)
    return {"provider": "eventpulse", "hour": hour, "intent": a["intent"], "steps": a["steps"], "answer": a["answer"],
            "refine_id": _queue_refine(question, a["answer"], hour, "attendee")}


def _briefing(hour):
    """The no-AI answer: the same facts the model would start from, laid out plainly."""
    sm = get_summary(hour)
    lines = [f"City at {sm['citywide_occupancy_pct']}% occupancy with {sm['spare_capacity']:,} beds free. "
             f"{sm['zones_saturated']} of {sm['zones_total']} zones are saturated; "
             f"{sm['alerts_critical']} critical and {sm['alerts_warning']} warning alerts."]
    for a in get_alerts(hour)[:3]:
        lines.append("- " + str(a.get("message") or a.get("zone_name") or a))
    return "\n".join(lines)


class CopilotRequest(BaseModel):
    question: str
    history: list = []


@app.get("/ai/config")
def ai_config():
    return {"enabled": ai.configured()}


@app.post("/organizer/copilot")
def organizer_copilot(body: CopilotRequest, who=Depends(require_organizer)):
    """Ask a question in plain words; the copilot reads the live model and answers."""
    _require_data()
    q = _copilot_calls[who["sub"]]
    now = time.time()
    while q and now - q[0] > 300:
        q.popleft()
    if len(q) >= 20:
        raise HTTPException(429, "That's a lot of questions. Give it a few minutes.")
    q.append(now)
    hour = _clock_hour()
    question = (body.question or "").strip()
    if not question:
        raise HTTPException(400, "Ask a question first.")
    if len(question) > 600:
        raise HTTPException(400, "Keep the question under 600 characters.")
    if not ai.configured():
        # Answer the question asked, from the live engines (engine/copilot.py). When the
        # Nugen-aligned EventPulse model is deployed it phrases the answer from those same
        # facts, and is only used if it adds no number the facts don't contain.
        grounded = copilot.answer(question, hour, _copilot_router_tools(hour))
        out = {"ai": False, "provider": "eventpulse", "hour": hour, "intent": grounded["intent"],
               "steps": grounded["steps"], "answer": grounded["answer"]}
        out["refine_id"] = _queue_refine(question, grounded["answer"], hour, "organizer")
        return out
    try:
        out = ai.run_copilot(body.question, _copilot_tools(hour), hour, body.history)
    except ai.AIError as e:
        raise HTTPException(502, str(e))
    return {"ai": True, "hour": hour, **out}


@app.get("/status/{token}/guidance")
def status_guidance(token: str, hour: int = None):
    """One short message for the attendee, right now. Plain text always; warmer when AI is on."""
    _require_data()
    h = _clock_hour() if hour is None else hour
    status = attendee_status(token, h)
    if "error" in status:
        raise HTTPException(404, status["error"])
    if nugen.configured(DATA_DIR) and status["event_state"] in ("arriving", "running", "exiting"):
        facts = _plan_facts(h, weather_state(h))
        if facts["event"] and facts["event"]["name"] == status["event_name"]:
            p = _domain_plan(facts)
            if p["source"] == "nugen" and p["attendee_message"]:
                return {"text": p["attendee_message"], "source": "nugen", "confidence": p["confidence"]}
    return ai.guidance(status)


_MEDIA = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".svg": "image/svg+xml"}


@app.post("/organizer/venues/{venue_zone_id}/map/suggest")
def suggest_gates(venue_zone_id: str, who=Depends(require_organizer)):
    """Read the uploaded floor plan and propose gate positions. Nothing is saved: the organiser
    reviews the proposals and adds the ones that are right."""
    _venue_call(venue_admin.require_edit, venue_zone_id, who["sub"])
    if not ai.configured():
        raise HTTPException(503, "Reading a plan needs AI switched on. Place the gates by hand instead.")
    for ext in MAP_EXTENSIONS:
        path = os.path.join(MAPS_DIR, f"{venue_zone_id}{ext}")
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = f.read()
            try:
                return {"gates": ai.suggest_gates(data, _MEDIA[ext])}
            except ai.AIError as e:
                raise HTTPException(502, str(e))
    raise HTTPException(404, "Upload a floor plan first.")


@app.get("/scenarios/cancel/{event_id}")
def scenario_cancel(event_id: str, hour: int = DEFAULT_HOUR):
    """What-if: cancel this event right now — who is already in, who has to be evacuated."""
    _require_data()
    result = simulate_event_cancellation(event_id, hour)
    if "error" in result:
        raise HTTPException(404, result["error"])
    return result


@app.get("/scenarios/surge/{event_id}")
def scenario_surge(event_id: str, extra: int, hour: int = DEFAULT_HOUR):
    """What-if: `extra` unexpected arrivals on top of what this event already expects."""
    _require_data()
    result = simulate_demand_surge(event_id, extra, hour)
    if "error" in result:
        raise HTTPException(404, result["error"])
    return result


@app.get("/scenarios/close-link/{link_id}")
def scenario_close_link(link_id: str, hour: int = DEFAULT_HOUR):
    """What-if: this transport link closes entirely — is there still a way through?"""
    _require_data()
    result = simulate_transport_closure(link_id, hour)
    if "error" in result:
        raise HTTPException(404, result["error"])
    return result


@app.get("/reverse-uber/{venue_zone_id}")
def get_reverse_uber(venue_zone_id: str, hour: int = DEFAULT_HOUR):
    """
    System-optimal egress split: what the naive, habit-driven per-gate load
    looks like against a capacity-balanced allocation, and how many people
    need to be actively redirected off the over-drawn gate to close the gap.
    """
    _require_data()
    return optimal_exit_distribution(venue_zone_id, hour)


@app.get("/incentives")
def get_incentives(hour: int = DEFAULT_HOUR):
    """
    Active demand-shifting offers: which crowded zones warrant one, which
    quieter zone each points to, and the discount sized to how much
    redirection is actually needed.
    """
    _require_data()
    return generate_incentives(current_hour=hour)


@app.get("/routes/{from_zone}/{to_zone}")
def routes(from_zone: str, to_zone: str, hour: int = DEFAULT_HOUR):
    """Ways between two zones at this hour, least congested first."""
    _require_data()
    return find_routes(from_zone, to_zone, hour)


@app.get("/reroute/{link_id}")
def reroute(link_id: str, hour: int = DEFAULT_HOUR):
    """Alternatives for one congested link, reached from a transport alert."""
    _require_data()
    out = reroute_for_link(link_id, hour)
    if out is None:
        raise HTTPException(status_code=404, detail=f"No link {link_id}")
    return out


@app.get("/restaurants")
def restaurants(hour: int = DEFAULT_HOUR):
    """
    Bookable-table inventory across every zone, biased toward quiet ones —
    the attendee-facing "go here while you wait" list the incentive engine's
    offers point at.
    """
    _require_data()
    return build_restaurants(current_hour=hour)


@app.get("/hotels")
def hotels(hour: int = DEFAULT_HOUR):
    """
    Bookable inventory across every zone, with live availability.

    The attendee-facing view of the same data the command view works from, so
    both sides agree on what is actually free.
    """
    _require_data()
    return build_hotels(current_hour=hour)


@app.get("/history")
def get_history(hour: int = DEFAULT_HOUR, window: int = 24, top: int = 5):
    """
    Occupancy trajectory for the busiest zones over a trailing window.

    Feeds the trend chart: enough history to show where a zone has come from,
    ending at the hour being viewed so the line grows as the clock advances.
    """
    _require_data()

    occ = pd.read_csv(os.path.join(DATA_DIR, "occupancy_timeline.csv"))
    zones = pd.read_csv(os.path.join(DATA_DIR, "zones.csv"))
    names = dict(zip(zones["zone_id"], zones["name"]))

    lo = max(0, hour - window)
    span = occ[(occ["hour_offset"] >= lo) & (occ["hour_offset"] <= hour)]

    # Rank by pressure at the current hour, not across the whole window.
    now = span[span["hour_offset"] == hour].sort_values("occupancy_pct", ascending=False)
    keep = list(now["zone_id"].head(top))

    series = []
    for zid in keep:
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


@app.get("/transport")
def get_transport(hour: int = DEFAULT_HOUR):
    """Transport links with their load at the given hour, for the map view."""
    _require_data()

    t = pd.read_csv(os.path.join(DATA_DIR, "transport.csv"))

    load_path = os.path.join(DATA_DIR, "transport_load.csv")
    if os.path.exists(load_path):
        load = pd.read_csv(load_path)
        snap = load[load["hour_offset"] == hour]
        t = t.merge(
            snap[["link_id", "current_load", "load_pct"]],
            on="link_id", how="left",
        )
        t["load_pct"] = t["load_pct"].fillna(0.0)
        t["current_load"] = t["current_load"].fillna(0).astype(int)

    return t.to_dict("records")


@app.get("/ml/forecast")
def ml_forecast(hour: int = None):
    """Gradient boosting forecast, 6 hours ahead."""
    if not ML_AVAILABLE:
        raise HTTPException(503, "scikit-learn not installed")

    _require_data()
    preds = model_predict(current_hour=hour)
    if preds is None:
        raise HTTPException(
            503, "Model not trained yet. Run: python -m engine.model"
        )
    return preds


@app.get("/ml/benchmark")
def ml_benchmark():
    """
    Model scorecard against the trend baseline on held-out data.

    Retrains on request. At this data size that takes under a second, and it
    means the figures shown are always the real ones rather than something
    cached from an earlier dataset.
    """
    if not ML_AVAILABLE:
        # Deployed without scikit-learn (serverless size limit): serve the scorecard it last produced.
        try:
            with open(os.path.join(DATA_DIR, "ml_benchmark.json")) as f:
                return json.load(f)
        except (OSError, ValueError):
            raise HTTPException(503, "scikit-learn not installed")

    _require_data()
    _model, _df, m = model_train(verbose=False)

    return {
        "horizon_hours": m["horizon_hours"],
        "train_rows": m["train_rows"],
        "test_rows": m["test_rows"],
        "split_at_hour": m["split_at_hour"],
        "baseline": {
            "name": "Trend projection",
            "mae": round(m["baseline_mae"], 2),
            "r2": round(m["baseline_r2"], 3),
        },
        "model": {
            "name": "Gradient boosting",
            "mae": round(m["model_mae"], 2),
            "r2": round(m["model_r2"], 3),
        },
        "mae_improvement_pct": m["improvement_pct"],
    }


# --------------------------------------------------------------------------
# Serve the dashboard
# --------------------------------------------------------------------------
# In deployment there is one service, not two: the API also serves the page, so
# the frontend can call it same-origin and there is a single URL to share.
# Mounted last so it never shadows an API route.
FRONTEND_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend"
)

if os.path.isdir(FRONTEND_DIR):
    @app.get("/")
    def dashboard():
        return FileResponse(os.path.join(FRONTEND_DIR, "home.html"))

    app.mount("/", StaticFiles(directory=FRONTEND_DIR), name="frontend")
