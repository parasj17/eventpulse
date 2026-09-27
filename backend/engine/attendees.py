"""
Attendee registration and the live status page.

A signed-in attendee registers for an event. That creates a registration with
its own unguessable token and emails them a link to `GET /status/{token}`, a
no-login page showing live conditions for their event and, once it ends, which
gate to head for. The token is the credential for that page, so the link works
from any device without signing in.

Registrations are keyed to the attendee's email (accounts live in
engine/accounts.py). Older registrations made with a phone number, before
accounts existed, are still readable and their links still work; they just are
not attached to an account.

Email goes out through engine/mailer.py: real SMTP when configured, a
clearly-labelled simulation otherwise. The link in the email must be reachable
from the attendee's device, so set EVENTPULSE_PUBLIC_URL to wherever the app is
hosted (the API falls back to the address the request arrived on, which is only
useful on a local network).

Run:  python -m engine.attendees
"""

import os
import time
import uuid

import pandas as pd

from engine import db
from engine.accounts import normalise_email
from engine.density import zone_density_for_venue
from engine.forecast import DATA_DIR
from engine.mailer import send_email
from engine.venues import EGRESS_SURGE_WINDOW_HOURS, analyse_all

ATTENDEES_CSV = "attendees.csv"
COLUMNS = ["attendee_token", "email", "event_id", "venue_zone_id", "registered_hour", "partner"]

# Registering twice for the same event reuses the existing link, and an email to
# the same address and event is not repeated inside this window. Every message
# is a real send, so registration must not be a way to flood someone's inbox.
# Kept in memory: a restart clears it, which is fine for a limit whose only job
# is to stop rapid repeats.
RESEND_COOLDOWN_SECONDS = 60
_last_sent = {}


def _path(data_dir):
    return os.path.join(data_dir or DATA_DIR, ATTENDEES_CSV)


DB_COLS = ["attendee_token", "email", "event_id", "venue_zone_id", "registered_hour", "partner", "legacy_phone"]
_migrated = set()


def _load_csv(data_dir):
    path = _path(data_dir)
    if not os.path.exists(path):
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(path, dtype=str).fillna("")
    # Registrations from before accounts existed carry a phone number and no email.
    for col in ("email", "partner"):          # older files predate these columns
        if col not in df.columns:
            df[col] = ""
    if "phone_number" in df.columns:
        df = df.rename(columns={"phone_number": "legacy_phone"})
    return df


def _upsert(df):
    rows = [tuple(str(r.get(c, "") if r.get(c, "") is not None else "") for c in DB_COLS) for r in df.to_dict("records")]
    if rows:
        db.executemany("insert into ep_attendees (" + ", ".join(DB_COLS) + ") values (" + ", ".join(["%s"] * len(DB_COLS)) + ") "
                       "on conflict (attendee_token) do update set " +
                       ", ".join(f"{c}=excluded.{c}" for c in DB_COLS[1:]), rows)


def _load(data_dir):
    """All registrations: Supabase (table ep_attendees) when configured, else the CSV."""
    if not db.enabled():
        return _load_csv(data_dir)
    if data_dir not in _migrated:              # copy in registrations made before Supabase, once
        old = _load_csv(data_dir)
        if len(old):
            known = {r[0] for r in db.execute("select attendee_token from ep_attendees", fetch=True)}
            _upsert(old[~old["attendee_token"].isin(known)])
        _migrated.add(data_dir)
    rows = db.execute("select " + ", ".join(DB_COLS) + " from ep_attendees", fetch=True)
    return pd.DataFrame(rows, columns=DB_COLS).fillna("") if rows else pd.DataFrame(columns=DB_COLS)


def _save(df, data_dir):
    if db.enabled():
        _upsert(df)
    else:
        df.to_csv(_path(data_dir), index=False)


def register_attendee(email, event_id, current_hour, base_url="", data_dir=None, send=True, partner=""):
    """
    Register one attendee for one event and email them their status link.
    Returns the registration plus the outcome of the email. `send=False` skips the
    email, for a partner that delivers the link inside its own app. `partner` records which
    platform registered them, so only that platform can read their status through the API.
    """
    data_dir = data_dir or DATA_DIR
    email = normalise_email(email)

    events = pd.read_csv(os.path.join(data_dir, "events.csv"))
    row = events[events["event_id"] == event_id]
    if row.empty:
        return {"error": f"No event {event_id}"}

    venue_zone_id = row.iloc[0]["zone_id"]
    event_name = row.iloc[0]["name"]

    attendees = _load(data_dir)
    existing = attendees[(attendees["email"] == email) & (attendees["event_id"] == event_id)]
    already = not existing.empty

    if already:
        reg = existing.iloc[0].to_dict()
        if partner and not reg.get("partner"):
            attendees.loc[existing.index[0], "partner"] = partner
            _save(attendees, data_dir)
            reg["partner"] = partner
    else:
        reg = {
            "attendee_token": uuid.uuid4().hex[:12],
            "email": email,
            "event_id": event_id,
            "venue_zone_id": venue_zone_id,
            "registered_hour": current_hour,
            "partner": partner,
        }
        attendees = pd.concat([attendees, pd.DataFrame([reg])], ignore_index=True)
        _save(attendees, data_dir)

    link = f"{base_url}/status.html?token={reg['attendee_token']}"
    ev = row.iloc[0]
    subject, body, html = registration_email(
        event_name=event_name, venue_name=str(ev.get("venue_name", "the venue")),
        start_hour=int(ev["start_hour"]), end_hour=int(ev["end_hour"]),
        status_link=link, app_link=f"{base_url}/home.html", current_hour=current_hour)

    key = (email, event_id)
    now = time.time()
    if not send:
        mail = {"provider": "none", "status": "not_requested", "delivered": False, "to": email,
                "subject": subject, "body": body, "note": "No email was sent: the partner delivers the link."}
    elif already and now - _last_sent.get(key, 0) < RESEND_COOLDOWN_SECONDS:
        mail = {"provider": "none", "status": "skipped", "delivered": False, "to": email,
                "subject": subject, "body": body,
                "note": "An email was sent moments ago, so it was not repeated."}
    else:
        mail = send_email(email, subject, body, html=html)
        if mail["status"] != "failed":
            _last_sent[key] = now

    return {"attendee_token": reg["attendee_token"], "email": email, "event_id": event_id,
            "event_name": event_name, "venue_zone_id": venue_zone_id,
            "status_link": link, "already_registered": already, "email_result": mail}


def when(hour):
    """The event clock as the app shows it: day number and time, e.g. 'Day 5, 10:00'."""
    return f"Day {hour // 24 + 1}, {hour % 24:02d}:00"


def registration_email(event_name, venue_name, start_hour, end_hour, status_link, app_link, current_hour=None):
    """Subject, plain text and HTML for the 'you're registered' email."""
    gates = start_hour - ARRIVAL_LEAD_HOURS
    until = ""
    if current_hour is not None and int(current_hour) < gates:
        h = gates - int(current_hour)
        until = f" (in {h} hour{'s' if h != 1 else ''})"
    subject = f"You're registered for {event_name} - gates open {when(gates)}"
    text = (
        f"You're registered for {event_name} at {venue_name}.\n\n"
        f"Gates open:  {when(gates)}{until}\n"
        f"Starts:      {when(start_hour)}\n"
        f"Ends:        {when(end_hour)}\n\n"
        f"Your live page (quietest gate on the way in, fastest way out after):\n{status_link}\n\n"
        f"Open EventPulse:\n{app_link}\n\n"
        f"Tap either link on your phone and it opens in your browser. No app or sign-in needed; keep this email.\n"
    )
    esc = lambda s: str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    row = lambda k, v: (f'<tr><td style="padding:6px 16px 6px 0;color:#6a6a6a;font-size:14px">{k}</td>'
                        f'<td style="padding:6px 0;font-size:15px;font-weight:700;color:#111">{v}</td></tr>')
    btn = lambda href, label, bg: (f'<a href="{esc(href)}" style="display:inline-block;background:{bg};color:#fff;text-decoration:none;'
                                   f'font-weight:700;font-size:15px;padding:12px 22px;border-radius:8px;margin:6px 8px 6px 0">{label}</a>')
    html = f"""<!doctype html><html><body style="margin:0;background:#f4f4f5;font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0"><tr><td align="center" style="padding:24px 12px">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="max-width:560px;background:#fff;border-radius:12px;overflow:hidden">
<tr><td style="background:#0a0a0a;padding:18px 24px;color:#fff;font-size:20px;font-weight:800;letter-spacing:.5px">Event<span style="color:#ff3b1f">Pulse</span></td></tr>
<tr><td style="padding:24px">
<div style="font-size:13px;color:#ff3b1f;font-weight:700;text-transform:uppercase;letter-spacing:1px">You're registered</div>
<h1 style="margin:6px 0 4px;font-size:24px;color:#111">{esc(event_name)}</h1>
<div style="color:#555;font-size:15px;margin-bottom:16px">{esc(venue_name)}</div>
<table role="presentation" cellpadding="0" cellspacing="0" style="margin-bottom:18px">
{row("Gates open", esc(when(gates)) + esc(until))}{row("Starts", esc(when(start_hour)))}{row("Ends", esc(when(end_hour)))}
</table>
<p style="font-size:15px;color:#333;line-height:1.5;margin:0 0 10px">Your live page tells you the quietest gate on the way in and the fastest way out afterwards.</p>
{btn(status_link, "Open my live page", "#ff3b1f")}{btn(app_link, "Open EventPulse", "#111")}
<p style="font-size:13px;color:#777;line-height:1.5;margin:18px 0 0">Tap a button on your phone and it opens in your browser. No app or sign-in needed. If the buttons don't work, copy this link:<br>
<a href="{esc(status_link)}" style="color:#ff3b1f;word-break:break-all">{esc(status_link)}</a></p>
</td></tr></table></td></tr></table></body></html>"""
    return subject, text, html


def list_registrations(email, data_dir=None):
    """Everything this email has registered for, with the link to each status page."""
    data_dir = data_dir or DATA_DIR
    email = normalise_email(email)
    attendees = _load(data_dir)
    mine = attendees[attendees["email"] == email]
    if mine.empty:
        return []

    events = pd.read_csv(os.path.join(data_dir, "events.csv")).set_index("event_id")
    out = []
    for _, r in mine.iterrows():
        ev = events.loc[r["event_id"]] if r["event_id"] in events.index else None
        out.append({
            "attendee_token": r["attendee_token"],
            "event_id": r["event_id"],
            "event_name": ev["name"] if ev is not None else r["event_id"],
            "venue_name": ev["venue_name"] if ev is not None else "",
            "start_hour": int(ev["start_hour"]) if ev is not None else None,
            "status_link": f"status.html?token={r['attendee_token']}",
        })
    return out


def get_attendee(token, data_dir=None):
    attendees = _load(data_dir)
    row = attendees[attendees["attendee_token"] == token]
    return None if row.empty else row.iloc[0].to_dict()


ARRIVAL_LEAD_HOURS = 4   # gates begin filling this long before the start


def _event_state(hour, start, end):
    if hour < start - ARRIVAL_LEAD_HOURS:
        return "upcoming"
    if hour < start:
        return "arriving"
    if hour <= end:
        return "running"
    if hour <= end + EGRESS_SURGE_WINDOW_HOURS:
        return "exiting"
    return "finished"


def event_status(event_id, current_hour, data_dir=None):
    """
    Live conditions for one event at `current_hour`: where it is in its life, the state of
    each gate, how full the venue is, and, once it ends, which gate to head for. Shared by
    the attendee's status page and the partner API.
    """
    data_dir = data_dir or DATA_DIR
    from engine.exit_redirect import recommend_exit  # imported here to avoid a load-time cycle

    hour = int(current_hour)
    venue = next((v for v in analyse_all(current_hour=hour, data_dir=data_dir)
                  if v["event_id"] == event_id), None)
    if venue is None:
        return {"error": "That event no longer exists"}

    start, end = venue["start_hour"], venue["end_hour"]
    state = _event_state(hour, start, end)

    gates = sorted(
        ({"gate_id": g["gate_id"], "name": g["name"], "live_status": g["live_status"],
          "phase": g["phase"], "load_ratio": g["load_ratio"], "load_pct": g.get("load_pct", 0),
          "redirect_to": [t["name"] for t in g.get("redirect_to", [])],
          "out_of_service": g.get("status", "open") != "open"} for g in (venue["gates"] or [])),
        key=lambda g: g["name"])

    inside = None
    bowl = next((z for z in zone_density_for_venue(venue["zone_id"], hour, data_dir) if z["kind"] == "bowl"), None)
    if bowl and state in ("arriving", "running", "exiting"):
        inside = {"people": bowl["people_count"], "capacity": bowl["capacity"], "pct": bowl["occupancy_pct"]}

    exits = recommend_exit(venue["zone_id"], hour, data_dir=data_dir)["options"] if state == "exiting" else []

    return {
        "event_id": event_id,
        "event_name": venue["name"],
        "venue_zone_id": venue["zone_id"],
        "venue_name": venue["venue_name"],
        "hour": hour,
        "start_hour": start,
        "end_hour": end,
        "event_state": state,
        "hours_to_start": max(0, start - hour),
        "hours_to_end": max(0, end - hour),
        "hours_since_end": max(0, hour - end),
        "inside": inside,
        "gates": gates,
        "best_exit": exits[0] if exits else None,
        "all_exits": exits,
    }


def attendee_status(token, current_hour, data_dir=None):
    """Everything the live status page shows for one registration at `current_hour`."""
    attendee = get_attendee(token, data_dir or DATA_DIR)
    if attendee is None:
        return {"error": "Unknown or expired link"}
    out = event_status(attendee["event_id"], current_hour, data_dir)
    if "error" not in out:
        out = {"attendee_token": token, **out}
    return out


if __name__ == "__main__":
    r = register_attendee("demo@example.com", "E5", current_hour=100, base_url="http://127.0.0.1:8080", data_dir="/tmp")
    print({k: v for k, v in r.items() if k != "email_result"})
