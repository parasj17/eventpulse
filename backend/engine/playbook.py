"""
The EventPulse operations playbook: how a crowd-operations lead turns a situation into
actions.

Two jobs, one source of truth:

1. **Teacher for the domain model.** `domain_corpus.py` runs the city's engines across
   many hours and weather scenarios, writes each situation in the fixed format below
   and pairs it with what this playbook says to do. That corpus is what the base
   model is aligned on through Nugen (see nugen_align.py).
2. **The fallback.** In the running app the aligned model writes the plan. If it is not
   configured, unreachable, or its answer fails the checks (it must only use numbers
   and places that are in the situation), this playbook's own answer is shown instead.

The situation format is the contract between the two: the model sees at inference
exactly the shape it was aligned on.
"""

import re

WARN_PCT = 75.0
CRIT_PCT = 95.0
LINK_CRIT_PCT = 92.0
HEAT_C = 35.0

SYSTEM_PROMPT = (
    "You are the EventPulse operations assistant for mega-event crowd orchestration. "
    "Read the situation and reply with the operator's actions, most urgent first, as a numbered list, "
    "then one line starting 'Attendee message:'. Use only numbers, gates and places given in the situation."
)


SYSTEM_PROMPT_COPILOT = (
    "You are the EventPulse operations copilot for mega-event crowd orchestration. Answer the organiser's question in "
    "two to four plain sentences using only the facts given. Keep every number, gate and place exactly as in the facts; "
    "do not add any. If the facts don't answer the question, say what they do show."
)


# ---------------------------------------------------------------- facts
def gather(system, twin=None, extras=None):
    """
    Collect the facts a plan is written from. `system` is the dict api._weather_system
    returns; `twin` a weather-twin result (optional); `extras` any of: alerts,
    reverse_uber, best_exit, incentives, transport.
    """
    extras = extras or {}
    zones = sorted(system["zones"], key=lambda z: -z["occupancy_pct"])
    v = system.get("venue")
    f = {
        "hour": system["hour"],
        "event": None,
        "zones": [{"name": z["name"], "pct": round(z["occupancy_pct"]), "spare": int(z.get("spare_capacity", 0))} for z in zones[:3]],
        "spare_beds": int(sum(z.get("spare_capacity", 0) for z in system["zones"])),
        "alerts": [a["message"] for a in (extras.get("alerts") or []) if a.get("severity") == "critical"][:2],
        "links": [],
        "gates": [],
        "rebalance": None,
        "best_exit": None,
        "offer": None,
        "weather": None,
    }
    if v:
        f["event"] = {"name": v["event_name"], "venue": v["venue_name"], "phase": v["phase"],
                      "people": int(v.get("people") or v.get("expected") or 0)}
    for l in sorted(extras.get("transport") or [], key=lambda l: -l.get("load_pct", 0))[:2]:
        f["links"].append({"name": f"{l['mode']} {l['from_name']} - {l['to_name']}", "pct": round(l["load_pct"])})
    for g in (extras.get("gates") or [])[:6]:
        f["gates"].append({"name": g["name"], "status": g.get("live_status", "open"),
                           "load": round(float(g.get("load_pct", g.get("load_ratio", 0) * 100))),
                           "to": [t["name"] for t in g.get("redirect_to", [])]})
    # The live redirection the gates are doing: a gate at 95% closes and its overflow goes to gates with room.
    shedding = sorted((g for g in (extras.get("gates") or []) if g.get("redirected_out", 0) > 0 and g.get("phase") == "egress"),
                      key=lambda g: -g["redirected_out"])
    if shedding:
        g0 = shedding[0]
        f["rebalance"] = {"from": g0["name"], "people": int(g0["redirected_out"]),
                          "to": [{"name": t["name"], "people": int(t["people"])} for t in g0.get("redirect_to", [])],
                          "load_before": round(g0.get("demand_pct", 0) / 100, 2), "load_after": round(g0.get("load_pct", 0) / 100, 2)}
    be = extras.get("best_exit")
    if be:
        f["best_exit"] = be["name"]
    offers = extras.get("incentives") or []
    if offers:
        o = offers[0]
        f["offer"] = {"from": o["source_zone_name"], "to": o["destination_zone_name"], "discount": int(o["discount_pct"])}
    if twin:
        sc = twin["scenario"]
        floods = sorted((z for z in twin["zones"] if z["flood_pct"]["p50"] >= 50), key=lambda z: -z["flood_pct"]["p50"])
        cut = [l for l in twin["links"] if l["capacity_pct_baseline"] - l["capacity_pct"]["p50"] >= 25]
        tv = twin.get("venue")
        f["weather"] = {
            "label": sc.get("label", "weather"), "rain": round(sc["rain_mm_h"]), "hours": sc["duration_h"],
            "temp": round(sc["temp_c"]), "wind": round(sc["wind_kmh"]),
            "flood": [{"name": z["name"], "pct": round(z["flood_pct"]["p50"])} for z in floods[:3]],
            "links_cut": len(cut),
            "stranded": int(round(twin["city"]["stranded"]["p50"], -1)),
            "stranded_high": int(round(twin["city"]["stranded"]["p90"], -1)),
            "unhoused": int(round(twin["city"]["unhoused"]["p50"], -1)),
            "clear_h": round(tv["hours_to_clear"]["p50"], 1) if tv else None,
            "clear_h_normal": round(tv["hours_to_clear_baseline"], 1) if tv else None,
            "heat_calls": int(round(tv["heat_incidents"]["p50"])) if tv else 0,
        }
        # Where stranded people can go: zones with rooms, nearest the venue first.
        rooms = sorted((z for z in system["zones"] if z.get("spare_capacity", 0) > 200), key=lambda z: z.get("distance_km", 99))
        f["weather"]["rooms"] = [{"name": z["name"], "spare": int(z["spare_capacity"])} for z in rooms[:2]]
    return f


# ---------------------------------------------------------------- the situation, as text
def _day(h):
    return f"day {h // 24 + 1}, {h % 24:02d}:00"


def situation(f):
    """The fixed, line-per-fact format the domain model is aligned on."""
    L = [f"Time: {_day(f['hour'])}."]
    e = f["event"]
    if e:
        phase = {"arriving": "people arriving", "inside": "under way", "leaving": "ended, crowd leaving"}[e["phase"]]
        L.append(f"Event: {e['name']} at {e['venue']}, {phase}, {e['people']:,} people.")
    else:
        L.append("Event: none in the next few hours.")
    L.append("Busiest zones: " + "; ".join(f"{z['name']} {z['pct']}%" for z in f["zones"]) + f". Spare beds citywide: {f['spare_beds']:,}.")
    if f["alerts"]:
        L.append("Critical alerts: " + " | ".join(f["alerts"]))
    if f["links"]:
        L.append("Busiest transport: " + "; ".join(f"{l['name']} {l['pct']}%" for l in f["links"]) + ".")
    # While a rebalance is running, the exit plan is the gate picture; the live gate line
    # (entry backlog) would only contradict it.
    if f["gates"] and not f["rebalance"]:
        L.append("Gates: " + "; ".join(f"{g['name']} {g['status']} at {g['load']}%"
                                       + (f", redirecting to {', '.join(g['to'])}" if g["to"] else "") for g in f["gates"]) + ".")
    if f["best_exit"]:
        L.append(f"Best exit right now: {f['best_exit']}.")
    r = f["rebalance"]
    if r:
        L.append(f"Exit plan: move {r['people']:,} from {r['from']} to " + ", ".join(f"{t['name']} ({t['people']:,})" for t in r["to"])
                 + f"; the busiest gate goes from {round(r['load_before'] * 100)}% of capacity to {round(r['load_after'] * 100)}%.")
    if f["offer"]:
        o = f["offer"]
        L.append(f"Offer available: {o['discount']}% off in {o['to']} for people heading to {o['from']}.")
    w = f["weather"]
    if w:
        bits = [f"{w['label']}: rain {w['rain']} mm/h for {w['hours']:g} h, {w['temp']} C, gusts {w['wind']} km/h"]
        if w["flood"]:
            bits.append("flood risk " + ", ".join(f"{z['name']} {z['pct']}%" for z in w["flood"]))
        if w["links_cut"]:
            bits.append(f"{w['links_cut']} transport links lose a quarter or more of capacity")
        if w["clear_h"] is not None and w["clear_h"] - (w["clear_h_normal"] or 0) >= 0.3:
            bits.append(f"venue takes {w['clear_h']} h to empty instead of {w['clear_h_normal']} h")
        if w["stranded"] >= 100:
            bits.append(f"about {w['stranded']:,} visitors stranded (up to {w['stranded_high']:,})")
        if w["unhoused"] >= 100:
            bits.append(f"{w['unhoused']:,} with no bed in the city")
        if w["heat_calls"] >= 1:
            bits.append(f"{w['heat_calls']} heat-related medical calls expected")
        if w["rooms"] and w["stranded"] >= 100:
            bits.append("rooms free in " + ", ".join(f"{z['name']} ({z['spare']:,})" for z in w["rooms"]))
        L.append("Weather twin: " + "; ".join(bits) + ".")
    return "\n".join(L)


# ---------------------------------------------------------------- what an expert does
def plan(f):
    """The playbook's answer: ordered actions and one attendee message."""
    acts = []
    e, w, r = f["event"], f["weather"], f["rebalance"]
    if w and w["flood"]:
        names = ", ".join(z["name"] for z in w["flood"])
        acts.append(f"Stop routing crowds through {names}: flood risk is {w['flood'][0]['pct']}% at {w['flood'][0]['name']}. "
                    f"Warn shuttle and bus drivers and post marshals at low-lying crossings.")
    if r:
        acts.append(f"Rebalance the exit: send {r['people']:,} people from {r['from']} to "
                    + ", ".join(f"{t['name']} ({t['people']:,})" for t in r["to"])
                    + f" on screens and PA, bringing the busiest gate from {round(r['load_before'] * 100)}% to {round(r['load_after'] * 100)}% of capacity.")
    elif e and e["phase"] == "leaving" and f["best_exit"]:
        acts.append(f"Point everyone leaving to {f['best_exit']}, the quickest open exit right now.")
    closed = [g for g in f["gates"] if g["status"] in ("closed", "restricted")]
    if e and e["phase"] == "arriving" and closed:
        open_g = [g["name"] for g in f["gates"] if g["status"] == "open"]
        acts.append(f"{closed[0]['name']} is {closed[0]['status']} at {closed[0]['load']}%: send new arrivals to "
                    f"{open_g[0] if open_g else 'the least loaded gate'} and open extra lanes where you can.")
    if w and w["clear_h"] is not None and w["clear_h"] - (w["clear_h_normal"] or 0) >= 0.3 and e:
        acts.append(f"Plan for a slower exit: {w['clear_h']} h instead of {w['clear_h_normal']} h. Stagger the release by stand "
                    f"and keep concourses open so people can wait inside.")
    if w and w["stranded"] >= 100:
        where = " and ".join(f"{z['name']} ({z['spare']:,} rooms)" for z in w["rooms"]) or "the nearest zones with rooms"
        acts.append(f"About {w['stranded']:,} visitors may be stranded (up to {w['stranded_high']:,}): hold rooms in {where} "
                    f"and tell partner hotels now.")
        if w["unhoused"] >= 100:
            acts.append(f"{w['unhoused']:,} will find no bed: open a shelter in the venue concourse and arrange late transport.")
    if w and w["temp"] >= HEAT_C:
        calls = f", expect {w['heat_calls']} medical calls" if w["heat_calls"] else ""
        acts.append(f"Heat at {w['temp']} C: open water points and shaded queues, and put medical teams at the gates{calls}.")
    if w and w["wind"] >= 60:
        acts.append(f"Gusts of {w['wind']} km/h: suspend shuttles and buses on exposed routes and move people to the metro.")
    for l in f["links"]:
        if l["pct"] >= LINK_CRIT_PCT:
            acts.append(f"{l['name']} is at {l['pct']}%: add services or hold departures and release the crowd in waves.")
            break
    full = [z for z in f["zones"] if z["pct"] >= CRIT_PCT]
    if full and f["offer"]:
        o = f["offer"]
        acts.append(f"{full[0]['name']} is at {full[0]['pct']}%: push the {o['discount']}% offer in {o['to']} to new arrivals.")
    elif full:
        acts.append(f"{full[0]['name']} is at {full[0]['pct']}%: send new arrivals to zones with spare beds ({f['spare_beds']:,} citywide).")
    for a in f["alerts"]:
        if "will not get in" in a:
            acts.append(f"Oversubscribed: {a} Tell ticket holders without a seat early and stop walk-ups.")
            break
    if not acts:
        top = f["zones"][0]
        acts.append(f"No action needed. Keep watching {top['name']} at {top['pct']}%.")

    msg = attendee_message(f)
    return {"actions": acts[:6], "attendee_message": msg}


def attendee_message(f):
    e, w = f["event"], f["weather"]
    tail = ""
    if w and w["rain"] >= 10:
        tail = " It's raining hard: allow extra time."
    elif w and w["temp"] >= HEAT_C:
        tail = " It's very hot: carry water."
    if not e:
        return "No event traffic right now." + tail
    if e["phase"] == "leaving":
        where = f["rebalance"]["to"][0]["name"] if f["rebalance"] else f["best_exit"]
        return (f"{e['name']} has ended. Leave via {where}: it is quicker right now." if where else
                f"{e['name']} has ended. Take your time; the gates are clearing.") + tail
    if e["phase"] == "arriving":
        open_g = [g["name"] for g in f["gates"] if g["status"] == "open"]
        return (f"Head to {open_g[0]} for the shortest queue." if open_g else "Every gate is busy: expect a wait.") + tail
    return f"{e['name']} is under way. We'll tell you the best way out as it ends." + tail


def plan_text(p):
    return "\n".join(f"{i + 1}. {a}" for i, a in enumerate(p["actions"])) + f"\nAttendee message: {p['attendee_message']}"


# ---------------------------------------------------------------- checking a model's answer
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _nums(text):
    return {n.replace(",", "").rstrip(".") for n in _NUM.findall(text)}


def parse(text):
    """Split a model answer into numbered actions and the attendee message."""
    acts, msg = [], ""
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^attendee message\s*:\s*(.+)$", line, re.I)
        if m:
            msg = m.group(1).strip()
            continue
        m = re.match(r"^(?:\d+[.)]|[-*•])\s*(.+)$", line)
        if m:
            acts.append(m.group(1).strip())
    return {"actions": acts, "attendee_message": msg}


def check(answer, situation_text):
    """
    An aligned model's plan is only shown if it is well formed and every number in it
    appears in the situation (plus small list numbers). Returns (ok, reason).
    """
    p = parse(answer)
    if not p["actions"]:
        return False, "no numbered actions"
    if len(p["actions"]) > 8 or any(len(a) > 400 for a in p["actions"]):
        return False, "too long"
    allowed = _nums(situation_text) | {str(i) for i in range(0, 11)}
    bad = [n for n in _nums("\n".join(p["actions"]) + " " + p["attendee_message"]) if n not in allowed]
    if bad:
        return False, f"numbers not in the situation: {', '.join(sorted(bad)[:4])}"
    return True, ""
