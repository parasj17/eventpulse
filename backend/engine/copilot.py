"""
The organiser copilot's question router.

Works out what a question is about (crowds, gates, exits, alerts, transport, stays,
events, forecast, weather, public reports, or a what-if), picks out the zone, gate,
event and numbers it names, asks the live engines for exactly that, and answers in a few
lines. Every number in an answer comes from the engines. The same facts are what the
Nugen-aligned model is given to phrase an answer (api.organizer_copilot), so the model
answers the question asked, from the data, and is checked against it.

`tools` is a dict of callables supplied by api.py (so this module has no import cycle):
zones, alerts, gates(zone), exits(zone), transport, hotels, events, forecast, recommend(zone),
simulate(zone, move), surge(event, extra), cancel(event), close_link(link),
weather_now(), weather_whatif(params), social().
"""

import re

WARN, CRIT = 75, 95


def when(h):
    return f"Day {h // 24 + 1}, {h % 24:02d}:00"


def _n(x):
    return f"{int(round(x)):,}"


# ---------------------------------------------------------------- understanding the question
INTENTS = [
    # (intent, keywords, weight); the highest total wins
    ("whatif_weather", ["if it rains", "rain", "storm", "flood", "cyclone", "heatwave", "heat wave", "cloudburst",
                        "monsoon", "mm", "temperature", "wind"], 1),
    ("surge", ["extra people", "more people", "surge", "turn up", "show up", "additional", "extra"], 1),
    ("cancel", ["cancel", "called off", "postpone", "evacuat"], 2),
    ("close_link", ["close the", "closes", "shut", "link down", "metro closes", "line closes", "closure"], 1),
    ("move", ["move ", "redirect", "shift ", "relocate", "send "], 1),
    ("gates", ["gate", "entrance", "entry", "queue"], 2),
    ("exits", ["exit", "leave", "way out", "get out", "egress"], 2),
    ("alerts", ["alert", "problem", "issue", "risk", "worry", "danger", "critical", "urgent", "wrong"], 1),
    ("crowd", ["crowd", "crowded", "crowds", "crowding", "busy", "busiest", "worst", "packed", "full", "occupancy", "pressure", "zone", "capacity"], 1),
    ("transport", ["transport", "metro", "train", "shuttle", "bus", "link", "traffic", "route", "lines", "line ", "get there",
                   "reach", "travel", "commute", "directions", "get to", "get home"], 2),
    ("stays", ["hotel", "stay", "room", "bed", "accommodation"], 2),
    ("events", ["event", "match", "concert", "start", "schedule", "next", "when is", "what's on", "ceremony", "final"], 1),
    ("forecast", ["forecast", "later", "next hours", "tonight", "tomorrow", "predict", "will be", "going to", "in 6", "ahead"], 1),
    ("weather", ["weather", "raining", "temperature now", "hot", "humid"], 1),
    ("social", ["social", "reddit", "mastodon", "news", "people saying", "reports", "tweets", "posts", "complain"], 2),
    ("help", ["help", "what can you", "how do i", "hello", "hi "], 1),
    ("status", ["how are we", "how is it going", "status", "overview", "summary", "situation", "update", "sitrep", "how's it"], 2),
]


def classify(q):
    t = " " + q.lower() + " "
    scores = {}
    def has(word):                      # whole words, so "busy" doesn't count as "bus"
        return re.search(r"(?<![a-z])" + re.escape(word.strip()) + r"(?![a-z])", t) is not None
    for intent, words, w in INTENTS:
        s = sum(w for word in words if has(word))
        if s:
            scores[intent] = scores.get(intent, 0) + s
    # What-ifs need a hypothetical; "is it raining" is a weather question, not a what-if.
    hypo = any(k in t for k in ("what if", "what happens", "suppose", "if it", "if there", "if we", "simulate", "scenario", "would"))
    if not hypo:
        for k in ("whatif_weather", "surge", "cancel", "close_link", "move"):
            scores.pop(k, None)
        if any(k in t for k in ("rain", "storm", "flood", "weather", "hot", "temperature")):
            scores["weather"] = scores.get("weather", 0) + 2
    else:
        if re.search(r"\d[\d,]*\s*(extra|more|additional)|(extra|more|additional)\s*\d", t):
            scores["surge"] = scores.get("surge", 0) + 3
        if re.search(r"\d+\s*mm|\bmm/h|degrees|°|\bc\b|km/h|heat|cyclone|cloudburst|storm|rain", t):
            scores["whatif_weather"] = scores.get("whatif_weather", 0) + 3
    if hypo and any(k in t for k in ("move ", "redirect", "shift ", "relocate")) and "rain" not in t:
        scores["move"] = scores.get("move", 0) + 3
    if hypo and any(k in t for k in ("close", "shut", "down")) and any(k in t for k in ("metro", "shuttle", "bus", "link", "line", "train")):
        scores["close_link"] = scores.get("close_link", 0) + 3
    return max(scores, key=scores.get) if scores else "overview"


def _find(names, q, exact=False):
    """The item whose name (or, unless exact, a distinctive word of it) appears in the question."""
    t = q.lower()
    best = None
    for key, name in names:
        n = name.lower()
        if n in t:
            if not best or len(n) > len(best[1]):
                best = (key, n)
    if best:
        return best[0]
    if exact:
        return None
    for key, name in names:
        words = [w for w in re.findall(r"[a-z]+", name.lower()) if len(w) > 3 and w not in ("gate", "zone", "concourse", "central")]
        if any(re.search(rf"\b{w}\b", t) for w in words):
            return key
    return None


def _number(q, near=None):
    nums = [int(x.replace(",", "")) for x in re.findall(r"\d[\d,]*", q)]
    return nums


# ---------------------------------------------------------------- answers
def answer(question, hour, tools):
    q = (question or "").strip()
    intent = classify(q)
    zones = tools["zones"]()
    zname = {z["id"]: z["name"] for z in zones}
    events = tools["events"]()
    evlist = events["all"]
    # Transport questions name modes ("metro"), which are also words in zone names: match zones exactly there.
    zone_id = _find([(z["id"], z["name"]) for z in zones], q, exact=intent in ("transport", "close_link"))
    event_id = _find([(e["event_id"], e["name"]) for e in evlist], q)
    # The venue to talk about: the one named, else the event nearest now.
    live = [e for e in evlist if e["start_hour"] - 4 <= hour <= e["end_hour"] + 3]
    ev = next((e for e in evlist if e["event_id"] == event_id), None) or (live[0] if live else (events["schedule"][0] if events["schedule"] else None))
    fn = globals().get("_a_" + intent, _a_overview)
    text, steps = fn(q=q, hour=hour, tools=tools, zones=zones, zname=zname, zone_id=zone_id, ev=ev, events=events)
    return {"intent": intent, "answer": text, "steps": [{"tool": s, "input": {}} for s in steps]}


def _a_crowd(q, hour, tools, zones, zname, zone_id, **_):
    if zone_id:
        z = next(z for z in zones if z["id"] == zone_id)
        recs = tools["recommend"](zone_id).get("options") or []
        lvl = "full" if z["occupancy_pct"] >= CRIT else "busy" if z["occupancy_pct"] >= WARN else "comfortable"
        s = f"{z['name']} is at {z['occupancy_pct']:.0f}% ({lvl}), {_n(z['spare_capacity'])} beds free."
        if recs and z["occupancy_pct"] >= WARN:
            r = recs[0]
            s += f" Best place to send overflow: {r['zone_name']}, {r['reason']}."
        return s, ["zone", "recommend"]
    top = sorted(zones, key=lambda z: -z["occupancy_pct"])[:3]
    s = "Busiest zones right now: " + "; ".join(f"{z['name']} {z['occupancy_pct']:.0f}%" for z in top) + "."
    worst = top[0]
    if worst["occupancy_pct"] >= WARN:
        recs = tools["recommend"](worst["id"]).get("options") or []
        if recs:
            s += f" {worst['name']} is the one to act on: send overflow to {recs[0]['zone_name']} ({recs[0]['reason']})."
    else:
        s += " Nothing is past 75%, so no zone needs redirecting yet."
    return s, ["zones", "recommend"]


def _venue_gates(tools, ev):
    return tools["gates"](ev["zone_id"]) if ev else []


def _a_gates(q, hour, tools, ev, **_):
    if not ev:
        return "No event is on or coming up, so the gates are idle.", ["events"]
    gates = _venue_gates(tools, ev)
    g_id = _find([(g["gate_id"], g["name"]) for g in gates], q)
    act = [g for g in gates if g.get("phase") in ("ingress", "egress")]
    if g_id:
        g = next(x for x in gates if x["gate_id"] == g_id)
        pct = g.get("load_pct", 0)
        s = f"{g['name']} at {ev['venue_name'] if 'venue_name' in ev else ev['zone_name']}: {g.get('live_status', 'open')}"
        s += f", {pct:.0f}% of capacity." if g.get("phase") in ("ingress", "egress") else ", no crowd at it right now."
        if g.get("redirect_to"):
            s += " It is closed at 95%; its overflow goes to " + ", ".join(f"{t['name']} ({_n(t['people'])})" for t in g["redirect_to"]) + "."
        return s, ["venue_gates"]
    if not act:
        return (f"The gates for {ev['name']} are idle right now. Arrivals start at {when(ev['start_hour'] - 4)}, "
                f"4 hours before the {when(ev['start_hour'])} start."), ["venue_gates", "events"]
    act.sort(key=lambda g: -g.get("load_pct", 0))
    worst = act[0]
    s = (f"{ev['name']}: the gate to watch is {worst['name']} at {worst.get('load_pct', 0):.0f}% ({worst['live_status']}). "
         + "Others: " + ", ".join(f"{g['name']} {g.get('load_pct', 0):.0f}%" for g in act[1:4]) + ".")
    if worst.get("redirect_to"):
        s += (f" It has closed at 95% and is sending {_n(worst['redirected_out'])} people to "
              + ", ".join(f"{t['name']} ({_n(t['people'])})" for t in worst["redirect_to"]) + ".")
    elif worst.get("load_pct", 0) >= WARN:
        s += " It's busy; if it reaches 95% it closes and the overflow is redirected automatically."
    return s, ["venue_gates"]


def _a_exits(q, hour, tools, ev, **_):
    if not ev:
        return "No event is on, so there's no exit to plan.", ["events"]
    opts = tools["exits"](ev["zone_id"]).get("options") or []
    if hour <= ev["end_hour"]:
        lead = f"{ev['name']} ends at {when(ev['end_hour'])}. If people left now, the best exits would be: "
    else:
        lead = f"{ev['name']} has ended. Best exits right now: "
    if not opts:
        return lead.rstrip(": ") + ".", ["exit_plan"]
    return lead + "; ".join(f"{i + 1}. {o['name']} ({o['reason'].lower()})" for i, o in enumerate(opts[:3])) + ".", ["exit_plan"]


def _a_alerts(q, hour, tools, **_):
    al = tools["alerts"]()
    if not al:
        return "No alerts: every zone, gate and transport link is within its limits.", ["alerts"]
    crit = [a for a in al if a["severity"] == "critical"]
    pick = (crit or al)[:3]
    s = f"{len(crit)} critical and {len(al) - len(crit)} warning alerts. Most urgent: " + " | ".join(a["message"] for a in pick) + "."
    return s, ["alerts"]


def _a_transport(q, hour, tools, zname, zone_id, **_):
    links = tools["transport"]()
    if zone_id:
        links = [l for l in links if zone_id in (l["from_zone"], l["to_zone"])]
    links = sorted(links, key=lambda l: -l["load_pct"])[:3]
    if not links:
        return "No transport links found for that.", ["transport"]
    s = ("Busiest links" + (f" at {zname[zone_id]}" if zone_id else "") + ": "
         + "; ".join(f"{l['mode']} {zname[l['from_zone']]} - {zname[l['to_zone']]} {l['load_pct']:.0f}%" for l in links) + ".")
    if links[0]["load_pct"] >= 92:
        s += " The top one is critical: add services or release people in waves."
    elif links[0]["load_pct"] >= 80:
        s += " The top one is past its 80% warning level."
    return s, ["transport"]


def _a_stays(q, hour, tools, zones, zname, zone_id, **_):
    hotels = [h for h in tools["hotels"]() if h["rooms_left"] > 0]
    if zone_id:
        hotels = [h for h in hotels if h["zone_id"] == zone_id] or hotels
    free = sorted(zones, key=lambda z: -z["spare_capacity"])[:3]
    near = sorted(hotels, key=lambda h: h["distance_to_venue_km"])[:3]
    s = (f"{_n(sum(z['spare_capacity'] for z in zones))} beds free citywide. Most room in: "
         + ", ".join(f"{z['name']} ({_n(z['spare_capacity'])})" for z in free) + ".")
    if near:
        s += " Closest stays with rooms to the venue: " + ", ".join(f"{h['name']} ({h['zone_name']}, {h['distance_to_venue_km']} km)" for h in near) + "."
    return s, ["hotels", "zones"]


def _a_events(q, hour, tools, ev, events, **_):
    e = ev
    if e is None:
        return "No events are scheduled from here on.", ["list_events"]
    if hour < e["start_hour"]:
        s = (f"{e['name']} at {e['zone_name']}: gates open {when(e['start_hour'] - 4)}, starts {when(e['start_hour'])} "
             f"(in {e['start_hour'] - hour} h), ends {when(e['end_hour'])}; {_n(e['expected_attendance'])} expected.")
    elif hour <= e["end_hour"]:
        s = f"{e['name']} is under way at {e['zone_name']} and ends at {when(e['end_hour'])}; {_n(e['expected_attendance'])} expected."
    else:
        s = f"{e['name']} ended at {when(e['end_hour'])}."
    later = [x for x in events["schedule"] if x["event_id"] != e["event_id"]][:2]
    if later:
        s += " After that: " + "; ".join(f"{x['name']} at {when(x['start_hour'])}" for x in later) + "."
    return s, ["list_events"]


def _a_forecast(q, hour, tools, zones, zname, zone_id, **_):
    fc = {f["zone_id"]: f for f in tools["forecast"]()}
    def at6(zid):
        tl = fc.get(zid, {}).get("timeline") or []
        return tl[min(6, len(tl) - 1)]["predicted_occupancy_pct"] if tl else None
    if zone_id:
        z = next(z for z in zones if z["id"] == zone_id)
        p = at6(zone_id)
        return f"{z['name']} is at {z['occupancy_pct']:.0f}% now and forecast at {p:.0f}% in 6 hours ({when(hour + 6)}).", ["forecast"]
    rows = sorted(((z, at6(z["id"])) for z in zones if at6(z["id"]) is not None), key=lambda r: -r[1])[:3]
    s = f"Forecast for {when(hour + 6)}: " + "; ".join(f"{z['name']} {p:.0f}% (now {z['occupancy_pct']:.0f}%)" for z, p in rows) + "."
    hot = [z for z, p in rows if p >= CRIT and z["occupancy_pct"] < CRIT]
    if hot:
        s += f" {hot[0]['name']} is heading to full: start redirecting before it gets there."
    return s, ["forecast"]


def _a_weather(q, hour, tools, **_):
    st = tools["weather_now"]()
    live = st.get("live") or {}
    c = live.get("city")
    if not c:
        return "Live weather isn't available right now; the twin is assuming calm weather.", ["live_weather"]
    stale = " (last reading; live updates paused briefly)" if live.get("stale") else ""
    s = (f"Now{stale}: {c['condition']}, {c['temp_c']:.0f}°C (feels {c['apparent_c']:.0f}°C), rain {c['rain_mm_h_max']} mm/h, "
         f"{c['rain_next_6h_mm_max']} mm expected in the next 6 h ({c['max_prob_next_6h']:.0f}% chance), gusts {c['gust_kmh_max']:.0f} km/h.")
    fx = [e for e in st.get("effects", []) if not e.get("folded")]
    s += (" Effect on the city: " + " ".join(e["text"] for e in fx[:2])) if fx else " It isn't changing how the city moves compared with a normal day."
    return s, ["live_weather", "weather_impact_now"]


def _a_social(q, hour, tools, **_):
    so = tools["social"]()
    if not so.get("available"):
        return "Couldn't reach the public feeds just now.", ["social"]
    c = so["counts_24h"]
    s = (f"In the last 24 h: {c['flooding']} flooding, {c['transport']} transport, {c['alert']} alert and {c['heat']} heat reports "
         f"(from {so['total']} weather-related posts and headlines).")
    if so.get("emerging"):
        e = so["emerging"][0]
        s += f" Emerging: {e['category']} reports are up {e['times_usual']}x in 6 h" + (f" around {', '.join(e['places'])}" if e["places"] else "") + "."
    if so["items"]:
        s += f" Latest: \"{so['items'][0]['text'][:120]}\"."
    return s, ["social"]


def _a_whatif_weather(q, hour, tools, zones, zname, zone_id, **_):
    t = q.lower()
    p = {"rain_mm_h": 0, "duration_h": 3, "temp_c": 30, "wind_kmh": 10}
    m = re.search(r"(\d+(?:\.\d+)?)\s*mm", t)
    if m:
        p["rain_mm_h"] = float(m.group(1))
    elif "cloudburst" in t:
        p.update(rain_mm_h=80, duration_h=2)
    elif "cyclone" in t:
        p.update(rain_mm_h=45, duration_h=6, wind_kmh=110)
    elif "heavy" in t or "flood" in t or "storm" in t:
        p["rain_mm_h"] = 40
    elif "rain" in t:
        p["rain_mm_h"] = 15
    m = re.search(r"(\d+(?:\.\d+)?)\s*(h|hr|hrs|hour|hours)\b", t)
    if m:
        p["duration_h"] = float(m.group(1))
    m = re.search(r"(\d+)\s*(°|degrees|deg|c\b)", t)
    if m:
        p["temp_c"] = float(m.group(1))
    elif "heat" in t:
        p["temp_c"] = 44
    m = re.search(r"(\d+)\s*km/?h", t)
    if m:
        p["wind_kmh"] = float(m.group(1))
    if zone_id:
        p.update(center={"zone_id": zone_id}, radius_km=3)
    r = tools["weather_whatif"](p)
    c, v = r["city"], r.get("venue")
    desc = (f"{p['rain_mm_h']:g} mm/h rain for {p['duration_h']:g} h" if p["rain_mm_h"] else "no rain") + \
           (f", {p['temp_c']:g}°C" if p["temp_c"] != 30 else "") + (f", {p['wind_kmh']:g} km/h gusts" if p["wind_kmh"] > 10 else "") + \
           (f" centred on {zname[zone_id]}" if zone_id else "")
    bits = []
    if c["max_flood_pct"]["p50"] >= 10:
        bits.append(f"worst flood risk {c['max_flood_pct']['p50']:.0f}%")
    if c["links_disrupted"]["p50"] >= 1:
        bits.append(f"{c['links_disrupted']['p50']:.0f} transport links cut to 60% or less")
    if v and v["hours_to_clear"]["p50"] - v["hours_to_clear_baseline"] >= 0.1:
        bits.append(f"{v['venue_name']} takes {v['hours_to_clear']['p50']:.1f} h to empty (normally {v['hours_to_clear_baseline']:.1f} h)")
    if c["stranded"]["p50"] >= 1:
        bits.append(f"about {_n(c['stranded']['p50'])} visitors stranded (likely {_n(c['stranded']['p10'])}-{_n(c['stranded']['p90'])})")
    if v and v["heat_incidents"]["p50"] >= 1:
        bits.append(f"about {v['heat_incidents']['p50']:.0f} heat-related medical calls at the venue")
    if v and v["people_baseline"] - v["people"]["p50"] >= 200:
        bits.append(f"{_n(v['people_baseline'] - v['people']['p50'])} fewer people turn up")
    s = f"Simulated {desc} (compared with a normal day, 400 runs): " + ("; ".join(bits) if bits else "no measurable change") + "."
    fx = [e for e in r["effects"] if not e.get("folded") and e["order"] >= 2 and "medical" not in e["id"]][:2]
    if fx:
        s += " Knock-on: " + " ".join(e["text"] for e in fx)
    s += " Nothing in the live system was changed."
    return s, ["weather_whatif"]


def _a_surge(q, hour, tools, ev, **_):
    if not ev:
        return "There's no upcoming event to add people to.", ["list_events"]
    nums = [n for n in _number(q) if n >= 50]
    extra = nums[0] if nums else 3000
    r = tools["surge"](ev["event_id"], extra)
    b, a = r["before"], r["after"]
    s = (f"If {_n(extra)} extra people turn up to {r['event_name']}: {_n(a['turned_away'])} would be turned away "
         f"(was {_n(b['turned_away'])}), and the peak entry queue grows from {_n(b['peak_queue'])} to {_n(a['peak_queue'])}.")
    if a["turned_away"] > b["turned_away"]:
        s += " Warn ticket holders without seats early and stop walk-ups."
    return s + " Simulated only.", ["surge_event"]


def _a_cancel(q, hour, tools, ev, **_):
    if not ev:
        return "There's no event to cancel.", ["list_events"]
    r = tools["cancel"](ev["event_id"])
    eg = r["before"]["egress"]
    s = (f"If {r['event_name']} were cancelled now, {_n(r['already_admitted'])} people already inside would have to leave: "
         f"about {eg['hours_to_clear_venue']} h to get out of the venue and {eg['hours_to_clear_zone']} h to clear the area "
         f"(transport carries {_n(eg['transport_capacity_per_hour'])} an hour). Simulated only.")
    return s, ["cancel_event"]


def _a_close_link(q, hour, tools, zones, zname, **_):
    links = tools["transport"]()
    named = [z["id"] for z in zones if z["name"].lower() in q.lower()]
    cand = [l for l in links if all(n in (l["from_zone"], l["to_zone"]) for n in named[:2])] if named else []
    mode = next((m for m in ("metro", "shuttle", "bus") if m in q.lower()), None)
    if mode:
        cand = [l for l in (cand or links) if l["mode"] == mode]
    link = sorted(cand or links, key=lambda l: -l["load_pct"])[0]
    r = tools["close_link"](link["link_id"])
    route = (r.get("after") or r.get("before") or {}).get("route") or {}
    s = f"If the {link['mode']} {zname[link['from_zone']]} - {zname[link['to_zone']]} closed (now {link['load_pct']:.0f}% loaded): "
    s += route.get("summary") or "no alternative route data."
    return s + " Simulated only.", ["close_link"]


def _a_move(q, hour, tools, zones, zname, zone_id, **_):
    src = zone_id or sorted(zones, key=lambda z: -z["occupancy_pct"])[0]["id"]
    nums = [n for n in _number(q) if n >= 10]
    move = nums[0] if nums else 500
    r = tools["simulate"](src, move)
    so = r["source"]
    s = (f"Moving {_n(move)} people out of {r['zone_name']} (75% follow the advice): {_n(r['relocated'])} relocate, "
         f"{r['zone_name']} goes from {so['occupancy_before_pct']:.0f}% to {so['occupancy_after_pct']:.0f}%. They land in "
         + ", ".join(f"{p['zone_name']} (+{_n(p['received'])}, to {p['occupancy_after_pct']:.0f}%)" for p in r["placements"][:3]) + ".")
    return s + " Simulated only.", ["simulate_move"]


def _a_help(q, hour, **_):
    return ("Ask me about: the busiest zones, a zone by name, which gate to worry about, the best exits, alerts, transport, "
            "hotels and free beds, the event schedule, the 6-hour forecast, the weather now, what people are reporting, "
            "or a what-if such as \"what if 3,000 extra people turn up\", \"what if 40 mm rain for 3 hours at Metro Interchange\", "
            "\"what if we move 800 people out of Stadium Precinct\" or \"what if the metro to Metro Interchange closes\"."), []


def _a_status(q, hour, tools, zones, ev, **kw):
    a, s1 = _a_alerts(q, hour, tools)
    c, s2 = _a_crowd(q, hour, tools, zones, kw["zname"], None)
    g = ""
    if ev and ev["start_hour"] - 4 <= hour <= ev["end_hour"] + 3:
        g, _ = _a_gates(q, hour, tools, ev)
    return f"Status at {when(hour)}. " + c + " " + a + (" " + g if g else ""), s1 + s2


def _a_overview(q, hour, tools, zones, ev, **kw):
    a, s1 = _a_alerts(q, hour, tools)
    c, s2 = _a_crowd(q, hour, tools, zones, kw["zname"], None)
    lead = "I wasn't sure what you meant, so here's the overview. "
    return lead + c + " " + a + " (Try: \"which gate should I worry about?\" or \"what if it rains 40 mm?\")", s1 + s2


# ---------------------------------------------------------------- attendees
ATTENDEE_MAP = {"gates": "entry", "exits": "exit", "events": "my_event", "weather": "weather", "whatif_weather": "weather",
                "stays": "stays", "crowd": "crowd", "transport": "transport", "help": "help", "status": "my_event",
                "alerts": "my_event", "forecast": "crowd", "social": "social"}


def answer_attendee(question, hour, tools, regs):
    """
    The attendee's copilot: about *their* event (the next or current one they registered for),
    in their terms: when to arrive, which gate to use, how to get out, the weather, where to stay.
    No organiser what-ifs.
    """
    q = (question or "").strip()
    intent = ATTENDEE_MAP.get(classify(q), "my_event")
    evs = tools["events"]()["all"]
    mine = [e for e in evs if e["event_id"] in {r["event_id"] for r in regs}]
    named = _find([(e["event_id"], e["name"]) for e in mine], q)
    upcoming = sorted([e for e in mine if e["end_hour"] + 3 >= hour], key=lambda e: e["start_hour"])
    ev = next((e for e in mine if e["event_id"] == named), None) or (upcoming[0] if upcoming else None)
    if ev is None and intent in ("entry", "exit", "my_event"):
        if not mine:
            return {"intent": intent, "answer": "You haven't registered for an event yet. Pick one under \"Register for an event\" and I can tell you when to arrive, which gate to use and how to get out.", "steps": []}
        return {"intent": intent, "answer": "Your events have all finished. Register for another to get live guidance.", "steps": []}
    fn = globals()["_t_" + intent]
    text, steps = fn(q=q, hour=hour, tools=tools, ev=ev, mine=mine)
    return {"intent": intent, "answer": text, "steps": [{"tool": s, "input": {}} for s in steps]}


def _venue(ev):
    return ev.get("zone_name") or "the venue"


def _t_my_event(q, hour, tools, ev, mine, **_):
    gates_open = ev["start_hour"] - 4
    if hour < gates_open:
        s = (f"{ev['name']} at {_venue(ev)}: gates open {when(gates_open)} (in {gates_open - hour} h), it starts {when(ev['start_hour'])} "
             f"and ends {when(ev['end_hour'])}. Arrive early in the gate window to skip the rush just before the start.")
    elif hour < ev["start_hour"]:
        s = f"Gates are open for {ev['name']}; it starts at {when(ev['start_hour'])}. " + _t_entry(q, hour, tools, ev, mine)[0]
    elif hour <= ev["end_hour"]:
        s = f"{ev['name']} is under way and ends at {when(ev['end_hour'])}. Ask me \"best way out?\" near the end."
    else:
        x = _t_exit(q, hour, tools, ev, mine)[0]
        s = f"{ev['name']} has ended. " + x[:1].upper() + x[1:]
    others = [e for e in mine if e["event_id"] != ev["event_id"] and e["end_hour"] >= hour]
    if others:
        s += " Your other events: " + "; ".join(f"{e['name']} at {when(e['start_hour'])}" for e in others[:2]) + "."
    return s, ["my_events"]


def _t_entry(q, hour, tools, ev, mine, **_):
    gates = tools["gates"](ev["zone_id"])
    live = [g for g in gates if g.get("status", "open") == "open"]
    if not live:
        return "No gates are open at the moment.", ["venue_gates"]
    if hour < ev["start_hour"] - 4:
        return (f"Gates for {ev['name']} open at {when(ev['start_hour'] - 4)}. Closer to then I'll tell you which one has the "
                f"shortest queue."), ["venue_gates"]
    openg = sorted([g for g in live if g.get("live_status") != "closed"], key=lambda g: g.get("load_pct", 0))
    if not openg:
        return "Every gate is full right now: wait a few minutes and ask again, you'll be sent to the first one with room.", ["venue_gates"]
    g = openg[0]
    s = f"Use {g['name']}: it's the quietest right now at {g.get('load_pct', 0):.0f}% of capacity."
    closed = [x["name"] for x in live if x.get("live_status") == "closed"]
    if closed:
        s += f" Avoid {', '.join(closed)}: closed to new arrivals because it's full."
    return s, ["venue_gates"]


def _t_exit(q, hour, tools, ev, mine, **_):
    opts = tools["exits"](ev["zone_id"]).get("options") or []
    if not opts:
        return "I don't have exit advice for your venue yet.", ["exit_plan"]
    o = opts[0]
    pre = "" if hour > ev["end_hour"] else f"{ev['name']} ends at {when(ev['end_hour'])}. If you left now: "
    s = pre + (f"head for" if pre else "Head for") + f" {o['name']} ({o['reason'].lower()})."
    if len(opts) > 1:
        s += f" Second choice: {opts[1]['name']}."
    return s, ["exit_plan"]


def _t_weather(q, hour, tools, ev, mine, **_):
    st = tools["weather_now"]()
    live = (st.get("live") or {})
    zs = {z["zone_id"]: z for z in live.get("zones", [])}
    z = zs.get(ev["zone_id"]) if ev else None
    if not z:
        c = live.get("city")
        if not c:
            return "I can't read the live weather right now.", ["live_weather"]
        return f"Across the city: {c['condition']}, {c['temp_c']:.0f}°C, rain {c['rain_mm_h_max']} mm/h.", ["live_weather"]
    s = f"At {_venue(ev)}: {z['condition']}, {z['temp_c']:.0f}°C (feels {z['apparent_c']:.0f}°C), rain {z['rain_mm_h']} mm/h, {z['rain_next_6h_mm']} mm expected in the next 6 h."
    if z["rain_mm_h"] >= 2 or z["rain_next_6h_mm"] >= 5:
        s += " Bring something waterproof and allow extra time."
    if z["apparent_c"] >= 35:
        s += " It's very hot: carry water."
    return s, ["live_weather"]


def _t_stays(q, hour, tools, ev, mine, **_):
    hotels = [h for h in tools["hotels"]() if h["rooms_left"] > 0]
    if ev:
        hotels.sort(key=lambda h: (h["zone_id"] != ev["zone_id"], h["distance_to_venue_km"]))
    near = hotels[:3]
    if not near:
        return "I can't see any rooms free right now.", ["hotels"]
    return ("Closest stays to your venue: " + "; ".join(f"{h['name']} in {h['zone_name']} ({h['distance_to_venue_km']} km, Rs {h['nightly_rate']:,}/night)" for h in near)
            + ". The Stays section of the dashboard sorts them by distance from you."), ["hotels"]


def _t_crowd(q, hour, tools, ev, mine, **_):
    zones = tools["zones"]()
    if ev:
        z = next((z for z in zones if z["id"] == ev["zone_id"]), None)
        if z:
            lvl = "very busy" if z["occupancy_pct"] >= CRIT else "busy" if z["occupancy_pct"] >= WARN else "comfortable"
            quiet = sorted((x for x in zones if x["occupancy_pct"] < WARN), key=lambda x: x["distance_to_venue_km"])[:1]
            s = f"The area around your venue ({z['name']}) is {lvl} at {z['occupancy_pct']:.0f}%."
            if quiet and z["occupancy_pct"] >= WARN:
                s += f" Nearest calmer area: {quiet[0]['name']} ({quiet[0]['occupancy_pct']:.0f}%)."
            return s, ["zones"]
    top = sorted(zones, key=lambda z: -z["occupancy_pct"])[0]
    return f"The busiest part of the city is {top['name']} at {top['occupancy_pct']:.0f}%.", ["zones"]


def _t_transport(q, hour, tools, ev, mine, **_):
    zone = ev["zone_id"] if ev else None
    links = [l for l in tools["transport"]() if zone in (l["from_zone"], l["to_zone"])] if zone else tools["transport"]()
    names = {z["id"]: z["name"] for z in tools["zones"]()}
    if not links:
        return "I don't have transport data for that.", ["transport"]
    quiet = sorted(links, key=lambda l: l["load_pct"])[0]
    busy = sorted(links, key=lambda l: -l["load_pct"])[0]
    other = lambda l: names[l["to_zone"] if l["from_zone"] == zone else l["from_zone"]]
    return (f"From your venue, the quietest way is the {quiet['mode']} to {other(quiet)} ({quiet['load_pct']:.0f}% full); "
            f"the busiest is the {busy['mode']} to {other(busy)} ({busy['load_pct']:.0f}%)."), ["transport"]


def _t_social(q, hour, tools, ev, mine, **_):
    return _a_social(q, hour, tools)


def _t_help(q, hour, tools, ev, mine, **_):
    return ("Ask me: when my event starts, which gate to use, the best way out, the weather at the venue, how busy it is, "
            "the quietest transport, or where to stay nearby."), []


# ---------------------------------------------------------------- is a model's answer faithful?
_NUMS = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _numset(t):
    return [n.replace(",", "").rstrip(".") for n in _NUMS.findall(t or "")]


def faithful(model_text, grounded, names, question="", event_names=()):
    """
    The aligned model's wording is shown only if it says the same thing as the facts:
    it leads with the same place/gate/event, keeps the headline number, adds no number
    or name of its own, doesn't claim the facts are missing, and is a real sentence.
    `names` are all zone, gate and event names. Returns (ok, reason).
    """
    m, g = (model_text or "").strip(), grounded or ""
    ml, gl = m.lower(), g.lower()
    if len(m.split()) < 6:
        return False, "too short"
    if any(k in ml for k in ("do not show", "does not show", "not provided", "no information", "don't show", "cannot determine")):
        return False, "said the facts don't cover it"
    gn, mn = _numset(g), _numset(m)
    small = {str(i) for i in range(11)}
    if [n for n in mn if n not in set(gn) | small]:
        return False, "used a number not in the live data"
    qn = set(_numset(question))
    head = next((n for n in gn if n not in small and n not in qn), None)     # the answer's own key number
    if head and head not in mn:
        return False, "left out the headline number"
    ev = {e.lower() for e in event_names}
    def first_name(t):
        hits = [(t.find(n.lower()), n) for n in names if n and n.lower() in t and n.lower() not in ev]
        return min(hits)[1] if hits else None
    gf, mf = first_name(gl), first_name(ml)
    if gf and mf and mf != gf:
        return False, "led with a different place or gate"
    if [n for n in names if n and n.lower() in ml and n.lower() not in gl]:
        return False, "named a place or gate not in the live data"
    return True, ""
