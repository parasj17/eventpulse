"""
The EventPulse domain corpus and benchmark, for aligning a base model through Nugen.

Three files, all plain text (Nugen's developer edition accepts text documents):

* `handbook.txt`: the operating rules EventPulse runs on (thresholds, gates, exits,
  transport, the weather twin, privacy, how to talk to attendees), with questions and
  answers.
* `cases.txt`: worked cases. Each is a real situation from the city's engines at one
  hour under one weather scenario, written in the fixed situation format, followed by
  the playbook's actions and attendee message. Even hours only.
* `benchmark.json`: held-out questions and answers (odd hours, plus handbook
  questions) that Nugen scores the model on before and after alignment.

Run through `python nugen_align.py corpus`.
"""

import json
import os

from engine import playbook, weather_twin

SCENARIOS = [
    None,                                                                           # the weather as normal
    {"label": "Heavy monsoon", "rain_mm_h": 25, "duration_h": 3, "temp_c": 27, "wind_kmh": 35},
    {"label": "Cloudburst over the venue", "rain_mm_h": 80, "duration_h": 2, "temp_c": 26, "wind_kmh": 45,
     "center": {"zone_id": "Z01"}, "radius_km": 3},
    {"label": "Heatwave", "rain_mm_h": 0, "duration_h": 5, "temp_c": 44, "wind_kmh": 12},
    {"label": "Cyclone", "rain_mm_h": 45, "duration_h": 6, "temp_c": 27, "wind_kmh": 110},
    {"label": "Light rain", "rain_mm_h": 5, "duration_h": 2, "temp_c": 28, "wind_kmh": 20},
    {"label": "Storm over the metro hub", "rain_mm_h": 60, "duration_h": 3, "temp_c": 27, "wind_kmh": 50,
     "center": {"zone_id": "Z04"}, "radius_km": 4},
]

HANDBOOK = """EVENTPULSE OPERATIONS HANDBOOK
Mega-event crowd orchestration: keeping a city's hotels, transport and venues from overloading, and getting every visitor in and out safely.

1. WHAT EVENTPULSE WATCHES
The city is divided into zones. Each zone has hotel capacity, and its occupancy is the share of beds taken. Transport links (metro, shuttle, bus) join zones; each has an hourly capacity and a live load. Venues (the Central Stadium in Stadium Precinct, the Old Town Arena in Old Town) have named gates, each with an entry and an exit capacity per hour.

2. THRESHOLDS
A zone at 75% occupancy or more is a warning: start moving people. At 95% or more it is critical: it is effectively full and new arrivals must be sent elsewhere.
A transport link at 80% load is a warning and at 92% is critical: add services or release people in waves.
A gate's load is the share of its capacity it is handling, and never goes past 100%. Below 75% it is open (green), from 75% it is busy (amber), and at 95% it closes to new people (red): everyone above 95% is redirected, first to gates under 75%, then to any gate under 95%. If every gate is full, the rest are held back and told to wait. Nobody sets this by hand.
Inside a venue, a floor area at its capacity is a warning and at twice its capacity is critical.

3. GATES AND EXITS
At least 4 gates must stay working at every venue. An organiser can take a gate out of service for maintenance, never below that minimum.
When an event ends everyone leaves at once. Individually "best" advice sends everyone to the same gate, so EventPulse balances the whole crowd: it moves people away from the over-drawn gate (the bottleneck) toward gates with headroom until every gate runs at about the same load. Announce the split on screens and the PA.
The best exit for one person is the open gate with the best mix of distance and spare capacity right now.
A quarter of the transport network's capacity is kept for ordinary travellers during an exit surge; the venue crowd can only use the rest.

4. REDIRECTING DEMAND
When a zone fills, a quieter zone nearby gets an offer (a discount at restaurants and stays) sized to how much relief is needed. Push the offer to people heading to the full zone.
Oversubscribed events: when more people are expected than there are seats, tell ticket holders without a seat early and stop walk-ups.

5. THE WEATHER TWIN
EventPulse reads live weather for every zone and public reports (flooding, transport trouble, heat) and runs them through a model learned from the city's history. In a typical shower rain multiplies transport demand by about 1.2 (people stop walking) and slows movement between zones to about 61% of normal (people stay where they are).
Heavy rain over hours floods low-lying zones first. Flooded zones take capacity off every link that touches them: road shuttles and buses lose most of it, the metro about half. Gusts above about 50 km/h slow road shuttles and buses; above 60 km/h suspend them on exposed routes and move people to the metro.
When the venue takes much longer to empty, people miss the last trains. Up to an hour's extra wait just means waiting; beyond that a growing share of leavers need a room for the night. Hold rooms in the nearest zones with spare beds and tell partner hotels early. If there are not enough beds, open a shelter in the venue concourse and arrange late transport.
Heat above 35 C: open water points, use shaded queues, and put medical teams at the gates. Fewer ticket holders come in heavy rain or heat.
Every weather prediction is a range (10th to 90th percentile of 400 simulated runs). Plan for the middle, prepare for the top of the range.
A what-if simulation never changes the live system.

6. PRIVACY
Location is only shared with the server when a visitor opts in. It is reduced to an area about 150 m across, forgotten after five minutes, never tied to an account, and only shown as a count once at least three people are in the same place.

7. TALKING TO ATTENDEES
One short message, one clear action, only facts you actually have: which gate, which exit, how long. Never invent a number, a gate or a place. Examples: "Leave via North Gate: it is quicker right now." "Head to East Concourse for the shortest queue." "It's raining hard: allow extra time."

8. HOW AN OPERATOR PLAN IS WRITTEN
Most urgent first, as a numbered list: safety (flooding, heat, closed gates), then the exit or entry flow, then the stranded and hotels, then transport, then demand offers. End with one line: "Attendee message: ...". Use only numbers, gates and places from the situation.
"""

HANDBOOK_QA = [
    ("At what occupancy is a zone critical in EventPulse?", "At 95% or more. From 75% it is a warning and people should start being moved."),
    ("When is a transport link critical?", "At 92% load. From 80% it is a warning; add services or release people in waves."),
    ("How is a gate's live status decided?", "From its load as a share of capacity: below 75% open, from 75% busy, and at 95% it closes and its overflow is redirected to gates with room. A gate never goes past 100%."),
    ("How many gates must stay working at a venue?", "At least 4."),
    ("Why doesn't EventPulse send everyone to the single best exit?", "Because everyone would go to the same gate. It balances the whole crowd across gates until they run at about the same load."),
    ("How much does rain change transport demand?", "In a typical shower it multiplies demand by about 1.2, because people stop walking."),
    ("How does rain change movement between zones?", "It slows it to about 61% of normal: people stay where they are."),
    ("What should an operator do when gusts exceed 60 km/h?", "Suspend shuttles and buses on exposed routes and move people to the metro."),
    ("What should be done for visitors stranded by a slow exit?", "Hold rooms in the nearest zones with spare beds and tell partner hotels early; if there are not enough beds, open a shelter in the venue concourse and arrange late transport."),
    ("What are the heat actions above 35 C?", "Open water points, use shaded queues and put medical teams at the gates."),
    ("Does a weather what-if change the live system?", "No. It is simulated on a copy."),
    ("How is visitor location kept private?", "Only shared on opt-in, reduced to about 150 m, forgotten after five minutes, never tied to an account, and only shown for groups of at least three."),
    ("How should an attendee message be written?", "One short message with one clear action, using only facts you have: which gate, which exit, how long."),
    ("What order should an operator plan follow?", "Safety first, then the exit or entry flow, then the stranded and hotels, then transport, then demand offers, ending with an attendee message."),
    ("How are weather predictions reported?", "As a range: the 10th to 90th percentile of 400 simulated runs."),
]


def _case(api, hour, sc):
    live = None
    twin = None
    dur = sc["duration_h"] if sc else 3
    zones = api.get_zones(hour)
    if sc:
        parsed = weather_twin.parse_scenario(sc, zones)
        twin = weather_twin.simulate(api._weather_system(hour, dur, live), parsed, weather_twin.normal_weather(dur))
    system = api._weather_system(hour, dur, live)
    names = {z["id"]: z["name"] for z in system["zones"]}
    transport = [{**l, "from_name": names[l["from_zone"]], "to_name": names[l["to_zone"]]} for l in system["links"]]
    extras = {"alerts": api.get_alerts(hour), "transport": transport, "incentives": api.get_incentives(hour)}
    v = system.get("venue")
    if v:
        extras["gates"] = api.get_gates(v["zone_id"], hour)
        extras["reverse_uber"] = api.get_reverse_uber(v["zone_id"], hour)
        opts = api.get_exit_redirect(v["zone_id"], hour).get("options") or []
        if v["phase"] == "leaving" and opts:
            extras["best_exit"] = opts[0]
    facts = playbook.gather(system, twin, extras)
    return facts, playbook.situation(facts), playbook.plan(facts)


def build(out_dir, api, hours=range(24, 145, 2), bench_hours=range(27, 145, 6), log=print):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "handbook.txt"), "w") as f:
        f.write(HANDBOOK)
        f.write("\nQUESTIONS AND ANSWERS\n")
        for q, a in HANDBOOK_QA:
            f.write(f"\nQ: {q}\nA: {a}\n")

    n = 0
    with open(os.path.join(out_dir, "cases.txt"), "w") as f:
        f.write("EVENTPULSE WORKED CASES\nEach case is a situation from the live system and the operator plan for it.\n")
        for i, h in enumerate(hours):
            for sc in (SCENARIOS[0], SCENARIOS[1 + i % (len(SCENARIOS) - 1)]):
                facts, text, p = _case(api, h, sc)
                n += 1
                f.write(f"\n=== CASE {n} ===\nSITUATION\n{text}\nOPERATOR PLAN\n{playbook.plan_text(p)}\n")
    log(f"cases.txt: {n} cases")

    bench = [{"question": q, "answer": a} for q, a in HANDBOOK_QA[::2]]
    for i, h in enumerate(bench_hours):
        sc = SCENARIOS[(i * 3) % len(SCENARIOS)]
        facts, text, p = _case(api, h, sc)
        bench.append({"question": f"{playbook.SYSTEM_PROMPT}\n\n{text}", "answer": playbook.plan_text(p)})
    with open(os.path.join(out_dir, "benchmark.json"), "w") as f:
        json.dump(bench, f, indent=2)
    log(f"benchmark.json: {len(bench)} questions")
    sizes = {name: os.path.getsize(os.path.join(out_dir, name)) for name in ("handbook.txt", "cases.txt", "benchmark.json")}
    return {"cases": n, "benchmark_questions": len(bench), "bytes": sizes}
