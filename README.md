# EventPulse — PS8 Mega-Event Hospitality Orchestration

Intelligent capacity and crowd management for cities hosting mega-events.

The city has spare capacity. It's just in the wrong place. EventPulse forecasts
where pressure is building, raises it before it becomes critical, and ranks
where to redirect people instead.

## Tech stack

**Backend** — Python 3.11, FastAPI + Uvicorn, pandas/numpy, scikit-learn (the
forecasting model), httpx, psycopg[binary] v3, python-multipart.

**Database / accounts** — Supabase Postgres (`ep_accounts`, `ep_attendees`,
row-level security), falling back automatically to local CSV
(`backend/data/accounts.csv`) if Supabase isn't configured. Passwords are
salted PBKDF2 hashes.

**Frontend** — plain HTML/CSS/JS, no build step and no `npm install`. React and
MapLibre GL JS are vendored directly (`frontend/vendor/`) as pre-built files,
with Babel vendored too for in-browser JSX, so the whole frontend runs
offline. The 3D city twin (`twin.html`) renders over OpenFreeMap ("Liberty"
style) or CARTO as a fallback basemap, both OpenStreetMap-based, and drops to
an offline 3D model if the map or WebGL isn't available.

**AI / ML** — an in-house gradient-boosted forecaster (scikit-learn) that
cuts error 52% against a linear-trend baseline. Domain reasoning (operator
plans, attendee messaging, the copilot) runs on **Nugen**
(`backend/nugen_align.py`, [docs.nugen.in](https://docs.nugen.in)): a base LLM
aligned on an EventPulse-specific operations corpus and 122 worked cases, with
a deterministic playbook fallback if Nugen is unreachable or an answer fails
validation. An Anthropic (Claude) key can power the copilot instead when set.

**Infra / deployment** — static frontend on **Vercel** (`frontend/vercel.json`);
the full stack (API + frontend) also ships as a single **Docker** container
(`Dockerfile`), portable to Render, Fly.io, Railway or Hugging Face Spaces. The
dataset and model are generated and trained at Docker *build* time rather than
on boot, for an instant cold start. Email goes through SMTP when configured,
with a simulated-email fallback otherwise.

## Deploying

See [DEPLOY.md](DEPLOY.md). Because the dataset is deterministic, every response
is precomputed by `export_static.py` and the dashboard deploys as a static site:
free to host, instant to load, and nothing that can sleep between someone getting
the link and opening it. Vercel is the recommended host. A `Dockerfile` is also
included for hosts that run containers, if the live API is wanted instead.

## Run it

Two terminals.

**Terminal 1 — backend**

```bash
cd backend
pip install -r requirements.txt
python generate_city.py
uvicorn api:app --reload
```

**Terminal 2 — frontend**

```bash
cd frontend
python -m http.server 8080
```

Open <http://127.0.0.1:8080>

React is vendored into `frontend/vendor/`, so there is no `npm install` and the
demo runs with no internet. That matters on hackathon wifi.

## What's built

| Phase | Component | File |
|---|---|---|
| 0 | Data contract | `contracts/mock-data.json` |
| 1 | Synthetic city generator | `backend/generate_city.py` |
| 2 | Forecasting engine (trend baseline) | `backend/engine/forecast.py` |
| 2b | ML forecaster (gradient boosting) | `backend/engine/model.py` |
| 3 | Risk detection & alerts | `backend/engine/risk.py` |
| 4 | Recommendation engine | `backend/engine/recommend.py` |
| 4b | What-if redistribution simulator | `backend/engine/simulate.py` |
| 5 | REST API | `backend/api.py` |
| A | Command interface (organizers) | `frontend/index.html` |
| B | Attendee interface | same file, Attendee tab |
| 6 | Hotel inventory | `backend/engine/hotels.py` |
| 7 | Venue capacity | `backend/engine/venues.py` |
| 8 | Alternative routing | `backend/engine/routing.py` |

Each engine module runs standalone for testing:

```bash
cd backend
python -m engine.forecast     # trend-line projections
python -m engine.model        # train the ML model and benchmark it
python -m engine.risk         # the alert feed
python -m engine.recommend    # ranked redirect options
python -m engine.simulate     # what-if: move N people, see the effect
python -m engine.hotels       # bookable inventory with live availability
python -m engine.venues       # seats, gate queues and egress clearance
python -m engine.routing      # alternative routes around a congested link
```

## How it works

**Forecasting** comes in two forms. The baseline fits a weighted linear trend
over recent occupancy and projects it forward — simple and explainable.

The ML forecaster (`engine/model.py`) predicts occupancy 6 hours ahead with
gradient boosting, and **cuts error by 52% against that baseline** on held-out
data (MAE 2.26 → 1.07 percentage points, R² 0.970 → 0.992).

Two things make that number trustworthy:

* **The split is chronological.** Train is the first 70% of hours, test is the
  last 30%. A random split on time-series data lets the model see the future
  through neighbouring rows and inflates the score.
* **The model predicts the change, not the level.** Predicting the level lets a
  model score well by echoing its own input — the first version did exactly
  that and *lost* to the trend line. Modelling the delta forces it to learn
  what drives movement.

What it learns, by permutation importance: time of day (the check-in and
check-out rhythm), distance to the next event, and combined event pressure.
Those are precisely the signals a trend line cannot see, which is why it wins.

**Venue capacity** treats an event as a constrained building rather than only a
source of demand, which turns out to be three problems that peak at different
times. Over-subscription is the simplest: the Semi Final sells 58,000 against
55,000 seats, so 3,000 people will not get in and the city absorbs them.
Ingress is a rate problem, not a volume one — 40% of a crowd arrives in the
final hour, and when that outruns the gates a queue builds; the Closing Concert
peaks at 9,000 people waiting outside an arena that is already full. Egress is
the sharpest and the one that couples venues to transport: everybody leaves at
once, and whether that clears depends on the links out of the zone rather than
on the venue at all.

That last figure exposed something the rest of the dashboard could not see.
Old Town Arena has two links at 4,200 an hour, so even a modest crowd takes
close to seven hours to clear, which is the last-mile connectivity problem the
brief names.

**Alternative routing** answers the question a congestion reading raises. The
dashboard could say a metro link was at 96%; it could not say what to do
instead. Routing treats the network as a graph and looks for a way around,
modelling two things that make the answer non-obvious: a congested link is
slower as well as fuller, so a longer route can be quicker, and changing
service costs time, so a detour has to be worth it.

It is honest about failure. During an egress surge every link out of the venue
saturates at once, and the answer comes back as "no alternative, stagger
departures" rather than four equally jammed options. Of thirty congested
corridors around the Semi Final, four have a usable alternative; the best is a
direct route at 100% against a two-leg route at 82% for eight minutes more.

**Risk detection** applies thresholds to those projections. A zone is critical
at 95% occupancy *or* if it breaches within 4 hours — that second condition is
the early warning the problem statement asks for. Transport links are checked
separately against their per-hour capacity.

**Recommendation** ranks alternatives nearest-first. Someone turned away from
the venue wants the closest room that still exists, not the emptiest one across
the city, so proximity leads and spare capacity breaks ties between zones at a
similar distance. A zone only drops out of the list once it is genuinely full or
sits beyond 8km, and every option carries its score breakdown so the dashboard
shows why a zone was picked rather than asserting it.

This cascades outward. While the venue has room it is the intake point. Once it
fills it can no longer take anyone, so the focus moves to the nearest zone that
can, and destinations are re-ranked from there. Stadium Precinct fills and Old
Town becomes the hub, 1.7km out; when Old Town fills, Riverside East takes over.
The hub walks away from the venue as the city saturates.

Thresholds are deliberately early. A zone starts getting onward directions at
45% and enters redirect at 75%: waiting until it is nearly full leaves no time
to move anyone.

**Simulation** closes the loop. A recommendation says where to send people; the
simulator tests what happens if you do it. Move 800 visitors out of a saturating
zone and it reports the new occupancy on both sides, where they land, and
whether the alert clears.

It models two things rather than hand-waving them. Receiving zones have a
ceiling, so visitors spill into the next-best option once one fills, and only a
share of people actually follow guidance, so the headline number is never the
number that moves.

**Two interfaces, one dataset.** The problem statement asks for a command
interface for organizers and operators alongside a planning interface for
attendees; the tabs in the masthead switch between them.

*Command* is the control room: the 3D city twin above the pressure map, then
venue floor, absorption grid, scenario sim, disruption scenarios and redirect
options in the main column; alerts, incentives, the orchestration log, the
event schedule, transport load and the model scorecard in the rail. The rail's
height varies with the alert count, so the two columns cannot be matched by a
fixed height. The last card in the main column takes up whatever slack is left,
which keeps them ending together without leaving bare background under it.

*Attendee* is the same inventory expressed as something a visitor would book.
Two properties per zone, with rating, nightly rate, rooms left and the saving
against staying at the venue. Badges are derived, not assigned: a property reads
HOT because its zone is filling, GEM because it is well rated and still quiet,
SAVE because it is materially cheaper than the venue zone. Both sides read the
same occupancy figures, so what the operator is balancing is what the visitor
sees.

## Pages

| Page | What it is |
|---|---|
| `home.html` | Landing page: two paths under the title (register as an attendee, or organise an event), one link to the live dashboard, plus about, features, FAQ and contact. Served at `/` |
| `attendee.html` | Attendees only: sign in or create an account, then register for events. Each registration emails a live status link |
| `organizer.html` | Organisers only: sign in or create an account, then gates, floor-map upload and the what-if simulator (cancel an event, demand surge, close a transport link, redirect visitors) |
| `status.html` | The live page an attendee's emailed link opens: countdown, crowd inside, every gate's status, and the best exit once the event ends. Updates every few seconds |
| `index.html` | The live dashboard. Header: Home, Digital Twin, Stays & Experiences. Live KPIs stay pinned to the top |
| `twin.html` | The city digital twin: a real, colourful 3D map (MapLibre over OpenFreeMap/OpenStreetMap, with CARTO as backup) with 3D buildings, zone towers, a pressure heat surface and transport arcs. With location on, the camera follows you as you walk ("Follow me"; dragging the map pauses it). Falls back to an offline 3D model if the map can't load |

The twin's map library is vendored in `frontend/vendor/`, but the basemap tiles
come from OpenFreeMap (colourful "Liberty" style), or CARTO if that is unreachable, both built on OpenStreetMap, so the real map needs internet. Without it
(or without WebGL) the twin quietly switches to the offline model.

Sign-in, registration and the status page call the live API, so they only work
with the backend running. On a static deploy the pages say so rather than failing
silently.

### Accounts

Attendees and organisers have separate accounts and separate sign-in pages. An
attendee account cannot open the organiser console and the reverse. Passwords are
stored as salted PBKDF2 hashes in `backend/data/accounts.csv`, and repeated failed
sign-ins on one account are throttled.

Anyone can create an organiser account by default, which is fine for a demo but
means anyone could then change gates. Set an access code to close that:

```bash
export EVENTPULSE_ORGANIZER_ACCESS_CODE=choose-something-secret
```

The sign-up form then asks for it. A built-in demo organiser
(`organizer@eventpulse.demo` / `eventpulse`, overridable with
`EVENTPULSE_ORGANIZER_EMAIL` / `EVENTPULSE_ORGANIZER_PASSWORD`) still works so a
fresh checkout is usable straight away.

### Accounts in Supabase

Attendee and organiser accounts (and attendee registrations) are stored in Supabase
Postgres, so they survive restarts and redeploys: sign up once, sign in anywhere. The
server connects with the database password from `backend/.supabase` (private, git-ignored):

```
SUPABASE_PROJECT_REF=ozostnxlxllotzmrnold
SUPABASE_DB_PASSWORD=...
SUPABASE_DB_HOST=aws-0-ap-southeast-1.pooler.supabase.com   # session pooler: IPv4
SUPABASE_DB_PORT=5432
```

Tables `ep_accounts` and `ep_attendees` are created automatically, with row-level security
on and no public policies, so only the server can read them (passwords are PBKDF2 hashes).
Accounts from the old `data/accounts.csv` are copied in once. Without these settings, or
with `EVENTPULSE_DB=off`, the app falls back to the CSV files.

### Email

Registering emails the attendee their status link. With no mail server configured
it only simulates the email (the page says so, and shows the message and link).
To send real ones, set these before starting the API:

```bash
export SMTP_HOST=smtp.example.com
export SMTP_PORT=587                       # default; 465 with SMTP_SECURITY=ssl
export SMTP_USER=you@example.com
export SMTP_PASSWORD=...                   # most providers want an app password
export SMTP_FROM="EventPulse <no-reply@yourdomain>"
export EVENTPULSE_PUBLIC_URL=https://your-host   # what the attendee's device can reach
```

The link in the email opens `status.html`, which calls the API, so
`EVENTPULSE_PUBLIC_URL` must point at a host that runs the API (see `DEPLOY.md`),
not just the static site.

### Partner API (for ticketing platforms)

The business model is B2B: the platform that sells the tickets plugs EventPulse in, and
its buyers get a live link inside that platform's own ticket email or app.

1. An organiser creates a key in the organiser console ("Partner access"). It is shown
   once; only a SHA-256 hash is stored (`data/partner_keys.csv`).
2. The platform's server calls the API with `X-API-Key: ep_live_...`:
   - `GET /partner/v1/events`: the organiser's events
   - `POST /partner/v1/attendees` `{event_id, email, send_email?}`: registers a buyer and returns their live link
   - `GET /partner/v1/events/{id}` and `GET /partner/v1/attendees/{token}`: live conditions plus a ready-to-show sentence
3. It shares the link with the buyer.

A key reaches only its organiser's venues, and a platform can only read buyers it
registered. Keys are revocable, at most 10 active per organiser, and rate limited
(`EVENTPULSE_PARTNER_RATE`, default 120 requests a minute). The full developer guide is
`frontend/partners.html`.

### EventPulse domain model (Nugen)

EventPulse runs its own AI model: a base model aligned through
[Nugen](https://docs.nugen.in) on EventPulse's operations, then used across the app.

```
Base model (e.g. Llama 3.2 3B)  ->  Nugen alignment on the EventPulse corpus
  ->  EventPulse domain model  ->  operator plans, attendee guidance, copilot
```

- **Corpus** (`backend/engine/domain_corpus.py`, built into `data/nugen_corpus/`): an
  operations handbook (thresholds, gates, exits, transport, weather twin, privacy, how to
  write to attendees) plus 122 worked cases. Each case is a real situation from this
  system's engines (zones, alerts, gates, exit rebalancing, offers, and the weather twin
  under seven weather scenarios) paired with the operator plan from the playbook
  (`engine/playbook.py`). A 28-question benchmark comes from held-out hours.
- **Pipeline** (`backend/nugen_align.py`): uploads the corpus, uploads the benchmark,
  aligns an alignment-ready base model, waits, deploys it, and records Nugen's before/after
  benchmark scores in `data/nugen_state.json`. Each step resumes if interrupted.
- **In the app:** the aligned model writes the operator plan (`GET /ai/plan`,
  `POST /weather-twin/plan`), shown in the twin's weather what-if, the dashboard's weather
  card and the organiser console. It also writes the attendee's "Right now" message
  and answers the copilot when there is no Anthropic key. Every answer is checked:
  it must be a numbered plan that uses only numbers from the situation it was given.
  Otherwise, or if Nugen is unreachable, the playbook's plan is shown and the page says so.
  `GET /ai/nugen` shows where the pipeline stands.

```bash
cd backend
export NUGEN_API_KEY=...                  # from https://platform.nugen.in (never written to disk)
python nugen_align.py models              # base models you can align
python nugen_align.py run                 # corpus -> upload -> benchmark -> align -> deploy -> evaluate
python nugen_align.py try --hour 106      # ask the aligned model about a live situation
```

Then start the API with `NUGEN_API_KEY` set. `NUGEN_MODEL_ID` overrides the model id.

### Weather twin

The digital twin has a weather layer that learns how weather moves through the city and
simulates what-ifs on a copy of the live state (it never changes the live system).

- **Live weather:** Open-Meteo, no key. Current conditions, the last 6 hours and the next
  24 hours for every zone, plus each zone's elevation (`engine/weather_live.py`).
- **Social signals** (`engine/social.py`, `GET /weather/social`): public posts and news
  about the weather, no keys needed.
  - Sources: Reddit (r/mumbai, r/india search and new-post feeds), Mastodon (weather
    hashtags and #mumbai/#monsoon filtered to weather talk, on mastodon.social and
    mastodon.online), and Google News search RSS.
  - Each item is read for what it reports (flooding, transport, alerts, heat, power, all
    clear), the part of a trip it touches (airport, trains, metro, roads, hotels, events),
    the traveller's reaction (frustrated, worried, relieved) and the localities it names,
    mapped to zones.
  - Trends: counts per category in 6-hour steps over two days, with direction, built from
    a rolling history (`data/social_history.jsonl`: ids, times, categories and zones only).
  - Emerging conditions: a kind of report running at twice its usual rate in the last 6 hours.
  - Reddit allows about one request a minute without an account, so each refresh asks one
    query in turn and keeps posts for a week (`data/social_reddit.json`).
  - Fresh flooding and disruption reports raise that zone's flood estimate in the twin
    (weight halves every 6 hours).
  - Shown in the dashboard's Social Pulse section, the twin's weather panel and map, and
    the attendee's status page ("What people are reporting").
- **Model:** Bayesian regressions learned from the city's history: rain multiplies
  transport demand by about 1.20 and slows movement between zones to about 61% (the data
  was generated with 1.18 and 0.62). What the history can't teach (flooding by elevation,
  wind, heat, staff access, attendance) starts from stated priors. Field reports from the
  organiser console update it and are saved (`data/weather_observations.csv`).
- **Cascade:** weather -> zones (rain, flood risk, heat) -> transport links, gates, staff,
  attendance -> stranded visitors, hotels, restaurants, alerts. 400 draws per run;
  results are 10th/50th/90th percentiles. `engine/weather_twin.py`.
- **Where it shows:** the twin (panel and map layer; `twin.html?weather=whatif` opens the
  what-if), a Weather section on the dashboard, field reports in the organiser console,
  and weather advice on the attendee's status page. The copilot can call it too.
- **API:** `GET /weather/live`, `GET /weather/social`, `GET /weather-twin/model`,
  `GET /weather-twin/state`, `POST /weather-twin/simulate`, `POST /organizer/weather/observe`.

It needs internet for live data; without it the twin says so and assumes calm weather.

### AI features

Three things use a language model. All are optional and off until you set a key;
each has a plain fallback so nothing depends on it.

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export EVENTPULSE_AI_MODEL=claude-sonnet-5      # default
```

| Feature | Where | What the model does | Without a key |
|---|---|---|---|
| Organiser copilot | organiser page, "Ask the copilot" | Answers plain-language questions by calling read-only tools on the live model (city summary, alerts, busiest zones, gates, exit plan, the what-if scenarios). It cannot change anything. | A plain briefing of the same headline numbers |
| Attendee guidance | "Right now" card on the status page | Rewrites one factual sentence warmly. The rewrite is discarded if it adds any number that was not in the facts. Only facts about the event go to the model, never the attendee's email or link. | The factual sentence itself |
| Floor-plan reading | organiser page, "Read gates from my plan" | Looks at the uploaded plan image (PNG/JPEG/WebP) and proposes gate positions. Nothing is saved until the organiser adds them. | Button is disabled; place gates by hand |

Limits: the copilot is rate limited to 20 questions per organiser per 5 minutes, and
questions are capped at 600 characters. `engine/ai.py` holds all of it.

### The live clock

The status page follows a shared simulated clock that steps through the event on its
own (`GET /clock`), one simulated hour every 6 seconds. Two settings help demos:

```bash
export EVENTPULSE_CLOCK_TICK_SECONDS=6      # real seconds per simulated hour
export EVENTPULSE_CLOCK_START_HOUR=96       # begin just before the Semi Final (hour 102)
```

## The dashboard

The signature element is the pressure bar. Solid fill is occupancy now, the
hatched extension is where the forecast takes it, and the white line is capacity.
When the hatching crosses the line, that zone is in trouble — the thesis of the
whole project in one visual.

Colour is a signal, not decoration: teal has headroom, amber is filling, coral
is at capacity. Bars are coloured by *risk* rather than current reading, so a
zone at 82% that will breach in 6 hours reads amber and matches its alert.

Click any zone or alert to see where to redirect people.

## Structure

```
eventpulse/
├── contracts/mock-data.json   Phase 0 — the agreed API shapes
├── backend/
│   ├── generate_city.py       Phase 1
│   ├── engine/                Phases 2–4
│   ├── api.py                 Phase 5
│   └── data/                  generated CSVs (git-ignored)
└── frontend/
    ├── index.html             the dashboard
    └── vendor/                React, vendored for offline use
```

## Still to build

- **Phase B — attendee view.** A second page: enter your destination zone, get a
  congestion warning and alternatives. The API already returns everything it
  needs; it's a frontend task.
- **Phase 7 — demo prep.** Seed the data so a breach lands on cue, script the
  walkthrough, record a backup.

## Not doing

Blockchain. It isn't required by the problem statement and a half-finished
component is worse than none. Worth one future-scope slide: a trust layer for
cross-vendor settlement and an auditable action log.
