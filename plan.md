# Internal Crowd Redirection — Plan

Extends EventPulse from city-scale zone management down to inside-the-venue
gate/exit management: multiple entries/exits, an organizer floor-map page,
live crowd density from device localization, and an AI redirect engine —
especially for the post-event egress surge.

## Problem restated

Today a venue has one `gate_throughput_per_hour` and one
`exit_throughput_per_hour` (`backend/engine/venues.py`). We need per-gate
granularity: named entries/exits placed on a real floor plan, live occupancy
per zone of that floor plan, and an engine that tells people (and organizers)
which exit is least crowded — especially in the post-event egress spike,
which is already flagged in the README as "the sharpest problem."

## 1. Data model — from one gate to many

New CSV-backed entities, same pattern as `backend/data/`:

- `gates.csv` — `gate_id, venue_id, name, type (entry|exit|both), capacity_per_hour, x, y, status (open|closed|restricted)`
- `floor_zones.csv` — `zone_id, venue_id, name, polygon (floor-map coords), capacity`
- `floor_maps.csv` — `venue_id, image_url/svg_url, width, height, scale`

`venues.py`'s `analyse_event` gets refactored so ingress/egress throughput is
`sum(open gates)` instead of a single scalar — same math, just per-gate
before aggregating. Mechanical change, not a rewrite.

## 2. Organizer page

New tab/page alongside the existing Command/Attendee tabs in
`frontend/index.html`:

- Upload a floor plan (image or SVG).
- Click-to-place gates/exits/entries on the map; set capacity and type.
- Toggle a gate open/closed live — the actual control organizers need during
  an incident ("close Exit C, it's overwhelmed").
- Define floor zones (concourses, stairwells, chokepoints) by drawing
  polygons over the map, so density attributes to a place, not just a gate.
- Persist via new `POST /organizer/venues/{id}/gates` and
  `POST /organizer/venues/{id}/map` endpoints in `api.py`.

## 3. Crowd localization — the hard, sensitive part

Two device populations, two signal sources:

- **Wifi-associated devices** — venue APs already know which AP (and often
  AP + RSSI) each device is attached to. Coarse position from nearest-AP, or
  finer via RSSI trilateration/fingerprinting with ≥3 APs per zone.
- **Non-wifi devices on cellular** — cannot be triangulated from the public
  network without carrier cooperation (DAS/small-cell sector load, or a
  carrier CDR feed). A real deployment needs either a data-sharing agreement
  with the venue's DAS/telco operator, or a venue-side BLE beacon layer as a
  substitute. Be upfront about this limitation when pitching: arbitrary
  phones cannot be located off the public cell network from an app alone.

Two build tracks given that constraint:

1. **Demo/hackathon track** — synthetic positions, same spirit as
   `generate_city.py`: a stream of fake device pings `(x, y, timestamp)` per
   venue, weighted toward realistic egress-surge clustering near exits. Lets
   the whole density/redirect pipeline be built and demoed with no real infra.
2. **Production track** — an ingestion adapter interface
   (`localization/sources/wifi_adapter.py`, `localization/sources/das_adapter.py`)
   so real feeds swap in later without touching the density/redirect engine.

Privacy: aggregate to zone-level counts immediately, discard raw per-device
trajectories beyond a short rolling window, anonymize/hash device
identifiers. This is a legal requirement in most jurisdictions for any
location-tracking-adjacent system, not just good practice.

## 4. Crowd density engine (`backend/engine/density.py`) — built

Rather than generating a separate synthetic device-ping stream with no
consumer yet, this reuses the per-gate live load already computed in Phase 9
(queue size during ingress, exit demand weighted by `default_exit_share`
during egress) and maps it onto named floor zones: one bowl/hall per venue,
one concourse per gate. It returns `people_count`, `capacity`,
`occupancy_pct` and `severity` per zone — a floor-plan-scale version of the
pressure-bar concept already used city-wide, exposed at `GET
/density/{venue_zone_id}?hour=`.

This is written against the same shape a real wifi/BLE localization feed
would need to produce anyway, so it is swappable later (Section 3's adapter
interface) without any caller changing. Verified: at the Semi Final's egress
hour, Main Gate Concourse reaches 170% of capacity (`critical` territory)
while the four side concourses sit at 85–90%, purely because more of the
crowd defaults to the gate they entered through — this is the imbalance the
redirect engine (Section 5) exists to correct.

## 5. Redirect engine (`backend/engine/exit_redirect.py`) — built

Mirrors `recommend.py`/`routing.py`'s pattern:

- Input: current density per zone, gate status/throughput, live queue per
  open exit.
- Output: per internal zone, a ranked list of exits by
  `distance × inverse congestion`, excluding closed/overwhelmed exits — the
  same nearest-first-with-tiebreak logic `recommend.py` already implements
  for city zones, run on the floor-plan graph instead.
- Special-cased **post-event egress mode**: once an event's `end_hour`
  passes, switch every zone's redirect targets to "nearest open exit,"
  recompute every 30–60s, and auto-raise a critical alert (reusing `risk.py`'s
  alert shape) the moment any exit's incoming flow exceeds
  `capacity_per_hour`.

Built at `GET /exit-redirect/{venue_zone_id}?hour=&from_floor_zone_id=`:
ranks every non-closed gate by a 0.4/0.6 blend of distance-from-current-
position and inverse congestion (`load_ratio` against `GATE_LOAD_CRITICAL`).
Verified live at the Semi Final's egress hour: the four side gates rank 1–4
(open, score ~0.40–0.43), Main Gate ranks last (`restricted`, score 0.20)
despite being the nearest gate to the bowl — the redirect-away-from-the-
popular-exit behaviour this phase exists to produce. Along the way,
`current_event_for_zone` was factored into `venues.py` as a shared helper —
`density.py` and `exit_redirect.py` both need "which of this venue's several
events is live at this hour," and `exit_redirect.py` originally got this
wrong by grabbing the first scheduled event regardless of hour.

## 6. Getting the redirect to people

Three channels, cheapest-first:

- **Digital signage / organizer dashboard** — live floor-map heatmap with
  arrows toward the recommended exit. Most feasible to build and demo now.
- **Attendee app push/SMS** — "Exit C is congested, use Exit A." Needs an
  attendee-side app or SMS gateway integration; stretch phase.
- **Staff radio/notification feed** — reuse the existing "orchestration log"
  concept from the Command tab so stewards see the same redirect
  instructions.

## 7. Minimum gate requirement

A single-entry/single-exit venue defeats the purpose of redirect (nowhere to
redirect *to*). Enforce a **minimum of 4–5 entry/exit gates per venue** as a
validation rule in the organizer page and in `gates.csv` ingestion — reject
(or warn hard on) a venue configured with fewer, since the redirect engine
degrades to "wait in the only line" below that.

## 8. 3D live heatmap

Both organizer and attendee views need a 3D crowd-density view of the venue,
not just the 2D pressure bars used city-wide:

- **Rendering**: a WebGL 3D layer (deck.gl or three.js) over the floor plan,
  with zone density mapped to height/color — a literal "pressure map" instead
  of a metaphorical bar. This is the one place the project's "no npm install,
  works on hackathon wifi" constraint has to bend: 3D rendering libraries are
  too large to hand-vendor cleanly. Options, in order of preference:
  1. Vendor a single minimal 3D lib (e.g. three.js core, ~150KB) into
     `frontend/vendor/` like React already is — keeps the offline guarantee.
  2. Fall back to a CDN `<script>` tag only if truly offline demo isn't a
     requirement anymore for this feature.
- **Data feed**: same `density.py` output as the 2D version — the 3D view is
  a renderer choice, not a new data pipeline.
- **Update cadence**: live, same 30–60s tick as the redirect engine (Section
  5), via polling or websocket.
- Applies at two scales: the *floor-plan* 3D heatmap (inside one venue) and
  optionally the *city-wide* 3D heatmap (zones across the event footprint),
  reusing the same renderer with different data sources.

## 9. Live, moving turn-by-turn guidance

The attendee status page (from the SMS/link flow) becomes a live navigator,
not just a status readout:

- **Locating the user**: browser Geolocation API (outdoors, GPS/network) for
  city-scale navigation; indoor position comes from the wifi/BLE localization
  pipeline (Section 3) once inside a venue with that infra, or a manual
  "which gate did you enter" fallback otherwise.
- **Indoor pathfinding**: model each venue's floor zones + gates as a graph
  and run A*/Dijkstra from the user's current zone to the recommended exit
  (`exit_redirect.py`'s output becomes a graph target, not just a label) —
  redraws the path each tick as the user's position updates.
- **Outdoor routing + mode of transport**: once outside, hand off to the
  existing `routing.py` (already does alternative-route-around-congestion on
  the transport graph) to route the user to their actual destination (hotel,
  transit stop, parking), and surface which mode — walk, metro, bus — the
  route implies, the same way `routing.py` already reasons about link
  capacity and speed.
- The map re-centers and re-draws the route continuously as the user's
  location changes — this is the "map should redirect with the user as
  they're moving" requirement.

## 10. Points of interest: hotels and restaurants near quiet zones

`hotels.py` already ranks nearby bookable inventory by zone occupancy. Extend
the same pattern to restaurants (`restaurants.csv`, same shape: zone, rating,
distance) and surface both as pins on the 3D map, biased toward zones that
are *not* congested — so "where should I go while I wait for the crowd to
clear" is answered visually on the same map as the congestion itself, not a
separate list.

## 11. Role-based access: organizer vs. attendee

Two distinct logins, distinct capabilities — this needs real auth, not just
separate UI tabs:

- **Organizer account**: create/edit venues, floor maps, gates (with the
  4–5 minimum enforced), open/close gates live, view the full 3D heatmap
  across all zones, see alerts and the orchestration log, trigger
  egress-mode overrides.
- **Attendee account** (created at registration, Section on SMS flow):
  view only their own live position/status, their personalized route and
  transport mode, nearby quiet-zone hotels/restaurants — no visibility into
  other attendees or organizer controls.
- Implementation: a minimal auth layer (`POST /auth/login`, JWT or session
  cookie, `role: organizer|attendee` claim), gating both API routes and which
  frontend build (organizer console vs. attendee navigator) gets served.

## 12. What-if simulator — extended scenarios — built

`simulate.py` already answers "move 800 people out of zone X, what happens" —
extend the same engine to run named catastrophic/disruptive scenarios end to
end through the existing forecast → risk → recommend chain, rather than
building separate one-off logic per scenario:

- **Event cancellation**: zero out the event's expected attendance mid-run,
  recompute egress on whoever already arrived, and surface the transport/zone
  impact of a sudden reverse-flow with no "planned" egress curve to smooth it.
- **Demand surge** (e.g. +10,000 unexpected arrivals at once): inject a spike
  into `ARRIVAL_PROFILE` for one hour and re-run `analyse_event` — this is
  the over-subscription and ingress-queue math `venues.py` already has,
  just fed an adversarial input instead of the planned one.
- **Transport link closure** (e.g. nearest metro station closes): zero out
  that link's `capacity_per_hour` in the transport graph and re-run
  `routing.py`'s alternative-route search plus `venues.py`'s
  `_transport_capacity_out` — this directly tests whether the "no
  alternative, stagger departures" failure mode from the README gets
  triggered.

Each scenario returns the same shape `simulate.py` already returns (new
occupancy, alerts cleared/raised, where people land) so the existing
what-if UI just gets a scenario picker instead of only a manual "move N
people" slider.

Built as `backend/engine/scenarios.py`, exposed at `GET
/scenarios/cancel/{event_id}`, `GET /scenarios/surge/{event_id}?extra=`, and
`GET /scenarios/close-link/{link_id}`. Implementation approach: rather than
adding scenario-specific branches to `venues.py`/`routing.py`, each scenario
copies the real dataset to a scratch directory, mutates exactly the one CSV
value the scenario describes (event's `expected_attendance`/`end_hour`, or a
dropped row in `transport.csv`), and re-runs the untouched engines against
it — the existing engines stay exactly as they are; the scenario is entirely
in which numbers they're handed. Verified: cancelling the Semi Final at
hour 101 evacuates only the 27,260 already inside (egress clears in 1.3h,
`ok`) instead of the full 55,000 (2.6h, `warning`), and total alerts drop
15→12; a +10,000 surge on the same event pushes turned-away from 3,000 to
13,000 and flips ingress severity from `warning` to `critical`.

Building the transport-closure scenario surfaced a real, pre-existing bug in
`routing.py`'s `find_routes`: when a link's removal leaves no *direct* route
between two zones (only interchange routes survive), the function crashed
trying to read `direct_route["bottleneck_pct"]` on a `None`. Fixed — `None`
direct routes are now handled as their own case, recommending the best
surviving interchange route instead. Confirmed no change to the normal path
(direct route present) via `python -m engine.routing`.

## 13. Personalized arrival windows

At registration, optionally collect: accommodation zone, preferred arrival
time, transport mode availability, and intended entry gate. From that,
predict the congestion curve for that specific route + gate + time combo
(reusing `forecast.py`'s projection and `venues.py`'s ingress queue model),
and recommend an arrival window that lands *before* the predicted peak —
e.g. event starts 7:00 PM, route congestion peaks 6:00–6:30 PM, so the
recommended window is 5:20–5:40 PM.

This only works as a system if arrival windows are **allocated across all
registrants**, not computed independently per person — otherwise everyone
gets told the same "arrive early" window and the peak just moves earlier.
So this needs a scheduling step: bucket registrants by entry gate, and
stagger recommended windows across the pre-event hours so the *aggregate*
arrival curve stays under each gate's throughput, rather than everyone
being individually optimized into the same slot.

## 14. Reverse-uber: system-optimal (not individually-optimal) routing — built

The redirect and routing engines built so far (Sections 5, 9, 20) each
optimize per user: nearest exit, fastest route. At mega-event scale that's
exactly what causes the problem — everyone routed onto the "best" option
floods it. This section changes the objective function from user-optimal to
system-optimal, the same distinction traffic engineering draws between
Wardrop user equilibrium and system optimum:

- Instead of ranking routes/exits by what's fastest *for this person*, rank
  by marginal cost to the network: how much would routing this person here
  increase congestion for everyone already using that path. This is a small
  but real change to `routing.py`/`exit_redirect.py`'s scoring function —
  add a network-load term, not a new engine.
- Concretely: hold back some fraction of users from the locally-fastest
  option and route them onto a slightly slower but under-used alternative,
  the way `recommend.py` already breaks ties by spare capacity rather than
  raw proximity — this generalizes that same idea from "which zone" to
  "which route, right now."
- This is the mechanism that makes Section 13's staggering and Section 5's
  exit redirect actually hold under load: individually-rational advice
  doesn't self-balance a crowd; the recommendation has to account for what
  everyone else is being told at the same moment.

Built as `backend/engine/reverse_uber.py`, exposed at `GET
/reverse-uber/{venue_zone_id}?hour=`. Rather than a per-person route score,
this operates on the whole egress crowd at once: it compares the naive,
habit-driven per-gate load (the `default_exit_share`-weighted numbers
`venues.py` already computes) against a capacity-proportional balanced
allocation, and reports exactly how many people need to shift off the
over-drawn gate to reach it. Worth noting explicitly: `venues.py`'s
venue-level `hours_to_clear_venue` figure already *assumes* this balance is
achieved — this module is what would actually have to happen for that
number to be true, and quantifies the gap. Verified at the Semi Final's
egress hour: Main Gate runs at 1.33x while the balanced target is 0.9x for
every gate, meaning 8,800 of the 27,500 people naturally heading for Main
Gate need to be redirected (2,200 each) to the four side gates to hit it.

## 15. Dynamic incentive engine — built

Sections 5, 9, and 14 all *inform or route* people. This section *changes
demand itself*: when a zone is trending toward overcrowded, actively pull
people away from it with an incentive, not just a warning.

- Trigger: `density.py`/`risk.py` flags a zone crossing a threshold (same
  severity levels already in use).
- Response: generate a time-boxed offer steering demand toward a specific
  under-crowded zone — e.g. "15% off at participating restaurants in Zone C"
  — sized by how much redirection is actually needed (a zone at 80% needs a
  smaller nudge than one at 98%), the inverse of ride-hailing surge pricing:
  discount the *under-demanded* option rather than upcharge the over-demanded
  one, since raising prices during a public safety-adjacent crowd event is
  the wrong lever.
- Needs a partner/merchant layer: participating businesses register an
  offer + capacity + zone; the engine picks which offers to surface based on
  current density, not a fixed schedule.
- Feedback loop: track whether an offer actually reduced flow toward the
  crowded zone (redemptions + subsequent density trend) so incentive sizing
  can adjust — this is the one component in the whole plan that needs a
  live feedback signal rather than a one-shot computation, so treat it as
  the last piece to build.

Built as `backend/engine/incentives.py`, exposed at `GET /incentives?hour=`.
No merchant registry exists in the project, so rather than inventing a
synthetic list of named businesses with no other consumer, this reuses the
zone-level economics `recommend.py`/`hotels.py` already carry (spare
capacity, distance) as the "participating zone" signal, sized on a linear
ramp from 5% at the warning threshold to 25% at critical. Verified at hour
110: Stadium Precinct at 100% generates 25% offers toward Old Town and
Riverside East; Metro Interchange, only moderately over warning, generates
just 11% toward its own nearest quiet zones — confirming the discount scales
with severity rather than being a flat number. The feedback loop (tracking
whether an offer actually worked) is explicitly not built — there is no real
redemption data to close it against yet.

## Phase plan

| Phase | Component | File |
|---|---|---|
| 9 ✅ | Multi-gate venue model (min 4–5 gates enforced), auto open/restricted/closed per gate driven by live queue/exit load, not a manual toggle | `backend/engine/venues.py`, `backend/generate_city.py` |
| 10 ✅ | Organizer floor-map page | `frontend/organizer.html` |
| 11+12 ✅ | Crowd density per floor zone (bowl + one concourse per gate) | `backend/engine/density.py`, `backend/generate_city.py` (`floor_zones.csv`) |
| 13 ✅ | Exit redirect engine | `backend/engine/exit_redirect.py` |
| 14 ✅ | Live 2D floor heatmap UI | frontend, Command tab addition |
| 15 ✅ | Role-based auth (organizer/attendee) | `backend/auth.py` |
| 16 ✅ | Attendee registration + SMS link flow (SMS simulated) | `backend/engine/attendees.py`, `frontend/status.html` |
| 17 | 3D heatmap renderer (venue + city scale) | `frontend/vendor/three.min.js`, map component |
| 18 | Indoor pathfinding (graph over floor zones/gates) | `backend/engine/indoor_routing.py` |
| 19 | Live user-tracking navigator (moving map, re-route on move) | attendee frontend |
| 20 | Outdoor routing + transport mode surfaced | extend `backend/engine/routing.py` |
| 21 (backend ✅, UI pending) | Nearby hotels/restaurants on map, biased to quiet zones | `backend/engine/restaurants.py` |
| 22 ✅ | What-if simulator: cancellation, surge, transport closure scenarios | `backend/engine/scenarios.py` |
| 23 | Personalized arrival window recommendation | `backend/engine/arrival_window.py` |
| 24 | Arrival-window scheduling across all registrants (stagger, not per-person) | extend Phase 23 |
| 25 ✅ | System-optimal ("reverse-uber") routing objective | `backend/engine/reverse_uber.py` |
| 26 ✅ | Dynamic incentive engine (merchant offers, density-triggered) | `backend/engine/incentives.py` |
| 27 (stretch) | Real wifi/DAS adapters | `backend/localization/sources/` |

Build order: 9 → 11 → 12 → 13 → 14 gets a fully demoable synthetic-data
version end to end first. 15 → 16 add real accounts and the SMS entry point.
17 → 21 build out the 3D/navigation/POI experience on top of that live data.
22 (what-if scenarios) can be built any time after Phase 9–13 since it only
extends the existing `simulate.py`/`routing.py` engines with new inputs.
23 → 24 need live density (Phase 12) and the registration flow (Phase 16)
before arrival windows mean anything. 25 depends on 23–24 existing to have
something to balance against. 26 depends on 12 (density) and a merchant
data model that doesn't exist yet anywhere in the project — build it last
among the "core AI" phases. 27 is the only phase with an external dependency
(telco/DAS partnership) and stays last overall.

## Frontend wiring pass (post Phase 26)

With every backend-only engine built (9, 11+12, 13, 14, 22, 25, 26), did a
pass wiring the ones that had no UI yet into the Command tab:

- **Venue Capacity card removed.** Its occupancy bar and gate-queue chart
  were dropped; the card was renamed `VenueFloor` and now only hosts Gates,
  Floor Density, and Exit Redirect — the three panels this project actually
  needs going forward.
- **Reverse-uber panel** added under Exit Redirect inside `VenueFloor`:
  shows the naive vs. balanced load per gate and how many people to shed or
  absorb, live.
- **Incentives panel** added to the rail, next to Alerts: ranked active
  offers with the zone-relief reasoning visible.
- **Disruption Scenarios card** added to the main stage: three preset
  buttons (cancel the Semi Final, +10,000 surge, close the Stadium↔South
  Market link) against the real `/scenarios/*` endpoints, showing the
  before/after diff inline.

All new components checked for brace/paren balance by hand (no Node/browser
available in this environment to run Babel directly) and both servers
re-verified live after each change.

## Auth, organizer page, registration, and static-export pass

Built in one pass, in dependency order, each verified live before moving on:

- **Phase 15 — auth (`backend/auth.py`).** Dependency-free HMAC-signed
  tokens rather than pulling in a JWT library, since a token here only needs
  to prove a role, not interoperate with anything external. One hardcoded
  organizer account (`organizer@eventpulse.demo` / `eventpulse`, overridable
  by env var). `require_organizer` now gates every destructive organizer
  endpoint. Verified: bad credentials rejected, valid login accepted,
  tampered token rejected, protected endpoints 401 without a token.

- **Gate protection + the 4-gate minimum, actually enforced.**
  `POST /organizer/gates/{id}/status` now requires organizer auth, and
  separately rejects taking a gate out of service if fewer than
  `MIN_OPEN_GATES` (4) would remain open — Section 7's requirement, enforced
  at the API layer, not just at data-generation time. Verified: with 5 gates,
  closing a 2nd one is correctly blocked once the 1st is already closed.

- **Phase 10 — organizer page (`frontend/organizer.html`).** Plain HTML/JS
  (no React/vendoring needed for one page) with: login, a venue picker, a
  floor-plan view (uploaded map image behind live-colored gate dots), the
  gate list with take-out-of-service/restore controls, and an add-gate form.
  Adding a gate automatically creates its matching concourse floor zone via
  the same backend logic used at data-generation time, so a new gate shows
  up in density/exit-redirect immediately — verified via curl: new gate
  appeared in `/gates`, matching concourse appeared in `/density`.
  Floor-map upload/download round-tripped byte-for-byte in testing.
  Linked from the main dashboard's nav.

- **Phase 16 — registration + status (`backend/engine/attendees.py`,
  `frontend/status.html`).** `POST /register` creates a token and an
  `attendees.csv` row; SMS sending is explicitly simulated (logged +
  returned as `sms_preview`), not wired to a real provider — there are no
  provider credentials in this project, and pretending to send a text while
  doing nothing would be worse than being honest about the gap. Swapping in
  Twilio or similar later is a one-function change behind the same
  interface. `GET /status/{token}` (used by `status.html`, a no-login page
  polled every 10s) returns the attendee's event state and, once egress
  starts, the ranked exit list from `exit_redirect.py`. Verified end to end:
  register → simulated SMS body includes the right link → status before
  egress shows "upcoming" → status after egress shows the ranked gates with
  Main Gate correctly deprioritized.

- **Phase 21 backend (`backend/engine/restaurants.py`).** Mirrors
  `hotels.py` exactly, plus a `quiet_pick` flag for the zones incentives.py's
  offers point at. **UI not built yet** — surfacing it in the Attendee tab
  is a small, low-risk follow-up, deliberately left out this pass rather
  than rushed.

- **Static export kept in sync.** `export_static.py` now also precomputes
  `density`, `exit_redirect`, `reverse_uber` per venue and `restaurants`/
  `incentives` city-wide into every hour's bundle, and the frontend
  components (`FloorDensity`, `ExitRedirect`, `ReverseUberPanel`,
  `IncentivesPanel`) accept a `staticData` prop and use it instead of a live
  fetch when running in static mode — otherwise these would silently 404 on
  a real Vercel deploy, which has no backend behind it. Ran the full export
  (121 hours) successfully and spot-checked the output. **Not verified**:
  an actual live Vercel deployment — that needs the user to run `vercel`
  themselves; what's confirmed here is that the generated JSON is correct
  and the frontend code path that would consume it is wired correctly.

- **`DisruptionScenarios` intentionally left live-only.** These are
  parameterized what-if queries, not naturally cacheable per hour the way
  the rest of the bundle is. Rather than leave it silently broken on a
  static deploy, it now detects `STATIC_MODE` and shows an explicit
  "needs the live API" message instead of dead buttons.

- Added `python-multipart` to `requirements.txt` (needed for the map-upload
  endpoint's file handling) — the only new dependency introduced this pass.

## What's genuinely still open

- **Phase 17 — 3D heatmap.** Needs either a vendored WebGL library (breaks
  the project's "no npm install, fully offline" guarantee unless done
  carefully) or a CDN dependency. Not attempted — this needs a deliberate
  choice from the user first (see the question already asked about this
  earlier in the plan), not a rushed pick.
- **Phase 18/19 — indoor pathfinding graph + live moving navigator.**
  `exit_redirect.py` currently uses straight-line floor-plan distance, which
  is a reasonable proxy for an open bowl-and-concourse layout but is not a
  real graph over corridors/stairwells, and there is no real indoor
  positioning source to feed a "moving" map with in this environment.
- **Phase 20 — outdoor transport mode.** Already effectively satisfied:
  `routing.py`'s `find_routes` has returned a `mode` per leg since Phase 8;
  nothing new was needed here, it just isn't surfaced as its own UI element.
- **Phase 23/24 — personalized arrival windows + staggered scheduling
  across registrants.** Not attempted this pass — this is a real
  multi-registrant optimization problem (bucket by gate, stagger without
  everyone converging on the same "arrive early" slot), and rushing it was
  judged higher-risk than leaving it explicitly open.
- **Phase 27 — real wifi/DAS localization.** Out of scope by design; needs
  a venue's actual network infrastructure or a telco/DAS partnership, which
  cannot exist inside this environment.

## Next step

The organizer/auth/registration flow and the static-export sync were the
highest-value remaining pieces and are now done and verified. What's left is
either a design decision only the user can make (3D rendering approach) or
genuinely large, separable pieces of work (indoor pathfinding, arrival-window
scheduling) best scoped as their own focused sessions rather than appended
here.
