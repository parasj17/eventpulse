# Deploying EventPulse

The dataset is synthetic and deterministic, so every response the dashboard can
ask for is knowable in advance. `export_static.py` writes them all to JSON, which
means the whole thing runs as a **static site**: no server, no cold start, and
nothing that can fall asleep between someone receiving the link and opening it.

That matters more than it sounds. Most free tiers that run Python sleep after
about fifteen minutes idle and take the better part of a minute to wake, so a
judge opening your link hours later would sit on a blank page.

---

## Build the static bundle

From `backend/`, in order:

```bash
python generate_city.py     # the dataset
python -m engine.model      # train and save the model
python export_static.py     # write frontend/data/*.json
```

The export takes a couple of minutes and produces roughly 30 MB: one 52 KB
bundle per event hour, a small file per zone per hour holding the simulator's
results, and `meta.json` for the model scorecard.

Simulations are kept separate because they are four fifths of the data but only
ever shown for the one selected zone. Inlining them meant every hour tick
downloaded work the viewer would never look at.

Rerun all three whenever you change the generator or the engines.

---

## Vercel (recommended)

Fast, free, global CDN, and no sleeping. The site is static, so there is no
build step to configure.

### Option A: CLI, quickest

```bash
npm install -g vercel
cd frontend
vercel
```

Answer the prompts: link to your account, accept the detected settings, and say
no to overriding the build command. Then promote it:

```bash
vercel --prod
```

You get a URL like `https://eventpulse.vercel.app`.

**Run this from inside `frontend`, not the project root.** Vercel serves what it
is pointed at, so `index.html`, `vendor/` and `data/` need to sit at the top of
what you deploy.

### Option B: GitHub import

1. Push the project to GitHub.
2. vercel.com → **Add New** → **Project** → import the repo.
3. Set **Root Directory** to `frontend`. This is the step people miss; without
   it Vercel looks at the repo root and finds no `index.html`.
4. Framework Preset: **Other**. Leave build and output settings empty.
5. Deploy.

Pushing to the branch redeploys automatically after that.

### What the config does

`frontend/vercel.json` sets long cache lifetimes on `data/` and `vendor/`, which
never change between deploys, and none on `index.html` so a redeploy is picked
up immediately.

---

## Alternatives

- **Netlify** — drag the `frontend` folder onto app.netlify.com/drop. No account
  needed for a temporary link, and the fastest route to something live.
- **Hugging Face Spaces** — create a Space with **SDK: Static** (Docker Spaces
  need a paid plan), then push the contents of `frontend` to it.
- **GitHub Pages** — push `frontend` to a repo, then Settings → Pages.

---

## Checking it worked

Open the link and confirm:

- The KPI strip shows numbers rather than skeleton blocks
- Zone Absorption lists ten tiles
- The clock advances on its own and alerts appear as it does
- Dragging the Scenario Sim slider updates instantly
- The **Attendee** tab scrolls to Experience Bundles

If every panel stays a loading skeleton, `data/` did not get deployed. Confirm
`https://<your-link>/data/hour-26.json` returns JSON. The usual cause is
deploying the project root instead of `frontend`.

---

## Running it locally

Unchanged. The page uses the live API on localhost and the static bundle
everywhere else, so the same file works both ways.

Two terminals:

```bash
cd backend
pip install -r requirements.txt
python generate_city.py
python -m engine.model
python -m uvicorn api:app --reload
```

```bash
cd frontend
python -m http.server 8080
```

<http://127.0.0.1:8080>

Or as one service, the way the Docker image runs it:

```bash
cd backend
python -m uvicorn api:app --reload
```

<http://127.0.0.1:8000>

The `Dockerfile` is still here if you later want the live API hosted somewhere
that runs containers, such as Render or Fly.

---

## What needs a live API (and what that means for email)

The static site on Vercel shows the dashboard, the twin and the landing page from
precomputed data. These need the live API, which a static deploy does not have:

- Attendee and organiser accounts, sign-in and registration
- The live status page an emailed link opens
- Organiser gate control, floor-map upload and the what-if simulator
- Disruption scenarios on the dashboard

So for emailed links to work, the API has to be hosted somewhere the attendee's
device can reach. The `Dockerfile` runs the API and the frontend together as one
service (Render, Fly, Railway and similar), which is the simplest way to do it. Set
the `SMTP_*` variables described in the README and set `EVENTPULSE_PUBLIC_URL` to
that service's address. Otherwise the link in the email would point at `localhost`
and be dead on the attendee's device.

Set `NUGEN_API_KEY` on the API service so the EventPulse domain model (aligned through Nugen, see the README) serves plans and guidance; keep `data/nugen_state.json` with the deploy so it knows the model id. To switch on the other AI features (copilot tools, floor-plan reading) set `ANTHROPIC_API_KEY` on the API service too; they need the live API, so they are absent on a static-only deploy.

Also set `EVENTPULSE_SECRET` (it signs sign-in tokens; the built-in default is only
for local use) and, unless you want anyone to be able to create organiser accounts,
`EVENTPULSE_ORGANIZER_ACCESS_CODE`.

Accounts and registrations are stored in `backend/data/accounts.csv` and
`backend/data/attendees.csv`, which many container hosts wipe on redeploy. Attach a
persistent disk if that matters.

The 3D twin's map tiles are fetched from OpenFreeMap (falling back to CARTO), both built on OpenStreetMap, in the viewer's
browser; nothing extra to deploy, but the map needs internet there.
