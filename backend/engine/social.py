"""
Social signals: what travellers, residents and the news are saying about the weather,
right now and over the last few days.

Public sources, none needing a key:

* **Reddit** (r/mumbai search feeds): travellers and residents asking about and
  reporting on commutes, trains, flooding and heat, in their own words.
* **Mastodon** hashtag timelines on more than one server: #MumbaiRains and similar,
  plus #mumbai and #monsoon filtered to weather talk.
* **Google News** search RSS: reports that usually lead with the places affected.

Every item is cleaned to plain text and read for:
  - what it reports: flooding, transport disruption, heavy rain or alerts, heat, power, all clear;
  - which part of a trip it touches: airport, trains, metro, roads, hotels, events;
  - the reaction in it: frustrated, worried, relieved, or just reporting;
  - where: the Mumbai localities it names, mapped to the nearest zone.

On top of that: trends (how often each kind of report comes up, per six hours over two
days, and whether it is rising), and emerging conditions (a kind of report running at
twice its usual rate in the last six hours). Each fetch is added to a small history
(data/social_history.jsonl: ids, times, categories and zones only, no text or authors),
so trends build up across refreshes.

Fresh flooding and disruption reports near a zone are evidence for the weather twin
(weather_twin.py); their weight halves every six hours.

Run:  python -m engine.social
"""

import html
import json
import math
import os
import re
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from email.utils import parsedate_to_datetime
from datetime import datetime

import httpx

from engine.recommend import haversine_km

MASTODON = [u for u in os.environ.get("EVENTPULSE_MASTODON_URL", "https://mastodon.social,https://mastodon.online").split(",") if u]
NEWS_RSS = os.environ.get("EVENTPULSE_NEWS_URL", "https://news.google.com/rss/search")
REDDIT = os.environ.get("EVENTPULSE_REDDIT_URL", "https://www.reddit.com")
TAGS = ["mumbairains", "mumbairain", "mumbaiweather", "mumbaimonsoon", "mumbaiflood"]
BROAD_TAGS = ["mumbai", "monsoon"]                      # busy tags: kept only when a post is about Mumbai weather
NEWS_QUERIES = ["Mumbai rain", "Mumbai waterlogging", "Mumbai rain traffic local trains", "Mumbai heat"]
# (subreddit, search) pairs; None means the subreddit's newest posts. Fetched one after
# another: Reddit refuses parallel requests from one address.
REDDIT_QUERIES = [("mumbai", "rain OR flood OR waterlogging OR flooded OR monsoon"),
                  ("mumbai", None),
                  ("mumbai", "traffic OR train OR local OR metro OR airport OR flight OR commute"),
                  ("mumbai", "heat OR humid OR hot OR weather"),
                  ("india", "mumbai rain OR mumbai flood OR mumbai waterlogging")]

USER_AGENT = "EventPulse/1.0 (crowd-safety hackathon demo)"
HISTORY_FILE = "social_history.jsonl"
HISTORY_DAYS = 14
CACHE_SECONDS = int(os.environ.get("EVENTPULSE_SOCIAL_CACHE", "600"))
TIMEOUT_S = 10
MAX_AGE_H = 24 * 14                 # older than this is history, not a signal
EVIDENCE_HALF_LIFE_H = 6            # a report's weight as evidence halves every 6 hours
MATCH_KM = 5                        # a locality this close to a zone counts for it

CATEGORIES = {
    "flooding": ["flood", "waterlog", "water-log", "water logging", "inundat", "knee-deep", "knee deep", "submerged",
                 "overflow", "water entered", "flooded", "high tide"],
    "transport": ["local train", "locals ", "train services", "trains delayed", "delayed", "disrupted", "suspended",
                  "cancelled", "canceled", "diverted", "traffic jam", "traffic snarl", "snarl", "metro services",
                  "flights", "bus services", "best buses", "road closed", "closed for traffic", "stranded"],
    "alert": ["red alert", "orange alert", "yellow alert", "imd", "heavy rain", "very heavy", "extremely heavy",
              "cloudburst", "thunderstorm", "cyclone", "downpour", "lashes", "lashed", "pouring", "torrential"],
    "heat": ["heatwave", "heat wave", "hot and humid", "scorching", "temperature soars", "heatstroke", "heat stroke",
             "sweltering", "humid"],
    "power": ["power cut", "power outage", "blackout", "no electricity", "power supply"],
    "clear": ["rain stops", "rains subside", "water recedes", "receded", "services resume", "resumed", "back to normal",
              "sunny", "clear skies"],
}
# Approximate centres of Mumbai localities that come up in weather reports.
PLACES = {
    "andheri": (19.1197, 72.8468), "kurla": (19.0726, 72.8845), "bandra": (19.0596, 72.8295), "bkc": (19.0674, 72.8678),
    "bandra kurla complex": (19.0674, 72.8678), "dadar": (19.0178, 72.8478), "sion": (19.0390, 72.8619),
    "chembur": (19.0522, 72.9005), "ghatkopar": (19.0860, 72.9081), "powai": (19.1176, 72.9060),
    "santacruz": (19.0810, 72.8410), "vile parle": (19.0990, 72.8490), "sakinaka": (19.1030, 72.8880),
    "saki naka": (19.1030, 72.8880), "marol": (19.1190, 72.8820), "worli": (19.0176, 72.8170),
    "lower parel": (18.9960, 72.8300), "mahim": (19.0400, 72.8400), "matunga": (19.0270, 72.8570),
    "khar": (19.0700, 72.8370), "hindmata": (19.0080, 72.8430), "milan subway": (19.0870, 72.8430),
    "andheri subway": (19.1160, 72.8470), "king's circle": (19.0290, 72.8570), "kings circle": (19.0290, 72.8570),
    "wadala": (19.0170, 72.8640), "vikhroli": (19.1110, 72.9280), "jogeshwari": (19.1380, 72.8480),
    "goregaon": (19.1640, 72.8490), "malad": (19.1860, 72.8480), "airport": (19.0896, 72.8656),
    "chakala": (19.1110, 72.8610), "kalina": (19.0760, 72.8640), "vidyavihar": (19.0790, 72.8970),
    "mulund": (19.1720, 72.9560), "colaba": (18.9067, 72.8147), "churchgate": (18.9322, 72.8264),
    "cst": (18.9398, 72.8355), "byculla": (18.9790, 72.8340), "parel": (19.0060, 72.8420),
}
_PLACE_RE = re.compile(r"\b(" + "|".join(sorted((re.escape(p) for p in PLACES), key=len, reverse=True)) + r")\b", re.I)
_TAG_RE = re.compile(r"<[^>]+>")
_MUMBAI = re.compile(r"mumbai|bombay|#mumbai", re.I)

_cache = {"at": 0, "data": None}


def clean(text):
    return re.sub(r"\s+", " ", html.unescape(_TAG_RE.sub(" ", text or ""))).strip()


# Transport and power trouble only counts when the item is about the weather; a story
# about the airport is not a flight disruption report.
WEATHER_CONTEXT = re.compile(r"\b(rains?|rainfall|raining|monsoon|weather|flood\w*|waterlog\w*|storms?|heat\w*|imd|"
                             r"downpour|cyclone|thunder\w*|humid\w*|temperature|winds?|high tide|red alert|orange alert)\b")
# Planning and budget stories mention floods without reporting one happening.
NOT_AN_EVENT = ["crore", "budget", "tender", "project", "policy", "plan ", "plans ", "proposal", "study finds", "anniversary"]


def classify(text):
    t = " " + text.lower() + " "
    cats = [c for c, words in CATEGORIES.items() if any(w in t for w in words)]
    if not WEATHER_CONTEXT.search(t):
        cats = [c for c in cats if c not in ("transport", "power")]
    if any(w in t for w in NOT_AN_EVENT):
        cats = [c for c in cats if c not in ("flooding", "transport")]
    return cats


TOPICS = {
    "airport": ["airport", "flight", "flights", "terminal", "t2 ", "t1 ", "boarding", "landed", "landing"],
    "trains": ["local train", "locals", "train", "trains", "railway", "platform", "harbour line", "western line", "central line", "station"],
    "metro": ["metro"],
    "roads": ["traffic", "road", "roads", "cab", "cabs", "uber", "ola", "auto", "rickshaw", "jam", "highway", "flyover", "bus", "best bus"],
    "hotels": ["hotel", "hotels", "staying at", "stay at", "place to stay", "room for", "rooms", "airbnb", "hostel", "check-in", "check in"],
    "events": ["stadium", "match", "concert", "event", "mandal", "festival", "ganpati", "visarjan", "procession", "crowd"],
}
MOODS = {
    "frustrated": ["stuck", "stranded", "nightmare", "worst", "hell", "hours", "ridiculous", "pathetic", "fed up", "again",
                   "shame", "useless", "angry", "why is", "no one", "every year", "struggle", "suffer"],
    "worried": ["stay safe", "be careful", "avoid", "scared", "worried", "unsafe", "danger", "please help", "any update",
                "is it safe", "should i", "will there be", "advice", "warning", "alert"],
    "relieved": ["resumed", "cleared", "back to normal", "finally", "relief", "lovely", "pleasant", "beautiful", "enjoy",
                 "enjoying", "cool breeze", "made it"],
}


def topics(text):
    t = " " + text.lower() + " "
    return [k for k, words in TOPICS.items() if any(w in t for w in words)]


def mood(text):
    t = " " + text.lower() + " "
    scores = {k: sum(t.count(w) for w in words) for k, words in MOODS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] else "reporting"


def locate(text, zones):
    """The zones the text points at, via the localities it names."""
    hits = {}
    for m in _PLACE_RE.finditer(text):
        name = m.group(1).lower()
        lat, lng = PLACES[name]
        best = min(zones, key=lambda z: haversine_km(lat, lng, float(z["lat"]), float(z["lng"])))
        d = haversine_km(lat, lng, float(best["lat"]), float(best["lng"]))
        if d <= MATCH_KM:
            hits.setdefault(best["zone_id"], {"zone_id": best["zone_id"], "place": m.group(1), "lat": lat, "lng": lng})
    return list(hits.values())


def _age_h(dt, now):
    return max(0.0, (now - dt.timestamp()) / 3600) if dt else None


def _get(url, **kw):
    return httpx.get(url, timeout=TIMEOUT_S, headers={"User-Agent": USER_AGENT}, follow_redirects=True, **kw)


def _mastodon_tag(base, tag, now):
    r = _get(f"{base}/api/v1/timelines/tag/{tag}", params={"limit": 40})
    r.raise_for_status()
    out = []
    for st in r.json():
        if st.get("reblog"):
            continue
        text = clean(st.get("content"))
        if not text:
            continue
        # Busy tags carry everything from food to politics: keep only Mumbai weather talk.
        if tag in BROAD_TAGS and not (WEATHER_CONTEXT.search(text.lower()) and (tag == "mumbai" or _MUMBAI.search(text) or _PLACE_RE.search(text))):
            continue
        try:
            dt = datetime.fromisoformat(st["created_at"].replace("Z", "+00:00"))
        except (KeyError, ValueError):
            dt = None
        out.append({"id": "m-" + str(st.get("uri") or st.get("id")), "source": "Mastodon", "kind": "traveller",
                    "channel": f"#{tag}", "author": (st.get("account") or {}).get("acct", ""), "text": text[:500],
                    "url": st.get("url") or st.get("uri"), "age_h": _age_h(dt, now), "tag": tag,
                    "engagement": int(st.get("favourites_count") or 0) + int(st.get("reblogs_count") or 0)
                    + int(st.get("replies_count") or 0)})
    return out


REDDIT_KEEP_DAYS = 7
REDDIT_STORE = "social_reddit.json"
_reddit_state = {"store": None, "turn": 0}


def _reddit_path():
    from engine.forecast import DATA_DIR
    return os.path.join(DATA_DIR, REDDIT_STORE)


def _reddit_all(now):
    """
    Reddit lets an address without an account make about one request a minute, so each
    refresh asks one of the queries in turn and keeps what it has seen for a week (on
    disk, so a restart doesn't lose it). Returns every kept post with its age brought up to date.
    """
    if _reddit_state["store"] is None:
        try:
            with open(_reddit_path()) as f:
                _reddit_state["store"] = json.load(f)
        except (OSError, ValueError):
            _reddit_state["store"] = {}
    store = _reddit_state["store"]
    sub, q = REDDIT_QUERIES[_reddit_state["turn"] % len(REDDIT_QUERIES)]
    _reddit_state["turn"] += 1
    errors = []
    try:
        for it in _reddit(sub, q, now):
            store[it["id"]] = {**it, "at": now - (it["age_h"] or 0) * 3600}
    except Exception as e:
        errors.append(f"Reddit r/{sub} {'new' if q is None else repr(q[:24])}: {type(e).__name__} (tries again next refresh)")
    for k in [k for k, v in store.items() if now - v["at"] > REDDIT_KEEP_DAYS * 86400]:
        del store[k]
    try:
        with open(_reddit_path() + ".tmp", "w") as f:
            json.dump(store, f)
        os.replace(_reddit_path() + ".tmp", _reddit_path())
    except OSError:
        pass
    return [{**{k: v for k, v in it.items() if k != "at"}, "age_h": max(0.0, (now - it["at"]) / 3600)} for it in store.values()], errors


def _reddit(sub, q, now):
    if q is None:
        r = _get(f"{REDDIT}/r/{sub}/new/.rss", params={"limit": 50})
    else:
        r = _get(f"{REDDIT}/r/{sub}/search.rss", params={"q": q, "restrict_sr": 1, "sort": "new", "t": "week"})
    r.raise_for_status()
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out = []
    for e in ET.fromstring(r.content).findall("a:entry", ns):
        title = clean(e.findtext("a:title", namespaces=ns))
        body = clean(e.findtext("a:content", namespaces=ns))
        body = re.sub(r"submitted by /u/\S+.*$", "", body).strip()
        stamp = e.findtext("a:published", namespaces=ns) or e.findtext("a:updated", namespaces=ns)
        try:
            dt = datetime.fromisoformat(stamp)
        except (TypeError, ValueError):
            dt = None
        link = e.find("a:link", ns)
        author = e.findtext("a:author/a:name", namespaces=ns) or ""
        out.append({"id": "r-" + (e.findtext("a:id", namespaces=ns) or title), "source": "Reddit", "kind": "traveller",
                    "channel": f"r/{sub}", "author": author.replace("/u/", "u/"),
                    "text": (title + (" \u2014 " + body[:300] if body else ""))[:500],
                    "title": title, "url": link.get("href") if link is not None else None,
                    "age_h": _age_h(dt, now), "tag": q or "new", "engagement": 0})
    return out


def _news_query(q, now):
    r = _get(NEWS_RSS, params={"q": q + " when:3d", "hl": "en-IN", "gl": "IN", "ceid": "IN:en"})
    r.raise_for_status()
    out = []
    for it in ET.fromstring(r.content).iter("item"):
        title = clean(it.findtext("title"))
        if not title:
            continue
        try:
            dt = parsedate_to_datetime(it.findtext("pubDate"))
        except (TypeError, ValueError):
            dt = None
        src = it.find("source")
        out.append({"id": "n-" + re.sub(r"\W+", "", title.lower())[:80], "source": "News", "kind": "news",
                    "channel": src.text if src is not None and src.text else "News", "author": "",
                    "text": title[:300], "url": it.findtext("link"), "age_h": _age_h(dt, now), "tag": q, "engagement": 0})
    return out


def _collect(now):
    """All sources at once. Returns (items, errors, per-source counts)."""
    jobs = [("Mastodon", f"{base.split('//')[-1]} #{tag}", _mastodon_tag, (base, tag, now))
            for base in MASTODON for tag in TAGS + BROAD_TAGS]
    jobs += [("News", f"'{q}'", _news_query, (q, now)) for q in NEWS_QUERIES]
    items, errors, counts = [], [], {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        reddit = pool.submit(_reddit_all, now)
        futs = [(src, label, pool.submit(fn, *args)) for src, label, fn, args in jobs]
        try:
            got, errs = reddit.result()
            items += got; errors += errs
            counts["Reddit"] = len(got)
            counts["_reddit_ok"] = not errs
        except Exception as e:
            errors.append(f"Reddit: {type(e).__name__}")
        for src, label, f in futs:
            try:
                got = f.result()
                items += got
                counts[src] = counts.get(src, 0) + len(got)
            except Exception as e:
                errors.append(f"{src} {label}: {type(e).__name__}")
    return items, errors, counts


def evidence_weight(age_h):
    return 0.0 if age_h is None else 0.5 ** (age_h / EVIDENCE_HALF_LIFE_H)


def analyse(raw, zones, now=None, history=None):
    """Dedupe, keep what's about Mumbai weather and recent, read each item, and summarise."""
    now = now or time.time()
    seen, out = set(), []
    for it in sorted(raw, key=lambda i: (i["age_h"] if i["age_h"] is not None else 1e9)):
        key = re.sub(r"\W+", "", (it.get("title") or it["text"]).lower())[:90]
        if key in seen or (it["age_h"] is not None and it["age_h"] > MAX_AGE_H):
            continue
        seen.add(key)
        # A news result has to be about Mumbai; hashtag and r/mumbai posts already are.
        if it["source"] == "News" and not (_MUMBAI.search(it["text"]) or _PLACE_RE.search(it["text"])):
            continue
        cats = classify(it["text"])
        if not cats:
            continue
        out.append({**it, "categories": cats, "topics": topics(it["text"]),
                    "mood": mood(it["text"]) if it["kind"] == "traveller" else None,
                    "places": locate(it["text"], zones), "weight": round(evidence_weight(it["age_h"]), 3),
                    "at": now - (it["age_h"] or 0) * 3600})

    recent = [i for i in out if i["age_h"] is not None and i["age_h"] <= 24]
    counts_24h = {c: sum(1 for i in recent if c in i["categories"]) for c in CATEGORIES}
    by_zone = {}
    for i in out:
        for p in i["places"]:
            z = by_zone.setdefault(p["zone_id"], {"zone_id": p["zone_id"], "flood_evidence": 0.0,
                                                   "disruption_evidence": 0.0, "reports": 0})
            z["reports"] += 1
            if "flooding" in i["categories"]:
                z["flood_evidence"] += i["weight"]
            if "transport" in i["categories"]:
                z["disruption_evidence"] += i["weight"]
    for z in by_zone.values():
        z["flood_evidence"] = round(z["flood_evidence"], 3)
        z["disruption_evidence"] = round(z["disruption_evidence"], 3)

    pool = _merge_history(history or [], out)
    pressure = sum(i["weight"] for i in out if set(i["categories"]) & {"flooding", "transport", "alert"})
    last48 = [i for i in out if i["age_h"] is not None and i["age_h"] <= 48]
    people = [i for i in last48 if i["kind"] == "traveller"]
    return {
        "items": out[:80],
        "counts_24h": counts_24h,
        "total": len(out), "recent_24h": len(recent),
        "by_source": {s: sum(1 for i in out if i["source"] == s) for s in ("Reddit", "Mastodon", "News")},
        "zones": sorted(by_zone.values(), key=lambda z: -(z["flood_evidence"] + z["disruption_evidence"])),
        "signal_level": round(1 - math.exp(-pressure / 3), 3),
        "moods_48h": {m: sum(1 for i in people if i["mood"] == m) for m in ("frustrated", "worried", "relieved", "reporting")},
        "topics_48h": {t: sum(1 for i in last48 if t in i["topics"]) for t in TOPICS},
        "trends": trends(pool, now),
        "emerging": emerging(pool, out, now),
    }


# ---------------------------------------------------------------- trends
BUCKET_H = 6
N_BUCKETS = 8                      # two days in six-hour steps


def trends(pool, now):
    """Per category: counts in each six-hour bucket over two days (oldest first), and
    whether the latest twelve hours run above or below the twelve before."""
    out = {}
    for c in CATEGORIES:
        b = [0] * N_BUCKETS
        for i in pool:
            if c not in i["categories"]:
                continue
            age = (now - i["at"]) / 3600
            k = int(age // BUCKET_H)
            if 0 <= k < N_BUCKETS:
                b[N_BUCKETS - 1 - k] += 1
        recent, before = sum(b[-2:]), sum(b[-4:-2])
        direction = "rising" if recent >= 2 and recent >= 1.5 * max(1, before) else (
            "falling" if before >= 2 and recent <= 0.5 * before else "steady")
        out[c] = {"buckets": b, "last_12h": recent, "previous_12h": before, "direction": direction}
    return out


def emerging(pool, items, now):
    """A kind of report running at twice its usual rate in the last six hours, with where."""
    out = []
    for c in ("flooding", "transport", "alert", "heat", "power"):
        last6 = [i for i in pool if c in i["categories"] and now - i["at"] <= 6 * 3600]
        before = [i for i in pool if c in i["categories"] and 6 * 3600 < now - i["at"] <= 48 * 3600]
        rate_now, rate_before = len(last6) / 6, len(before) / 42
        if len(last6) >= 2 and rate_now >= 2 * max(rate_before, 1 / 42):
            fresh = [i for i in items if c in i["categories"] and i["age_h"] is not None and i["age_h"] <= 6]
            places = sorted({p["place"] for i in fresh for p in i["places"]})
            zones = sorted({p["zone_id"] for i in fresh for p in i["places"]})
            out.append({"category": c, "reports_6h": len(last6), "times_usual": round(rate_now / max(rate_before, 1 / 42), 1),
                        "places": places[:5], "zones": zones, "example": fresh[0]["text"][:200] if fresh else None,
                        "example_url": fresh[0]["url"] if fresh else None})
    return out


# ---------------------------------------------------------------- history (for trends)
def _history_path():
    from engine.forecast import DATA_DIR
    return os.path.join(DATA_DIR, HISTORY_FILE)


def load_history(now=None):
    now = now or time.time()
    rows = []
    try:
        with open(_history_path()) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if now - r["at"] <= HISTORY_DAYS * 86400:
                    rows.append(r)
    except OSError:
        pass
    return rows


def save_history(rows):
    tmp = _history_path() + ".tmp"
    try:
        with open(tmp, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        os.replace(tmp, _history_path())
    except OSError:
        pass


def _merge_history(history, items):
    """History rows plus today's items, one row per item id. Only what trends need."""
    byid = {r["id"]: r for r in history}
    for i in items:
        byid[i["id"]] = {"id": i["id"], "at": round(i["at"]), "source": i["source"], "kind": i["kind"],
                         "categories": i["categories"], "zones": [p["zone_id"] for p in i["places"]]}
    return list(byid.values())


def fetch(zones, now=None, force=False):
    now = now or time.time()
    if not force and _cache["data"] is not None and now - _cache["at"] < CACHE_SECONDS:
        return _cache["data"]
    raw, errors, counts = _collect(now)
    # Available means a source answered this time; saved Reddit posts alone don't count.
    reddit_ok = counts.pop("_reddit_ok", False)
    available = bool(counts.get("Mastodon") or counts.get("News") or reddit_ok)
    history = load_history(now)
    data = {"available": available, "fetched_at": now,
            "sources": ["Reddit (r/mumbai, r/india)", "Mastodon (mastodon.social, mastodon.online)", "Google News"],
            "fetched": counts, "errors": errors, **analyse(raw, zones, now, history)}
    if available:
        save_history(_merge_history(history, data["items"]))
    if not available and _cache["data"] and _cache["data"].get("available") and now - _cache["at"] < 3600:
        return {**_cache["data"], "stale": True}
    _cache.update(at=now, data=data)
    return data


if __name__ == "__main__":
    import pandas as pd
    from engine.forecast import DATA_DIR
    zs = pd.read_csv(os.path.join(DATA_DIR, "zones.csv")).to_dict("records")
    t0 = time.time()
    s = fetch(zs)
    print(f"{time.time() - t0:.1f}s  fetched {s['fetched']}  kept {s['total']} by source {s['by_source']}")
    print("24h:", s["counts_24h"], "| moods:", s["moods_48h"], "| topics:", s["topics_48h"])
    print("trends:", {c: (t["buckets"], t["direction"]) for c, t in s["trends"].items() if sum(t["buckets"])})
    print("emerging:", s["emerging"])
    print("errors:", s["errors"][:5])
    for i in s["items"][:14]:
        print(f"  [{i['source']:8} {i['channel'][:14]:14}] {i['age_h'] and round(i['age_h'],1)}h {i['categories']} {i['topics']} {i['mood']} "
              f"{[p['place'] for p in i['places']]} | {i['text'][:80]}")
