/*
 * EventPulse: the attendee's location, shared by every page.
 *
 * Asks the browser for the visitor's position as soon as a page loads, keeps
 * watching it as they move, and turns it into things a visitor can use: how busy
 * the area around them is, and which stays are closest right now.
 *
 * The position stays in the browser. Nothing here sends it anywhere; sharing an
 * anonymous, aggregated reading to help map crowds is a separate, explicit choice
 * (see presence.js). The ranking and congestion logic below is plain functions over
 * data the page already has (zones, hotels, transport), so it works the same on the
 * static deploy, with no backend.
 */
(function (root) {
  "use strict";

  var CONSENT_KEY = "eventpulse_geo";          // "on" | "off": the visitor's choice, remembered
  var CACHE_KEY = "eventpulse_geo_last";       // last fix, so the next page starts with a position
  var CACHE_MAX_AGE_MS = 10 * 60 * 1000;

  var state = { status: "idle", lat: null, lng: null, accuracy: null, heading: null, speed: null, ts: 0, fromCache: false, error: "" };
  var listeners = [];
  var extras = [];                // other modules add controls to the status pill
  var watchId = null;

  // ------------------------------------------------------------------ pure logic
  function distanceKm(lat1, lng1, lat2, lng2) {
    var r = 6371, p1 = lat1 * Math.PI / 180, p2 = lat2 * Math.PI / 180;
    var dp = (lat2 - lat1) * Math.PI / 180, dl = (lng2 - lng1) * Math.PI / 180;
    var a = Math.sin(dp / 2) * Math.sin(dp / 2) + Math.cos(p1) * Math.cos(p2) * Math.sin(dl / 2) * Math.sin(dl / 2);
    return 2 * r * Math.asin(Math.sqrt(a));
  }

  // The zone whose centre is nearest, with how far away it is.
  function nearestZone(zones, lat, lng) {
    var best = null;
    (zones || []).forEach(function (z) {
      var km = distanceKm(lat, lng, z.lat, z.lng);
      if (!best || km < best.km) best = { zone: z, km: km };
    });
    return best;
  }

  // Walking for short hops, otherwise a ride with a few minutes' wait added.
  function travel(km) {
    if (km <= 1.2) return { minutes: Math.max(1, Math.round(km / 5 * 60)), mode: "walk" };
    return { minutes: Math.round(6 + km / 18 * 60), mode: "ride" };
  }

  // How busy an area is, in words a visitor can act on.
  function assess(occupancyPct) {
    if (occupancyPct >= 95) return { level: "full", label: "Full", advice: "Avoid this area if you can. Head somewhere quieter." };
    if (occupancyPct >= 85) return { level: "crowded", label: "Crowded", advice: "Very busy. Expect queues." };
    if (occupancyPct >= 60) return { level: "filling", label: "Filling up", advice: "Getting busy. Worth moving soon." };
    return { level: "calm", label: "Calm", advice: "Plenty of room." };
  }

  // The most loaded transport link touching a zone, if any.
  function busiestLink(links, zoneId) {
    var best = null;
    (links || []).forEach(function (l) {
      if (l.from_zone !== zoneId && l.to_zone !== zoneId) return;
      if (!best || l.load_pct > best.load_pct) best = l;
    });
    return best;
  }

  /*
   * Stays near a position, nearest first. Only distance counts: we don't have live
   * room availability from the hotels, so nothing here claims one is filling up or
   * has rooms left. `zonesById` is accepted for older callers and ignored.
   */
  function rankHotels(hotels, lat, lng, zonesById, limit) {
    var out = [];
    (hotels || []).forEach(function (h) {
      if (h.lat == null || h.lng == null) return;
      var km = distanceKm(lat, lng, h.lat, h.lng);
      var t = travel(km);
      out.push({
        hotel: h, km: km, minutes: t.minutes, mode: t.mode, score: km,
        distance: km < 1 ? Math.round(km * 1000 / 10) * 10 + " m" : km.toFixed(1) + " km",
        reason: (km < 1 ? Math.round(km * 1000 / 10) * 10 + " m" : km.toFixed(1) + " km") + " away · about " + t.minutes + " min " +
          (t.mode === "walk" ? "on foot" : "by car"),
      });
    });
    out.sort(function (x, y) { return x.km - y.km; });
    return limit ? out.slice(0, limit) : out;
  }

  // ------------------------------------------------------------------ browser state
  function store(area, key, val) {
    try {
      var s = area === "local" ? root.localStorage : root.sessionStorage;
      if (val === undefined) return s.getItem(key);
      if (val === null) s.removeItem(key); else s.setItem(key, val);
    } catch (e) { /* storage blocked: works for this page only */ }
    return null;
  }

  function emit() {
    listeners.slice().forEach(function (fn) { try { fn(state); } catch (e) { /* one listener must not break the rest */ } });
  }

  function set(patch) {
    Object.keys(patch).forEach(function (k) { state[k] = patch[k]; });
    emit();
  }

  function onFix(p) {
    var fix = { lat: p.coords.latitude, lng: p.coords.longitude, accuracy: p.coords.accuracy, ts: p.timestamp || Date.now() };
    store("session", CACHE_KEY, JSON.stringify(fix));
    // Heading (degrees from north) and speed (m/s) come from the device while it moves; null otherwise.
    var heading = Number.isFinite(p.coords.heading) ? p.coords.heading : null;
    var speed = Number.isFinite(p.coords.speed) ? p.coords.speed : null;
    set({ status: "granted", lat: fix.lat, lng: fix.lng, accuracy: fix.accuracy, heading: heading, speed: speed,
          ts: fix.ts, fromCache: false, error: "" });
  }

  function onError(e) {
    if (e && e.code === 1) { store("local", CONSENT_KEY, "off"); set({ status: "denied", error: "Location is blocked for this site." }); }
    else set({ status: state.lat != null ? "granted" : "unavailable", error: "Couldn't get your location just now." });
  }

  function start() {
    if (!root.navigator || !root.navigator.geolocation) { set({ status: "unavailable", error: "This device can't share a location." }); return; }
    store("local", CONSENT_KEY, "on");
    if (watchId != null) return;
    set({ status: state.lat != null ? "granted" : "asking" });
    // Keep watching: suggestions follow the visitor as they move. A moderate
    // maximumAge lets the device reuse a fix instead of waking the GPS every time.
    watchId = root.navigator.geolocation.watchPosition(onFix, onError, { enableHighAccuracy: true, maximumAge: 5000, timeout: 20000 });
  }

  function stop() {
    if (watchId != null && root.navigator && root.navigator.geolocation) root.navigator.geolocation.clearWatch(watchId);
    watchId = null;
    store("local", CONSENT_KEY, "off");
    store("session", CACHE_KEY, null);
    set({ status: "off", lat: null, lng: null, accuracy: null, ts: 0, fromCache: false, error: "" });
  }

  // Called on every page load. Honours an earlier "off", otherwise asks straight away.
  function autoStart() {
    if (store("local", CONSENT_KEY) === "off") { set({ status: "off" }); return; }
    try {
      var c = JSON.parse(store("session", CACHE_KEY) || "null");
      if (c && Date.now() - c.ts < CACHE_MAX_AGE_MS) set({ lat: c.lat, lng: c.lng, accuracy: c.accuracy, ts: c.ts, fromCache: true, status: "granted" });
    } catch (e) { /* ignore a corrupt cache */ }
    start();
  }

  function onChange(fn) {
    listeners.push(fn);
    fn(state);
    return function () { listeners = listeners.filter(function (f) { return f !== fn; }); };
  }

  // ------------------------------------------------------------------ small status pill
  var CSS = ".ep-geo{position:fixed;left:14px;bottom:14px;z-index:9999;display:flex;align-items:center;gap:10px;max-width:calc(100vw - 28px);" +
    "background:rgba(16,16,16,.94);border:1px solid #2a2a2a;border-radius:999px;padding:7px 8px 7px 14px;color:#F2F2F2;" +
    "font:500 12.5px -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;box-shadow:0 6px 24px rgba(0,0,0,.5)}" +
    ".ep-geo .d{width:9px;height:9px;border-radius:50%;background:#6A6A6A;flex:none}.ep-geo.on .d{background:#4DA3FF}" +
    ".ep-geo.bad .d{background:#D9A441}.ep-geo button{font:inherit;font-weight:700;cursor:pointer;color:#F2F2F2;background:#242424;" +
    "border:1px solid #333;border-radius:999px;padding:5px 12px}.ep-geo button:hover{background:#2f2f2f}" +
    ".ep-geo button:focus-visible{outline:2px solid #8C8C8C;outline-offset:2px}";

  function mountBanner() {
    if (root.parent !== root) return;                      // an embedded frame leaves this to the page around it
    if (root.document.getElementById("ep-geo")) return;
    var style = root.document.createElement("style"); style.textContent = CSS; root.document.head.appendChild(style);
    var box = root.document.createElement("div");
    box.id = "ep-geo"; box.className = "ep-geo"; box.setAttribute("role", "status");
    root.document.body.appendChild(box);

    function paint(s) {
      box.textContent = "";
      var dot = root.document.createElement("span"); dot.className = "d";
      var text = root.document.createElement("span");
      var btn = root.document.createElement("button"); btn.type = "button";
      var cls = "ep-geo";
      if (s.status === "granted") {
        cls += " on";
        text.textContent = s.fromCache && !s.accuracy ? "Locating you..." : "Using your location" + (s.accuracy ? " · ±" + Math.round(s.accuracy) + " m" : "");
        btn.textContent = "Stop"; btn.onclick = stop;
      } else if (s.status === "asking") {
        text.textContent = "Allow location to see crowding and stays near you"; btn.style.display = "none";
      } else if (s.status === "denied") {
        cls += " bad"; text.textContent = "Location is blocked. Allow it in your browser to see what's near you";
        btn.textContent = "Try again"; btn.onclick = start;
      } else if (s.status === "unavailable") {
        cls += " bad"; text.textContent = s.error || "Location isn't available"; btn.textContent = "Retry"; btn.onclick = start;
      } else {
        text.textContent = "Location is off"; btn.textContent = "Turn on"; btn.onclick = start;
      }
      box.className = cls;
      box.appendChild(dot); box.appendChild(text);
      extras.forEach(function (fn) { try { fn(box, s); } catch (e) { /* keep the pill working */ } });
      if (btn.style.display !== "none") box.appendChild(btn);
    }
    onChange(paint);
  }

  root.EPLocate = {
    distanceKm: distanceKm, nearestZone: nearestZone, travel: travel, assess: assess,
    busiestLink: busiestLink, rankHotels: rankHotels,
    start: start, stop: stop, autoStart: autoStart, onChange: onChange, mountBanner: mountBanner,
    addBannerExtra: function (fn) { extras.push(fn); },
    refresh: emit,
    get: function () { return state; },
  };
})(typeof window !== "undefined" ? window : this);
