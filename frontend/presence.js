/*
 * EventPulse: sharing your position anonymously, to help map crowds.
 *
 * Off by default and only ever switched on by the visitor. When on, this sends the
 * visitor's position, at most every 15 seconds or when they have moved about 40 m, with a
 * random per-tab id. The server reduces it to a coarse grid cell on arrival and keeps that
 * for five minutes (see backend/engine/presence.py). Turning it off, or closing the
 * page, removes the visitor straight away.
 *
 * Needs the live API. Where there is none (a static deploy) it stays quietly unavailable
 * and the switch never appears.
 */
(function (root) {
  "use strict";

  var KEY = "eventpulse_presence";          // "on" only after an explicit choice
  var SID_KEY = "eventpulse_sid";
  var SEND_EVERY_MS = 15000;
  var MOVE_M = 40;
  var API = root.location.port === "8080" ? "http://127.0.0.1:8000" : "";

  var state = { available: false, on: false, demo: false };
  var lastSent = 0, lastPos = null, sid = null, inflight = Promise.resolve();

  function store(area, key, val) {
    try {
      var s = area === "local" ? root.localStorage : root.sessionStorage;
      if (val === undefined) return s.getItem(key);
      if (val === null) s.removeItem(key); else s.setItem(key, val);
    } catch (e) { /* storage blocked */ }
    return null;
  }

  function sessionId() {
    if (sid) return sid;
    sid = store("session", SID_KEY);
    if (!sid) {
      var a = new Uint8Array(12);
      (root.crypto || root.msCrypto).getRandomValues(a);
      sid = Array.prototype.map.call(a, function (b) { return ("0" + b.toString(16)).slice(-2); }).join("");
      store("session", SID_KEY, sid);
    }
    return sid;
  }

  function moved(a, b) { return !a || root.EPLocate.distanceKm(a.lat, a.lng, b.lat, b.lng) * 1000 >= MOVE_M; }

  function send(g) {
    lastSent = Date.now(); lastPos = { lat: g.lat, lng: g.lng };
    inflight = root.fetch(API + "/presence", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sid: sessionId(), lat: g.lat, lng: g.lng }),
    }).catch(function () { /* the crowd map is a nicety; never disturb the page */ });
  }

  function leave() {
    if (!state.available) return;
    // keepalive lets the request outlive the page. (sendBeacon can't carry JSON across origins.)
    root.fetch(API + "/presence/leave", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sid: sessionId() }), keepalive: true }).catch(function () {});
  }

  function onLocate(g) {
    if (!state.on || g.lat == null) return;
    if (Date.now() - lastSent >= SEND_EVERY_MS || moved(lastPos, g)) send(g);
  }

  function setOn(v) {
    state.on = !!v;
    store("local", KEY, v ? "on" : "off");
    if (v) { var g = root.EPLocate.get(); if (g.lat != null) send(g); }
    else {
      lastPos = null; lastSent = 0;
      inflight.then(leave);          // a send still on its way must land first, or it would outlive the opt-out
    }
    root.EPLocate.refresh();
  }

  // The switch lives in the location pill, next to the button that stops location.
  function bannerExtra(box, s) {
    if (!state.available || s.status !== "granted") return;
    var b = root.document.createElement("button");
    b.type = "button";
    b.textContent = state.on ? "Sharing anonymously ✓" : "Help map crowds";
    b.title = state.on
      ? "Your position is shared anonymously as a coarse area and forgotten after 5 minutes. Click to stop."
      : "Share your position anonymously so crowds at places nobody has mapped can show up. Off until you choose.";
    b.onclick = function () { setOn(!state.on); };
    box.appendChild(b);
  }

  function init() {
    state.on = store("local", KEY) === "on";
    if (root.EPLocate && root.EPLocate.addBannerExtra) root.EPLocate.addBannerExtra(bannerExtra);
    root.addEventListener("pagehide", function () { if (state.on) leave(); });
    return root.fetch(API + "/presence/config").then(function (r) { return r.ok ? r.json() : null; }).then(function (c) {
      if (c) { state.available = true; state.demo = !!c.demo_enabled; state.minGroup = c.min_group; state.ttl = c.ttl_seconds; }
      if (root.EPLocate) { root.EPLocate.onChange(onLocate); root.EPLocate.refresh(); }
    }).catch(function () { /* no backend here */ });
  }

  root.EPPresence = { init: init, get: function () { return state; }, setOn: setOn, leave: leave, api: API };
})(window);
