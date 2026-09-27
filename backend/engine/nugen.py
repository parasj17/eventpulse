"""
Nugen Intelligence: the client EventPulse uses to align a base model on its own
domain and to run that aligned model inside the app.

    Base model  ->  Nugen alignment on the EventPulse corpus  ->  domain model
    ->  used for operator action plans, attendee guidance and the copilot

The alignment pipeline itself is `backend/nugen_align.py`. It records what it did
(document, benchmark, alignment, model ids, evaluation scores) in
`data/nugen_state.json`; the app reads the model id from there, or from the
NUGEN_MODEL_ID environment variable. The API key is only ever read from the
environment (NUGEN_API_KEY) and is never written to disk.

Docs: https://docs.nugen.in   Cookbook: https://github.com/nugen-in/nugen-cookbook
"""

import json
import os
import re
import time

import httpx

from engine.forecast import DATA_DIR

BASE_URL = os.environ.get("NUGEN_BASE_URL", "https://api.nugen.in")
STATE_FILE = "nugen_state.json"
TIMEOUT_S = 60


class NugenError(RuntimeError):
    """Safe to show: never contains the key."""


KEY_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".nugen_key")


def api_key():
    """From NUGEN_API_KEY, or backend/.nugen_key (owner-only file, never served: only
    frontend/ is published)."""
    key = os.environ.get("NUGEN_API_KEY")
    if key:
        return key.strip()
    try:
        with open(KEY_FILE) as f:
            return f.read().strip() or None
    except OSError:
        return None


def masked_key():
    k = api_key()
    return f"{k[:10]}\u2022\u2022\u2022\u2022{k[-2:]}" if k and len(k) > 14 else None


# ---------------------------------------------------------------- state
def state_path(data_dir=None):
    return os.path.join(data_dir or DATA_DIR, STATE_FILE)


def load_state(data_dir=None):
    try:
        with open(state_path(data_dir)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_state(state, data_dir=None):
    state = {k: v for k, v in state.items() if "key" not in k.lower()}      # belt and braces: no secrets on disk
    tmp = state_path(data_dir) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, state_path(data_dir))


def model_id(data_dir=None):
    return os.environ.get("NUGEN_MODEL_ID") or load_state(data_dir).get("model_id")


def configured(data_dir=None):
    """Inference is possible: a key, and an aligned model to call."""
    return bool(api_key() and model_id(data_dir))


# ---------------------------------------------------------------- HTTP
def _headers():
    key = api_key()
    if not key:
        raise NugenError("NUGEN_API_KEY is not set.")
    return {"Authorization": f"Bearer {key}", "accept": "application/json"}


def request(method, path, *, json_body=None, data=None, files=None, timeout=TIMEOUT_S):
    try:
        r = httpx.request(method, BASE_URL + path, headers=_headers(), json=json_body, data=data, files=files, timeout=timeout)
    except httpx.HTTPError as e:
        raise NugenError(f"Couldn't reach Nugen ({type(e).__name__}).")
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail") or r.text
        except ValueError:
            detail = r.text
        raise NugenError(f"Nugen {method} {path} failed ({r.status_code}): {str(detail)[:300]}")
    try:
        return r.json()
    except ValueError:
        return {"raw": r.text}


# ---------------------------------------------------------------- inference
BREAKER_S = 120            # after Nugen fails or times out, skip it this long instead of making users wait
_down = {"until": 0.0, "why": ""}


def available():
    return time.time() >= _down["until"]


def chat(messages, model=None, max_tokens=400, temperature=0.2, timeout=25):
    """
    One non-streaming chat completion from the aligned model.
    Returns {"text", "model", "confidence", "latency_ms"}. Gives up after `timeout` seconds,
    and for two minutes after any failure raises at once, so nothing waits on a slow Nugen.
    """
    if not available():
        raise NugenError(f"Nugen is paused after a recent failure ({_down['why']}).")
    model = model or model_id()
    if not model:
        raise NugenError("No aligned Nugen model yet. Run nugen_align.py.")
    t0 = time.time()
    try:
        body = request("POST", "/api/v3/inference/chat/completions", json_body={
            "model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature, "stream": False},
            timeout=timeout)
    except NugenError as e:
        _down.update(until=time.time() + BREAKER_S, why=str(e)[:80])
        raise
    choices = body.get("choices") or []
    msg = (choices[0].get("message") or {}) if choices else {}
    text = msg.get("content") if isinstance(msg.get("content"), str) else (choices[0].get("text", "") if choices else "")
    # Reasoning models may think out loud first; only the answer after the thinking is used.
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    text = re.sub(r"^.*</think>", "", text, flags=re.S).strip()
    if not text:
        raise NugenError("Nugen returned an empty answer.")
    return {"text": text.strip(), "model": body.get("model", model), "confidence": body.get("confidence_score"),
            "latency_ms": int((time.time() - t0) * 1000)}


# ---------------------------------------------------------------- is the key accepted?
_verify_cache = {"at": 0, "data": None}
VERIFY_CACHE_S = 300


def verify(data_dir=None, force=False):
    """
    Ask Nugen, with the configured key, what this account has: proof the key is accepted,
    the base models it can align, and where EventPulse's documents, benchmark and alignment
    stand. The last successful check is saved, so the site can still say when the key was
    last confirmed if Nugen is briefly unreachable. Nothing here is assumed or made up.
    """
    now = time.time()
    if not force and _verify_cache["data"] and now - _verify_cache["at"] < VERIFY_CACHE_S:
        return _verify_cache["data"]
    state = load_state(data_dir)
    out = {"has_key": bool(api_key()), "key": masked_key(), "checked_at": now}
    if not api_key():
        out.update(accepted=False, reason="No Nugen API key on this server.")
        return out
    try:
        base = request("GET", "/api/v3/models/base", timeout=20).get("models", [])
    except NugenError as e:
        last = state.get("verified")
        out.update(accepted=None, reason=str(e), last_verified=last)
        _verify_cache.update(at=now, data=out)
        return out
    out["accepted"] = True
    out["alignment_ready_models"] = [m["model_id"] for m in base if m.get("alignment_ready")]
    try:
        docs = request("GET", "/api/v3/documents/list", timeout=20).get("documents", [])
        mine = [d for d in docs if d.get("document_id") in (state.get("document_ids") or [])]
        out["documents"] = [{"name": d.get("document_name"), "status": d.get("status")} for d in mine]
    except NugenError:
        pass
    try:
        if state.get("benchmark_id"):
            b = request("GET", f"/api/v3/benchmarks/{state['benchmark_id']}", timeout=20)
            out["benchmark"] = {"name": b.get("benchmark_name"), "status": b.get("status"), "samples": b.get("n_samples")}
    except NugenError:
        pass
    try:
        projects = request("GET", "/api/v3/alignment-projects/list", timeout=20).get("alignment_projects", [])
        if projects:
            # The alignment that made our model, if there is one; otherwise the latest attempt.
            mine = state.get("alignment_id")
            latest = next((p for p in projects if p["alignment_id"] == mine), None) or max(projects, key=lambda p: p.get("created_at", ""))
            detail = request("GET", f"/api/v3/alignment-projects/{latest['alignment_id']}", timeout=20)
            out["alignment"] = {"id": latest["alignment_id"], "status": detail.get("status"), "error": detail.get("error"),
                                "attempts": len(projects), "base_model_id": latest.get("base_model_id"),
                                "trained": bool(detail.get("gpu_training_completed")), "model_id": detail.get("model_id")}
    except NugenError:
        pass
    # Keep the aligned model serving: Nugen undeploys it after an evaluation, so redeploy it
    # (at most every 10 minutes) and say so.
    mid = model_id(data_dir)
    if mid:
        try:
            m = request("GET", f"/api/v3/models/{mid}", timeout=20)
            out["model"] = {"id": mid, "deployment_status": m.get("deployment_status")}
            if m.get("deployment_status") == "UNDEPLOYED" and now - state.get("redeploy_at", 0) > 600:
                request("POST", f"/api/v3/models/{mid}/deployment", timeout=20)
                state["redeploy_at"] = now
                out["model"]["deployment_status"] = "DEPLOYING"
        except NugenError:
            pass
    state["verified"] = {"at": now, "key": out["key"], "alignment_ready_models": out["alignment_ready_models"]}
    save_state(state, data_dir)
    _verify_cache.update(at=now, data=out)
    return out
