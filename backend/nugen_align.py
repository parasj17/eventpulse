"""
Align a base model on EventPulse's domain through Nugen, and deploy it for the app.

    Base model -> Nugen alignment on the EventPulse corpus -> EventPulse domain model
    -> the app uses it for operator plans, attendee guidance and the copilot

Usage (from backend/):
    export NUGEN_API_KEY=...            # from https://platform.nugen.in (never written to disk)
    python nugen_align.py corpus        # build the corpus and benchmark (no key needed)
    python nugen_align.py models        # list base models you can align
    python nugen_align.py run           # upload, benchmark, align, wait, deploy, evaluate
    python nugen_align.py status        # where it's up to (re-run `run` to resume)
    python nugen_align.py try           # ask the aligned model about the live situation

Each step records its ids in data/nugen_state.json, so `run` resumes where it left off.
Base model: --base <id>, or the first alignment-ready model from PREFERRED.
"""

import argparse
import json
import os
import sys
import time

from engine import domain_corpus, nugen, playbook
from engine.forecast import DATA_DIR

CORPUS_DIR = os.path.join(DATA_DIR, "nugen_corpus")
PREFERRED = ["llama-v3p2-3b-instruct", "llama-v3p2-3b-reasoning", "qwen-v2p5-3b-instruct", "qwen-v2p5-1p5b-instruct",
             "qwen-v2p5-0p5b-instruct"]
POLL_S = float(os.environ.get("NUGEN_POLL_SECONDS", "15"))
ALIGN_NAME = "EventPulse crowd operations"


def say(*a):
    print(*a, flush=True)


def stage(state, name, **info):
    state.setdefault("stages", {})[name] = {"at": time.time(), **info}
    state["updated_at"] = time.time()
    nugen.save_state(state)


def cmd_corpus(_):
    import api                                                   # the live engines, for real situations
    info = domain_corpus.build(CORPUS_DIR, api, log=say)
    state = nugen.load_state()
    state["corpus"] = info
    state["benchmark_questions"] = info["benchmark_questions"]
    stage(state, "corpus", **info)
    say(f"Corpus in {CORPUS_DIR}")


def base_models():
    return nugen.request("GET", "/api/v3/models/base").get("models", [])


def cmd_models(_):
    for m in base_models():
        say(f"{m['model_id']:<34} {m.get('parameters', ''):>6}  alignment_ready={m.get('alignment_ready')}  "
            f"{'(on request)' if m.get('available_on_request') else ''}")


def pick_base(requested):
    models = [m for m in base_models() if m.get("alignment_ready") and m.get("type", "llm") == "llm"]
    ids = [m["model_id"] for m in models]
    if requested:
        if requested not in ids:
            raise SystemExit(f"{requested} is not an alignment-ready base model here. Try: {', '.join(ids)}")
        return next(m for m in models if m["model_id"] == requested)
    for p in PREFERRED:
        if p in ids:
            return next(m for m in models if m["model_id"] == p)
    if not models:
        raise SystemExit("No alignment-ready base models on this account.")
    return models[0]


def wait(label, fn, done, failed=("FAILED",), timeout_s=None, stop=()):
    t0 = time.time()
    last = None
    while True:
        r = fn()
        s = str(r.get("status") or r.get("deployment_status") or "").upper()
        if s != last:
            prog = r.get("progress")
            say(f"  {label}: {s}" + (f" ({prog}%)" if prog is not None else ""))
            last = s
        if s in done or s in stop:
            return r
        if s in failed:
            raise SystemExit(f"{label} failed: {json.dumps(r)[:400]}")
        if timeout_s and time.time() - t0 > timeout_s:
            raise SystemExit(f"{label} is still {s} after {int(timeout_s)} s. Run again later to resume.")
        time.sleep(POLL_S)


def cmd_run(args):
    state = nugen.load_state()
    if args.fresh:
        # A different Nugen account can't see the old one's uploads: start again, keep the corpus.
        keep = {k: state[k] for k in ("corpus", "benchmark_questions") if k in state}
        old = state.get("failed_alignments", []) + ([{"alignment_id": state["alignment_id"]}] if state.get("alignment_id") else [])
        state = {**keep, "previous_attempts": old, "stages": {k: v for k, v in state.get("stages", {}).items() if k == "corpus"}}
        nugen.save_state(state)
        say("Starting fresh: uploads and alignment will be redone on this key's account.")
    if "corpus" not in state.get("stages", {}) or not os.path.exists(os.path.join(CORPUS_DIR, "cases.txt")):
        cmd_corpus(args)
        state = nugen.load_state()

    # 1. Upload the domain documents.
    if not state.get("document_ids"):
        say("Uploading the EventPulse corpus...")
        files = []
        for name in ("handbook.txt", "cases.txt"):
            files.append(("files", (name, open(os.path.join(CORPUS_DIR, name), "rb"), "text/plain")))
        r = nugen.request("POST", "/api/v3/documents/create", files=files,
                          data={"categories": ["eventpulse-operations", "eventpulse-operations"],
                                "names": ["EventPulse operations handbook", "EventPulse worked cases"]}, timeout=300)
        state["document_ids"] = r["document_ids"]
        stage(state, "documents_uploaded", document_ids=r["document_ids"])
    for d in state["document_ids"]:
        wait(f"document {d}", lambda d=d: nugen.request("GET", f"/api/v3/documents/{d}/status"), {"READY"}, timeout_s=1800)
    stage(state, "documents_ready")

    # 2. Upload the benchmark it will be judged on, before and after.
    if not state.get("benchmark_id"):
        say("Uploading the EventPulse benchmark...")
        with open(os.path.join(CORPUS_DIR, "benchmark.json"), "rb") as fh:
            r = nugen.request("POST", "/api/v3/benchmarks/upload",
                              files={"file": ("eventpulse_benchmark.json", fh, "application/json")},
                              data={"name": "EventPulse operations benchmark", "document_id": state["document_ids"][-1],
                                    "description": "Held-out EventPulse situations and handbook questions."}, timeout=120)
        state["benchmark_id"] = r["benchmark_id"]
        stage(state, "benchmark_uploaded", benchmark_id=r["benchmark_id"])
    wait("benchmark", lambda: nugen.request("GET", f"/api/v3/benchmarks/{state['benchmark_id']}/status"),
         {"READY", "EVALUATED", "COMPLETED"}, timeout_s=1800)

    # 3. Align the base model on the corpus. Nugen's training service can fail to start a
    # job (it answered 502 once); a failed attempt is recorded and a fresh one started.
    attempts = 0
    while True:
        if not state.get("alignment_id"):
            base = pick_base(args.base)
            say(f"Aligning {base['model_id']} on the EventPulse corpus...")
            r = nugen.request("POST", "/api/v3/alignment-projects/create", json_body={
                "alignment_name": ALIGN_NAME, "base_model_id": base["model_id"], "document_ids": state["document_ids"],
                "benchmark_id": state["benchmark_id"],
                "description": "EventPulse: turn a live crowd, transport, venue and weather situation into an operator "
                               "action plan and a one-line attendee message, using only the facts given."})
            state.update(alignment_id=r["alignment_id"], alignment_name=ALIGN_NAME, base_model_id=base["model_id"],
                         base_model_name=base.get("model_name"))
            stage(state, "alignment_started", alignment_id=r["alignment_id"], base_model_id=base["model_id"])
        r = wait("alignment", lambda: nugen.request("GET", f"/api/v3/alignment-projects/{state['alignment_id']}/status"),
                 {"READY", "EVALUATED", "DEPLOYING", "EVALUATING", "UNDEPLOYED"}, failed=(), timeout_s=args.timeout,
                 stop=("FAILED", "STOPPED"))
        if r.get("status") not in ("FAILED", "STOPPED"):
            break
        detail = nugen.request("GET", f"/api/v3/alignment-projects/{state['alignment_id']}")
        say(f"  Nugen reports: {detail.get('error')} {detail.get('stage_failures') or ''}")
        state.setdefault("failed_alignments", []).append(
            {"alignment_id": state["alignment_id"], "error": detail.get("error"), "stages": detail.get("stage_failures")})
        state.pop("alignment_id", None)
        nugen.save_state(state)
        attempts += 1
        if attempts >= args.retries:
            raise SystemExit(f"Alignment failed {attempts} times on Nugen's side. Run again later to retry.")
        say(f"  Retrying in {int(args.retry_wait)} s (attempt {attempts + 1} of {args.retries})...")
        time.sleep(args.retry_wait)
    state["alignment_status"] = r.get("status")
    stage(state, "alignment_done", status=r.get("status"))

    # 4. Find the aligned model.
    if not state.get("model_id"):
        models = nugen.request("GET", "/api/v3/models/aligned").get("domain_aligned_models", [])
        mine = [m for m in models if m.get("alignment_id") == state["alignment_id"]]
        if not mine:
            raise SystemExit("Alignment finished but no aligned model is listed yet. Run again in a minute.")
        state["model_id"] = mine[0]["model_id"]
        stage(state, "model_created", model_id=state["model_id"])
    say(f"Aligned model: {state['model_id']}")

    # 5. Deploy it for inference.
    m = nugen.request("GET", f"/api/v3/models/{state['model_id']}")
    if m.get("deployment_status") != "DEPLOYED":
        if m.get("deployment_status") not in ("DEPLOYING",):
            say("Deploying...")
            nugen.request("POST", f"/api/v3/models/{state['model_id']}/deployment")
        wait("deployment", lambda: nugen.request("GET", f"/api/v3/models/{state['model_id']}"), {"DEPLOYED"},
             failed=(), timeout_s=args.timeout)
    state["deployment_status"] = "DEPLOYED"
    stage(state, "deployed")

    # 6. Evaluation: how it scored on the benchmark before and after alignment.
    m = nugen.request("GET", f"/api/v3/models/{state['model_id']}")
    state["performance_metrics"] = m.get("performance_metrics")
    ev = (m.get("evaluation_data") or {})
    if ev.get("evaluation_id"):
        try:
            state["evaluation"] = nugen.request("GET", f"/api/v3/evaluations/{ev['evaluation_id']}/results")
        except nugen.NugenError as e:
            state["evaluation"] = {"note": str(e)}
    stage(state, "evaluated")
    say("Performance:", json.dumps(state.get("performance_metrics"), indent=1))
    say("Done. The app now uses this model (restart the API if it's running).")


def cmd_status(_):
    s = nugen.load_state()
    say(json.dumps({k: v for k, v in s.items() if k != "evaluation"}, indent=1, default=str))
    if s.get("alignment_id") and nugen.api_key():
        say(json.dumps(nugen.request("GET", f"/api/v3/alignment-projects/{s['alignment_id']}/status"), indent=1))


def cmd_try(args):
    import api
    facts = api._plan_facts(args.hour)
    text = playbook.situation(facts)
    say("SITUATION\n" + text + "\n")
    r = nugen.chat([{"role": "system", "content": playbook.SYSTEM_PROMPT}, {"role": "user", "content": text}])
    ok, why = playbook.check(r["text"], text)
    say(f"ALIGNED MODEL ({r['model']}, confidence {r['confidence']}, {r['latency_ms']} ms)\n{r['text']}\n")
    say("Passes the checks." if ok else f"Would be replaced by the playbook: {why}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["corpus", "models", "run", "status", "try"])
    p.add_argument("--base", help="base model id to align")
    p.add_argument("--timeout", type=float, default=None, help="give up waiting after this many seconds (resume later)")
    p.add_argument("--hour", type=int, default=104, help="hour for `try`")
    p.add_argument("--retries", type=int, default=3, help="alignment attempts before giving up")
    p.add_argument("--fresh", action="store_true", help="forget earlier uploads and alignments (e.g. after switching to another account's key)")
    p.add_argument("--retry-wait", type=float, default=90, help="seconds between alignment attempts")
    args = p.parse_args()
    if args.command not in ("corpus",) and not nugen.api_key():
        sys.exit("Set NUGEN_API_KEY first (from https://platform.nugen.in).")
    try:
        {"corpus": cmd_corpus, "models": cmd_models, "run": cmd_run, "status": cmd_status, "try": cmd_try}[args.command](args)
    except nugen.NugenError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
