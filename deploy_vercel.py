"""
Deploy EventPulse to Vercel through its REST API (no Node or Vercel CLI needed).

    frontend/        -> static pages on Vercel's CDN
    api/index.py     -> the FastAPI backend as one Python function (includes backend/)
    secrets          -> Vercel environment variables, read from the local private files
                        (backend/.supabase, backend/.mail, backend/.nugen_key); never uploaded as files

Usage:
    echo 'YOUR_VERCEL_TOKEN' > backend/.vercel_token      # vercel.com/account/tokens
    python deploy_vercel.py [--name eventpulse] [--team TEAM_ID]
"""

import argparse
import fnmatch
import hashlib
import json
import os
import sys
import time

import httpx

ROOT = os.path.dirname(os.path.abspath(__file__))
API = "https://api.vercel.com"


def token():
    t = os.environ.get("VERCEL_TOKEN")
    if not t:
        try:
            t = open(os.path.join(ROOT, "backend", ".vercel_token")).read().strip()
        except OSError:
            sys.exit("Put a Vercel token in backend/.vercel_token (vercel.com/account/tokens) or VERCEL_TOKEN.")
    return t


def kv(path):
    out = {}
    try:
        for line in open(path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    except OSError:
        pass
    return out


def env_vars():
    b = os.path.join(ROOT, "backend")
    e = {}
    s = kv(os.path.join(b, ".supabase"))
    for k in ("SUPABASE_PROJECT_REF", "SUPABASE_DB_PASSWORD", "SUPABASE_DB_HOST", "SUPABASE_DB_PORT"):
        if s.get(k):
            e[k] = s[k]
    m = kv(os.path.join(b, ".mail"))
    for k in ("GMAIL_USER", "GMAIL_APP_PASSWORD", "SMTP_FROM"):
        if m.get(k):
            e[k] = m[k]
    try:
        e["NUGEN_API_KEY"] = open(os.path.join(b, ".nugen_key")).read().strip()
    except OSError:
        pass
    e["EVENTPULSE_CLOCK_EPOCH"] = str(int(time.time()))      # one shared "now" for every instance
    e["EVENTPULSE_DEMO_CROWD"] = "1"
    return e


def ignored(rel, patterns):
    for p in patterns:
        p = p.rstrip("/")
        if fnmatch.fnmatch(rel, p) or rel.startswith(p + "/") or fnmatch.fnmatch(os.path.basename(rel), p.replace("**/", "")):
            return True
    return False


def files():
    pats = [l.strip() for l in open(os.path.join(ROOT, ".vercelignore")) if l.strip() and not l.startswith("#")]
    pats += [".git", ".DS_Store", "backend/.vercel_token", "deploy_vercel.py", ".vercel"]
    out = []
    for top in ("frontend", "backend", "api"):
        for d, dirs, fs in os.walk(os.path.join(ROOT, top)):
            dirs[:] = [x for x in dirs if x != "__pycache__"]
            for f in fs:
                full = os.path.join(d, f)
                rel = os.path.relpath(full, ROOT)
                if not ignored(rel, pats) and not f.endswith(".pyc"):
                    out.append(rel)
    out += ["vercel.json", "requirements.txt"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="eventpulse")
    ap.add_argument("--team", default=None)
    a = ap.parse_args()
    H = {"Authorization": f"Bearer {token()}"}
    q = {"teamId": a.team} if a.team else {}
    c = httpx.Client(timeout=120, headers=H)

    chk = c.get(f"{API}/v9/projects", params={**q, "limit": 1})      # team-scoped tokens can't read /v2/user
    if chk.status_code != 200:
        sys.exit(f"Vercel refused the token ({chk.status_code}): {chk.text[:200]}")

    # project
    r = c.get(f"{API}/v9/projects/{a.name}", params=q)
    if r.status_code == 404:
        r = c.post(f"{API}/v10/projects", params=q, json={"name": a.name, "framework": None})
        if r.status_code >= 300:
            sys.exit(f"Couldn't create the project: {r.text[:300]}")
        print("Created project", a.name)
    proj = r.json()
    pid = proj["id"]

    # environment variables (secrets stay encrypted on Vercel)
    ev = env_vars()
    existing = {x["key"]: x["id"] for x in c.get(f"{API}/v9/projects/{pid}/env", params=q).json().get("envs", [])}
    for k, v in ev.items():
        if k in existing:
            c.delete(f"{API}/v9/projects/{pid}/env/{existing[k]}", params=q)
        rr = c.post(f"{API}/v10/projects/{pid}/env", params=q,
                    json={"key": k, "value": v, "type": "encrypted", "target": ["production", "preview"]})
        if rr.status_code >= 300:
            print("  env", k, "failed:", rr.text[:150])
    print("Environment variables set:", ", ".join(sorted(ev)))

    # Files: describe them all, and upload only the ones Vercel doesn't have yet.
    fl = files()
    manifest, blobs = [], {}
    for rel in fl:
        data = open(os.path.join(ROOT, rel), "rb").read()
        sha = hashlib.sha1(data).hexdigest()
        manifest.append({"file": rel, "sha": sha, "size": len(data)})
        blobs[sha] = data
    body = {"name": a.name, "project": pid, "target": "production", "files": manifest,
            "projectSettings": {"framework": None, "buildCommand": None, "installCommand": None, "outputDirectory": "frontend"}}
    dq = {**q, "skipAutoDetectionConfirmation": "1"}
    r = c.post(f"{API}/v13/deployments", params=dq, json=body)
    if r.status_code == 400 and (r.json().get("error") or {}).get("code") == "missing_files":
        missing = r.json()["error"].get("missing") or []
        print(f"Uploading {len(missing)} new or changed files (of {len(fl)})")
        for i, sha in enumerate(missing):
            for attempt in range(3):
                up = c.post(f"{API}/v2/files", params=q, content=blobs[sha],
                            headers={"x-vercel-digest": sha, "Content-Type": "application/octet-stream"})
                if up.status_code < 300:
                    break
                time.sleep(2)
            else:
                sys.exit(f"Upload failed: {up.text[:200]}")
            if (i + 1) % 200 == 0:
                print(f"  uploaded {i + 1}/{len(missing)}")
        r = c.post(f"{API}/v13/deployments", params=dq, json=body)
    else:
        print(f"Vercel already has all {len(fl)} files")
    if r.status_code >= 300:
        sys.exit(f"Deployment refused: {r.text[:500]}")
    d = r.json()
    did = d["id"]
    print("Deploying", d.get("url"))
    state = None
    for _ in range(180):
        s = c.get(f"{API}/v13/deployments/{did}", params=q).json()
        if s.get("readyState") != state:
            state = s.get("readyState")
            print("  ", state)
        if state in ("READY", "ERROR", "CANCELED"):
            break
        time.sleep(5)
    if state != "READY":
        ev = c.get(f"{API}/v3/deployments/{did}/events", params={**q, "limit": 60}).json()
        for e in (ev if isinstance(ev, list) else []):
            t = (e.get("payload") or {}).get("text") or e.get("text")
            if t:
                print("   |", t[:200])
        sys.exit("Deployment failed.")
    aliases = s.get("alias") or []
    print("\nLive:", "https://" + (aliases[0] if aliases else s["url"]))


if __name__ == "__main__":
    main()
