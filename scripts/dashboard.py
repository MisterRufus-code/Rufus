#!/usr/bin/env python3
"""
dashboard.py — Rufus's approval queue and status dashboard: one page
answering "what's actually happening with this pipeline," without scrolling
PowerShell logs — and the ONLY place a video can be approved for upload.

main.py no longer auto-uploads anything (RUFUS_AUTO_UPLOAD=1 is the explicit
opt-out back to the old behavior). Every rendered video lands here as
'pending'; a reviewer (you, or whoever you've shared this with) clicks
Approve to actually publish it, or Reject to never publish it. Title/
description are editable before that click.

Reads/writes rufus.db (WAL mode — safe alongside a live main.py run) and
reads media_library/debug/<run_id>/ for the per-run script/voiceover/
keyframes. No external assets (self-contained HTML/CSS/inline SVG — nothing
to break when you're not on the local network). No login of its own.

DEFAULT IS LOOPBACK-ONLY (127.0.0.1) — not reachable from your home WiFi at
all, only from this PC. This page also has "/system" routes that can START
AND KILL PROCESSES on this machine (launch a run, stop ComfyUI), so it needs
more than "no login" once those exist — loopback-only means literally
nothing on the network can reach ANY route here, including those.

Run:
    python scripts\\dashboard.py
    → http://localhost:8765            (this machine only, by default)

For access from your phone or another device, do NOT change
RUFUS_DASHBOARD_HOST to 0.0.0.0 and do NOT port-forward — either one puts
the process-control routes on your open WiFi/the internet with no login.
Use Tailscale instead (free, ~2 min):
    1. Install Tailscale on this PC and on your phone, sign into the same account.
    2. On this PC:  tailscale serve --bg 8765
       (proxies http://127.0.0.1:8765 onto your private tailnet over https,
       auto-renewed cert, only devices signed into YOUR tailnet can reach it —
       the dashboard itself never has to leave loopback.)
    3. On your phone (with Tailscale connected): open the https URL
       `tailscale serve status` prints.
    `tailscale serve --bg off` to stop sharing it again.

Environment:
  RUFUS_DASHBOARD_HOST   127.0.0.1 (default — loopback-only; 0.0.0.0 opens
                         it to your whole LAN, not recommended once
                         process-control routes exist — use tailscale serve
                         instead)
  RUFUS_DASHBOARD_PORT   8765
"""

import html
import json
import os
import re
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import console
console.force_utf8()   # the dashboard prints ✓/✗ too — see console.py
import paths
from urllib.parse import quote as _urlquote

import requests
from filelock import FileLock, Timeout
from flask import Flask, abort, g, make_response, redirect, request, send_from_directory

sys.path.insert(0, str(Path(__file__).parent))
import auth
import db_manager
import run_progress

# When this dashboard process started — reported as uptime by /api/status, so
# a phone can tell "the box has been up for 3 days" from "it just rebooted".
_STARTED_AT = time.time()

ROOT       = Path(__file__).parent.parent
DEBUG_ROOT = paths.debug_root()

app = Flask(__name__)

UPLOAD_THRESHOLD_DEFAULT = 8   # visual reference line on the score sparkline

# Hard floor enforced in approve_video() below — no video below this score
# can be approved for upload, even by a human clicking the button, whether
# it's a misclick or a reviewer other than you (Tailscale-shared access).
# Keep in sync with main.py's HARD_MIN_UPLOAD_SCORE (duplicated, not
# imported, so this file has zero import-time dependency on main.py).
HARD_MIN_UPLOAD_SCORE = 7


# ── Process control (/system) — status, launch, cancel ───────────────────────
# Loopback-only binding (see module docstring) already keeps these off the
# network by default; _require_localhost() is defense-in-depth against
# someone later widening RUFUS_DASHBOARD_HOST without re-reading why.

def _require_localhost() -> None:
    """Legacy guard, kept ONLY for when auth is off.

    Once config/users.json exists, loopback proves nothing: `tailscale serve`
    terminates TLS and proxies to 127.0.0.1, so every tailnet visitor arrives
    as loopback and would sail through this. In that mode the routes rely on
    auth.require() instead, and this becomes a no-op.
    """
    if auth.auth_enabled():
        return
    if request.remote_addr not in ("127.0.0.1", "::1"):
        abort(403)


# Routes reachable without being signed in: the login page itself, and a
# liveness probe the watchdog hits (it has no token and must not be a 401).
# google_login_start/callback must be reachable before anyone is signed in —
# that's the entire point of a login route.
PUBLIC_ENDPOINTS = {"login", "healthz", "static",
                   "google_login_start", "google_login_callback"}


@app.before_request
def _authenticate():
    """Resolve identity once per request and refuse anonymous traffic.

    Runs before every route so no handler can forget. Stores the user on
    Flask's `g` so _head() and the permission helpers don't re-read the file
    per call.
    """
    g.rufus_user = auth.current_user()
    if request.endpoint in PUBLIC_ENDPOINTS:
        return None
    if g.rufus_user is None:
        # A browser gets the sign-in page; anything scripted gets a clean 401.
        if "text/html" in (request.headers.get("Accept") or ""):
            return redirect("/login")
        abort(401)
    return None


@app.after_request
def _persist_token(response):
    """Turn a ?token=… sign-in link into a cookie, once.

    The partner opens one link on their phone and stays signed in; the token
    stops trailing in every subsequent URL (and out of browser history and
    any screenshot they take). HttpOnly so page scripts can't read it,
    SameSite=Strict so another site can't drive a POST here with their
    cookie attached — this dashboard's mutating routes are plain HTML forms
    with no CSRF token of their own, and Strict is what closes that.
    """
    token = request.args.get("token", "").strip()
    if token and auth.auth_enabled() and auth.user_for_token(token):
        response.set_cookie(auth.COOKIE_NAME, token, max_age=auth.COOKIE_MAX_AGE,
                            httponly=True, samesite="Strict",
                            secure=request.headers.get("X-Forwarded-Proto") == "https")
    return response


@app.route("/login")
def login():
    """Explains how to get in. Deliberately gives nothing away — no user list,
    no hint about whether a token was close, since this page is reachable by
    anyone who can route to the dashboard."""
    if not auth.auth_enabled():
        return redirect("/")
    if getattr(g, "rufus_user", None):
        return redirect("/")

    error = request.args.get("error", "")
    error_html = f'<div class="msg error">{_esc(error)}</div>' if error else ""

    google_html = ""
    if auth.google_oauth_enabled():
        google_html = """
        <a class="btn save" style="text-decoration:none;display:inline-block;margin-bottom:6px"
           href="/auth/google/start">Sign in with Google</a>
        <p class="muted" style="margin:4px 0 18px">Only works if the owner already
           added your Google account. — or use a token link instead —</p>
        """

    body = f"""
    <h2 style="margin-top:14px">Sign in</h2>
    {error_html}
    {google_html}
    <p class="muted">This dashboard needs a personal sign-in link. Ask the
       channel owner for yours — it looks like
       <code>https://…/?token=…</code> and only has to be opened once per
       device.</p>
    <form method="get" action="/">
      <label for="token">Access token</label>
      <input class="field" type="password" id="token" name="token"
             autocomplete="current-password" placeholder="paste your token">
      <button class="btn save" type="submit">Sign in</button>
    </form>
    """
    return PAGE_STYLE + '<header><h1>🎬 ThePaperTrails</h1></header>\n<main>\n' \
        + body + PAGE_TAIL, 401


@app.route("/auth/google/start")
def google_login_start():
    """Kick off the redirect-to-Google leg. 404s (not a plain error page)
    when Google sign-in isn't configured — the route simply doesn't exist in
    that setup, same as any other feature-flagged path."""
    if not auth.google_oauth_enabled():
        abort(404)
    try:
        flow = auth.build_google_flow()
    except auth.AuthError as e:
        return redirect(f"/login?error={_urlquote(str(e))}")
    state = auth.new_oauth_state()
    auth_url, _ = flow.authorization_url(
        access_type="online", include_granted_scopes="true",
        state=state, prompt="select_account")
    return redirect(auth_url)


@app.route("/auth/google/callback")
def google_login_callback():
    """Where Google sends the browser back. Every failure path redirects to
    /login with a reason rather than raising — a stack trace here would be
    shown to someone mid sign-in attempt, not a developer."""
    if not auth.google_oauth_enabled():
        abort(404)

    if request.args.get("error"):
        return redirect("/login?error=" + _urlquote(
            "Google sign-in was cancelled or denied."))

    if not auth.consume_oauth_state(request.args.get("state", "")):
        return redirect("/login?error=" + _urlquote(
            "That sign-in attempt expired or was already used — try again."))

    code = request.args.get("code", "")
    if not code:
        return redirect("/login?error=" + _urlquote(
            "Google did not return an authorization code."))

    try:
        flow = auth.build_google_flow()
        flow.fetch_token(code=code)
        claims = auth.verify_google_id_token(flow.credentials.id_token)
    except Exception as e:
        return redirect("/login?error=" + _urlquote(f"Google sign-in failed: {e}"))

    if not claims.get("email_verified", False):
        return redirect("/login?error=" + _urlquote(
            "That Google account has no verified email."))

    email = claims.get("email", "")
    user = auth.find_user_by_email(email)
    if user is None:
        # Deliberately does NOT create an account — an unrecognized Google
        # identity gets refused exactly like a wrong token. Google vouches for
        # WHO they are; it never decides WHETHER they're allowed in.
        return redirect("/login?error=" + _urlquote(
            f"{email} is not on the access list — ask the owner to add it "
            f"in Settings → Users."))

    resp = make_response(redirect("/"))
    resp.set_cookie(auth.COOKIE_NAME, user["token"], max_age=auth.COOKIE_MAX_AGE,
                    httponly=True, samesite="Strict",
                    secure=request.headers.get("X-Forwarded-Proto") == "https")
    return resp


@app.route("/logout")
def logout():
    resp = make_response(redirect("/login"))
    resp.delete_cookie(auth.COOKIE_NAME)
    return resp


@app.route("/healthz")
def healthz():
    """Unauthenticated liveness probe for the watchdog. Says nothing about the
    pipeline — just that this process is answering."""
    return {"ok": True}, 200


def _current_job() -> dict | None:
    """The open project's stage, how far in, and how long is left.

    WHY NOT THE RUN PROGRESS ALREADY IN THIS PAYLOAD. That tracks main.py's
    eight steps and is written by a full pipeline run. Drawing a gallery from
    the wizard is not one of those, so while thirty-two pictures were being
    rendered the bar said "Idle — not making a video" and the owner, watching
    ComfyUI churn in another window, quite reasonably asked what was going on.
    """
    try:
        import db_manager as dbm
        for p in dbm.projects(status="open", limit=5):
            prog = _project_progress(p)
            if not prog.get("working"):
                continue
            return {
                "title": p.get("title") or "",
                "stage": prog.get("stage"),
                "done": prog.get("done", 0),
                "total": prog.get("total", 0),
                "label": prog.get("label", ""),
                "eta_seconds": prog.get("eta_seconds"),
                "project": p["id"],
            }
    except Exception:
        pass
    return None


# Logs that are about the dashboard rather than about a run. Named rather
# than pattern-matched: a new run kind should appear in the bar automatically,
# and only these two are known not to belong.
_NOT_A_RUN_LOG = {"dashboard.log", "watchdog.log"}

# Longer than any single picture takes and shorter than a person's patience.
# One draw is ~19s on the owner's 3090; four minutes of silence on an
# unfinished set is not a slow picture, it is a dead renderer.
STALLED_AFTER_SECONDS = 240

_GPU_CACHE = {"at": 0.0, "value": None}


def _gpu_temp() -> int | None:
    """The GPU's temperature in Celsius, or None if it cannot be read.

    nvidia-smi, because it ships with the driver and needs no admin rights.
    CPU temperature is deliberately NOT here: on Windows it needs WMI or
    LibreHardwareMonitor and usually an elevated process, and a field that is
    blank forever is worse than a field that does not exist.

    Cached for ten seconds. This is polled by every open tab on a five-second
    timer, and spawning a process per tab per poll to read one number is a
    cost with nothing to show for it.
    """
    import time as _t
    if _t.time() - _GPU_CACHE["at"] < 10:
        return _GPU_CACHE["value"]
    val = None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=4)
        line = (out.stdout or "").strip().splitlines()
        if line:
            val = int(line[0].strip())
    except Exception:
        val = None
    _GPU_CACHE.update(at=_t.time(), value=val)
    return val


def _newest_log_lines(n: int = 3) -> list[str]:
    """The last few lines of whichever log was written to most recently.

    THE OWNER ASKED FOR THIS BY NAME: "if it possible see few lines of terminal
    of comfy/the run". This is the closest honest version — the dashboard
    cannot read ComfyUI's console, which belongs to another process it did not
    start, but every run it launches writes its own log, and that log says the
    same things in the pipeline's own words.

    Newest file by modification time, so it follows whatever is actually
    running without being told which stage that is.
    """
    try:
        d = paths.log_dir()
        # NOT dashboard.log. The dashboard redirects its own stdout there
        # and Flask writes an access line for every poll — including the poll
        # that draws this bar — so it is always the most recently modified file
        # in the directory and always won. The owner watched his status bar
        # report `GET /api/status HTTP/1.1 200` back at him while a gallery was
        # rendering. What belongs here is what the RUN is saying.
        files = [f for f in d.glob("*.log")
                 if f.is_file() and f.name not in _NOT_A_RUN_LOG]
        if not files:
            return []
        newest = max(files, key=lambda f: f.stat().st_mtime)
        # Only if it is recent. A three-day-old log presented as live status is
        # a lie the bar tells at a glance.
        if time.time() - newest.stat().st_mtime > 900:
            return []
        tail = newest.read_text(encoding="utf-8", errors="replace").splitlines()
        return [ln.rstrip() for ln in tail[-n:] if ln.strip()]
    except Exception:
        return []


@app.route("/api/status")
def api_status():
    """Live machine + pipeline state, polled by the status bar on every page.

    Answers the three things you actually want to know from a phone: is the PC
    up (you got a reply at all), is it making a video right now (and how far
    in), and what's waiting for you. Authenticated like everything else — run
    topics and niches are not public.

    Cheap by design, because it's polled: two file-existence checks, one
    3-second ComfyUI probe, and one small aggregate query.
    """
    auth.require("view")

    channels = _channels() or ["default"]
    runs = []
    for cid in channels:
        prog = run_progress.read(cid) or {}
        # The lock is authoritative for "is something running" — a progress
        # file can be stale or missing (a run started before this feature
        # existed), but the lock is held by the live process itself.
        running = _run_in_progress(cid)
        runs.append({
            "channel": cid,
            "running": running,
            "step": prog.get("step", 0),
            "total": prog.get("total", run_progress.TOTAL_STEPS),
            "label": prog.get("label", ""),
            "niche": prog.get("niche", ""),
            "topic": prog.get("topic", ""),
            "elapsed_seconds": int(prog.get("elapsed_seconds", 0)),
            "age_seconds": int(prog.get("age_seconds", 0)),
            # "Lock held but the progress file went quiet" = the run almost
            # certainly died without releasing. Worth surfacing, not hiding.
            "stale": bool(running and prog.get("stale")),
            "status": prog.get("status", ""),
            "detail": prog.get("detail", ""),
        })

    stats = _stats(limit=200)
    return {
        "ok": True,
        "server_time": time.time(),
        # The PC is obviously on if this reply arrived; what's worth reporting
        # is how long it's been up and whether the GPU service is actually
        # available, which is what decides if a run would even work now.
        "uptime_seconds": int(time.time() - _STARTED_AT),
        "comfyui": _comfyui_reachable(),
        # Empty when this is the venv interpreter. Reported rather than only
        # printed at startup, because the startup line scrolls away in a log
        # nobody opens and this is the fact that explains a dashboard missing
        # packages it should have.
        "interpreter_warning": _wrong_interpreter(),
        "busy": any(r["running"] for r in runs),
        "gpu_temp_c": _gpu_temp(),
        # What the machine is saying right now, in its own words.
        "log_tail": _newest_log_lines(),
        # The stage a person is actually waiting on, with its own count and
        # estimate — the run-progress numbers above describe main.py's eight
        # steps, which say nothing while a gallery is drawing.
        "job": _current_job(),
        "runs": runs,
        "queue": {
            "pending": stats.get("pending", 0),
            "approved": stats.get("uploaded", 0),
            "rejected": stats.get("rejected", 0),
        },
    }, 200


def _comfy_host() -> str:
    # Duplicated from comfy_client._host() rather than imported — same
    # reasoning as HARD_MIN_UPLOAD_SCORE above: this file stays free of an
    # import-time dependency on the ComfyUI client module.
    return os.environ.get("COMFY_HOST", "http://localhost:8188").rstrip("/")


def _comfyui_reachable() -> bool:
    try:
        r = requests.get(f"{_comfy_host()}/system_stats", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _lock_path(channel_id: str) -> Path:
    # Exact naming main.py's _acquire_lock() uses — same lock, checked
    # non-blockingly instead of held.
    return ROOT / f"rufus.{channel_id}.lock.lock"


def _run_in_progress(channel_id: str) -> bool:
    """True if main.py (any entry point, anywhere) currently holds this
    channel's run lock. Works regardless of who started the run — a manual
    run.bat, Task Scheduler, or this dashboard — because it checks the same
    FileLock main.py itself uses, non-blockingly."""
    lock = FileLock(str(_lock_path(channel_id)))
    try:
        lock.acquire(timeout=0)
    except Timeout:
        return True
    except Exception:
        return False   # fail open — never let a lock-check bug block the page
    lock.release()
    return False


# ── Settings editor (/settings) ──────────────────────────────────────────────
# Most Rufus tunables are read via os.environ.get(...) in the process that's
# actually running — this dashboard editing a value can't retroactively
# change an already-running process, and won't reach a Task-Scheduler-
# launched run_scheduled.bat unless that file is separately updated to
# source this same JSON (documented in the /settings page itself, not done
# automatically here). What it DOES do immediately: every run this
# dashboard launches (_launch_run below — /system/run, /request-topic,
# /trending's queue button) picks these up as env overrides.

SETTINGS_FILE = ROOT / "config" / "dashboard_settings.json"

# (env var, label, kind, help). kind: "bool" (tri-state — blank means "don't
# override, use whatever's already configured") or "select:opt1,opt2,...".
# EVERY KNOB THE OWNER ACTUALLY TURNS, in one form.
#
# WHY THIS GREW. The schema held six switches, so running a video the way this
# channel now runs it meant seven `$env:` lines in PowerShell before every
# single run — and getting one of them wrong (a cmd `set` in a PowerShell
# prompt, a stale RUFUS_BEAT_MOTION from an earlier experiment) is invisible
# until the video comes out wrong. The owner's instruction was plain: the
# software is run from the dashboard, not from a terminal. So anything worth
# setting is here, grouped by the decision it belongs to.
#
# `kind` is one of:
#   bool               on / off / (default)
#   select:a,b,c       a fixed set
#   text               free text
#   secret             free text, rendered masked (webhooks, tokens)
#   number             free text, validated as a number on save
SETTINGS_GROUPS = [
    ("Look", "How the pictures are drawn.", [
        # EVERY PRESET IN THE FILE, not the three that existed when this list
        # was written. A look the picker cannot offer is a look nobody uses —
        # the tests below build the same list from config/styles.json, so the
        # two cannot drift again.
        ("RUFUS_STYLE", "Style preset",
         "select:stickman,stickman_lean,stickman_micro,ink_explainer,flat_vector,ink_woodcut,"
         "paper_cut,chalkboard,retro_print,storybook,thumbnail",
         "Named look from config/styles.json, appended to every image prompt "
         "byte for byte. Leave at (default) to use the niche's own style_suffix. "
         "Render one of each on the Style page before choosing."),
        ("RUFUS_STILLS_DETAIL", "Style override (literal)", "text",
         "A style block written out in full. Beats the preset above — for a "
         "one-off experiment that does not deserve a name yet."),
        ("RUFUS_SHOT_LAST", "Shot description last", "bool",
         "Put the style block first and this shot's own sentence at the END of "
         "the prompt. A probe showed the shot losing to the style: asked for a "
         "leopard drawn in full it returned a rock and a tail, and asked for a "
         "close-up with raised brows it returned a wide shot with the brows "
         "down — while the same prompt with no style block drew both exactly "
         "as asked. The shot is ~130 characters against the block's ~3,000-5,000. "
         "This changes the order and nothing else. Probe it before a real run."),
        ("RUFUS_STILLS_ONLY", "Stills only", "bool",
         "Force images-only, overriding Wan/Hunyuan/LTX/SVD all at once. On "
         "this hardware a motion clip is minutes; a still is seconds."),
    ]),
    ("Pictures per video", "How many, and how they move.", [
        ("SD_CLIPS", "Beats (pictures)", "number",
         "One storyboard shot, one prompt and one cut each. Left empty it is "
         "computed from the script — about one per five spoken words, floor "
         "10, ceiling 30."),
        ("RUFUS_BEAT_MOTION", "How a beat moves", "select:cut,kenburns,i2i,hero,i2v",
         "cut = several stills hard-cut inside the beat (what stills-only "
         "picks on its own). kenburns = one still, zoom only. i2v = a real "
         "motion model per beat, measured in minutes each on this box."),
        ("RUFUS_FRAMES_PER_BEAT", "Stills per beat", "number",
         "Only in cut/i2i mode. Six is the most the progression arc has steps "
         "for; beyond that raise the beat count instead."),
        ("RUFUS_FRAME_GATE", "Re-roll bad frames", "bool",
         "Render a picture again when it comes back as a grid of panels or as "
         "a subject on blank paper, the way you would by hand. Costs a "
         "re-render each time it fires; up to two per picture."),
        ("RUFUS_VISION_GATE", "Ask a vision model too", "bool",
         "With the above: show every frame to a local vision model and re-roll "
         "the ones that do not match their prompt or that came back with "
         "lettering in them. Seconds per frame, and it wants the card ComfyUI "
         "is using."),
    ]),
    ("Word-synced inserts", "Small pictures that pop in on the word.", [
        ("RUFUS_INSERTS", "Insert layer", "bool",
         "Off by default in a run that already has many beats — inserts are a "
         "second set of pictures on top of them."),
        ("RUFUS_INSERT_MODE", "Insert mode", "select:nouns,phrases",
         "nouns = a picture per drawable noun, capped by the script's "
         "vocabulary. phrases = a picture per clause, tiling the narration."),
        ("RUFUS_INSERT_MAX", "Most inserts", "number",
         "Ceiling for one video. In noun mode the script's vocabulary is "
         "usually the real limit, so raising this alone changes nothing."),
        ("RUFUS_PHRASE_WORDS", "Words per insert", "number",
         "Phrase mode only. Four is about a second and a half of narration."),
    ]),
    ("Sound", "The synthesized layer. Every gain is 0-1.", [
        ("RUFUS_SFX", "Sound effects", "bool",
         "Off drops the whole layer — hit, bubble and riser together."),
        ("RUFUS_BUBBLE_GAIN", "Bubble (every cut)", "number",
         "Plays on every cut, so it compounds fast. 0.05 is felt without "
         "stepping on the narration."),
        ("RUFUS_HIT_GAIN", "Hit (on the hook)", "number",
         "The sub-bass punch under the first line. Plays once, 0.03s in."),
        ("RUFUS_RISER_GAIN", "Riser (into the payoff)", "number",
         "The swell leading into the final beat. Plays once, and its weight "
         "follows the tone of the beat it introduces."),
        ("RUFUS_TTS", "Voice engine", "select:elevenlabs,kokoro,edge",
         "ElevenLabs needs a cloned voice on a free account; Kokoro is local "
         "and free and is what this channel has been using."),
    ]),
    ("Where it goes", "Notifications and publishing.", [
        ("RUFUS_DISCORD_WEBHOOK", "Discord webhook", "secret",
         "Server Settings → Integrations → Webhooks → New Webhook → Copy URL. "
         "Every finished video is posted there with its score and hold reason, "
         "and the mp4 itself when it fits under Discord's attachment limit."),
        ("RUFUS_DISCORD_UPLOAD", "Attach the mp4", "bool",
         "Off posts a link only. On is the default and is what makes the "
         "phone useful — the video plays in the channel."),
        ("RUFUS_DASHBOARD_URL", "Dashboard URL", "text",
         "The ADDRESS ONLY — https://host.tailnet.ts.net, no ?token= on the "
         "end. Every Discord and ntfy alert deep-links to the video's own "
         "page with it, so a token pasted here gets posted into that chat "
         "channel several times a day. Anything after ? is stripped and "
         "warned about for that reason. Sign-in links come from "
         "scripts/auth.py, not from here."),
        ("RUFUS_NTFY_TOPIC", "ntfy topic", "text",
         "Free phone push, no account: install the ntfy app and subscribe to "
         "a topic nobody else would guess."),
        ("RUFUS_PRIVACY", "When it goes live", "select:public,private,unlisted",
         "public puts an approved video up immediately. private schedules it "
         "for the next peak hour instead — YouTube publishes it itself, and "
         "the Tracking page lists what is waiting. Scheduling is only "
         "possible on a private upload; that is YouTube's rule. On Windows it "
         "also needs the tzdata package, without which there is no schedule "
         "and the video stays private until you publish it by hand."),
        ("RUFUS_MIN_UPLOAD_SCORE", "Hold below score", "number",
         "Nothing auto-uploads regardless; this decides what the review queue "
         "flags as held."),
    ]),
    ("Engines", "Which models may run at all.", [
        ("RUFUS_RENDERER", "Renderer", "select:ffmpeg,remotion",
         "remotion needs `cd remotion && npm install` once; any failure falls "
         "back to ffmpeg, so a render always completes."),
        ("RUFUS_WAN", "Wan 2.2", "bool",
         "Text-to-video and image-to-video. A mixture-of-experts model whose "
         "weights stream from disk on 16GB of RAM — minutes per clip here."),
        ("RUFUS_HUNYUAN", "Hunyuan 1.5", "bool",
         "Image-to-video. Its template is exported and its models load, so "
         "this is one switch away from producing motion."),
        ("RUFUS_LTX", "LTX 2.3", "bool",
         "Image-to-video, the fastest of the three and the least faithful to "
         "the still it was given."),
        ("RUFUS_T2V", "Wan text-to-video", "bool",
         "Only ever renders the hero beat, and only in hero mode."),
        ("RUFUS_CHARACTER_MODE", "Recurring character", "bool",
         "Global switch for character_engine.py. No niche currently has one "
         "configured, so this does nothing until config/niches.json does."),
    ]),
    ("Writing", "The script, and what may reach the queue.", [
        ("RUFUS_STORYBOARD", "Storyboard", "bool",
         "Off falls back to per-beat prompts written without seeing the story "
         "— which is what produced ten unrelated pictures."),
        ("RUFUS_STORYBOARD_REPAIR", "Re-plan repeated shots", "bool",
         "After the shots are planned, measures whether one object is the "
         "subject of more than half of them and asks for new shots for the "
         "surplus — the \"why is everything coins\" fix. One extra model call, "
         "only on runs that need it."),
        ("RUFUS_SCRIPT_ARCHITECT", "Story architect", "bool",
         "The plan pass that finds a filmable moment before any prose is "
         "written."),
        ("RUFUS_SUPERVISOR", "Supervisors", "bool",
         "The seed and fact gates. Off means nothing is held for review."),
        ("RUFUS_SEED_TRIES", "Seed attempts", "number",
         "How many sources to try before accepting one the supervisor "
         "rejected. Default 4."),
        ("RUFUS_VISION", "Look at the pictures", "bool",
         "After a run, a vision model opens every keyframe and checks it "
         "against the prompt that made it: does it show what was asked, is "
         "there garbled lettering, is the face the one the storyboard "
         "specified. The only check that sees pixels — everything else here "
         "reads text. Costs seconds per frame, so it is off by default."),
        ("RUFUS_VISION_MODEL", "Vision model", "text",
         "qwen2.5vl:7b on your own 3090 through Ollama, or gpt-4o-mini in the "
         "cloud. Needs RUFUS_LLM_BASE set for a local one."),
        ("RUFUS_LLM_BASE", "Local model endpoint", "text",
         "http://localhost:11434/v1 for Ollama. Set it and the scripts, the "
         "storyboard, the gates and the picture review all run on your own "
         "GPU instead of the API. Empty = OpenAI."),
        ("RUFUS_LLM_MODEL", "Local model", "text",
         "The model name your endpoint serves, e.g. qwen3:14b. Per-stage "
         "overrides exist too — see docs/ENVIRONMENT.md."),
        ("RUFUS_OPENALEX", "Papers as sources", "bool",
         "Peer-reviewed abstracts from OpenAlex, tried before the discussion "
         "threads. An abstract states a year, names its authors and carries "
         "the study's own figures — the three things the fact gate keeps "
         "rejecting scripts for lacking."),
        ("RUFUS_OPENALEX_MAILTO", "OpenAlex contact email", "text",
         "Any address you read. OpenAlex's \"polite pool\" gives requests "
         "carrying a contact address a much higher rate limit — it is not a "
         "signup and not a key, just a parameter they ask for. Empty means "
         "the anonymous pool, which is the one that returns 429, and a run "
         "that cannot reach OpenAlex leans on sources the fact gate rejects "
         "more often. This said \"nothing breaks without it\" until a run "
         "died in Step 1 for want of a seed."),
        ("RUFUS_NEWSPAPERS", "Newspaper source", "bool",
         "Library of Congress scans. Turn off if the endpoint stays dead."),
    ]),
]

# Flat view, for the save handler and anything that only needs the keys.
SETTINGS_SCHEMA = [row for _title, _blurb, rows in SETTINGS_GROUPS for row in rows]
SETTINGS_KINDS = {key: kind for key, _label, kind, _help in SETTINGS_SCHEMA}


def _load_settings() -> dict:
    """What the settings page has saved.

    Reads the file directly rather than going through settings_store.load(),
    because THIS is the editor: it must show a key the loader would filter out
    (so the owner can see and delete it) rather than hiding it. settings_store
    is the reader every run uses; the difference is deliberate."""
    # utf-8-sig: PowerShell's `Set-Content -Encoding utf8` leaves a BOM, and
    # the editor has to be able to OPEN a file someone edited by hand — being
    # unable to is how a whole configuration silently reverts to defaults.
    try:
        return json.loads(SETTINGS_FILE.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError, ValueError):
        return {}


def _save_settings(values: dict) -> None:
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps(values, indent=2), encoding="utf-8")


# Processes THIS dashboard launched, keyed by channel id — the only ones it
# can cancel. A run started elsewhere (Task Scheduler, a manual run.bat) has
# no Popen handle here; _run_in_progress() above still reports it as
# running (shared lock file), but cancelling it needs the actual process
# handle, which only exists for runs this page started.
_LAUNCHED: dict[str, subprocess.Popen] = {}


def _launch_run(*, niche: str | None = None, topic: str | None = None,
                channel: str | None = None, script_file: str | None = None,
                gallery_id: int | None = None,
                hook_tone: str | None = None) -> tuple[subprocess.Popen, Path]:
    """Fire-and-forget: a genuinely separate OS process, not an in-process
    call — this Flask app runs threaded=False (see approve_video's
    _scoped_env note), so a call that blocks for the 5-45+ minutes a video
    can take would freeze every other request. No --skip-upload: safety
    comes from the review-queue gate itself, same as run_scheduled.bat.
    Output goes to a log file (not the dashboard's own stdout) so a
    dashboard-launched run doesn't interleave console output with whatever
    else is running; the single shared launch path behind both
    /request-topic and /system/run. Settings saved via /settings layer on
    top of the current env (they don't replace it) as overrides for THIS
    child process only."""
    cmd = [sys.executable, str(ROOT / "scripts" / "main.py")]
    if niche:
        cmd += ["--niche", niche]
    if topic:
        cmd += ["--topic", topic]
    if channel:
        cmd += ["--channel", channel]
    if script_file:
        # A script the owner chose in review. main.py skips its writer for
        # this one run; everything after it — fact gate, storyboard, render —
        # is the ordinary path.
        cmd += ["--script", script_file]
    env = os.environ.copy()
    env.update(_load_settings())
    if gallery_id is not None:
        # AFTER _load_settings, deliberately. A saved setting is a default for
        # every run; this is a fact about THIS one, and a stale RUFUS_GALLERY
        # left in the settings file would quietly render every future video
        # from one old set of pictures.
        env["RUFUS_GALLERY"] = str(gallery_id)
    if hook_tone:
        env["RUFUS_HOOK_TONE"] = hook_tone
    # THE CHILD'S STDOUT IS A FILE, so Python has no console to ask and falls
    # back to the system ANSI code page — cp1255 here, which has no ✗ and no
    # em-dash. A real run died mid-report on exactly that. The .bat launchers
    # set these; a dashboard-launched run inherits whatever started the
    # dashboard, which is not guaranteed to be one of them.
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    log_dir = paths.log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"dashboard_run_{int(time.time())}.log"
    with open(log_path, "wb") as logf:
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=logf, env=env,
                               stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    _LAUNCHED[channel or "default"] = proc
    return proc, log_path


def _launch_candidates(*, topic: str, proposal_id: int | None,
                       channel: str | None = None,
                       project_id: int | None = None) -> Path:
    """Write the script candidates for `topic` in a separate OS process.

    A SUBPROCESS FOR THE SAME REASON A RUN IS ONE. This Flask app runs
    threaded=False, and three scripts is one to three minutes of model calls —
    long enough that doing it inline freezes every other request, including the
    page the person is waiting on. They come back to /scripts when it is done,
    which is also why the page has to say plainly that nothing is there yet
    rather than looking empty and finished.
    """
    cmd = [sys.executable, str(ROOT / "scripts" / "script_candidates.py"), topic]
    if proposal_id is not None:
        cmd += ["--proposal", str(proposal_id)]
    if project_id is not None:
        cmd += ["--project", str(project_id)]
    env = os.environ.copy()
    env.update(_load_settings())
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    log_dir = paths.log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"candidates_{int(time.time())}.log"
    with open(log_path, "wb") as logf:
        subprocess.Popen(cmd, cwd=str(ROOT), stdout=logf, env=env,
                         stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    return log_path


def _launch_galleries(*, script_file: str, candidate_id: int | None,
                      topic: str = "") -> Path:
    """Draw the gallery variants for a chosen script, in a separate process.

    Two complete galleries is about forty minutes of the 3090 — far past the
    point where doing it inline would freeze the dashboard, and past the point
    where a person should sit and watch. They come back to /galleries.
    """
    cmd = [sys.executable, str(ROOT / "scripts" / "gallery_variants.py"),
           script_file]
    if candidate_id is not None:
        cmd += ["--candidate", str(candidate_id)]
    if topic:
        cmd += ["--topic", topic]
    env = os.environ.copy()
    env.update(_load_settings())
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    log_dir = paths.log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"galleries_{int(time.time())}.log"
    with open(log_path, "wb") as logf:
        subprocess.Popen(cmd, cwd=str(ROOT), stdout=logf, env=env,
                         stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    return log_path


def _launch_voice_takes(*, script_file: str, set_id: int,
                        topic: str = "") -> Path:
    """Record the three hook reads in a separate process.

    Shorter than the other two launches — three eight-second lines is seconds,
    not minutes — but still out of the request, because a TTS backend that
    stalls would hang the page rather than the take.
    """
    cmd = [sys.executable, str(ROOT / "scripts" / "voice_takes.py"),
           script_file, "--set", str(set_id)]
    if topic:
        cmd += ["--topic", topic]
    env = os.environ.copy()
    env.update(_load_settings())
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    log_dir = paths.log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"voice_takes_{int(time.time())}.log"
    with open(log_path, "wb") as logf:
        subprocess.Popen(cmd, cwd=str(ROOT), stdout=logf, env=env,
                         stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    return log_path


def _cancel_run(channel: str | None = None) -> bool:
    """Terminate a run this dashboard launched. Returns False (no-op) for a
    run it has no handle to, or one that already finished."""
    proc = _LAUNCHED.get(channel or "default")
    if proc is None or proc.poll() is not None:
        return False
    try:
        proc.terminate()
    except OSError:
        return False
    return True


# ── Data access (read-only) ───────────────────────────────────────────────────

def _channels() -> list[str]:
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        import channel_config
        return channel_config.list_channels()
    except Exception:
        return []


def _recent_videos(limit: int = 60, channel: str | None = None,
                   status: str | None = None) -> list[dict]:
    q = ("SELECT id, upload_date, niche, script_hook, title, score, "
         "hold_reason, youtube_id, run_id, channel, upload_status, "
         "created_at, uploaded_at, decided_by FROM videos")
    where, args = [], []
    if channel:
        where.append("channel = ?"); args.append(channel)
    if status:
        where.append("upload_status = ?"); args.append(status)
    if where:
        q += " WHERE " + " AND ".join(where)
    q += " ORDER BY id DESC LIMIT ?"
    args.append(limit)
    try:
        with db_manager._conn() as c:
            rows = c.execute(q, args).fetchall()
    except Exception:
        return []
    cols = ["id", "upload_date", "niche", "script_hook", "title", "score",
            "hold_reason", "youtube_id", "run_id", "channel", "upload_status",
            "created_at", "uploaded_at", "decided_by"]
    return [dict(zip(cols, r)) for r in rows]


def _video_detail(video_id: int) -> dict | None:
    q = ("SELECT id, upload_date, niche, script_hook, script_full, scene_desc, "
         "seed_type, seed_source, seed_content, seed_url, youtube_id, video_file, score, "
         "run_id, score_specificity, score_hook, score_compression, score_loop, "
         "score_human, attempts_used, final_temperature, score_reasoning, "
         "title, channel, hold_reason, description, upload_status, "
         # The evidence for which of two rows really owns a shared link.
         # Without it the duplicate audit called the ONE genuine upload
         # "never uploaded" and offered to clear the only correct row.
         "uploaded_at, decided_by "
         "FROM videos WHERE id = ?")
    try:
        with db_manager._conn() as c:
            row = c.execute(q, (video_id,)).fetchone()
    except Exception:
        return None
    if not row:
        return None
    cols = ["id", "upload_date", "niche", "script_hook", "script_full",
            "scene_desc", "seed_type", "seed_source", "seed_content", "seed_url",
            "youtube_id", "video_file", "score", "run_id",
            "score_specificity", "score_hook", "score_compression",
            "score_loop", "score_human", "attempts_used", "final_temperature",
            "score_reasoning", "title", "channel", "hold_reason",
            "description", "upload_status", "uploaded_at", "decided_by"]
    return dict(zip(cols, row))


def _stats(limit: int = 100, channel: str | None = None) -> dict:
    rows = _recent_videos(limit=limit, channel=channel)
    total = len(rows)
    if not total:
        return {"total": 0, "avg_score": 0.0, "hold_rate": 0.0,
                "uploaded": 0, "held": 0, "pending": 0, "rejected": 0}
    scores = [r["score"] for r in rows if r["score"] is not None]
    pending  = sum(1 for r in rows if r["upload_status"] == "pending")
    approved = sum(1 for r in rows if r["upload_status"] == "approved")
    rejected = sum(1 for r in rows if r["upload_status"] == "rejected")
    return {
        "total": total,
        "avg_score": round(sum(scores) / len(scores), 1) if scores else 0.0,
        "hold_rate": round(100 * pending / total, 1),   # legacy key, kept for template compat
        "uploaded": approved,
        "held": pending,
        "pending": pending,
        "rejected": rejected,
    }


def _top_rejections(limit: int = 8, channel: str | None = None) -> list[dict]:
    q = ("SELECT rejected_reason, COUNT(*) c FROM script_attempts "
         "WHERE accepted = 0 AND rejected_reason IS NOT NULL AND rejected_reason != ''")
    args: list = []
    if channel:
        q += " AND channel = ?"
        args.append(channel)
    q += " GROUP BY rejected_reason ORDER BY c DESC LIMIT ?"
    args.append(limit)
    try:
        with db_manager._conn() as c:
            rows = c.execute(q, args).fetchall()
    except Exception:
        return []
    return [{"reason": r[0], "count": r[1]} for r in rows]


def _orphaned_debug_runs(limit: int = 40) -> list[dict]:
    """Debug folders with NO matching videos.run_id — a run that started
    (RUFUS_DEBUG wrote script/keyframes) but crashed before reaching Step 6's
    DB save. These are invisible everywhere else in the app; that's exactly
    the "every failure, not just the successes" gap this page closes."""
    if not DEBUG_ROOT.is_dir():
        return []
    try:
        with db_manager._conn() as c:
            known = {r[0] for r in c.execute(
                "SELECT run_id FROM videos WHERE run_id IS NOT NULL").fetchall()}
    except Exception:
        known = set()

    orphans = []
    try:
        entries = sorted((d for d in DEBUG_ROOT.iterdir() if d.is_dir()),
                         key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        return []
    for d in entries:
        if d.name in known:
            continue
        try:
            files = sorted(f.name for f in d.iterdir() if f.is_file())
        except OSError:
            files = []
        preview = ""
        script_file = d / "script.txt"
        if script_file.exists():
            try:
                preview = script_file.read_text(encoding="utf-8", errors="replace")[:300]
            except OSError:
                pass
        orphans.append({"run_id": d.name, "mtime": d.stat().st_mtime,
                        "files": files, "preview": preview})
        if len(orphans) >= limit:
            break
    return orphans


# How far back the front page looks for a run that died. Long enough that the
# morning's screen still knows last night's scheduled run never produced
# anything; short enough that a crash from last week is not still shouting on a
# page that gets opened every day. A banner that is always on is a banner
# nobody reads.
FAILURE_WINDOW_SECONDS = 36 * 3600


def _recent_failures(window: float = FAILURE_WINDOW_SECONDS) -> list[dict]:
    """Runs that ended badly recently, newest first.

    A FINISHED VIDEO ANNOUNCES ITSELF — it appears in the pending list with a
    thumbnail and the count above it goes up. A run that DIED announces
    nothing: Step 6 is what writes the `videos` row, so a run that fell over
    before it leaves no row, the pending count is unchanged, and the front page
    looks exactly the way it looked yesterday. The owner reads an unchanged
    screen as "nothing ran last night" — when in fact something ran, spent
    eleven minutes, and died in Step 1 with no seed.

    Two sources, because they catch different deaths:

      run_progress    the run reached its finally-block and called
                      finish("failed"), so we know which of the seven steps it
                      died at and what the error said.
      orphan folders  the process was killed hard enough that the
                      finally-block never ran, so the only trace left behind
                      is a debug folder with no matching row.

    A run that is STILL GOING is not a failure, and its debug folder has no
    database row either — so live run_ids are excluded explicitly. Without
    that, every visit during a run would report that run as a crash.
    """
    now = time.time()
    out: list[dict] = []
    seen: set[str] = set()
    live: set[str] = set()

    try:
        progress = run_progress.read_all()
    except Exception:
        progress = []
    for p in progress:
        rid = (p.get("run_id") or "").strip()
        status = (p.get("status") or "").strip()
        stalled = bool(p.get("stale"))
        if status == "running" and not stalled:
            if rid:
                live.add(rid)
            continue
        # "cancelled" is deliberately NOT a failure. Somebody pressed stop;
        # reporting a person's own decision back to them as a problem is how a
        # notice area turns into something you learn to scroll past.
        if status != "failed" and not stalled:
            continue
        when = float(p.get("updated_at") or 0)
        if now - when > window:
            continue
        if rid:
            seen.add(rid)
        out.append({
            "run_id": rid,
            "when": when,
            "channel": (p.get("channel") or "").strip(),
            "step": int(p.get("step") or 0),
            "total": int(p.get("total") or run_progress.TOTAL_STEPS),
            "label": (p.get("label") or "").strip(),
            "why": (p.get("detail") or "").strip(),
            "stalled": stalled,
        })

    for o in _orphaned_debug_runs(limit=20):
        rid = o["run_id"]
        if rid in seen or rid in live or now - o["mtime"] > window:
            continue
        seen.add(rid)
        out.append({"run_id": rid, "when": o["mtime"], "channel": "",
                    "step": 0, "total": run_progress.TOTAL_STEPS,
                    "label": "", "why": "", "stalled": False})

    return sorted(out, key=lambda r: r["when"], reverse=True)


def _failure_notice() -> str:
    """The crashed-run block for the front page, or "" when nothing failed —
    a quiet week must show a quiet page, or this becomes furniture."""
    try:
        fails = _recent_failures()
    except Exception as e:                       # never break the front page
        print(f"[dashboard] failure summary unavailable: {e}")
        return ""
    if not fails:
        return ""

    now = time.time()
    rows = ""
    for f in fails[:4]:
        if f["stalled"]:
            where = (f"went quiet at step {f['step']}/{f['total']}"
                     if f["step"] else "went quiet before its first step")
        elif f["step"]:
            where = f"stopped at step {f['step']}/{f['total']}"
            if f["label"]:
                where += f" ({_esc(f['label'])})"
        else:
            where = "never reached the database"
        ago = _human_gap(max(0.0, now - f["when"])) if f["when"] else "?"
        who = f' · {_esc(f["channel"])}' if f["channel"] else ""
        why = (f'<div class="muted" style="margin-top:4px">'
               f'{_esc(f["why"][:240])}</div>') if f["why"] else ""
        rows += (f'<div style="margin:8px 0">'
                 f'<strong>{_esc(f["run_id"] or "a run")}</strong>'
                 f'<span class="muted"> · {ago} ago{who} · </span>{where}'
                 f'{why}</div>')

    more = (f'<div class="muted">and {len(fails) - 4} more</div>'
            if len(fails) > 4 else "")
    plural = "" if len(fails) == 1 else "s"
    return (f'<div class="card" style="width:100%;margin-bottom:18px">'
            f'<span class="badge held">{len(fails)} run{plural} failed '
            f'recently</span>'
            f'<div style="margin-top:8px">{rows}{more}</div>'
            f'<div class="muted" style="margin-top:6px">'
            f'<a href="/failures">every failure, with what it wrote →</a> · '
            f'<a href="/logs">the logs →</a></div></div>')


def _rejected_attempts(limit: int = 200, channel: str | None = None,
                       niche: str | None = None, phase: str | None = None) -> list[dict]:
    """Every rejected hook/body attempt (script_attempts already logs these —
    the homepage only ever showed the top-8 aggregate; this is the full,
    filterable browser."""
    q = ("SELECT ts, niche, phase, attempt_n, hook, body, rejected_reason, channel "
         "FROM script_attempts WHERE accepted = 0 AND rejected_reason IS NOT NULL "
         "AND rejected_reason != ''")
    args: list = []
    if channel:
        q += " AND channel = ?"; args.append(channel)
    if niche:
        q += " AND niche = ?"; args.append(niche)
    if phase:
        q += " AND phase = ?"; args.append(phase)
    q += " ORDER BY id DESC LIMIT ?"
    args.append(limit)
    try:
        with db_manager._conn() as c:
            rows = c.execute(q, args).fetchall()
    except Exception:
        return []
    cols = ["ts", "niche", "phase", "attempt_n", "hook", "body",
            "rejected_reason", "channel"]
    return [dict(zip(cols, r)) for r in rows]


# Root-cause taxonomy for script_attempts.rejected_reason — a fixed, small
# set of buckets instead of counting distinct free-text strings (which
# mostly differ only by which specific word got banned). After enough
# volume this answers "which STAGE of the pipeline is actually the
# bottleneck" at a glance instead of requiring someone to read every row.
# Order matters: checked top-to-bottom, first match wins (e.g. a banned-
# phrase rejection on a hook attempt is "safety", not "weak_hook").
_REJECTION_CATEGORIES = [
    ("safety",          ("banned phrase", "hedging", "conspiracy")),
    ("accuracy",        ("specificity", "invented", "fabricat", "sensory")),
    ("weak_hook",       ("forbidden opener", "hook too short", "hook too long")),
    ("loose_structure", ("loop no echo", "opinion word", "sentences too long",
                        "sentences too short", "cadence", "too few sentences")),
    ("boring",          ("boring", "no tension", "flat")),
]

# supervisor.py's three gates (seed_gate, fact_check, footage_gate) return
# free-form LLM prose, not script_writer's controlled-vocabulary strings —
# keyword matching against that prose would be unreliable. Their PHASE alone
# already says exactly what kind of failure it is, so those are categorized
# directly instead of by keyword.
_PHASE_CATEGORY = {
    "seed_gate":    "weak_seed",
    "fact_check":   "accuracy",
    "footage_gate": "footage_drift",
}
_CATEGORY_ORDER = ([c for c, _ in _REJECTION_CATEGORIES]
                  + ["weak_seed", "footage_drift", "other"])


def _categorize_rejection(reason: str, phase: str | None = None) -> str:
    if phase in _PHASE_CATEGORY:
        return _PHASE_CATEGORY[phase]
    r = (reason or "").lower()
    for category, keywords in _REJECTION_CATEGORIES:
        if any(k in r for k in keywords):
            return category
    return "other"


def _rejection_category_counts(channel: str | None = None) -> list[dict]:
    """Aggregate ALL rejected attempts (not just the last N shown in the
    browser) into the fixed taxonomy above — covers every gate in the
    pipeline (hook/body phases AND the three supervisor gates), so this
    can answer e.g. "is Hook Scorer or Fact-check the real bottleneck"."""
    reasons = _rejected_attempts(limit=100_000, channel=channel)
    from collections import Counter
    counts = Counter(_categorize_rejection(r["rejected_reason"], r["phase"]) for r in reasons)
    total = sum(counts.values())
    if not total:
        return []
    order = _CATEGORY_ORDER
    return [{"category": c, "count": counts[c],
            "pct": round(100 * counts[c] / total, 1)}
           for c in order if counts.get(c)]


def _distinct(column: str) -> list[str]:
    try:
        with db_manager._conn() as c:
            rows = c.execute(
                f"SELECT DISTINCT {column} FROM script_attempts "
                f"WHERE {column} IS NOT NULL ORDER BY {column}").fetchall()
        return [r[0] for r in rows]
    except Exception:
        return []


def _fmt_ts(epoch: float) -> str:
    from datetime import datetime
    return datetime.fromtimestamp(epoch).strftime("%Y-%m-%d %H:%M")


# Real audience performance, not just the internal LLM score. The data pipe
# already exists (analytics_fetcher.py fetches views/watch% via the YouTube
# API, report.py's correlate() already buckets score-vs-views for the log
# digest) — this is that same signal in the browser, since nothing here ever
# queried the `metrics` table before.
MIN_VIDEOS_FOR_CORRELATION = 5   # matches report.py's --correlate guard


def _performance_rows(channel: str | None = None, days: int = 90) -> list[dict]:
    """videos LEFT JOIN latest metrics row per video — same shape as
    report.py's _latest_metrics_join() / feedback_analyzer.py's query.
    LEFT JOIN (not INNER) so a video with no metrics yet still shows up
    with blank views/watch%, rather than silently vanishing."""
    q = """
        SELECT v.id, v.upload_date, v.niche, v.channel,
               COALESCE(v.title, v.script_hook) AS title, v.score, v.youtube_id,
               m.views, m.watch_pct, m.likes
        FROM videos v
        LEFT JOIN (
            SELECT video_id, views, watch_pct, ctr, likes
            FROM metrics WHERE id IN (SELECT MAX(id) FROM metrics GROUP BY video_id)
        ) m ON m.video_id = v.id
        WHERE v.upload_date >= date('now', ?)
    """
    args: list = [f"-{days} days"]
    if channel:
        q += " AND v.channel = ?"
        args.append(channel)
    q += " ORDER BY v.id DESC"
    try:
        with db_manager._conn() as c:
            rows = c.execute(q, args).fetchall()
    except Exception:
        return []
    cols = ["id", "upload_date", "niche", "channel", "title", "score",
            "youtube_id", "views", "watch_pct", "likes"]
    return [dict(zip(cols, r)) for r in rows]


def _score_vs_views(rows: list[dict]) -> list[dict]:
    """Bucket by score, average views per bucket — does the 1-10 gate
    actually predict real performance? Same question report.py's
    correlate() answers for the log digest."""
    from collections import defaultdict
    buckets: dict[int, list[int]] = defaultdict(list)
    for r in rows:
        if r["score"] is not None and r["views"] is not None:
            buckets[r["score"]].append(r["views"])
    return [{"score": s, "avg_views": round(sum(vs) / len(vs)), "n": len(vs)}
           for s, vs in sorted(buckets.items())]


def _debug_assets(run_id: str | None) -> list[dict]:
    """Files in this run's debug folder (script/voiceover/keyframes). Every run
    keeps these now (not just RUFUS_DEBUG=1 runs), and they're retained
    permanently as the quality-review record."""
    if not run_id:
        return []
    d = DEBUG_ROOT / run_id
    if not d.is_dir():
        return []
    out = []
    for f in sorted(d.iterdir()):
        if f.is_file():
            out.append({"name": f.name, "size_kb": max(1, f.stat().st_size // 1024)})
    return out


def _image_prompts(run_id: str | None) -> list[dict]:
    """The per-beat image-generation prompts for this run, paired with their
    keyframe. comfy_client/sd_client write NN.png (the still) + NN.txt (the
    exact FLUX/SD prompt that produced it) per beat. This surfaces the full
    script→images chain on the review page instead of leaving each prompt as a
    file the reviewer has to download one by one.

    Returns [{n, prompt, image}] ordered by beat, where `image` is the NN.png
    name (served via /debug/<run_id>/<image>) or None if only the prompt exists."""
    if not run_id:
        return []
    d = DEBUG_ROOT / run_id
    if not d.is_dir():
        return []
    out = []
    for txt in sorted(d.glob("[0-9]*.txt")):
        try:
            raw = txt.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue
        # Files are written as "FLUX PROMPT:\n<prompt>" — strip the label.
        prompt = raw.split(":", 1)[1].strip() if raw.lower().startswith("flux prompt") else raw
        png = txt.with_suffix(".png")
        out.append({"n": txt.stem, "prompt": prompt,
                    "image": png.name if png.is_file() else None})
    return out


def _gallery_images(limit: int = 60) -> list[dict]:
    """Every keyframe still across every recent run, newest first — a
    browsable portfolio instead of hunting through one video's detail page
    at a time. Same source _orphaned_debug_runs()/_debug_assets() already
    read (paths.debug_root()), just flattened across runs."""
    if not DEBUG_ROOT.is_dir():
        return []
    out: list[dict] = []
    try:
        run_dirs = sorted((d for d in DEBUG_ROOT.iterdir() if d.is_dir()),
                         key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        return []
    for run_dir in run_dirs:
        try:
            pngs = sorted(run_dir.glob("[0-9]*.png"))
        except OSError:
            continue
        for png in pngs:
            out.append({"run_id": run_dir.name, "image": png.name,
                       "mtime": png.stat().st_mtime})
            if len(out) >= limit:
                return out
    return out


# ── Rendering helpers (no template engine — a handful of small f-strings) ────

def _esc(s) -> str:
    return html.escape(str(s)) if s is not None else ""


def _when_cell(stamp) -> str:
    """A timestamp as date over time, for a table cell.

    ONE FUNCTION FOR BOTH PAGES on purpose. The front page and /history answer
    the same question and a hand-copy of this formatting is how one of them
    ends up showing seconds, or a fake midnight, after somebody edits the
    other. Rows written before created_at existed are a bare date and say so
    rather than being padded out to look like they carry a time.
    """
    text = str(stamp or "").strip()
    if not text:
        return '<span class="muted">&mdash;</span>'
    parts = text.split()
    if len(parts) < 2:
        return f'{_esc(parts[0])}<br><span class="muted">no time</span>'
    return f'{_esc(parts[0])}<br><strong>{_esc(parts[1][:5])}</strong>'


def _score_color(score) -> str:
    if score is None:
        return "#888"
    if score >= 8:
        return "#22c55e"
    if score >= 6:
        return "#eab308"
    return "#ef4444"


def _sparkline_svg(scores: list[int], width: int = 320, height: int = 64,
                   threshold: int = UPLOAD_THRESHOLD_DEFAULT) -> str:
    """Score trend, oldest → newest, left to right. Self-contained inline SVG
    — no chart library, nothing that can fail to load remotely."""
    if not scores:
        return "<p class='muted'>No scored videos yet.</p>"
    n = len(scores)
    pad = 6
    step_x = (width - 2 * pad) / max(1, n - 1)

    def y(s):
        s = max(0, min(10, s))
        return height - pad - (s / 10) * (height - 2 * pad)

    points = " ".join(f"{pad + i*step_x:.1f},{y(s):.1f}" for i, s in enumerate(scores))
    thresh_y = y(threshold)
    dots = "".join(
        f'<circle cx="{pad + i*step_x:.1f}" cy="{y(s):.1f}" r="3" '
        f'fill="{_score_color(s)}"><title>{s}/10</title></circle>'
        for i, s in enumerate(scores)
    )
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="Score trend">'
        f'<line x1="{pad}" y1="{thresh_y:.1f}" x2="{width-pad}" y2="{thresh_y:.1f}" '
        f'stroke="#6b7280" stroke-dasharray="4,4" stroke-width="1"/>'
        f'<polyline points="{points}" fill="none" stroke="#3b82f6" stroke-width="2"/>'
        f'{dots}</svg>'
    )


PAGE_STYLE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ThePaperTrails</title>
<style>
  /* ONE PALETTE, DEFINED ONCE. The previous stylesheet hardcoded #171a21 and
     #2a2d34 in a dozen rules and patched light mode with a dozen more
     one-off media queries — so every new component had to remember to bring
     its own light-mode override, and the ones that forgot (the log viewer's
     dark pre block on a white page) were unreadable. Tokens make the default
     correct instead of remembered. */
  :root {
    color-scheme: dark light;
    --bg:      #0f1115;
    --surface: #171a21;
    --raised:  #1d212a;
    --border:  #2a2d34;
    --text:    #e5e7eb;
    --dim:     #9ca3af;
    --accent:  #3b82f6;
    --ok:      #22c55e;
    --warn:    #eab308;
    --bad:     #ef4444;
    /* A HUE PER JOB. The nav already groups into Make / Review / Measure /
       System; giving each one a colour means the page you are on is legible
       before you have read a word of it. One accent for everything is tidy
       and tells you nothing. */
    --make:    #8b5cf6;
    --review:  #f59e0b;
    --measure: #06b6d4;
    --system:  #64748b;
    /* Scoped by .card.t-* and h2.s-* below, with a root default so a use that
       escapes its scope degrades to a neutral edge instead of resolving to
       nothing and silently dropping the whole declaration. */
    --tone:    var(--dim);
    /* Set inline per name chip, from the name itself. Same root default as
       --tone and for the same reason: a use that escapes its element resolves
       to a neutral colour instead of resolving to nothing, which silently
       drops the whole declaration. */
    --who:     var(--dim);
    --radius:  10px;   /* panels: cards, tables, code, images */
    --radius-sm: 8px;  /* controls: buttons, fields, messages  */
    --shadow:  0 1px 2px rgba(0,0,0,.28);
  }
  @media (prefers-color-scheme: light) {
    :root {
      --bg: #f6f7f9; --surface: #ffffff; --raised: #ffffff;
      --border: #e3e6ea; --text: #14171c; --dim: #5f6672;
      /* Slightly deeper on white: the dark-mode values are chosen to glow
         against #0f1115 and go weak on a light page. */
      --make: #7c3aed; --measure: #0891b2; --system: #475569;
      --shadow: 0 1px 2px rgba(16,24,40,.06), 0 1px 3px rgba(16,24,40,.08);
    }
  }

  * { box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
         Helvetica, Arial, sans-serif; margin: 0; background: var(--bg);
         color: var(--text); -webkit-font-smoothing: antialiased;
         /* A wash rather than a flat field. Fixed so it does not slide
            around under a long table, and layered UNDER var(--bg) as a
            colour so a browser that ignores the gradient still gets the
            right background rather than white. */
         background-image:
           radial-gradient(900px 500px at 12% -8%,
             color-mix(in srgb, var(--accent) 13%, transparent), transparent 70%),
           radial-gradient(700px 420px at 92% 0%,
             color-mix(in srgb, var(--make) 11%, transparent), transparent 70%);
         background-attachment: fixed; }
  @media (prefers-reduced-motion: reduce) {
    * { animation-duration: 0.01ms !important; animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important; }
  }
  a { color: var(--accent); }
  :focus-visible { outline: 2px solid var(--accent); outline-offset: 2px;
                   border-radius: 4px; }

  header { position: sticky; top: 0; z-index: 20; padding: 12px 24px;
           background: linear-gradient(to right,
             color-mix(in srgb, var(--accent) 10%, transparent),
             color-mix(in srgb, var(--make) 8%, transparent)),
             color-mix(in srgb, var(--bg) 88%, transparent);
           backdrop-filter: saturate(180%) blur(10px);
           border-bottom: 1px solid var(--border);
           display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
  header a { color: inherit; text-decoration: none; }
  main { padding: 22px 24px 60px; max-width: 1140px; margin: 0 auto; }
  h1 { font-size: 18px; margin: 0; letter-spacing: -0.01em; }
  h2 { font-size: 12px; color: var(--dim); text-transform: uppercase;
       letter-spacing: 0.08em; margin: 30px 0 10px; font-weight: 700; }

  .cards { display: flex; gap: 12px; flex-wrap: wrap; }
  .card { background: var(--surface); border: 1px solid var(--border);
          border-radius: var(--radius); padding: 14px 18px; min-width: 120px;
          box-shadow: var(--shadow); }
  .card .num { font-size: 26px; font-weight: 700; letter-spacing: -0.02em; }
  .card .label { font-size: 12px; color: var(--dim); }

  /* FOUR IDENTICAL GREY BOXES MADE YOU READ FOUR LABELS. The numbers already
     mean four different things — waiting on you, published, thrown away, how
     good they were — and the palette already has a colour for each of those
     meanings. Using it costs nothing and is read before the words are. */
  .card.tone { border-left: 3px solid var(--tone); }
  .card.tone .num { color: var(--tone); }
  .card.tone { background: linear-gradient(to bottom right,
                 color-mix(in srgb, var(--tone) 9%, transparent), transparent 60%),
                 var(--surface); }
  .card.t-pending { --tone: var(--warn); }
  .card.t-ok      { --tone: var(--ok); }
  .card.t-bad     { --tone: var(--bad); }
  .card.t-info    { --tone: var(--accent); }

  /* A heading knows which job it belongs to. */
  h2.sec { border-left: 3px solid var(--tone, var(--border)); padding-left: 10px; }
  h2.s-make    { --tone: var(--make); color: var(--make); }
  h2.s-review  { --tone: var(--review); color: var(--review); }
  h2.s-measure { --tone: var(--measure); color: var(--measure); }

  table { width: 100%; border-collapse: collapse; margin-top: 6px;
          background: var(--surface); border: 1px solid var(--border);
          border-radius: var(--radius); overflow: hidden;
          box-shadow: var(--shadow); }
  th, td { text-align: left; padding: 10px 12px;
           border-bottom: 1px solid var(--border); font-size: 14px;
           vertical-align: top; }
  tr:last-child td { border-bottom: none; }
  th { color: var(--dim); font-weight: 700; font-size: 11px;
       text-transform: uppercase; letter-spacing: 0.06em; }
  tbody tr:hover td, tr:hover td { background: color-mix(in srgb, var(--accent) 7%, transparent); }
  a.row-link { color: inherit; text-decoration: none; display: block; }

  .badge { display: inline-block; padding: 2px 9px; border-radius: 999px;
           font-size: 11px; font-weight: 700; letter-spacing: 0.02em;
           text-transform: uppercase; }
  .badge.ok      { background: color-mix(in srgb, var(--ok) 16%, transparent);  color: var(--ok); }
  .badge.held    { background: color-mix(in srgb, var(--bad) 16%, transparent); color: var(--bad); }
  .badge.pending { background: color-mix(in srgb, var(--warn) 18%, transparent);color: var(--warn); }

  .muted { color: var(--dim); font-size: 13px; line-height: 1.5; }
  code { background: color-mix(in srgb, var(--dim) 14%, transparent);
         padding: 1px 5px; border-radius: 4px; font-size: 12px; }
  pre { background: var(--surface); border: 1px solid var(--border);
        border-radius: var(--radius); box-shadow: var(--shadow);
        color: var(--text); }

  .msg { padding: 11px 14px; border-radius: var(--radius-sm); margin-bottom: 14px;
         font-size: 14px; border: 1px solid transparent; }
  .msg.ok    { background: color-mix(in srgb, var(--ok) 12%, transparent);
               border-color: color-mix(in srgb, var(--ok) 30%, transparent); color: var(--ok); }
  .msg.error { background: color-mix(in srgb, var(--bad) 12%, transparent);
               border-color: color-mix(in srgb, var(--bad) 30%, transparent); color: var(--bad); }

  .actions { margin: 16px 0; display: flex; gap: 12px; flex-wrap: wrap; }
  .btn { border: 1px solid var(--border); background: var(--raised);
         color: var(--text); border-radius: var(--radius-sm); padding: 10px 18px;
         font-size: 14px; font-weight: 600; cursor: pointer;
         transition: transform .06s ease, filter .12s ease; }
  .btn:hover { filter: brightness(1.08); }
  .btn:active { transform: translateY(1px); }
  /* HALF OF "SLOW" IS A BUTTON THAT LOOKS DEAD. Approve, Draw them, Re-cut and
     Regen all hand off to something that takes real time, and until the page
     navigated there was no evidence the click had landed — so people click
     again, which on Approve is a second upload attempt. Disabling on submit
     costs one attribute and removes both problems. */
  .btn[disabled], .btn.working { opacity: .72; cursor: progress;
                                 filter: none; transform: none; }
  .btn.working::before { content: ""; display: inline-block; width: 11px;
                         height: 11px; margin-right: 7px; vertical-align: -1px;
                         border: 2px solid currentColor; border-right-color: transparent;
                         border-radius: 50%; animation: spin .7s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* Rows and cards move rather than snap. */
  tbody tr td, tr td { transition: background .12s ease; }
  .card, .thumbcard, .orphan { transition: transform .12s ease, box-shadow .12s ease; }
  .card:hover, .orphan:hover { transform: translateY(-1px);
                               box-shadow: 0 4px 14px rgba(0,0,0,.16); }

  /* A WIDE TABLE SCROLLS INSIDE ITSELF, NOT BY DRAGGING THE PAGE WITH IT.
     Six columns on a 390px screen pushed the whole document sideways: the
     status bar's text ran off the right edge and the Status column — the one
     that says whether a video is waiting on you — was simply not on screen,
     with nothing to suggest it existed. */
  .tablewrap { overflow-x: auto; -webkit-overflow-scrolling: touch;
               border-radius: var(--radius); }
  .tablewrap > table { margin-top: 0; }

  /* Filter box above a long table. */
  .tablefilter { display: block; width: 100%; max-width: 320px; margin: 10px 0 0;
                 padding: 8px 11px; border-radius: var(--radius-sm);
                 border: 1px solid var(--border); background: var(--bg);
                 color: inherit; font: inherit; font-size: 13px; }
  .btn.approve { background: var(--ok);     color: #06210f; border-color: transparent; }
  .btn.reject  { background: var(--bad);    color: #2a0a0a; border-color: transparent; }
  .btn.save    { background: var(--accent); color: #06122a; border-color: transparent; }

  .field { display: block; width: 100%; margin: 6px 0 14px; padding: 9px 11px;
           border-radius: var(--radius-sm); border: 1px solid var(--border);
           background: var(--bg); color: inherit; font-family: inherit;
           font-size: 14px; }
  select { padding: 8px 10px; border-radius: var(--radius-sm); border: 1px solid var(--border);
           background: var(--bg); color: inherit; font: inherit; }
  label { font-size: 11px; color: var(--dim); text-transform: uppercase;
          letter-spacing: 0.06em; }

  /* The four decisions. A row of equal words tells you nothing about order or
     state, so this carries both: a number for the step and a badge for how
     many choices are actually waiting behind it. Scrolls sideways rather than
     wrapping — on a phone four steps that reflow into two rows stop reading as
     a sequence, which is the only thing this element is for. */
  .flow { display: flex; gap: 4px; margin: 0 0 20px; overflow-x: auto;
          padding-bottom: 4px; -webkit-overflow-scrolling: touch; }
  .flow-step { display: flex; align-items: center; gap: 8px; flex: 0 0 auto;
               padding: 8px 12px; border-radius: var(--radius-sm);
               border: 1px solid var(--border); background: var(--surface);
               color: var(--dim); text-decoration: none; font-size: 13px;
               white-space: nowrap; }
  .flow-step:hover { color: inherit; border-color: var(--accent); }
  .flow-step.here { color: inherit; border-color: var(--accent);
                    background: var(--bg); font-weight: 600; }
  .flow-i { display: inline-flex; align-items: center; justify-content: center;
            width: 18px; height: 18px; border-radius: 50%; font-size: 11px;
            background: var(--border); color: var(--dim); }
  .flow-step.here .flow-i { background: var(--accent); color: #06122a; }
  .flow-n { min-width: 18px; text-align: center; padding: 1px 6px;
            border-radius: 999px; font-size: 11px; font-weight: 600;
            background: var(--accent); color: #06122a; }
  /* A zero is information too — it says "nothing waiting here", which is
     different from a badge that is simply absent and could mean anything. */
  .flow-n.zero { background: transparent; color: var(--dim); font-weight: 400; }

  /* THE HOME GRID. Auto-fit rather than a fixed column count: the same markup
     is one column on the phone this is actually reviewed from and four on a
     desktop, without a media query per breakpoint. */
  .tiles { display: grid; gap: 12px; margin: 0 0 20px;
           grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); }
  .tile { display: flex; flex-direction: column; gap: 4px;
          padding: 14px 16px; border-radius: var(--radius);
          border: 1px solid var(--border); background: var(--surface);
          box-shadow: var(--shadow); text-decoration: none; color: inherit;
          /* The group's hue as an edge rather than a fill: it names the tile
             without competing with the words in it, and it survives the light
             theme where a filled panel would have to fight for contrast. */
          border-left: 3px solid var(--tone); }
  .tile:hover { border-color: var(--tone); background: var(--raised); }
  .tile-t { display: flex; align-items: center; gap: 8px; font-size: 15px;
            font-weight: 600; }
  .tile-b { font-size: 13px; color: var(--dim); }
  /* The number is in the label too, for anything that does not render colour
     or size — a badge alone is a fact only sighted users get. */
  .tile-n { min-width: 20px; padding: 1px 7px; border-radius: 999px;
            font-size: 12px; font-weight: 600;
            background: var(--tone); color: #0b1220; }
  /* One shot per card: a header line saying which shot and what of, then the
     draws side by side. The draws grid is TWO columns and stays two on a
     phone — the entire point is comparing them at a glance, and stacking them
     turns one comparison into two acts of memory. */
  .shot { border: 1px solid var(--border); border-radius: var(--radius);
          background: var(--surface); margin: 10px 0; overflow: hidden; }
  .shot-h { display: flex; gap: 8px; align-items: baseline; padding: 8px 12px;
            border-bottom: 1px solid var(--border); }
  .shot-n { flex: none; min-width: 20px; height: 20px; border-radius: 999px;
            background: var(--make); color: #06122a; font-size: 12px;
            font-weight: 600; text-align: center; line-height: 20px; }
  .shot-p { font-size: 13px; color: var(--dim); }
  .draws { display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px;
           padding: 8px; }
  .draw { border: 2px solid transparent; border-radius: var(--radius-sm);
          overflow: hidden; background: var(--bg); }
  .draw.won { border-color: var(--ok); }
  .draw img { width: 100%; display: block; aspect-ratio: 9 / 16;
              object-fit: cover; }
  .draw-f { display: flex; gap: 8px; align-items: center;
            justify-content: space-between; padding: 4px 8px; }
  .draw-v { font-size: 12px; color: var(--dim); font-weight: 600; }
  .draw-w { font-size: 12px; color: var(--ok); font-weight: 600; }
  .pick { font-size: 12px; padding: 4px 8px; }
  .by { font-size: 11px; font-weight: 600; padding: 2px 7px;
        border-radius: 999px; white-space: nowrap; color: var(--who);
        background: color-mix(in srgb, var(--who) 15%, transparent); }
  .notes { list-style: none; padding: 0; margin: 10px 0 0;
           display: grid; gap: 8px; max-width: 720px; }
  .note { display: grid; grid-template-columns: 1fr auto; gap: 8px 12px;
          align-items: center; background: var(--surface);
          border: 1px solid var(--border); border-left: 3px solid var(--tone);
          border-radius: var(--radius); padding: 10px 14px; }
  .note.p-high   { --tone: var(--bad); }
  .note.p-normal { --tone: var(--accent); }
  .note.p-low    { --tone: var(--border); }
  .note.is-done  { opacity: .58; }
  .note.is-done .note-b { text-decoration: line-through; }
  .note-b { grid-column: 1; font-size: 15px; line-height: 1.45; }
  .note-m { grid-column: 1; display: flex; gap: 8px; align-items: center;
            flex-wrap: wrap; font-size: 12px; }
  .note form { grid-column: 2; grid-row: 1 / span 2; }
  .note-bell { color: var(--accent); font-size: 11px; }
  .hi { display: flex; flex-wrap: wrap; gap: 8px 16px; align-items: baseline;
        padding: 16px 20px; margin-bottom: 16px; border-radius: var(--radius);
        border: 1px solid var(--border);
        background: linear-gradient(120deg,
          color-mix(in srgb, var(--make) 16%, transparent),
          color-mix(in srgb, var(--measure) 12%, transparent)),
          var(--surface); }
  .hi-h { font-size: 26px; font-weight: 600; margin: 0; letter-spacing: -.01em; }
  .hi-q { font-size: 14px; color: var(--dim); }
  .hi-q a { color: inherit; font-weight: 600; text-decoration: none;
            border-bottom: 1px solid var(--border); }
  .hi-q a:hover { color: var(--text); border-bottom-color: var(--accent); }
  /* The drawing room. Empty frames from the first poll, so the shape of the
     job is visible before any of it has arrived. */
  .draw-head { display: flex; justify-content: space-between; align-items: flex-end;
               gap: 16px; flex-wrap: wrap; margin: 12px 0 8px; }
  .draw-n { font-size: 26px; font-weight: 600; letter-spacing: -.01em; }
  .draw-of { color: var(--dim); font-weight: 400; font-size: 15px; }
  .draw-rate { font-size: 12px; font-variant-numeric: tabular-nums; }
  .draw-grid { display: grid; gap: 8px;
               grid-template-columns: repeat(auto-fill, minmax(96px, 1fr)); }
  .draw-cell { position: relative; }
  .draw-slot { aspect-ratio: 9 / 16; border-radius: var(--radius-sm);
               background: var(--surface); border: 1px solid var(--border);
               overflow: hidden; display: grid; place-items: center; }
  .draw-slot img { width: 100%; height: 100%; object-fit: cover; display: block; }
  /* An arrived picture announces itself once, then stops moving. */
  .draw-cell.in .draw-slot { border-color: var(--ok); animation: land .45s ease-out; }
  @keyframes land { from { opacity: 0; transform: scale(.96); }
                    to   { opacity: 1; transform: none; } }
  .draw-tag { position: absolute; left: 4px; top: 4px; font-size: 11px;
              font-weight: 600; color: var(--dim);
              background: color-mix(in srgb, var(--bg) 78%, transparent);
              border-radius: 4px; padding: 0 4px; }
  #draw.stalled .draw-n { color: var(--bad); }
  #draw.stalled #draw-sub { color: var(--bad); }
  .draw-log { margin-top: 20px; font-size: 11px; line-height: 1.5;
              color: var(--dim); background: var(--surface);
              border: 1px solid var(--border); border-radius: var(--radius);
              padding: 12px; max-height: 132px; overflow: auto;
              white-space: pre-wrap; word-break: break-word; }
  @media (prefers-reduced-motion: reduce) {
    .draw-cell.in .draw-slot { animation: none; }
  }
  .tile.t-make    { --tone: var(--make); }
  .tile.t-review  { --tone: var(--review); }
  .tile.t-measure { --tone: var(--measure); }
  .tile.t-system  { --tone: var(--system); }

  .filters { margin: 12px 0; }
  .filters a { margin-right: 10px; font-size: 13px; text-decoration: none; }
  .back { text-decoration: none; font-size: 14px; }
  .script { white-space: pre-wrap; font-size: 15px; line-height: 1.6;
            background: var(--surface); border: 1px solid var(--border);
            border-radius: var(--radius); box-shadow: var(--shadow);
            padding: 14px; }
  .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  /* A GRID ITEM DEFAULTS TO min-width:auto, WHICH MEANS "never shrink below
     your content". So a 1fr column holding a six-column table stayed as wide
     as the table wanted and pushed the whole document 20px sideways on a
     390px screen — even with the table itself in an overflow:auto wrapper,
     because the wrapper could not shrink either. min-width:0 is what lets
     "one fraction of the available space" actually mean that. */
  .grid2 > * { min-width: 0; }
  @media (max-width: 760px) { .grid2 { grid-template-columns: 1fr; } }
  .assets a { display: inline-block; margin: 4px 8px 4px 0; font-size: 13px;
              text-decoration: none; }

  .navlink { color: var(--dim); text-decoration: none; font-size: 14px;
             margin-left: 14px; padding: 5px 2px; border-bottom: 2px solid transparent; }
  .navlink:hover { color: var(--accent); border-bottom-color: var(--accent); }

  /* The grouped overflow. Closed it is one word; open it is four labelled
     columns. No script — see _head for why. */
  .navmore { display: inline-block; position: relative; margin-left: 14px;
             vertical-align: baseline }
  .navmore > summary { cursor: pointer; color: var(--dim); font-size: 14px;
                       list-style: none; padding: 2px 8px;
                       border: 1px solid var(--border); border-radius: var(--radius-sm) }
  .navmore > summary::-webkit-details-marker { display: none }
  .navmore > summary::after { content: " ▾"; opacity: .6 }
  .navmore[open] > summary { color: var(--accent); border-color: var(--accent) }
  .navmore-body { position: absolute; z-index: 40; top: calc(100% + 8px);
                  left: 0; min-width: 460px; display: grid;
                  grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 4px 20px;
                  background: var(--surface); border: 1px solid var(--border);
                  border-radius: var(--radius); box-shadow: var(--shadow);
                  padding: 14px 16px }
  .navgroup { display: flex; flex-direction: column; gap: 4px; min-width: 0 }
  .navgroup-t { font-size: 11px; letter-spacing: .08em; text-transform: uppercase;
                color: var(--dim); opacity: .7; margin-bottom: 2px }
  .navgroup .navlink { border-bottom: 0; padding: 5px 0 }

  .orphan { background: var(--surface); border: 1px solid var(--border);
            border-radius: var(--radius); box-shadow: var(--shadow);
            padding: 12px 14px; margin-bottom: 10px; }

  /* Live status bar — polls /api/status, no page reload */
  /* Pinned along the bottom of the window, above everything, on every page.
     A strip you glance at while reading something else cannot be a strip that
     scrolls away. body gets padding to match so the last row of a long page is
     never hidden underneath it. */
  #livebar { position: fixed; left: 0; right: 0; bottom: 0; z-index: 40;
             display: flex; gap: 16px; flex-wrap: wrap; align-items: center;
             background: color-mix(in srgb, var(--surface) 94%, transparent);
             backdrop-filter: blur(8px);
             border-top: 1px solid var(--border);
             padding: 8px 18px; font-size: 13px;
             box-shadow: 0 -2px 12px -6px rgba(0,0,0,.4); }
  body { padding-bottom: 76px; }
  /* The last few lines of whatever is running, in the machine's own words.
     Fixed height so a chatty log cannot push the page around. */
  #livelog { flex: 1 1 100%; margin: 0; font-family: ui-monospace,
             "SFMono-Regular", Menlo, monospace; font-size: 11px;
             line-height: 1.45; color: var(--dim); max-height: 42px;
             overflow: hidden; white-space: pre-wrap; word-break: break-word; }
  .temp { font-variant-numeric: tabular-nums; }
  .temp.hot { color: var(--warn); font-weight: 600; }
  #livebar .dot { width: 9px; height: 9px; border-radius: 50%;
                  display: inline-block; margin-right: 6px; vertical-align: middle; }
  .dot.on   { background: var(--ok);     box-shadow: 0 0 0 3px color-mix(in srgb, var(--ok) 18%, transparent); }
  .dot.off  { background: var(--bad);    box-shadow: 0 0 0 3px color-mix(in srgb, var(--bad) 18%, transparent); }
  .dot.warn { background: var(--warn);   box-shadow: 0 0 0 3px color-mix(in srgb, var(--warn) 18%, transparent); }
  .dot.busy { background: var(--accent); animation: pulse 1.4s ease-in-out infinite; }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.35; } }
  #livebar .item { white-space: nowrap; }
  .progress { height: 6px; background: var(--border); border-radius: 999px;
              overflow: hidden; min-width: 140px; flex: 1 1 140px; }
  .progress > i { display: block; height: 100%; background: var(--accent);
                  border-radius: 999px; transition: width .4s ease; }

  .whoami { margin-left: auto; font-size: 12px; color: var(--dim); }
  .whoami .role { background: color-mix(in srgb, var(--accent) 16%, transparent);
                  color: var(--accent); padding: 2px 8px; border-radius: 999px;
                  font-weight: 700; margin-left: 6px; }

  .thumbgrid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
               gap: 16px; margin-top: 12px; }
  .thumbcard { background: var(--surface); border: 1px solid var(--border);
               border-radius: var(--radius); overflow: hidden;
               box-shadow: var(--shadow); transition: transform .1s ease; }
  .thumbcard:hover { transform: translateY(-2px); }
  .thumbcard img { width: 100%; display: block; background: var(--bg); }
  .thumbcard .meta { padding: 8px 10px; font-size: 12px; color: var(--dim); }

  @media (max-width: 760px) {
    header { padding: 10px 14px; }
    main { padding: 14px 14px 48px; }
    .navlink { display: inline-block; margin: 6px 12px 0 0; }
    /* On a phone the overflow drops INTO the flow rather than floating over
       it: a 460px panel positioned absolutely on a 390px screen is a panel
       with half of itself off the edge. */
    .navmore { display: block; margin: 10px 0 0; position: static }
    .navmore-body { position: static; min-width: 0; grid-template-columns: 1fr 1fr;
                    margin-top: 8px }
    .navgroup .navlink { padding: 9px 0; }
    .whoami { margin-left: 0; display: block; margin-top: 8px; }
    /* SIX COLUMNS ON A 390px SCREEN, AND TWO OF THEM SAY NOTHING THERE. The
       preview strip is a row of "—" until a run has keyframes, and the niche
       is the same word on every row of a single-channel queue. Dropping them
       leaves Made / Hook / Score / Status, which is the decision, and it fits
       without sideways scrolling. Marked by class rather than by column index
       because `previews` changes how many columns there are, and an nth-child
       rule would silently hide the wrong one. */
    .c-preview, .c-niche { display: none; }
    /* The status bar wrapped to nowhere: white-space:nowrap on a 390px screen
       put the end of every message past the right edge. */
    #livebar { font-size: 12px; gap: 8px; padding: 6px 12px; }
    #livebar .item { white-space: normal; }
    #livelog { display: none; }   /* no room for it, and it is the least of it */
    body { padding-bottom: 92px; }
    /* Tap targets. The review queue is worked from a phone. */
    .btn { padding: 12px 20px; }
    th, td { padding: 12px 10px; }
    /* Approve / Reject / Download stack full width instead of sharing a row
       three ways — on a 390px screen that row gives each of them about a
       thumb's width, and the two that are not Download are irreversible. */
    .actions { flex-direction: column; align-items: stretch; }
    .actions form, .actions .btn { width: 100%; }
    .actions .btn { display: block; text-align: center; }
  }
  .fmt-switch { display:inline-flex; gap:0; margin-left:10px; vertical-align:middle;
                border:1px solid var(--border); border-radius:var(--radius-sm); overflow:hidden }
  .fmt-switch form { margin:0 }
  button.fmt { border:0; background:var(--surface); color:var(--dim);
               padding:5px 11px; font-size:13px; cursor:pointer; font-family:inherit }
  button.fmt:hover { color:var(--text) }
  button.fmt.on { background:var(--accent); color:#fff; font-weight:600 }
  .fmt-badge { margin-left:10px; color:var(--dim); font-size:13px }
  .style-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(230px,1fr));
                gap:16px; margin-top:16px }
  .style-card { border:1px solid var(--border); border-radius:var(--radius);
                box-shadow:var(--shadow); overflow:hidden;
                background:var(--surface); display:flex; flex-direction:column }
  .style-card.on { border-color:var(--accent); box-shadow:0 0 0 2px var(--accent) }
  .style-card img { width:100%; aspect-ratio:16/9; object-fit:cover; display:block;
                    background:var(--bg) }
  .style-card .noimg { width:100%; aspect-ratio:16/9; display:flex; align-items:center;
                       justify-content:center; color:var(--dim); font-size:13px;
                       background:var(--bg); text-align:center; padding:10px }
  .style-card .body { padding:10px 12px; display:flex; flex-direction:column; gap:8px;
                      flex:1 }
  .style-card h4 { margin:0; font-size:15px }
  .style-card p { margin:0; font-size:13px; color:var(--dim); line-height:1.45;
                  flex:1 }
  .style-card .row { display:flex; gap:8px }
</style></head><body>
"""

# Nav entries gated by permission — a partner never sees Settings or System,
# because a link they can only get a 403 from is worse than no link at all.
NAV_ITEMS = [
    ("/create",     "✦ Make a video",                     "view"),
    ("/generate",   "▶ Quick run (no choosing)",          "generate"),
    ("/thumbnails", "🎨 Thumbnails",                      "thumbnail"),
    ("/styles",     "🎨 Style",                           "settings"),
    ("/scout",      "🛰 Scout",                           "view"),
    ("/scripts",    "📝 Choose a script",                 "view"),
    ("/galleries",  "🖼 Choose the pictures",             "view"),
    ("/voice",      "🎙 Choose how it opens",             "view"),
    # "settings" and not "view", to match /styles exactly. Both are rare
    # identity decisions that change every future run, and a page whose every
    # button a viewer would get a 403 from is a page they should not be
    # offered. It is also what keeps Setup empty for a viewer, which is the
    # case the empty-group rule is tested against.
    ("/voices",     "🗣 Narrator",                        "settings"),
    ("/bench",      "🔬 Workflow bench",                  "settings"),
    ("/failures",   "⚠ Failures &amp; rejected attempts", "view"),
    ("/trending",   "🔥 Trending",                        "view"),
    ("/gallery",    "🖼 Gallery",                         "view"),
    ("/measure",    "📊 Measure",                         "view"),
    ("/history",    "🕰 History",                         "view"),
    ("/message",    "📣 Send a message",                  "generate"),
    ("/review",     "⏳ Awaiting review",                 "view"),
    ("/logs",       "📜 Logs",                            "view"),
    ("/system",     "🖥 System",                          "system"),
    ("/settings",   "⚙ Settings",                         "settings"),
]

# SIXTEEN LINKS IS NOT A NAVIGATION, IT IS AN INVENTORY.
#
# Flat, they wrapped to two rows on a laptop and filled an entire phone screen
# before any content appeared — on a dashboard whose review queue is worked
# from a phone. They are not sixteen equal things either; they are four jobs,
# and only three of them are opened daily.
#
# The registry above stays flat and stays the source of truth (four tests
# unpack it, and a page is registered by adding one line to it). This is a
# VIEW of it. The invariant that matters is that every registered page appears
# somewhere here — a page that exists and is unreachable is worse than one
# that was never written, and a test enforces it.
NAV_PRIMARY = ("/create", "/gallery", "/measure")

# GROUPED BY WHAT YOU CAME HERE TO DO, and the four choosing pages had been
# filed under Measure — next to the analytics, because that is where a
# permission technicality put them rather than where anybody would look. They
# are not measurements. They are the four decisions a video is made of, in
# order, and Make is where a person goes to make one.
#
# Nine links under one heading is also not a group, it is a list. Measure holds
# five now, and the pipeline reads as a sequence rather than an alphabet.
#
# /logs moved to Review for the same reason: a log is the record of a run, and
# the question it answers ("what happened to that video") is the one Review
# asks. It also leaves Setup holding nothing a viewer may open, which is what
# keeps the empty-group rule testable with a real case.
NAV_GROUPS = (
    # MAKE IS WHERE YOU START ONE. /create walks all five decisions; the four
    # per-stage pages are not steps any more, they are QUEUES — every pending
    # script across every project, every gallery waiting on a base. Useful, and
    # a different question from "make me a video", which is why seven links
    # under one heading felt wrong before it broke the group-size rule.
    ("Make",    ("/create", "/generate", "/thumbnails")),
    ("Queues",  ("/scout", "/scripts", "/galleries", "/voice")),
    ("Review",  ("/review", "/gallery", "/history", "/failures", "/message")),
    ("Logs",    ("/logs",)),
    ("Measure", ("/measure", "/trending")),
    ("Setup",   ("/styles", "/voices", "/bench", "/system", "/settings")),
)

# ── the four decisions, and which of them is waiting for you ─────────────────
#
# THE PROBLEM A TAB LIST CANNOT SOLVE. The pipeline is sequential — a topic
# becomes three scripts, a script becomes two galleries, a gallery becomes
# three reads — and a nav bar renders that as four equal words with no order
# and no state. So you open /voice, find it empty, and have no way to tell
# whether that means "nothing to do" or "you have not done step three yet".
#
# The bar below is the answer: the same four steps in order, on every page they
# concern, each carrying the number of decisions actually waiting behind it.
# A person can see where they are needed without opening anything.
FLOW_STEPS = (
    ("/scout",     "Topic"),
    ("/scripts",   "Script"),
    ("/galleries", "Pictures"),
    ("/voice",     "Voice"),
)


def _flow_counts() -> dict:
    """How many decisions are waiting at each step.

    Fail-open to zeros and never raises: this renders at the top of six pages,
    and a database hiccup must cost the badges, not the page.
    """
    out = {"/scout": 0, "/scripts": 0, "/galleries": 0, "/voice": 0}
    try:
        import db_manager as dbm
        out["/scout"] = dbm.pending_proposal_count()
        # SETS, NOT ROWS. Three scripts on one topic is ONE decision; counting
        # the cards would say 3 and send someone looking for three topics.
        out["/scripts"] = len({(c["proposal_id"], c["topic"])
                               for c in dbm.candidates(status="pending",
                                                       limit=200)})
        out["/galleries"] = len(dbm.gallery_sets(status="pending", limit=50))
        out["/voice"] = len({t["set_id"] for t in
                             dbm.voice_takes(status="pending", limit=200)})
    except Exception:
        pass
    return out


def _flow_bar(current: str = "") -> str:
    """The four steps in order, with what is waiting at each."""
    counts = _flow_counts()
    cells = []
    for i, (href, label) in enumerate(FLOW_STEPS, start=1):
        n = counts.get(href, 0)
        here = " here" if href == current else ""
        badge = (f'<span class="flow-n">{n}</span>' if n
                 else '<span class="flow-n zero">0</span>')
        cells.append(f'<a class="flow-step{here}" href="{href}">'
                     f'<span class="flow-i">{i}</span>{label}{badge}</a>')
    return f'<nav class="flow" aria-label="the four decisions">{"".join(cells)}</nav>'



# Polls /api/status and rewrites the bar in place. Vanilla JS and inline —
# the dashboard is deliberately self-contained (no CDN, no build step) so it
# keeps working on a phone with no internet beyond the tailnet.
LIVEBAR_JS = """
<script>
(function () {
  var el = document.getElementById('livebar');
  if (!el) return;
  function fmt(s) {
    s = Math.max(0, Math.round(s));
    if (s < 60) return s + 's';
    var m = Math.floor(s / 60);
    if (m < 60) return m + 'm';
    return Math.floor(m / 60) + 'h ' + (m % 60) + 'm';
  }
  function render(d) {
    var bits = [];
    bits.push('<span class="item"><span class="dot on"></span>PC on'
              + ' <span class="muted">(up ' + fmt(d.uptime_seconds) + ')</span></span>');
    bits.push('<span class="item"><span class="dot ' + (d.comfyui ? 'on' : 'off')
              + '"></span>GPU ' + (d.comfyui ? 'ready' : 'offline') + '</span>');
    var active = (d.runs || []).filter(function (r) { return r.running; });
    // GPU TEMPERATURE, when the driver will say. Absent rather than blank
    // when it will not — a field that never fills is noise.
    if (d.gpu_temp_c !== null && d.gpu_temp_c !== undefined) {
      var hot = d.gpu_temp_c >= 80 ? ' hot' : '';
      bits.push('<span class="item temp' + hot + '">GPU '
                + d.gpu_temp_c + '\\u00B0C</span>');
    }
    // THE STAGE A PERSON IS ACTUALLY WAITING ON. `runs` covers main.py's eight
    // steps; drawing a gallery from the wizard is not one of them, so this bar
    // used to say "Idle" through forty minutes of rendering.
    var j = d.job;
    if (j) {
      var jt = j.total
        ? j.done + '/' + j.total + ' ' + (j.label || '')
        : (j.label || 'starting') + '\\u2026';
      var jpct = j.total ? Math.round((j.done / j.total) * 100) : 0;
      bits.push('<span class="item"><span class="dot busy"></span>'
                + '<b>' + (j.title || j.stage) + '</b> \\u2014 ' + jt
                + (j.eta_seconds != null
                   ? ' <span class="muted">(~' + fmt(j.eta_seconds) + ' left)</span>'
                   : '') + '</span>');
      bits.push('<span class="progress"><i style="width:' + jpct + '%"></i></span>');
    }
    if (!active.length && !j) {
      bits.push('<span class="item"><span class="dot on"></span>Idle \\u2014 not making a video</span>');
    } else if (active.length) {
      active.forEach(function (r) {
        var pct = r.total ? Math.round((r.step / r.total) * 100) : 0;
        var cls = r.stale ? 'warn' : 'busy';
        var txt = r.stale
          ? 'stuck? no update for ' + fmt(r.age_seconds)
          : 'step ' + r.step + '/' + r.total + ' \\u2014 ' + (r.label || 'working');
        bits.push('<span class="item"><span class="dot ' + cls + '"></span>'
                  + '<b>' + r.channel + '</b> ' + txt
                  + ' <span class="muted">(' + fmt(r.elapsed_seconds) + ')</span></span>');
        bits.push('<span class="progress"><i style="width:' + pct + '%"></i></span>');
      });
    }
    var q = d.queue || {};
    bits.push('<span class="item"><a class="navlink" style="margin:0" href="/review">'
              + (q.pending || 0) + ' awaiting review</a></span>');
    // The last few lines of whatever is running, in the machine's own words.
    // textContent, not innerHTML: a log line is arbitrary text and a stray
    // angle bracket in a prompt must not become markup.
    if (d.log_tail && d.log_tail.length) {
      bits.push('<pre id="livelog"></pre>');
    }
    el.innerHTML = bits.join('');
    var lg = document.getElementById('livelog');
    if (lg) { lg.textContent = d.log_tail.join('\\n'); }
  }
  function poll() {
    fetch('/api/status', { headers: { 'Accept': 'application/json' } })
      .then(function (r) { return r.ok ? r.json() : Promise.reject(r.status); })
      .then(render)
      .catch(function () {
        el.innerHTML = '<span class="item"><span class="dot off"></span>'
          + 'Cannot reach the PC \\u2014 it may be asleep, off, or off the tailnet.</span>';
      });
  }
  poll();
  // 5s while a run is active is responsive without being chatty; the endpoint
  // is a couple of file checks plus one small query.
  setInterval(poll, 5000);
})();
</script>
"""



def _format_switch() -> str:
    """Short ⇄ Long, in the header, on every page.

    WHY IT LIVES UP HERE and not in Settings. It is not a preference like the
    seed count — it decides the aspect ratio, the script length, the number of
    pictures and how long the GPU is busy, and it is the one thing the owner
    said they wanted to change per video rather than per channel. A setting
    buried three pages deep that changes everything is a setting people forget
    is set, and this dashboard has already been bitten by exactly that with
    SD_CLIPS.

    Shown to whoever may change settings; everyone else sees the CURRENT
    format as plain text, because knowing what the next run will make is
    useful even when you cannot change it.
    """
    import video_format
    current = video_format.name()
    if not auth.can("settings"):
        return (f'<span class="fmt-badge" title="the shape of the next run">'
                f'{_esc(video_format.profile()["label"])}</span>')
    out = ['<span class="fmt-switch" title="the shape of the next run">']
    for fid in ("short", "long"):
        prof = video_format.PROFILES[fid]
        on = " on" if fid == current else ""
        label = "Shorts" if fid == "short" else "Long-form"
        out.append(
            f'<form method="post" action="/format" style="display:inline">'
            f'<input type="hidden" name="format" value="{fid}">'
            f'<button class="fmt{on}" type="submit" '
            f'title="{_esc(prof["width"])}×{_esc(prof["height"])}, '
            f'{_esc(prof["words_min"])}–{_esc(prof["words_max"])} words">'
            f'{label}</button></form>')
    out.append("</span>")
    return "".join(out)


def _head() -> str:
    """Page header with the nav this user may actually use.

    Replaces the old module-level PAGE_HEAD constant: nav now depends on who
    is asking, which a constant can't express. Hiding a link is cosmetic —
    every route enforces its own permission besides.
    """
    allowed = {href: label for href, label, perm in NAV_ITEMS if auth.can(perm)}

    def _link(href: str) -> str:
        return (f'<a class="navlink" href="{href}">{allowed[href]}</a>'
                if href in allowed else "")

    primary = "".join(_link(h) for h in NAV_PRIMARY)

    # <details> and not a script: this dashboard is deliberately
    # self-contained with no build step, and a menu that needs JavaScript to
    # open is a menu that does not open when the JavaScript fails. It is also
    # keyboard-reachable and screen-reader-sane for free.
    #
    # A group whose every link this user lacks permission for is omitted
    # entirely rather than rendered empty — the same reason a single link they
    # would only get a 403 from is hidden.
    sections = []
    for title, hrefs in NAV_GROUPS:
        rows = "".join(_link(h) for h in hrefs if h in allowed)
        if rows:
            sections.append(f'<div class="navgroup"><span class="navgroup-t">'
                            f'{title}</span>{rows}</div>')
    more = (f'<details class="navmore"><summary>More</summary>'
            f'<div class="navmore-body">{"".join(sections)}</div></details>'
            if sections else "")
    links = primary + more + "\n"
    user = getattr(g, "rufus_user", None) or {}
    who = ""
    if user:
        who = (f'<span class="whoami">{_esc(user.get("name", "?"))}'
               f'<span class="role">{_esc(user.get("role", "?"))}</span>'
               f' · <a class="navlink" href="/logout" style="margin-left:6px">sign out</a></span>')
    return (PAGE_STYLE + '<header><a href="/"><h1>🎬 ThePaperTrails</h1></a>\n'
            + links + _format_switch() + who + "</header>\n<main>\n"
            # PINNED TO THE BOTTOM, ON EVERY PAGE. It used to sit inside <main>
            # at the top, which meant it scrolled away exactly when a long page
            # was being read — and the one thing it answers ("is the machine
            # doing anything, and how far in") is the thing you want while
            # looking at something else. The sketch put it along the bottom for
            # that reason.
            + '<div id="livebar"><span class="item">'
              '<span class="dot warn"></span>checking…</span></div>\n'
            + LIVEBAR_JS)


# Vanilla, inline, no build step — the same reason LIVEBAR_JS is. Two small
# behaviours, both of which degrade to "the page works as it did" if the script
# never runs.
INTERACT_JS = """
<script>
(function () {
  // 1. A BUTTON THAT LOOKS DEAD GETS CLICKED TWICE. Approve, Draw them,
  //    Re-cut and Regen all hand off to something that takes real time, and
  //    until the page navigated there was no evidence the click had landed.
  //    On Approve a second click is a second upload attempt.
  document.addEventListener('submit', function (e) {
    var form = e.target;
    if (!form || form.dataset.noBusy) return;
    var btn = form.querySelector('button[type=submit], button:not([type])');
    if (!btn || btn.disabled) return;
    // Let the browser send the form first: a button disabled synchronously in
    // the submit handler is not included in the POST body, which silently
    // drops any name/value it carried.
    setTimeout(function () {
      btn.classList.add('working');
      btn.disabled = true;
    }, 0);
  }, true);

  // 2. Filter a long table without a round trip. Added only where a table is
  //    long enough to be worth it, so short pages gain nothing to ignore.
  document.querySelectorAll('table').forEach(function (table) {
    var rows = table.tBodies.length ? table.tBodies[0].rows : table.rows;
    if (rows.length < 12) return;
    var box = document.createElement('input');
    box.className = 'tablefilter';
    box.type = 'search';
    box.placeholder = 'Filter these ' + (rows.length - 1) + ' rows\u2026';
    // Outside the scroll container, or the filter box scrolls away sideways
    // with the table it filters.
    var anchor = table.parentNode.classList.contains('tablewrap')
      ? table.parentNode : table;
    anchor.parentNode.insertBefore(box, anchor);
    box.addEventListener('input', function () {
      var q = box.value.toLowerCase();
      for (var i = 1; i < rows.length; i++) {
        rows[i].style.display =
          (!q || rows[i].textContent.toLowerCase().indexOf(q) !== -1) ? '' : 'none';
      }
    });
  });
})();
</script>
"""

PAGE_TAIL = "</main>" + INTERACT_JS + "</body></html>"


# ── Routes ─────────────────────────────────────────────────────────────────────

def _status_badge(status: str) -> str:
    if status == "approved":
        return '<span class="badge ok">approved</span>'
    if status == "rejected":
        return '<span class="badge held">rejected</span>'
    return '<span class="badge pending">pending</span>'


def _run_keyframes(run_id: str | None, limit: int = 4) -> list[str]:
    """First few keyframe filenames for a run, for an inline preview strip.

    Cheap (one directory glob, no image decoding) because the queue calls it
    once per row."""
    if not run_id:
        return []
    folder = DEBUG_ROOT / run_id
    try:
        if not folder.is_dir():
            return []
        return [p.name for p in sorted(folder.glob("[0-9]*.png"))[:limit]]
    except OSError:
        return []


# A STABLE COLOUR PER PERSON, from the name itself. Two people share this
# channel, and a name printed in the same grey as everything else is a word you
# have to read; a name that is always the same colour is one you recognise
# without reading. Derived from the string rather than assigned, so adding a
# third person needs no table and no migration — and so the colour a name has
# is the same on every page it appears on.
_BY_HUES = ("var(--make)", "var(--measure)", "var(--review)", "var(--accent)",
            "var(--ok)", "var(--system)")


def _by_hue(name: str) -> str:
    return _BY_HUES[sum(ord(ch) for ch in name) % len(_BY_HUES)]


def _by_badge(name: str | None) -> str:
    """Who decided, as a small tinted chip. An em dash when nobody is recorded.

    Blank means "not recorded" — a decision made before this column existed, or
    with auth off — and it says so with a dash rather than inventing a name,
    because an anonymous row and a row somebody actually owns must not look
    alike.
    """
    name = (name or "").strip()
    if not name:
        return '<span class="muted">—</span>'
    return (f'<span class="by" style="--who: {_by_hue(name)}">'
            f'{_esc(name)}</span>')


def _videos_table(videos: list[dict], *, previews: bool = False) -> str:
    """The queue table. `previews` adds a strip of that run's actual keyframes
    to each row — approving a video is a judgement about how it LOOKS, and
    making that call previously meant opening every row one at a time."""
    if not videos:
        return "<p class='muted'>Nothing here.</p>"
    rows = ""
    for v in videos:
        score = v["score"]
        score_html = (f'<span style="color:{_score_color(score)};font-weight:700">{score}/10</span>'
                      if score is not None else "—")
        title = _esc((v["title"] or v["script_hook"] or "")[:70])
        preview_cell = ""
        if previews:
            frames = _run_keyframes(v.get("run_id"))
            if frames:
                imgs = "".join(
                    f'<img src="/debug/{_esc(v["run_id"])}/{_urlquote(f)}?w=120" loading="lazy" '
                    f'alt="" style="width:38px;height:66px;object-fit:cover;'
                    f'border-radius:4px;margin-right:3px">'
                    for f in frames)
                preview_cell = (f'<td class="c-preview"><a class="row-link" '
                                f'href="/video/{v["id"]}" '
                                f'style="display:flex">{imgs}</a></td>')
            else:
                preview_cell = '<td class="c-preview"><span class="muted">—</span></td>'
        went_out = (f'<br><span class="muted">out '
                    f'{_esc(str(v["uploaded_at"]).split(" ")[-1][:5])}</span>'
                    if v.get("uploaded_at") and " " in str(v["uploaded_at"])
                    else "")
        rows += (f'<tr>{preview_cell}<td><a class="row-link" href="/video/{v["id"]}">'
                 f'{_when_cell(v.get("created_at") or v.get("upload_date"))}'
                 f'{went_out}</a></td>'
                 f'<td class="c-niche"><a class="row-link" href="/video/{v["id"]}">{_esc(v["niche"])}</a></td>'
                 f'<td><a class="row-link" href="/video/{v["id"]}">{title}</a></td>'
                 f'<td>{score_html}</td><td>{_status_badge(v["upload_status"])}</td>'
                 f'<td class="c-by">{_by_badge(v.get("decided_by"))}</td></tr>\n')
    preview_th = '<th class="c-preview">Preview</th>' if previews else ""
    return (f'<div class="tablewrap"><table><tr>{preview_th}<th>Made</th>'
            f'<th class="c-niche">Niche</th><th>Hook / Title</th>'
            f'<th>Score</th><th>Status</th><th class="c-by">By</th></tr>'
            f"{rows}</table></div>")


def _msg_banner() -> str:
    # `msg` is a synonym for `ok`. Two spellings exist because half the
    # redirects in this file already used one and half the newer ones reach for
    # the other, and a success banner that silently does not appear is a worse
    # outcome than accepting both.
    ok_msg  = request.args.get("ok") or request.args.get("msg")
    err_msg = request.args.get("error")
    if ok_msg:
        return f'<div class="msg ok">{_esc(ok_msg)}</div>'
    if err_msg:
        return f'<div class="msg error">{_esc(err_msg)}</div>'
    return ""


# ── a home page rather than a top-of-a-long-scroll ───────────────────────────
#
# The index was the review queue with everything else stacked underneath it: a
# flow bar, a banner, advice, the queue, a topic form, filters, stat cards, a
# sparkline and two tables. Every block earns its place and none of them
# answers the question somebody opening the dashboard actually has, which is
# "where do I go".
#
# So the page opens with the places you can go, as tiles: what each one is for,
# and the number waiting behind it where there is one. The detail blocks stay
# exactly where they were, below.
#
# COLOUR CARRIES THE GROUP, which the palette was already built for — the nav
# has had --make, --review, --measure and --system since it was grouped, and
# nothing used them at this size. A tile you can find by its colour before you
# have read it is the whole reason to have four hues rather than one accent.
#
# ACCESSIBLE MEANS THE ORDINARY THINGS. Real anchors, so they tab and they open
# in a new tab if somebody wants that; the count in the text rather than only
# in a coloured dot; aria-labels that read as a sentence; and contrast that
# survives the light theme, where these hues go weak against white and the
# palette already compensates.

_HOME_TILES = (
    # (href, title, what it is for, group, count key)
    #
    # FOUR, FROM THE OWNER'S OWN SKETCH. This was ten — every page in the nav,
    # tiled — which put a launcher above a launcher and made the home page a
    # second copy of the menu. The sketch asks a narrower question: what are
    # the four things worth putting on the front of this, and the answer is one
    # box to start work, one to find a subject, one to be told what to fix, and
    # one to see what went out. Everything else is a nav click away and was
    # never worth the vertical space.
    ("/create",   "Make a video",     "Five decisions, one at a time",
     "make",    "create"),
    ("/trending", "Trending topics",  "Rising on Google, money and history",
     "make",    None),
    ("/measure",  "What to change",   "The one thing worth fixing next",
     "measure", None),
    ("/gallery",  "Recent uploads",   "The last five that went out",
     "review",  None),
)


def _home_tile_counts(stats: dict) -> dict:
    counts = {"pending": stats.get("pending", 0)}
    try:
        flow = _flow_counts()
        counts.update(flow)
        import db_manager as dbm
        counts["create"] = len(dbm.projects(status="open", limit=20))
    except Exception:
        pass
    return counts


def _home_tiles(stats: dict) -> str:
    allowed = {href for href, _l, perm in NAV_ITEMS if auth.can(perm)}
    counts = _home_tile_counts(stats)
    cells = []
    for href, title, blurb, group, key in _HOME_TILES:
        if href != "/" and href not in allowed:
            continue
        # THE MEASURE TILE SAYS THE THING, not the category. A box labelled
        # "what to change" that does not say what to change is a signpost to a
        # signpost — and the front page used to carry the actual line, which is
        # the one piece of that block worth keeping when the block goes.
        if href == "/measure":
            try:
                items, ready = _advice_now()
                if items:
                    blurb = items[0]["title"]
                    if len(items) > 1:
                        blurb += f" · and {len(items) - 1} more"
            except Exception as e:
                print(f"[dashboard] advice tile unavailable: {e}")
        n = counts.get(key) if key else None
        badge = ""
        label = title
        if n:
            badge = f'<span class="tile-n">{n}</span>'
            label = f"{title}, {n} waiting"
        target = "#review" if href == "/" else href
        cells.append(
            f'<a class="tile t-{group}" href="{target}" '
            f'aria-label="{_esc(label)}. {_esc(blurb)}">'
            f'<span class="tile-t">{_esc(title)}{badge}</span>'
            f'<span class="tile-b">{_esc(blurb)}</span></a>')
    if not cells:
        return ""
    return f'<div class="tiles">{"".join(cells)}</div>'


def _greeting() -> str:
    """"Good evening, Daniel" — the hour and the name, from the sketch.

    The name comes from whoever is signed in, which is the point of it now
    that two people share this. Nobody signed in gets the greeting without a
    name rather than a guess or a placeholder.
    """
    hour = time.localtime().tm_hour
    part = ("morning" if 5 <= hour < 12 else
            "afternoon" if 12 <= hour < 18 else "evening")
    who = _whoami()
    return f"Good {part}, {_esc(who)}" if who else f"Good {part}"


def _waiting_line() -> str:
    """What actually needs a person, in one line, each part a link.

    THE BAND WAS SAYING THE TIME OF DAY AND NOTHING ELSE. It is the largest
    thing on the page; spending it on a greeting alone wastes the one place
    the eye lands first. So the greeting keeps the left of it and this fills
    the rest — and when there is genuinely nothing waiting it says so, which
    is also worth knowing at a glance.
    """
    bits = []
    try:
        flow = _flow_counts()
        # The plural is carried per word rather than by adding "s", because
        # "gallery" does not take one and the page said "2 gallerys".
        for href, one, many in (("/scripts", "script", "scripts"),
                                ("/galleries", "gallery", "galleries"),
                                ("/voice", "read", "reads")):
            n = flow.get(href, 0)
            if n:
                bits.append(f'<a href="{href}">{n} {one if n == 1 else many}</a>')
    except Exception:
        pass
    try:
        import db_manager as dbm
        n = dbm.open_note_count()
        if n:
            bits.append(f'<a href="/message">{n} note{"" if n == 1 else "s"}</a>')
    except Exception:
        pass
    try:
        pend = _stats().get("pending", 0)
        if pend:
            bits.append(f'<a href="/review">{pend} to review</a>')
    except Exception:
        pass
    if not bits:
        return '<span class="hi-q">Nothing is waiting on you.</span>'
    return '<span class="hi-q">' + " · ".join(bits) + " waiting</span>"


@app.route("/")
def index():
    channel = request.args.get("channel") or None
    stats   = _stats(channel=channel)
    videos  = _recent_videos(limit=60, channel=channel)
    # "5 Recent uploaded videos" in the sketch means UPLOADED. `videos` is
    # everything made, and 86 of 111 of those never left the queue — a list
    # titled "recent uploads" that is mostly things which never went out is
    # the kind of wrong that looks right.
    recent_out = [v for v in videos if v.get("youtube_id")]
    channels = _channels()

    # oldest -> newest for the trend line
    scored = [v["score"] for v in reversed(videos) if v["score"] is not None]

    filt_html = ""
    if channels:
        links = [f'<a href="/">all channels</a>']
        for ch in channels:
            links.append(f'<a href="/?channel={_esc(ch)}">{_esc(ch)}</a>')
        filt_html = f'<div class="filters">{"".join(links)}</div>'

    cards = f"""
    <div class="cards">
      <a class="card tone t-pending" href="/review" style="text-decoration:none;color:inherit"><div class="num">{stats['pending']}</div><div class="label">awaiting review</div></a>
      <div class="card tone t-ok"><div class="num">{stats['uploaded']}</div><div class="label">approved / uploaded</div></div>
      <div class="card tone t-bad"><div class="num">{stats['rejected']}</div><div class="label">rejected</div></div>
      <div class="card tone t-info"><div class="num">{stats['avg_score']}</div><div class="label">avg score</div></div>
    </div>
    """

    # MINIMAL, TO THE SKETCH. This page had grown ten tiles, four stat cards,
    # a failure panel, an advice card, a topic form, a channel filter, a
    # sparkline, a sixty-row review table and a rejection list — eleven blocks,
    # each one added because it was worth knowing, together adding up to a page
    # nobody could read. The sketch asks for four: who you are and what needs
    # you, four ways in, the numbers, and what went out.
    #
    # Everything removed still exists. The review queue moved to /review rather
    # than being deleted, the failures to /failures, the trend and the
    # rejections to /measure. A front page is a place to start from, not the
    # place everything has to be.
    body = f"""
    <section class="hi">
      <h1 class="hi-h">{_greeting()}</h1>
      {_waiting_line()}
    </section>
    {_msg_banner()}
    {_home_tiles(stats)}
    {filt_html}
    <h2 class="sec s-measure">Analytics and stats</h2>
    {cards}
    {_sparkline_svg(scored)}
    <h2 class="sec s-review">Last five out</h2>
    {_videos_table(recent_out[:5])}
    """
    return _head() + body + PAGE_TAIL


@app.route("/review")
def review_queue():
    """Everything waiting on an approve or a reject.

    Lifted off the front page, where sixty rows with keyframe strips were the
    tallest thing on it by an order of magnitude. It is the most important
    list in the dashboard and it earns its own page — the home page carries
    the count and a link, which is what the count is for.
    """
    auth.require("view")
    channel = request.args.get("channel") or None
    pending = _recent_videos(limit=200, channel=channel, status="pending")
    rejects = _top_rejections(channel=channel)

    if rejects:
        items = "".join(f"<li>{_esc(r['reason'])} &mdash; <b>{r['count']}&times;</b></li>"
                        for r in rejects)
        reject_html = (f'<details style="margin-top:26px"><summary>Most common '
                       f'script rejections</summary><ul>{items}</ul></details>')
    else:
        reject_html = ""

    if pending:
        table = _videos_table(pending, previews=True)
    else:
        table = ('<p class="muted">Nothing is waiting on you. Queue a topic on '
                 '<a href="/create">Make a video</a>, or leave it to the '
                 'schedule.</p>')

    body = f"""
    <a class="back" href="/">&larr; back</a>
    <h2 style="margin-top:14px">Awaiting your review ({len(pending)})</h2>
    {_msg_banner()}
    {_failure_notice()}
    {table}
    {reject_html}
    """
    return _head() + body + PAGE_TAIL


@app.route("/failures")
def failures():
    """Every failure the automation produced, not just the successes — a
    crashed run has NO row in `videos` at all (it never reached Step 6), so
    without this page it's invisible everywhere else in the app."""
    channel = request.args.get("channel") or None
    niche   = request.args.get("niche") or None
    phase   = request.args.get("phase") or None

    orphans = _orphaned_debug_runs()
    rejects = _rejected_attempts(channel=channel, niche=niche, phase=phase)
    categories = _rejection_category_counts(channel=channel)

    niche_links = "".join(
        f'<a href="/failures?niche={_esc(n)}">{_esc(n)}</a> ' for n in _distinct("niche"))
    phase_links = "".join(
        f'<a href="/failures?phase={_esc(p)}">{_esc(p)}</a> ' for p in _distinct("phase"))
    filt_html = (f'<div class="filters"><a href="/failures">all niches</a> {niche_links}'
                f'<br><a href="/failures">all phases</a> {phase_links}</div>')

    orphan_html = "<p class='muted'>No crashed/incomplete runs found — every RUFUS_DEBUG run reached the database.</p>"
    if orphans:
        blocks = ""
        for o in orphans:
            file_links = "".join(
                f'<a href="/debug/{_esc(o["run_id"])}/{_esc(f)}" target="_blank">{_esc(f)}</a> '
                for f in o["files"]
            ) or "<span class='muted'>(no files saved)</span>"
            preview = f"<div class='muted' style='margin:6px 0'>{_esc(o['preview'])}</div>" if o["preview"] else ""
            blocks += (f'<div class="orphan"><b>{_esc(o["run_id"])}</b> '
                      f'<span class="muted">· {_fmt_ts(o["mtime"])}</span>'
                      f'{preview}<div class="assets">{file_links}</div></div>\n')
        orphan_html = blocks

    reject_html = "<p class='muted'>No rejected attempts recorded.</p>"
    if rejects:
        rows = ""
        for r in rejects:
            preview = _esc((r["body"] or r["hook"] or "")[:90])
            cat = _esc(_categorize_rejection(r["rejected_reason"], r["phase"]))
            rows += (f"<tr><td class='muted'>{_esc(r['ts'])}</td>"
                     f"<td>{_esc(r['niche'])}</td><td>{_esc(r['phase'])}</td>"
                     f"<td><span class='badge pending'>{cat}</span></td>"
                     f"<td>{_esc(r['rejected_reason'])}</td><td>{preview}</td></tr>\n")
        reject_html = (f"<table><tr><th>When</th><th>Niche</th><th>Phase</th>"
                       f"<th>Category</th><th>Reason</th><th>Preview</th></tr>{rows}</table>")

    category_html = "<p class='muted'>Not enough rejected attempts yet to show a breakdown.</p>"
    if categories:
        bars = ""
        for c in categories:
            bars += (f"<div style='margin:6px 0'>"
                     f"<span style='display:inline-block;width:130px'>{_esc(c['category'])}</span>"
                     f"<span style='display:inline-block;width:200px;background:var(--border);"
                     f"border-radius:4px;overflow:hidden;vertical-align:middle'>"
                     f"<span style='display:block;height:10px;width:{c['pct']}%;"
                     f"background:var(--accent)'></span></span> "
                     f"<b>{c['count']}</b> <span class='muted'>({c['pct']}%)</span></div>\n")
        category_html = bars

    body = f"""
    <a class="back" href="/">← back</a>
    <h2 style="margin-top:14px">Crashed / incomplete runs ({len(orphans)})</h2>
    <p class="muted">Debug folders with no matching database row — the run
       started (RUFUS_DEBUG was on) but never finished (Step 4/5 failure,
       crash, or a stopped process).</p>
    {orphan_html}
    <h2>Bottleneck breakdown (all-time)</h2>
    <p class="muted">Every rejected attempt, ever, grouped into a fixed
       taxonomy instead of counted by exact wording — this is what actually
       answers "which stage of the pipeline is the bottleneck" once there's
       enough volume.</p>
    {category_html}
    <h2>Rejected script attempts</h2>
    {filt_html}
    {reject_html}
    """
    return _head() + body + PAGE_TAIL


# ── one page for one question ────────────────────────────────────────────────
#
# FOUR PAGES ASKING VARIATIONS OF "WHAT DOES THE DATA SAY". Tracking asks
# whether the loop is even closed, Performance asks whether the score predicts
# anything, Insights asks what keeps going wrong, Advice asks what to change.
# Each is a good page. Four tabs is four places to look for one answer, and the
# owner said so plainly: the info is good but not organised.
#
# So they compose into one, in the order a person actually asks — what should I
# do, what keeps breaking, do the numbers agree, is there any data at all. The
# old routes still exist and redirect to the matching section, because a
# bookmark that 404s is a worse tidy-up than four tabs.
_BACK_LINKS = ('<a class="back" href="/">← back</a>',
               '<a class="back" href="/">&larr; back</a>')


def _section(anchor: str, title: str, body: str) -> str:
    """One former page as a section of the merged one.

    The back link is stripped rather than edited out of each body: these four
    functions still each own their own markup, and reaching into all of them to
    delete the same line four times is how the next person ends up with three
    of them fixed.
    """
    for link in _BACK_LINKS:
        body = body.replace(link, "")
    return (f'<section id="{anchor}" style="margin-top:26px">'
            f'<h2 style="margin-top:0">{title}</h2>{body}</section>')


_MEASURE_SECTIONS = (
    ("what-to-change", "What to change", "_advice_body"),
    ("what-goes-wrong", "What keeps going wrong", "_insights_body"),
    ("does-score-predict", "Does the score predict anything", "_performance_body"),
    ("is-the-loop-closed", "Is the loop closed", "_tracking_body"),
    ("links-that-collide", "Links claimed twice", "_duplicate_links_body"),
)


def _duplicate_links_body() -> str:
    """Videos claiming a YouTube link another video also claims.

    WHY THIS EARNS A SECTION. Analytics joins view counts on youtube_id, so a
    link recorded against two videos gives both the numbers of whichever one
    really owns it. In this channel six rows carried kGVAHaObJ38 and all six
    were credited with a seventh video's views — the same count, the same
    likes, a watch percentage of zero. Zero watch time is fatal to the
    engagement score, so all six sorted to the bottom and every one of them
    was written into losing_hooks and fed back into the hook prompt.

    feedback_analyzer now refuses to score them, which stops the bleeding but
    does not clean the wound: the rows are still wrong and still missing from
    the learning. This is where they get fixed.
    """
    import db_manager as dbm
    dupes = dbm.duplicate_youtube_ids()
    if not dupes:
        return ('<p class="muted">Every YouTube link belongs to exactly one '
                'video. Nothing to fix.</p>')

    blocks = ""
    for d in dupes:
        rows = ""
        for vid in d["video_ids"]:
            v = _video_detail(vid) or {}
            title = _esc((v.get("title") or v.get("script_hook") or "")[:70])
            went = v.get("uploaded_at")
            # The one with an upload timestamp is the one that really went out.
            mark = ('<span class="badge ok">has an upload time</span>'
                    if went else '<span class="badge held">never uploaded</span>')
            clear = ""
            if auth.can("approve"):
                clear = (f'<form method="post" action="/video/{vid}/unlink" '
                         f'style="display:inline" onsubmit="return confirm('
                         f'\'Take the link off video {vid}? It goes back to '
                         f'pending.\');">'
                         f'<button class="btn reject" type="submit" '
                         f'style="padding:4px 10px">not this one</button></form>')
            rows += (f'<tr><td><a class="row-link" href="/video/{vid}">'
                     f'#{vid}</a></td><td>{title}</td><td>{mark}</td>'
                     f'<td>{clear}</td></tr>')
        blocks += (f'<p class="muted" style="margin-top:16px"><code>'
                   f'{_esc(d["youtube_id"])}</code> is claimed by '
                   f'<b>{d["count"]}</b> videos. Only one of them is really '
                   f'that link; the rest were never published and are being '
                   f'credited with its views.</p>'
                   f'<div class="tablewrap"><table><tr><th>Video</th>'
                   f'<th>Hook / Title</th><th>Evidence</th><th></th></tr>'
                   f'{rows}</table></div>')
    return (blocks + '<p class="muted" style="margin-top:14px">Clearing a link '
            'puts that video back in the review queue and returns it to the '
            'learning. Its fetched metrics are kept &mdash; they are a record '
            'of what happened, not a claim about this video.</p>')


@app.route("/video/<int:video_id>/unlink", methods=["POST"])
def unlink_video(video_id: int):
    """Take a wrong YouTube link off a video."""
    auth.require("approve")
    if not db_manager.clear_youtube_id(video_id, by=_whoami()):
        return _redirect_detail(video_id, error="that video has no link on it")
    return _redirect_detail(video_id, ok=(
        "link removed — back in the review queue, and back in the learning "
        "on the next analyser run"))


@app.route("/measure")
def measure_page():
    """Everything the numbers say, in the order a person asks it.

    Fail-open per section: one that raises is reported in place and the other
    three still render. Four questions behind one link is only an improvement
    if a single broken query cannot take all four down with it — which is
    exactly what merging them would otherwise buy.
    """
    auth.require("view")
    jump = " · ".join(f'<a href="#{a}">{t}</a>' for a, t, _fn in _MEASURE_SECTIONS)
    parts = []
    for anchor_id, title, fn_name in _MEASURE_SECTIONS:
        try:
            parts.append(_section(anchor_id, title, globals()[fn_name]()))
        except Exception as e:
            parts.append(_section(anchor_id, title,
                                  f'<div class="msg error">{_esc(str(e))}</div>'))
    body = f"""
    <a class="back" href="/">← back</a>
    <h2 style="margin-top:14px">Measure</h2>
    {_msg_banner()}
    <p class="muted">{jump}</p>
    {"".join(parts)}
    """
    return _head() + body + PAGE_TAIL


# The old links, kept alive. A tidy-up that breaks every bookmark and every
# link in a log line is not a tidy-up.
@app.route("/advice")
def advice_page():
    return redirect("/measure#what-to-change")


@app.route("/insights")
def insights_page():
    return redirect("/measure#what-goes-wrong")


@app.route("/performance")
def performance():
    return redirect("/measure#does-score-predict")


@app.route("/tracking")
def tracking_page():
    return redirect("/measure#is-the-loop-closed")


def _performance_body():
    """Score vs real audience performance — does the internal 1-10 gate
    actually predict views/watch%? analytics_fetcher.py already fetches this
    data and report.py's correlate() already computes this same signal for
    the log digest; this is that signal in the browser, since dashboard.py
    never queried the `metrics` table before this route existed."""
    channel = request.args.get("channel") or None
    rows = _performance_rows(channel=channel)

    channel_links = "".join(
        f'<a href="/performance?channel={_esc(c)}">{_esc(c)}</a> '
        for c in sorted({r["channel"] for r in rows if r["channel"]}))
    filt_html = f'<div class="filters"><a href="/performance">all channels</a> {channel_links}</div>'

    with_metrics = [r for r in rows if r["views"] is not None]
    correlation_html = (f"<p class='muted'>Need ≥{MIN_VIDEOS_FOR_CORRELATION} "
                        f"videos with metrics to correlate (have {len(with_metrics)}).</p>")
    if len(with_metrics) >= MIN_VIDEOS_FOR_CORRELATION:
        bars = ""
        for b in _score_vs_views(with_metrics):
            bars += (f"<div style='margin:6px 0'>"
                     f"<span style='display:inline-block;width:60px'>{b['score']}/10</span>"
                     f"<b>{b['avg_views']}</b> <span class='muted'>avg views "
                     f"(n={b['n']})</span></div>\n")
        correlation_html = bars

    table_html = "<p class='muted'>No uploaded videos in the last 90 days.</p>"
    if rows:
        trs = ""
        for r in rows:
            views = r["views"] if r["views"] is not None else "—"
            watch = f"{r['watch_pct']:.0f}%" if r["watch_pct"] is not None else "—"
            likes = r["likes"] if r["likes"] is not None else "—"
            link = (f'<a href="/video/{r["id"]}">{_esc(r["title"] or "(untitled)")}</a>'
                    if r["id"] else _esc(r["title"] or ""))
            trs += (f"<tr><td class='muted'>{_esc(r['upload_date'] or '')}</td>"
                   f"<td>{_esc(r['niche'] or '')}</td><td>{link}</td>"
                   f"<td>{r['score'] if r['score'] is not None else '—'}/10</td>"
                   f"<td>{views}</td><td>{watch}</td><td>{likes}</td></tr>\n")
        table_html = (f"<table><tr><th>Date</th><th>Niche</th><th>Title</th>"
                     f"<th>Score</th><th>Views</th><th>Watch%</th><th>Likes</th></tr>{trs}</table>")

    body = f"""
    <a class="back" href="/">← back</a>
    <h2 style="margin-top:14px">Does the score predict real performance?</h2>
    <p class="muted">Average views per internal score bucket — pulled from
       the same YouTube Analytics data analytics_fetcher.py already collects.</p>
    {correlation_html}
    <h2>Videos (last 90 days)</h2>
    {filt_html}
    {table_html}
    """
    return body


@app.route("/system")
def system_status():
    """Status + process control for THIS PC's automation: is ComfyUI up, is
    a run in progress per channel, launch a run, cancel one this dashboard
    started. Loopback-only binding is the primary guard (see module
    docstring); /system/run and /system/cancel additionally self-check
    remote_addr as defense in depth."""
    auth.require("system")
    channels = _channels()
    comfy_up = _comfyui_reachable()

    rows = ""
    for cid in channels:
        running = _run_in_progress(cid)
        can_cancel = running and _LAUNCHED.get(cid, _LAUNCHED.get("default"))
        can_cancel = bool(can_cancel and can_cancel.poll() is None)
        status_badge = ("<span class='badge pending'>running</span>" if running
                        else "<span class='badge approved'>idle</span>")
        cancel_btn = (f'<form method="post" action="/system/cancel" style="display:inline">'
                     f'<input type="hidden" name="channel" value="{_esc(cid)}">'
                     f'<button type="submit">Cancel</button></form>' if can_cancel else "")
        rows += (f"<tr><td>{_esc(cid)}</td><td>{status_badge}</td><td>{cancel_btn}</td></tr>\n")
    channels_html = (f"<table><tr><th>Channel</th><th>Status</th><th></th></tr>{rows}</table>"
                     if rows else "<p class='muted'>No channels configured.</p>")

    comfy_badge = ("<span class='badge approved'>reachable</span>" if comfy_up
                  else "<span class='badge pending'>not reachable</span>")

    channel_options = "".join(f'<option value="{_esc(c)}">{_esc(c)}</option>' for c in channels)

    body = f"""
    <a class="back" href="/">← back</a>
    <h2 style="margin-top:14px">ComfyUI</h2>
    <p>{comfy_badge} <span class="muted">({_esc(_comfy_host())})</span></p>
    <h2>Channels</h2>
    {channels_html}
    <h2>Run a video now</h2>
    <form method="post" action="/system/run" style="display:flex;gap:10px;flex-wrap:wrap;align-items:flex-end">
      {f'''<div><label for="sys-channel">Channel</label>
        <select class="field" style="margin:6px 0 0" id="sys-channel" name="channel">
          <option value="">(default)</option>{channel_options}
        </select></div>''' if channels else ""}
      <div><label for="sys-niche">Niche override</label>
        <input class="field" style="margin:6px 0 0" type="text" id="sys-niche" name="niche"
               placeholder="(optional)"></div>
      <button class="btn save" type="submit" style="height:38px">Start</button>
    </form>
    <p class="muted">Cancelling only works for a run started from this page
       — a Task Scheduler run or a manual run.bat run has no process handle
       here to stop. If one of those actually crashed rather than just
       running long, its channel still shows "running" until its .lock file
       is removed — check /failures for a crashed/orphaned run before
       deleting a lock file by hand.</p>
    """
    return _head() + body + PAGE_TAIL


def _available_niches() -> list[str]:
    """Niche ids from config/niches.json, or [] if unreadable. A dropdown
    built from this can't offer a niche that doesn't exist — unlike the free-
    text field it replaces, which silently no-ops on a typo (the pipeline
    falls back to the schedule's default niche with no error shown here)."""
    try:
        data = json.loads((ROOT / "config" / "niches.json").read_text(encoding="utf-8"))
        return list(data.get("niches", {}).keys())
    except (OSError, json.JSONDecodeError, KeyError):
        return []


@app.route("/generate")
def generate_page():
    """The partner-facing entry point: describe a video, start it, watch it run.

    Separate from /system on purpose. /system is process control for the
    machine (kill a run, inspect ComfyUI) and stays owner-only; this page is
    just "make me a video," which is exactly the slice a collaborator needs
    and the only slice they should have.

    Niche is a dropdown (not free text) and a "pick a look" gallery is
    embedded directly here, rather than requiring a trip to /thumbnails and
    back — a collaborator without shell/System access previously had no way
    to build a genuinely CUSTOM video (a specific niche, a specific visual
    style) beyond typing a topic string into a box and hoping.
    """
    auth.require("generate")
    channels = _channels()
    channel_options = "".join(f'<option value="{_esc(c)}">{_esc(c)}</option>' for c in channels)
    running = [c for c in channels if _run_in_progress(c)]
    comfy_up = _comfyui_reachable()

    niches = _available_niches()
    niche_options = "".join(f'<option value="{_esc(n)}">{_esc(n)}</option>' for n in niches)

    status = ("<p class='muted'>Nothing running right now.</p>" if not running else
              "<p>" + " ".join(f"<span class='badge pending'>{_esc(c)} running</span>"
                               for c in running) + "</p>")
    gpu_warn = ("" if comfy_up else
                "<div class='msg error'>ComfyUI is not reachable — a run started "
                "now will fall back to stock footage instead of GPU stills.</div>")

    # "Pick a look" — browse recently generated images and drop one straight
    # into the topic field with one click, instead of retyping its prompt by
    # hand or leaving this page to find it on /thumbnails.
    gallery_html = "<p class='muted'>No generated images yet — try Thumbnails first.</p>"
    try:
        import image_gen
        imgs = [i for i in image_gen.recent_images(limit=24) if i.get("prompt")]
        if imgs:
            # data-prompt, not an inline JS string literal built from the
            # prompt text: HTML entities in an attribute value are decoded by
            # the browser's HTML parser BEFORE any inline JS runs, so a
            # prompt containing an apostrophe would have closed the JS string
            # early (html.escape() is correct for an ATTRIBUTE VALUE, but
            # that's not the same thing as correct for a JS STRING LITERAL
            # embedded inside one — reading it back via `.dataset` at click
            # time sidesteps the mismatch entirely, no JS escaping needed).
            cards = "".join(
                f'<div class="thumbcard pick-look" data-prompt="{_esc(i["prompt"])}">'
                f'<img src="/thumbnails/file/{_urlquote(i["name"])}" loading="lazy" alt="">'
                f'<div class="meta">{_esc(i["prompt"][:70])}</div>'
                f'</div>' for i in imgs)
            gallery_html = (
                f'<div class="thumbgrid">{cards}</div>'
                '<script>'
                'document.querySelectorAll(".pick-look").forEach(function(el){'
                '  el.style.cursor="pointer";'
                '  el.addEventListener("click", function(){'
                '    var f = document.getElementById("gen-topic");'
                '    f.value = el.dataset.prompt;'
                '    f.scrollIntoView({behavior:"smooth"});'
                '  });'
                '});'
                '</script>'
            )
    except Exception:
        pass

    body = f"""
    <a class="back" href="/">← back</a>
    <h2 style="margin-top:14px">Make a video</h2>
    {gpu_warn}
    {_msg_banner()}
    {status}
    <p class="muted">A full run takes roughly 5–45 minutes depending on the
       motion engine. It renders on the owner's RTX 3090 and lands in the
       review queue when it's done — it is not published automatically.</p>
    <form method="post" action="/system/run">
      {f'''<label for="gen-channel">Channel</label>
      <select class="field" id="gen-channel" name="channel">
        <option value="">(default)</option>{channel_options}
      </select>''' if channels else ""}
      <label for="gen-topic">Topic (optional — leave blank to let Rufus pick, or click a look below)</label>
      <input class="field" type="text" id="gen-topic" name="topic"
             placeholder="e.g. why the 1929 crash started in a Florida swamp">
      <label for="gen-niche">Niche (optional)</label>
      <select class="field" id="gen-niche" name="niche">
        <option value="">(default)</option>{niche_options}
      </select>
      <button class="btn save" type="submit">Start the run</button>
    </form>
    <h2>Pick a look (optional)</h2>
    <p class="muted">Click an image to build the video's topic around it —
       generate more first on the <a href="/thumbnails">Thumbnails</a> page.</p>
    {gallery_html}
    <h2>Recent</h2>
    {_videos_table(_recent_videos(limit=10))}
    """
    return _head() + body + PAGE_TAIL


@app.route("/system/run", methods=["POST"])
def system_run():
    """Start a run. Permission is "generate", not "system" — a partner may
    make videos; only an owner may control the machine."""
    auth.require("generate")
    _require_localhost()
    channel = request.form.get("channel", "").strip() or None
    niche   = request.form.get("niche", "").strip() or None
    topic   = request.form.get("topic", "").strip() or None
    back    = "/generate" if request.form.get("from") != "system" else "/system"
    from channel_config import load_channel
    resolved_id = load_channel(channel).id
    if _run_in_progress(resolved_id):
        return redirect(f"{back}?error=A+run+is+already+in+progress+for+that+channel")
    try:
        _launch_run(niche=niche, topic=topic, channel=channel)
    except Exception as e:
        return redirect(f"{back}?error={_urlquote(f'Could not start the run: {e}')}")
    return redirect(f"{back}?ok=Run+started")


@app.route("/system/cancel", methods=["POST"])
def system_cancel():
    auth.require("cancel")
    _require_localhost()
    channel = request.form.get("channel", "").strip() or None
    _cancel_run(channel)
    return redirect("/system")


# ── Thumbnails / image generation ────────────────────────────────────────────
# Generating a picture is a much shorter loop than a whole video (seconds, not
# tens of minutes) so unlike _launch_run this is done INLINE and the result is
# shown immediately. That's only safe because the app runs threaded=False and
# a stills render is bounded — a video render here would freeze the dashboard.

@app.route("/history")
def history_page():
    """When each video was made, when it went out, and what happened to it.

    THE QUESTION THIS ANSWERS. The queue shows what is waiting and Tracking
    shows what is performing; neither says WHEN. `upload_date` is a date with
    no time, and it means "made" for a pipeline upload and "published" for a
    hand-published one — so even the day it gave you was the wrong day half
    the time. Two honest columns beat one ambiguous one.

    Rows from before those columns existed show a date and no time, because
    that is all that was recorded. Inventing 00:00 would read as midnight.
    """
    auth.require("view")
    rows = db_manager.history(limit=300)

    _when = _when_cell

    badge = {"approved": "ok", "pending": "pending", "rejected": "bad"}
    trs = ""
    for v in rows:
        status = v.get("upload_status") or "pending"
        link = (f'<a href="https://youtu.be/{_esc(v["youtube_id"])}" '
                f'target="_blank" rel="noopener">&#9654; watch</a>'
                if v.get("youtube_id") else "")
        sched = (f'<br><span class="muted">scheduled {_esc(v["publish_at"])}</span>'
                 if v.get("publish_at") else "")
        why = (f'<br><span class="muted">{_esc(v["hold_reason"])}</span>'
               if v.get("hold_reason") and status != "approved" else "")
        trs += (
            f'<tr>'
            f'<td class="muted">#{v["id"]}</td>'
            f'<td>{_when(v.get("created_at"))}</td>'
            f'<td>{_when(v.get("uploaded_at"))}{sched}</td>'
            f'<td><a href="/video/{v["id"]}">'
            f'{_esc((v.get("title") or v.get("script_hook") or "")[:70])}</a></td>'
            f'<td class="muted">{_esc(v.get("channel") or "")}</td>'
            f'<td><span class="badge {badge.get(status, "pending")}">'
            f'{_esc(status)}</span>{why}</td>'
            f'<td>{link}</td>'
            f'</tr>')

    live = sum(1 for v in rows if v.get("youtube_id"))
    timed = sum(1 for v in rows if v.get("uploaded_at"))
    table = (f'<table><tr><th>#</th><th>Made</th><th>Went out</th>'
             f'<th>Title</th><th>Channel</th><th>Status</th><th></th></tr>'
             f'{trs}</table>' if trs else
             "<p class='muted'>No videos yet.</p>")

    body = f"""
    <a class="back" href="/">&larr; back</a>
    <h2 style="margin-top:14px">History</h2>
    <p class="muted">{len(rows)} video(s), newest first &middot; {live} live on
       YouTube &middot; {timed} with an exact upload time. Rows made before this
       page existed show the date only &mdash; that is all that was recorded for
       them, and a made-up time would be worse than a blank.</p>
    {table}
    """
    return _head() + body + PAGE_TAIL


@app.route("/thumbnails")
def thumbnails_page():
    auth.require("thumbnail")
    import comfy_client
    import image_gen

    comfy_up = _comfyui_reachable()
    warn = ("" if comfy_up else
            "<div class='msg error'>ComfyUI is not reachable at "
            f"{_esc(_comfy_host())} — start it on the host PC before generating.</div>")

    # An image whose prompt was saved can seed a whole video — pick the look
    # first, then let the pipeline build a script around that subject. Without
    # a stored prompt there's nothing to hand the script writer, so that
    # button only appears where it would actually work.
    can_generate = auth.can("generate")
    cards = ""
    for img in image_gen.recent_images(limit=36):
        name = _urlquote(img["name"])
        # The composed file when there is one, the raw background otherwise —
        # so a card shows the thing that would actually go on YouTube.
        shown = _urlquote(img["composed"] or img["name"])
        make_btn = ""
        if can_generate and img["prompt"]:
            make_btn = (
                f'<form method="post" action="/thumbnails/make-video" style="margin-top:6px">'
                f'<input type="hidden" name="name" value="{_esc(img["name"])}">'
                f'<button class="btn save" type="submit" style="padding:5px 10px;font-size:12px">'
                f'🎬 Make a video from this</button></form>')
        del_btn = ""
        if auth.can("delete"):
            del_btn = (
                f'<form method="post" action="/thumbnails/delete" '
                f'style="margin-top:6px" onsubmit="return confirm('
                f'\'Delete this image? It also stops offering itself as a '
                f'topic on the Make page.\');">'
                f'<input type="hidden" name="name" value="{_esc(img["name"])}">'
                f'<button class="btn" type="submit" '
                f'style="padding:5px 10px;font-size:12px">🗑 Delete</button></form>')

        # THE HEADLINE, TYPED HERE, RECOMPOSED INSTANTLY. No GPU — the picture
        # already exists and Pillow draws the words on it in about a tenth of
        # a second, so trying five headlines against one background costs
        # nothing.
        head_form = (
            f'<form method="post" action="/thumbnails/compose" '
            f'style="margin-top:6px;display:flex;gap:6px">'
            f'<input type="hidden" name="name" value="{_esc(img["name"])}">'
            f'<input class="field" style="margin:0;font-size:12px;padding:5px 8px" '
            f'type="text" name="headline" value="{_esc(img["headline"])}" '
            f'placeholder="headline (3-5 words)" maxlength="90">'
            f'<button class="btn save" type="submit" '
            f'style="padding:5px 10px;font-size:12px">Set</button></form>')

        # WHAT IT LOOKS LIKE IN THE FEED. 168x94 is the size a thumbnail is
        # actually competing at on a phone, and it is the only size that
        # decides anything. The page used to show one size, full width, which
        # is the size nobody ever sees it at.
        feed = ""
        if img["composed"]:
            feed = (f'<img src="/thumbnails/file/{shown}?w=240" loading="lazy" '
                    f'alt="" title="what it looks like in the feed" '
                    f'style="width:168px;height:94px;object-fit:cover;'
                    f'border-radius:4px;margin-top:6px">')

        when = time.strftime("%d %b %H:%M", time.localtime(img["mtime"]))
        cards += (
            f'<div class="thumbcard">'
            f'<a href="/thumbnails/file/{shown}" target="_blank">'
            f'<img src="/thumbnails/file/{shown}?w=480" loading="lazy" alt=""></a>'
            f'<div class="meta">{_esc(img["prompt"][:90] or img["name"])}<br>'
            f'<a href="/thumbnails/file/{shown}?download=1">⬇ Save to phone</a>'
            f' · {img["kb"]}KB · {when}{feed}{head_form}{make_btn}{del_btn}'
            f'</div></div>')
    gallery = (f'<div class="thumbgrid">{cards}</div>' if cards else
               "<p class='muted'>Nothing generated yet.</p>")

    presets = sorted(comfy_client.style_presets())
    style_options = "".join(
        f'<option value="{_esc(k)}" {"selected" if k == "thumbnail" else ""}>'
        f'{_esc(k)}</option>' for k in presets)

    body = f"""
    <a class="back" href="/">← back</a>
    <h2 style="margin-top:14px">Make a thumbnail</h2>
    {warn}
    {_msg_banner()}
    <p class="muted">Draws the picture on the RTX 3090 and burns the headline
       onto it — the niche accent bar, the words in Anton with an outline, and
       the recurring character's badge once this niche has one. Runs in the
       background; refresh when it is done. <b>The headline can be retyped on
       any finished image without redrawing it</b>, so try several.</p>
    <form method="post" action="/thumbnails/generate">
      <label for="tp">Describe the picture</label>
      <input class="field" type="text" id="tp" name="prompt" required
             placeholder="a cracked hourglass spilling gold coins across a desk">
      <label for="th">Headline (3-5 words — this is what gets the click)</label>
      <input class="field" type="text" id="th" name="headline" maxlength="90"
             placeholder="The bank that printed itself">
      <div style="display:flex;gap:12px;flex-wrap:wrap">
        <div style="flex:1;min-width:180px">
          <label for="tstyle">Look</label>
          <select class="field" id="tstyle" name="style">{style_options}</select>
        </div>
        <div style="flex:1;min-width:180px">
          <label for="tshape">Shape</label>
          <select class="field" id="tshape" name="shape">
            <option value="landscape">Landscape {image_gen.THUMB_W}×{image_gen.THUMB_H} (YouTube)</option>
            <option value="frame">{image_gen.FRAME_W}×{image_gen.FRAME_H} (video frame)</option>
          </select>
        </div>
        <div style="min-width:120px">
          <label for="tcount">How many</label>
          <select class="field" id="tcount" name="count">
            <option value="1">1</option><option value="3" selected>3</option>
            <option value="5">5</option>
          </select>
        </div>
      </div>
      <button class="btn save" type="submit">Draw them</button>
    </form>
    <h2>Generated</h2>
    {gallery}
    """
    return _head() + body + PAGE_TAIL


def _launch_thumb(prompt: str, headline: str, count: int, *,
                  frame: bool = False, style: str = ""):
    """Render thumbnail backgrounds in a subprocess. (proc, log_path).

    THE SAME SHAPE AS _launch_recut AND FOR THE SAME REASON. This used to call
    image_gen.generate_image() inline and the page waited for it — which under
    threaded=False froze the dashboard for everyone for the length of a GPU
    render, and the code said so out loud ("that wait freezes the dashboard for
    everyone") without doing anything about it. Threading fixes the freezing;
    it does not make a browser tab sit on an open connection for ninety seconds
    any less silly, and it cannot render three variants at once.

    The style override belongs to THIS render and not to the channel: the run
    style is whatever the videos are being made in, and asking for a thumbnail
    must not quietly change it. _scoped_env cannot do that here because the
    value has to reach a CHILD process, so it goes in that child's environment
    and nowhere else.
    """
    cmd = [sys.executable, str(ROOT / "scripts" / "image_gen.py"), prompt,
           "--count", str(max(1, min(count, 6)))]
    if headline.strip():
        cmd += ["--headline", headline.strip()]
    if frame:
        cmd += ["--frame"]
    env = os.environ.copy()
    env.update(_load_settings())
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    if style:
        env["RUFUS_STYLE"] = style
        env.pop("RUFUS_STILLS_DETAIL", None)   # a literal override outranks it
    log_dir = paths.log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"thumb_{int(time.time())}.log"
    with open(log_path, "wb") as logf:
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=logf, env=env,
                                stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL)
    return proc, log_path


@app.route("/thumbnails/generate", methods=["POST"])
def thumbnails_generate():
    auth.require("thumbnail")
    import image_gen

    prompt = request.form.get("prompt", "").strip()
    if not prompt:
        return redirect("/thumbnails?error=Describe+the+image+first")
    # A video run owns the GPU for its whole duration, so a thumbnail asked
    # for now would sit in ComfyUI's queue behind it. Refuse immediately with
    # the reason rather than queueing a job that cannot start.
    busy = [c for c in _channels() if _run_in_progress(c)]
    if busy:
        return redirect("/thumbnails?error=" + _urlquote(
            f"A video run is using the GPU ({', '.join(busy)}). "
            f"Thumbnails have to wait for it to finish — try again shortly."))

    # "frame" is the video's own shape, whatever format the next run is.
    # "portrait" is the value the form used to send, and older bookmarks and
    # any open tab still do, so it keeps meaning the same thing.
    want_frame = request.form.get("shape") in ("frame", "portrait")
    headline = request.form.get("headline", "").strip()
    style = request.form.get("style", "").strip()
    try:
        count = int(request.form.get("count", "1"))
    except ValueError:
        count = 1

    try:
        _proc, log = _launch_thumb(prompt, headline, count,
                                   frame=want_frame, style=style)
    except Exception as e:
        return redirect(f"/thumbnails?error={_urlquote(f'Could not start: {e}')}")
    return redirect("/thumbnails?ok=" + _urlquote(
        f"Drawing {count} — refresh in a minute (log: {log.name})"))


@app.route("/thumbnails/compose", methods=["POST"])
def thumbnails_compose():
    """Put different words on a background that already exists.

    THE FAST HALF, KEPT SEPARATE FROM THE SLOW ONE. Drawing the picture is
    seconds of GPU; drawing the words on it is about a tenth of a second of
    Pillow. Keeping them in one button meant every headline you wanted to try
    cost another render, so nobody tried a second one.
    """
    auth.require("thumbnail")
    import image_gen

    name = request.form.get("name", "").strip()
    headline = request.form.get("headline", "").strip()
    # Matched against the listing rather than trusted as a path — this value
    # reaches the filesystem otherwise.
    match = next((i for i in image_gen.recent_images(limit=500)
                  if i["name"] == name), None)
    if match is None:
        return redirect("/thumbnails?error=" + _urlquote("No such image."))
    src = paths.thumbnails_dir() / match["name"]
    if not image_gen.set_headline(src, headline):
        return redirect("/thumbnails?error=" + _urlquote(
            "Could not compose that headline — see the dashboard log."))
    return redirect("/thumbnails?ok=" + _urlquote(f"Composed “{headline[:50]}”"))


@app.route("/thumbnails/make-video", methods=["POST"])
def thumbnails_make_video():
    """Start a full video run using a generated image's prompt as the topic.

    The "pick the picture first" flow: browse the gallery, find a look that
    works, and let the pipeline write a script around that subject. It reuses
    the ordinary --topic path, so the result goes through every existing gate
    (fact-check, QC, score) and lands in the review queue like any other run —
    choosing an image changes what the video is ABOUT, it does not skip any
    of the checks or publish anything.
    """
    auth.require("generate")
    import image_gen

    name = request.form.get("name", "").strip()
    # Match against the listing rather than trusting the posted name as a
    # path — this value reaches the filesystem otherwise.
    match = next((i for i in image_gen.recent_images(limit=500) if i["name"] == name), None)
    if match is None:
        return redirect("/thumbnails?error=" + _urlquote("No such image."))
    if not match["prompt"]:
        return redirect("/thumbnails?error=" + _urlquote(
            "That image has no saved prompt, so there's nothing to build a script from."))

    busy = [c for c in _channels() if _run_in_progress(c)]
    if busy:
        return redirect("/thumbnails?error=" + _urlquote(
            f"A run is already in progress ({', '.join(busy)}) — wait for it to finish."))

    try:
        _launch_run(topic=match["prompt"])
    except Exception as e:
        return redirect(f"/thumbnails?error={_urlquote(f'Could not start the run: {e}')}")
    return redirect("/?ok=" + _urlquote(
        f'Started a video from "{match["prompt"][:60]}" — it will appear here for review when done.'))


@app.route("/thumbnails/delete", methods=["POST"])
def thumbnails_delete():
    """Remove one generated image and its saved prompt.

    WHY THIS IS NOT COSMETIC. /make's "Pick a look" gallery drops a stored
    prompt straight into the topic field on click, so every image ever
    generated stays a live, one-click suggestion for what the next video
    should be ABOUT. A test render, a bad idea, a duplicate — each keeps
    offering itself forever, and the only way to stop one was to reach the
    filesystem.

    The .txt sidecar goes with the .png on purpose: leaving it behind would
    keep the prompt in the gallery's own metadata with no picture to explain
    it, which is a worse state than either having both or having neither.
    """
    auth.require("delete")
    import image_gen

    name = request.form.get("name", "").strip()
    # Matched against the listing rather than trusted as a path — this value
    # reaches the filesystem, and unlike make-video the operation is a delete.
    match = next((i for i in image_gen.recent_images(limit=500)
                  if i["name"] == name), None)
    if match is None:
        return redirect("/thumbnails?error=" + _urlquote("No such image."))

    folder = paths.thumbnails_dir().resolve()
    png = (folder / match["name"]).resolve()
    # Belt and braces: the listing can only yield names from this folder, but
    # a delete that resolves outside it should never run even if that changes.
    if png.parent != folder:
        return redirect("/thumbnails?error=" + _urlquote("Refusing that path."))
    removed = []
    for f in (png, png.with_suffix(".txt")):
        try:
            f.unlink()
            removed.append(f.name)
        except FileNotFoundError:
            pass
        except OSError as e:
            return redirect("/thumbnails?error=" + _urlquote(f"Could not delete: {e}"))
    return redirect("/thumbnails?ok=" + _urlquote(f"Deleted {removed[0] if removed else name}"))


@app.route("/thumbnails/file/<path:filename>")
def thumbnail_file(filename):
    """Serve one generated image. `?download=1` forces a Save dialog instead of
    rendering inline — the difference between looking at it on a phone and
    actually getting it into the camera roll.

    `?w=` SERVES IT AT THE SIZE IT IS SHOWN AT. This route sent the full PNG,
    every time, with no cache header, into a 220px card. A gallery of 36
    generated images averaging ~1MB each is ~35 MEGABYTES fetched to draw
    postage stamps, and then fetched again on the next visit because nothing
    told the browser it could keep them. Over the tailnet from a phone that is
    the whole of "the dashboard is slow".

    The cache this uses is not new — _thumb_of has served /debug/ keyframes
    this way for a while. This route simply never asked it. The download link
    deliberately does NOT pass a width: "save to phone" means the real file.
    """
    auth.require("download")
    folder = paths.thumbnails_dir().resolve()
    if not folder.is_dir():
        abort(404)
    download = request.args.get("download") == "1"
    if not download:
        small = _thumb_of(folder, filename, request.args.get("w", type=int))
        if small is not None:
            return send_from_directory(small.parent, small.name,
                                       max_age=_IMAGE_MAX_AGE)
    return send_from_directory(folder, filename, as_attachment=download,
                               max_age=None if download else _IMAGE_MAX_AGE)


def _setting_field(key: str, kind: str, val: str) -> str:
    """One input, chosen by kind. Everything renders as a control the owner can
    actually use — the previous version made every setting a dropdown, which
    meant a webhook URL or a beat count simply could not be entered here."""
    if kind == "bool":
        opts = [("", "(default — don't override)"), ("1", "on"), ("0", "off")]
    elif kind.startswith("select:"):
        opts = [("", "(default)")] + [(o, o) for o in kind.split(":", 1)[1].split(",")]
    else:
        # A secret is masked but NOT hidden from the owner: they need to see
        # that a webhook is set, and a password field that silently shows an
        # empty box for a stored value is how people paste it in twice.
        typ = "password" if kind == "secret" else "text"
        place = {"number": "e.g. 24", "secret": "paste the URL",
                 "text": "(default)"}.get(kind, "(default)")
        return (f'<input class="field" style="margin:0" type="{typ}" name="{key}" '
                f'value="{_esc(val)}" placeholder="{place}" '
                f'autocomplete="off" spellcheck="false">')
    options = "".join(
        f'<option value="{_esc(v)}" {"selected" if val == v else ""}>{_esc(t)}</option>'
        for v, t in opts)
    return f'<select name="{_esc(key)}">{options}</select>'


@app.route("/settings")
def settings():
    """Every tunable, grouped by the decision it belongs to.

    THE POINT OF THIS PAGE, in the owner's words: the software is run from the
    dashboard, not from a terminal. Seven `$env:` lines before every run is not
    a workflow — it is seven chances to leave a stale value from an earlier
    experiment somewhere no log will mention.

    Applies to EVERY launch path — this dashboard, run.bat, run_scheduled.bat,
    a Task Scheduler entry, a bare `python scripts/main.py`. main.py reads the
    same file at startup (see settings_store), because a settings page obeyed
    by some launchers and not others is worse than none: it teaches the owner
    to trust a form that is sometimes ignored.
    """
    auth.require("settings")
    if request.args.get("reset") == "1":
        _require_localhost()
        _save_settings({})
        return redirect("/settings?msg=" +
                        _urlquote("Cleared — every setting is back to the "
                                  "pipeline default."))
    values = _load_settings()
    n_set = sum(1 for k in SETTINGS_KINDS if values.get(k))

    sections = ""
    for title, blurb, rows_spec in SETTINGS_GROUPS:
        rows = ""
        for key, label, kind, help_text in rows_spec:
            val = values.get(key, "")
            marker = ' <span class="badge ok">set</span>' if val else ""
            rows += (
                f'<tr><td style="width:38%"><strong>{_esc(label)}</strong>{marker}'
                f'<div class="muted" style="margin-top:2px">{_esc(help_text)}</div></td>'
                f'<td style="vertical-align:top">{_setting_field(key, kind, val)}'
                f'<div class="muted" style="margin-top:4px"><code>{_esc(key)}</code></div>'
                f'</td></tr>\n')
        sections += (f'<h2>{_esc(title)}</h2>'
                     f'<p class="muted" style="margin:0 0 6px">{_esc(blurb)}</p>'
                     f'<table>{rows}</table>')

    body = f"""
    <a class="back" href="/">← back</a>
    <h2 style="margin-top:14px">Settings</h2>
    {_msg_banner()}
    <p class="muted">These are the channel's defaults, and every way of
       starting a run obeys them — this page, <code>run.bat</code>, the
       scheduled task. An empty field means "don't override", so the
       pipeline's own default wins; and a variable you set in a terminal beats
       what is saved here, for that run only.
       <strong>{n_set}</strong> of {len(SETTINGS_KINDS)} currently set.</p>
    <form method="post" action="/settings/save">
      {sections}
      <div style="position:sticky;bottom:0;padding:12px 0;background:inherit">
        <button class="btn save" type="submit">Save all</button>
        <button class="btn" type="submit" formaction="/settings/test-notify"
                formnovalidate style="margin-left:8px">Send a test notification</button>
        <a class="muted" href="/settings?reset=1" style="margin-left:14px"
           onclick="return confirm('Clear every override and go back to pipeline defaults?')">
           reset everything to defaults</a>
      </div>
    </form>
    <h2>Dashboard users</h2>
    <p class="muted">Add or remove who can reach this dashboard — a partner,
       a viewer, or another owner — without a terminal.</p>
    <a class="btn save" style="text-decoration:none;display:inline-block" href="/settings/users">Manage users →</a>
    """
    return _head() + body + PAGE_TAIL


def _users_table_rows(users: list[dict]) -> str:
    rows = ""
    for u in users:
        via = (f'Google: {_esc(u["google_email"])}' if u.get("google_email")
               else "token link")
        name = _esc(u.get("name", ""))
        # THE ROLE IS EDITABLE IN PLACE, not revoke-and-re-add. A role gets
        # revised — somebody added as a partner turns out to be running the
        # channel with you — and the old route handed them a new sign-in link
        # and killed the one they already had. A role change is not a new
        # person.
        current = u.get("role", "")
        options = "".join(
            f'<option value="{r}"{" selected" if r == current else ""}>{r}</option>'
            for r in auth.ROLES)
        role_cell = (
            f'<form method="post" action="/settings/users/role" '
            f'style="display:flex;gap:6px;align-items:center">'
            f'<input type="hidden" name="name" value="{name}">'
            f'<select name="role" class="field" style="margin:0;padding:3px 6px;'
            f'width:auto" aria-label="Role for {name}">{options}</select>'
            f'<button type="submit" style="padding:4px 9px">Set</button>'
            f'</form>')
        rows += (
            f'<tr><td>{name}</td><td>{role_cell}</td><td>{via}</td>'
            f'<td style="white-space:nowrap">'
            f'<form method="post" action="/settings/users/link" style="display:inline">'
            f'<input type="hidden" name="name" value="{name}">'
            f'<button type="submit">Reprint link</button></form> '
            f'<form method="post" action="/settings/users/revoke" style="display:inline" '
            f'onsubmit="return confirm(\'Revoke {name}? This takes effect immediately.\');">'
            f'<input type="hidden" name="name" value="{name}">'
            f'<button class="btn reject" type="submit" style="padding:4px 10px">Revoke</button>'
            f'</form></td></tr>')
    return rows


@app.route("/settings/users")
def settings_users():
    """Owner-only user management, backed by the exact same add_user() /
    revoke_user() the CLI uses (scripts/auth.py) — this is a form in front of
    those, not a second implementation of the rules."""
    auth.require("manage_users")
    users = auth._load_users()
    rows = _users_table_rows(users)
    table = (f'<table><tr><th>Name</th><th>Role</th><th>Signs in via</th><th></th></tr>{rows}</table>'
             if rows else "<p class='muted'>No users yet — auth is off "
             "(legacy loopback-owner mode). Adding one here turns it on for everyone.</p>")

    link_val = request.args.get("link", "")
    link_name = request.args.get("name", "")
    link_html = ""
    if link_val:
        link_html = (f'<div class="msg ok">Sign-in link for {_esc(link_name)}: '
                    f'<code style="user-select:all">{_esc(link_val)}</code><br>'
                    f'<span class="muted">Send this privately — it IS the password.</span></div>')

    google_note = ("" if auth.google_oauth_enabled() else
                  "<p class='muted'>Google sign-in isn't set up — leave the "
                  "email field blank and share the printed link instead, or "
                  "see the README's \"Google Sign-In\" section to enable it.</p>")

    body = f"""
    <a class="back" href="/settings">← back to settings</a>
    <h2 style="margin-top:14px">Dashboard users</h2>
    {_msg_banner()}
    {link_html}
    {table}
    <h2>Add a user</h2>
    {google_note}
    <form method="post" action="/settings/users/add">
      <label for="uname">Name</label>
      <input class="field" type="text" id="uname" name="name" required placeholder="james">
      <label for="urole">Role</label>
      <select class="field" id="urole" name="role">
        <option value="partner">partner — generate videos/thumbnails, cannot publish</option>
        <option value="viewer">viewer — read-only, cannot generate</option>
        <option value="owner">owner — full control, including adding/revoking users</option>
      </select>
      <label for="uemail">Google email (optional)</label>
      <input class="field" type="email" id="uemail" name="google_email"
             placeholder="lets them sign in with Google instead of a link — optional">
      <button class="btn save" type="submit">Add</button>
    </form>
    """
    return _head() + body + PAGE_TAIL


@app.route("/settings/users/add", methods=["POST"])
def settings_users_add():
    auth.require("manage_users")
    name = request.form.get("name", "").strip()
    role_name = request.form.get("role", "partner").strip()
    google_email = request.form.get("google_email", "").strip() or None
    try:
        user = auth.add_user(name, role_name, google_email=google_email)
    except auth.AuthError as e:
        return redirect("/settings/users?error=" + _urlquote(str(e)))
    link = f"{auth._base_url()}/?token={user['token']}"
    return redirect(
        f"/settings/users?ok={_urlquote(f'Added {name} as {role_name}.')}"
        f"&link={_urlquote(link)}&name={_urlquote(name)}")


# ── Say something to the other person ────────────────────────────────────────

@app.route("/message")
def message_page():
    """Leave a note. Optionally ping it. Tick it off when it is done.

    WHAT THIS IS NOT. notify.py already pushes automatically — a finished
    video, a failed run, the analytics summary. Those are the machine telling
    you something. This is the other direction and the missing one: two people
    now share this channel, and there was no way for one of them to say "the
    pictures on 105 are wrong, redo them before you approve it".

    A NOTIFICATION IS NOT A RECORD, which is why the note is stored first and
    pushed second. Discord and ntfy are a tap on the shoulder — read once,
    scrolled past, gone by morning. Half of what passes between two people
    running a channel is not "look at this now", it is "do not forget this",
    and that needs somewhere to live until somebody actually does it.

    So the ping is a checkbox, not the point. A note nobody was pinged about
    is still a note; a ping nobody kept is nothing an hour later.

    IT SIGNS ITSELF, from the same auth.current_user() the decision columns
    use. A note in a shared list that does not say who wrote it makes the
    reader guess, and there are exactly two candidates.
    """
    auth.require("generate")
    import db_manager as dbm
    import notify
    backends = notify.configured()

    where = (f'<span class="muted">Ticking &ldquo;ping&rdquo; also sends it to '
             f'<b>{_esc(", ".join(backends))}</b>.</span>' if backends else
             '<span class="muted">No Discord webhook or ntfy topic is set, so '
             'nothing can be pushed &mdash; notes still save. Add one on '
             '<a href="/settings">Settings</a>.</span>')

    ping = ""
    if backends:
        ping = ('<label style="display:flex;gap:7px;align-items:center;'
                'font-weight:400"><input type="checkbox" name="ping" value="1" '
                'checked> ping it</label>')

    form = f"""
    <form method="post" action="/message/send" style="display:grid;gap:12px;max-width:600px">
      <div>
        <label for="body">Note</label>
        <textarea class="field" id="body" name="body" rows="3" required
                  placeholder="the pictures on 105 are wrong &mdash; redo them before approving"></textarea>
      </div>
      <div style="display:flex;gap:14px;flex-wrap:wrap;align-items:flex-end">
        <div>
          <label for="priority">Urgency</label>
          <select class="field" id="priority" name="priority" style="margin:6px 0 0">
            <option value="low">quiet</option>
            <option value="normal" selected>normal</option>
            <option value="high">urgent</option>
          </select>
        </div>
        {ping}
        <button class="btn save" type="submit" style="height:38px">Save</button>
      </div>
    </form>"""

    open_notes = dbm.notes(done=False, limit=60)
    done_notes = dbm.notes(done=True, limit=15)

    def row(n, done=False):
        who = _by_badge(n.get("author"))
        when = _esc(str(n.get("created_at") or "")[:16])
        bell = ('<span class="note-bell" title="was pushed">&#9679;</span>'
                if n.get("notified") else "")
        pr = n.get("priority") or "normal"
        if done:
            by = n.get("done_by")
            tail = (f'<span class="muted">done by {_esc(by)}</span>'
                    if by else '<span class="muted">done</span>')
            act = (f'<form method="post" action="/message/{n["id"]}/reopen">'
                   f'<button type="submit">reopen</button></form>')
        else:
            tail = ""
            act = (f'<form method="post" action="/message/{n["id"]}/done">'
                   f'<button class="btn save" type="submit" '
                   f'style="padding:4px 11px">done</button></form>')
        return (f'<li class="note p-{_esc(pr)}{" is-done" if done else ""}">'
                f'<div class="note-b">{_esc(n["text"])}</div>'
                f'<div class="note-m">{who}<span class="muted">{when}</span>'
                f'{bell}{tail}</div>{act}</li>')

    if open_notes:
        open_html = (f'<h2 style="margin-top:30px">Still open '
                     f'({len(open_notes)})</h2>'
                     f'<ul class="notes">'
                     f'{"".join(row(n) for n in open_notes)}</ul>')
    else:
        open_html = ('<h2 style="margin-top:30px">Still open (0)</h2>'
                     '<p class="muted">Nothing outstanding.</p>')

    done_html = ""
    if done_notes:
        done_html = (f'<details style="margin-top:26px"><summary>Recently done '
                     f'({len(done_notes)})</summary><ul class="notes">'
                     f'{"".join(row(n, done=True) for n in done_notes)}'
                     f'</ul></details>')

    body = f"""
    <a class="back" href="/">&larr; back</a>
    <h2 style="margin-top:14px">Notes &amp; messages</h2>
    <p class="muted">For telling the other person something, and for the things
       neither of you should forget. The automatic alerts &mdash; finished
       videos, failed runs, the analytics summary &mdash; send themselves and
       are not this. {where}</p>
    {_msg_banner()}
    {form}
    {open_html}
    {done_html}
    """
    return _head() + body + PAGE_TAIL


@app.route("/message/send", methods=["POST"])
def message_send():
    auth.require("generate")
    import db_manager as dbm
    import notify
    text = (request.form.get("body") or "").strip()
    if not text:
        return redirect("/message?error=" + _urlquote("Type something first."))
    priority = (request.form.get("priority") or "normal").strip()
    if priority not in ("low", "normal", "high"):
        priority = "normal"
    want_ping = request.form.get("ping") == "1"

    who = _whoami()
    # STORED BEFORE IT IS PUSHED. A failed webhook must not cost the note —
    # that is the half that has to survive, and the push is the part that can
    # be retried by simply saying it again.
    pushed = False
    if want_ping:
        pushed = notify.send(f"{who} says" if who else "Note from the dashboard",
                             text, url=notify._dashboard_url(),
                             priority=priority)
    dbm.add_note(text, author=who, priority=priority, notified=pushed)

    if want_ping and not pushed:
        return redirect("/message?error=" + _urlquote(
            "Saved, but nothing went out — either no backend is configured or "
            "the send failed. The Logs page has the reason."))
    if pushed:
        return redirect("/message?msg=" + _urlquote(
            f"Saved and sent via {', '.join(notify.configured())}."))
    return redirect("/message?msg=" + _urlquote("Saved."))


@app.route("/message/<int:note_id>/done", methods=["POST"])
def message_done(note_id: int):
    auth.require("generate")
    import db_manager as dbm
    if not dbm.finish_note(note_id, by=_whoami()):
        return redirect("/message?error=" + _urlquote(
            "Already done — somebody got there first."))
    return redirect("/message?msg=" + _urlquote("Ticked off."))


@app.route("/message/<int:note_id>/reopen", methods=["POST"])
def message_reopen(note_id: int):
    auth.require("generate")
    import db_manager as dbm
    dbm.reopen_note(note_id)
    return redirect("/message?msg=" + _urlquote("Back on the list."))


@app.route("/settings/users/role", methods=["POST"])
def settings_users_role():
    """Change a role in place, backed by the same auth.set_role() the CLI uses.

    The last-owner guard lives in auth, not here — a rule enforced in one of
    two front doors is a rule the other front door has never heard of.
    """
    auth.require("manage_users")
    name = (request.form.get("name") or "").strip()
    role_name = (request.form.get("role") or "").strip()
    try:
        user = auth.set_role(name, role_name)
    except auth.AuthError as e:
        return redirect("/settings/users?error=" + _urlquote(str(e)))
    if user is None:
        return redirect("/settings/users?error="
                        + _urlquote(f"No user called {name!r}."))
    return redirect("/settings/users?ok=" + _urlquote(
        f"{name} is now {role_name}. Their sign-in link is unchanged."))


@app.route("/settings/users/revoke", methods=["POST"])
def settings_users_revoke():
    auth.require("manage_users")
    name = request.form.get("name", "").strip()
    status = auth.revoke_user(name)
    if status == "last_owner":
        return redirect("/settings/users?error=" + _urlquote(
            "Refusing — that's the last owner; you'd lock yourself out."))
    if status == "not_found":
        return redirect("/settings/users?error=" + _urlquote("No such user."))
    return redirect(f"/settings/users?ok={_urlquote(f'Revoked {name}.')}")


@app.route("/settings/users/link", methods=["POST"])
def settings_users_link():
    auth.require("manage_users")
    name = request.form.get("name", "").strip()
    for u in auth._load_users():
        if u.get("name") == name:
            link = f"{auth._base_url()}/?token={u['token']}"
            return redirect(f"/settings/users?link={_urlquote(link)}&name={_urlquote(name)}")
    return redirect("/settings/users?error=" + _urlquote("No such user."))


@app.route("/styles")
def styles_page():
    """Pick the look, by looking at it.

    A style is 1,500 words of prose describing line weight, palette and how a
    background behaves. Choosing between seven of those from a dropdown means
    choosing by NAME, which is how a channel spent weeks in a beige wash
    nobody had asked for — the words said "soft muted flat colours" and only
    a rendered frame said what that meant.

    So the previews are RENDERED, not shipped. One fixed scene, the same for
    every style, through the same ComfyUI stills workflow the videos use, so
    the differences on this page are differences you will actually get. A
    style with no preview yet says so and offers to make one; nothing here
    pretends to show you art it has not produced.
    """
    auth.require("settings")
    import comfy_client
    import video_format

    presets = comfy_client.style_presets()
    current = (os.environ.get("RUFUS_STYLE") or "").strip()
    prev_dir = _style_preview_dir()
    busy = [c for c in _channels() if _run_in_progress(c)]

    cards = ""
    for sid in sorted(presets):
        text = presets[sid]
        # The opening sentence is the style's own summary of itself.
        blurb = text.split(".")[0][:180] + "."
        img = prev_dir / f"{sid}.png"
        if img.exists():
            thumb = (f'<img src="/styles/preview/{_esc(sid)}?v={int(img.stat().st_mtime)}" '
                     f'alt="{_esc(sid)} preview" loading="lazy">')
        else:
            thumb = ('<div class="noimg">no preview yet —<br>render one to see '
                     'this style</div>')
        on = " on" if sid == current else ""
        pick = ("<span class='muted' style='font-size:12px'>in use</span>"
                if sid == current else
                f'<form method="post" action="/styles/use" style="display:inline">'
                f'<input type="hidden" name="style" value="{_esc(sid)}">'
                f'<button type="submit">Use this</button></form>')
        render = (f'<form method="post" action="/styles/preview" style="display:inline">'
                  f'<input type="hidden" name="style" value="{_esc(sid)}">'
                  f'<button type="submit" class="ghost">'
                  f'{"Re-render" if img.exists() else "Render preview"}</button></form>')
        cards += (f'<div class="style-card{on}">{thumb}<div class="body">'
                  f'<h4>{_esc(sid)}</h4><p>{_esc(blurb)}</p>'
                  f'<div class="row">{pick}{render}</div></div></div>\n')

    note = ""
    if busy:
        note = (f'<p class="muted">A run is using the GPU ({_esc(", ".join(busy))}), '
                f'so previews have to wait for it to finish.</p>')
    elif not _comfyui_reachable():
        note = ('<p class="muted">ComfyUI is not answering, so previews cannot '
                'be rendered right now. Picking a style still works.</p>')

    body = f"""
    <a class="back" href="/">← back</a>
    <h2 style="margin-top:14px">Style</h2>
    <p class="muted">The look every picture is rendered in. Previews are real
       renders of one fixed scene — the same scene for every style — through
       the same workflow the videos use, so what you see here is what you get.
       Applies to the next run in {_esc(video_format.profile()["label"])}.</p>
    {_msg_banner()}
    {note}
    <div class="style-grid">{cards}</div>
    """
    return _head() + body + PAGE_TAIL


def _style_preview_dir() -> Path:
    import paths
    d = paths.media_root() / "style_previews"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ONE SCENE FOR ALL OF THEM. A picker where each card shows a different
# subject compares subjects, not styles — the whole point is that the only
# variable between these frames is the style block.
STYLE_PREVIEW_SCENE = ("Two people stand at a market stall on a street, one "
                       "handing a coin to the other, crates and a low wall "
                       "behind them, hills on the horizon under an open sky.")


@app.route("/styles/preview", methods=["POST"])
def styles_preview():
    auth.require("settings")
    import comfy_client
    import image_gen

    sid = (request.form.get("style") or "").strip()
    presets = comfy_client.style_presets()
    if sid not in presets:
        return redirect("/styles?error=" + _urlquote("Unknown style"))

    busy = [c for c in _channels() if _run_in_progress(c)]
    if busy:
        return redirect("/styles?error=" + _urlquote(
            f"A video run is using the GPU ({', '.join(busy)}) — try again "
            f"when it finishes."))

    # add_detail=False, then the style appended by hand: the automatic suffix
    # is whatever RUFUS_STYLE currently is, and a picker that previewed the
    # style you already have would be a picker in name only.
    try:
        path = image_gen.generate_image(
            f"{STYLE_PREVIEW_SCENE} {presets[sid]}",
            width=image_gen.THUMB_W, height=image_gen.THUMB_H,
            add_detail=False, timeout=image_gen.WEB_TIMEOUT)
    except Exception as e:
        return redirect("/styles?error=" + _urlquote(f"Render failed: {e}"))
    if path is None:
        return redirect("/styles?error=" + _urlquote(
            "Render failed — check ComfyUI is running and a stills workflow "
            "is exported to config/stills_api.json"))

    try:
        import shutil
        shutil.copyfile(path, _style_preview_dir() / f"{sid}.png")
    except OSError as e:
        return redirect("/styles?error=" + _urlquote(f"Could not save it: {e}"))
    return redirect("/styles?msg=" + _urlquote(f"Rendered a {sid} preview."))


@app.route("/styles/preview/<style_id>")
def styles_preview_image(style_id: str):
    auth.require("view")
    d = _style_preview_dir()
    name = f"{Path(style_id).name}.png"
    if not (d / name).exists():
        abort(404)
    return send_from_directory(str(d), name)


@app.route("/styles/use", methods=["POST"])
def styles_use():
    """Save the style the way every launch path reads it."""
    auth.require("settings")
    import comfy_client
    sid = (request.form.get("style") or "").strip()
    if sid not in comfy_client.style_presets():
        return redirect("/styles?error=" + _urlquote("Unknown style"))
    values = _load_settings()
    values["RUFUS_STYLE"] = sid
    _save_settings(values)
    os.environ["RUFUS_STYLE"] = sid
    return redirect("/styles?msg=" + _urlquote(f"Next run renders in {sid}."))


# ── The scout ────────────────────────────────────────────────────────────────

@app.route("/scout")
def scout_page():
    """What the agent noticed, and what it wants to make about it.

    EVERY PROPOSAL SHOWS ITS EVIDENCE, and that is the whole difference between
    reviewing an agent's work and rubber-stamping it. "Make a video about the
    Panic of 1893" is an instruction. "Neighbour published this, it did 9x
    their own median, and this channel has not covered it" is something a
    person can disagree with.

    The scout never renders. Approving one starts an ordinary run on that
    topic, through every gate any other video goes through.
    """
    auth.require("view")
    try:
        import db_manager as dbm
        pending = dbm.proposals(status="pending", limit=30)
        decided = [p for p in dbm.proposals(status=None, limit=40)
                   if p["status"] != "pending"][:10]
        watching = dbm.rising(limit=8)
    except Exception as e:
        body = (f'<a class="back" href="/">← back</a><h2 style="margin-top:14px">'
                f'Scout</h2><div class="msg error">{_esc(str(e))}</div>')
        return _head() + body + PAGE_TAIL

    cards = ""
    for p in pending:
        buttons = ""
        if auth.can("generate"):
            buttons = (
                f'<form method="post" action="/scout/{p["id"]}/approve" '
                f'style="display:inline"><button class="btn save" '
                f'type="submit">Write 3 scripts</button></form> '
                f'<form method="post" action="/scout/{p["id"]}/reject" '
                f'style="display:inline"><button type="submit">Not this'
                f'</button></form>')
        # NO SCRIPT ON THIS CARD ANY MORE. It used to show "the script it
        # wrote" in a <details> — prose bought for every proposal, displayed,
        # and then thrown away even on approval, because approving launched an
        # ordinary run that writes its own. A proposal is a topic and the
        # evidence that chose it; the writing happens once, on the one that
        # survives this page.
        cards += (
            f'<div class="card" style="width:100%;margin-bottom:12px">'
            f'<div style="display:flex;justify-content:space-between;gap:12px">'
            f'<strong>{_esc(p["topic"] or "—")}</strong>'
            f'<span class="muted">{_esc((p["created_at"] or "")[:16])}</span>'
            f'</div>'
            f'<p class="muted" style="margin:6px 0">{_esc(p["evidence"] or "")}</p>'
            f'<div style="margin-top:10px">{buttons}</div></div>')

    if not pending:
        cards = ('<p class="muted">Nothing waiting. The scout proposes when a '
                 'watched channel publishes something that beat its own median '
                 'and this channel has not covered it — a quiet week is a real '
                 'answer, not a failure.</p>')

    seen = ""
    for w in watching:
        seen += (f'<tr><td>{_esc(w["channel_title"] or "")}</td>'
                 f'<td>{_esc((w["title"] or "")[:70])}</td>'
                 f'<td class="muted">{w["outperformance"]:.1f}x</td>'
                 f'<td class="muted">{w["views"]:,}</td>'
                 f'<td class="muted">{w["sightings"]}</td></tr>')
    seen_html = (f'<h2>What it is watching</h2><table><tr><th>Channel</th>'
                 f'<th>Video</th><th>vs their median</th><th>Views</th>'
                 f'<th>Seen</th></tr>{seen}</table>' if seen else "")

    old = ""
    for p in decided:
        old += (f'<tr><td>{_esc((p["topic"] or "")[:60])}</td>'
                f'<td class="muted">{_esc(p["status"])}</td>'
                f'<td class="muted">{_esc(p["decided_at"] or "")}</td></tr>')
    old_html = (f'<h2>Already decided</h2><table><tr><th>Topic</th>'
                f'<th></th><th>When</th></tr>{old}</table>' if old else "")

    body = f"""
    <a class="back" href="/">← back</a>
    <h2 style="margin-top:14px">Scout</h2>
    {_flow_bar("/scout")}
    {_msg_banner()}
    <p class="muted">Watches the channels in <code>config/competitors.json</code>,
       scores every video against <em>its own channel's median</em> — 20k views
       on a channel that averages 3k is the interesting one, not 50k on a
       channel that averages 200k — and proposes what to make. A proposal is a
       topic and its evidence, and costs almost nothing. Choosing one writes
       three scripts about it, one per hook style, and they land on
       <a href="/scripts">Choose a script</a>.</p>
    {cards}
    {seen_html}
    {old_html}
    """
    return _head() + body + PAGE_TAIL


@app.route("/scout/<int:proposal_id>/approve", methods=["POST"])
def scout_approve(proposal_id: int):
    """Approve a topic → three scripts about it, for a person to rule between.

    THIS NO LONGER STARTS A RENDER, and that is the whole point of the split. A
    render is hours of the 3090 and it used to be committed to here, from a
    topic card, with the only script anyone had seen already discarded. Now the
    expensive irreversible step sits one page further on, behind a choice
    between three finished scripts.
    """
    auth.require("generate")
    try:
        import db_manager as dbm
        row = next((p for p in dbm.proposals(status="pending", limit=200)
                    if p["id"] == proposal_id), None)
        if not row:
            return redirect("/scout?error=" + _urlquote(
                "that proposal is not pending — already decided?"))
        if not dbm.decide_proposal(proposal_id, "approved"):
            return redirect("/scout?error=" + _urlquote("could not record it"))
        log_path = _launch_candidates(topic=row["topic"],
                                      proposal_id=proposal_id,
                                      channel=row["channel"])
    except Exception as e:
        return redirect("/scout?error=" + _urlquote(f"could not start: {e}"))
    return redirect("/scripts?msg=" + _urlquote(
        f'Writing three scripts about "{row["topic"]}" — a minute or two. '
        f'Reload this page. Log: logs/{log_path.name}'))


@app.route("/scout/<int:proposal_id>/reject", methods=["POST"])
def scout_reject(proposal_id: int):
    """Rejected proposals stay in the table on purpose — they are what stops
    the scout proposing the same idea again next pass."""
    auth.require("generate")
    import db_manager as dbm
    if not dbm.decide_proposal(proposal_id, "rejected"):
        return redirect("/scout?error=" + _urlquote("already decided"))
    return redirect("/scout?msg=" + _urlquote(
        "Rejected — it will not be proposed again."))


# ── Choosing a script ────────────────────────────────────────────────────────

@app.route("/scripts")
def scripts_page():
    """Three scripts on one topic. Pick the one that gets made.

    WHAT THIS PAGE IS ACTUALLY FOR, beyond the obvious. Nothing published
    through this pipeline has view counts yet, so feedback_analyzer has never
    run and config/learnings.json does not exist — which means the only
    judgement anywhere in the script loop is a score the writer assigns itself
    against thresholds it also owns. Every click here is the thing that score
    cannot be: a person comparing three finished scripts and preferring one.
    The two that lose are kept, because a preference is a pair.
    """
    auth.require("view")
    try:
        import db_manager as dbm
        pending = dbm.candidates(status="pending", limit=60)
    except Exception as e:
        body = (f'<a class="back" href="/">← back</a><h2 style="margin-top:14px">'
                f'Choose a script</h2><div class="msg error">{_esc(str(e))}</div>')
        return _head() + body + PAGE_TAIL

    # Grouped by the topic they are alternatives for. A flat list of nine
    # scripts across three topics is not three choices, it is one confusing
    # one — the whole value is in seeing the siblings side by side.
    sets: dict = {}
    for c in pending:
        sets.setdefault((c["proposal_id"], c["topic"]), []).append(c)

    blocks = ""
    for (prop_id, topic), rows in sets.items():
        cards = ""
        for c in rows:
            button = ""
            if auth.can("generate"):
                button = (f'<form method="post" action="/scripts/{c["id"]}'
                          f'/choose" style="display:inline">'
                          f'<button class="btn save" type="submit">'
                          f'Make this one</button></form>')
            words = len((c["script"] or "").split())
            # THE GATES LABEL, THEY DO NOT REJECT. A score below the bar used
            # to throw a script away and retry on a different angle — deciding
            # the thing this page exists for, and binning good scripts on the
            # way. The number is shown instead. The fact gate is the one that
            # still matters after a person has read the script, because they
            # can judge the writing and cannot check the figure against the
            # source, so its warning is loud and names the claim.
            warn = ""
            if not c.get("fact_ok", 1):
                warn = (f'<div class="msg error" style="margin:8px 0">'
                        f'⚠ the source does not support this: '
                        f'{_esc(c.get("fact_reason") or "unstated")}</div>')
            cards += (
                f'<div class="card" style="width:100%;margin-bottom:10px">'
                f'<div style="display:flex;justify-content:space-between;'
                f'gap:12px"><strong>{_esc(c["hook"] or "—")}</strong>'
                f'<span class="muted">{_esc(c["hook_style"] or "unpinned")} · '
                f'{c["score"]}/10 · {words}w · ${c["cost_usd"]:.3f}</span></div>'
                f'{warn}'
                f'<pre style="white-space:pre-wrap;font-size:13px;'
                f'margin:8px 0">{_esc(c["script"] or "")}</pre>'
                f'<div>{button}</div></div>')
        blocks += (f'<h2 style="margin-top:22px">{_esc(topic or "—")}</h2>'
                   f'<p class="muted">{len(rows)} script(s) — one per hook '
                   f'style. The score is shown, not enforced: nothing here was '
                   f'thrown away for missing a bar, because that is your call. '
                   f'A red warning means the source does not support a claim '
                   f'in it — the one thing reading it cannot tell you. '
                   f'Choosing one records the other(s) as passed over.</p>'
                   f'{cards}')

    if not sets:
        blocks = ('<p class="muted">Nothing waiting. Pick a topic on '
                  '<a href="/scout">Scout</a> and three scripts about it land '
                  'here in a minute or two — or use '
                  '<a href="/generate">Make a video</a> for a topic of your '
                  'own.</p>')

    try:
        import db_manager as dbm
        decided = [c for c in dbm.candidates(limit=60)
                   if c["status"] != "pending"][:12]
    except Exception:
        decided = []
    old = ""
    for c in decided:
        old += (f'<tr><td>{_esc((c["topic"] or "")[:40])}</td>'
                f'<td class="muted">{_esc(c["hook_style"] or "")}</td>'
                f'<td class="muted">{_esc((c["hook"] or "")[:60])}</td>'
                f'<td class="muted">{c["score"]}/10</td>'
                f'<td class="muted">{_esc(c["status"])}</td></tr>')
    old_html = (f'<h2 style="margin-top:26px">Already ruled on</h2>'
                f'<p class="muted">Kept on purpose — the one that lost is half '
                f'of every pair.</p><table><tr><th>Topic</th><th>Style</th>'
                f'<th>Hook</th><th>Score</th><th></th></tr>{old}</table>'
                if old else "")

    body = f"""
    <a class="back" href="/">← back</a>
    <h2 style="margin-top:14px">Choose a script</h2>
    {_flow_bar("/scripts")}
    {_msg_banner()}
    {blocks}
    {old_html}
    """
    return _head() + body + PAGE_TAIL


@app.route("/scripts/<int:candidate_id>/choose", methods=["POST"])
def scripts_choose(candidate_id: int):
    """Chosen → two galleries drawn for THIS script, to choose between next.

    The script is written to a file because that is the seam main.py already
    has: --script makes it skip its writer, so the script a person actually
    read is the one the video is built from. Passing a topic instead would have
    the writer produce a fourth script nobody chose.

    The render still does not start here. It starts once the pictures have been
    chosen too — which is the same principle one stage further on: the
    irreversible expensive step waits behind the last human judgement, not the
    first.
    """
    auth.require("generate")
    try:
        import db_manager as dbm
        chosen = dbm.choose_candidate(candidate_id, by=_whoami())
        if not chosen:
            return redirect("/scripts?error=" + _urlquote(
                "that one is not pending — already decided?"))
        out_dir = paths.log_dir() / "chosen_scripts"
        out_dir.mkdir(parents=True, exist_ok=True)
        script_file = out_dir / f"candidate_{candidate_id}.txt"
        script_file.write_text(chosen["script"] or "", encoding="utf-8")
        log_path = _launch_galleries(script_file=str(script_file),
                                     candidate_id=candidate_id,
                                     topic=chosen["topic"] or "")
    except Exception as e:
        return redirect("/scripts?error=" + _urlquote(f"could not start: {e}"))
    return redirect("/galleries?msg=" + _urlquote(
        f'Drawing two galleries for "{chosen["topic"]}" — about forty minutes '
        f'of the GPU, so come back rather than wait. Log: logs/{log_path.name}'))


# ── Choosing the pictures ────────────────────────────────────────────────────

_GALLERY_TAG_RE = re.compile(r"^\s*\[SHOT\s*=\s*\w+\s*\]\s*")
# The framing sentence is written by the storyboard and is one of a handful of
# fixed strings, identical on every shot that shares a distance. Printed in
# full it is the first ninety characters of every line on the page, so the
# part that says what THIS picture is of starts past where the strip ends.
_FRAMING_RE = re.compile(
    r"^(wide shot|medium shot|close shot|close detail)\b[^.]*\.\s*", re.I)


def _shot_line(prompt: str, limit: int = 110) -> str:
    """One readable line describing the shot, for the strip above the draws.

    Three things come off: the [SHOT=…] tag, which is a note to the prompt
    builder and is stripped before the model sees it; the framing sentence,
    which is boilerplate repeated verbatim across every shot at that distance;
    and the tail past what fits. What is left is the sentence that differs
    between one shot and the next, which is the only part worth reading when
    you are deciding between two pictures of it.
    """
    text = _GALLERY_TAG_RE.sub("", (prompt or "").strip())
    stripped = _FRAMING_RE.sub("", text).strip()
    text = stripped or text          # a prompt that is ONLY framing keeps it
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


@app.route("/galleries")
def galleries_page():
    """Two complete galleries, side by side, one row per shot.

    THE SHOT IS THE UNIT OF CHOICE, and that is what makes two variants enough.
    A gallery of sixteen pictures is sixteen independent draws — A comes back
    best on shot 3 and worst on shot 9, B the other way round — so picking a
    whole bundle throws away the good half of the other. Pick a base in one
    click, then correct the shots where the other one won.
    """
    auth.require("view")
    try:
        import db_manager as dbm
        sets = dbm.gallery_sets(status="pending", limit=6)
    except Exception as e:
        body = (f'<a class="back" href="/">← back</a><h2 style="margin-top:14px">'
                f'Choose the pictures</h2><div class="msg error">'
                f'{_esc(str(e))}</div>')
        return _head() + body + PAGE_TAIL

    blocks = ""
    for gs in sets:
        try:
            images = dbm.gallery_images(gs["id"])
        except Exception:
            images = []
        by_beat: dict = {}
        for im in images:
            by_beat.setdefault(im["beat_index"], []).append(im)

        bases = ""
        if auth.can("generate"):
            for v in range(int(gs["n_variants"] or 2)):
                bases += (f'<form method="post" action="/galleries/{gs["id"]}'
                          f'/base/{v}" style="display:inline">'
                          f'<button type="submit">Take all from {chr(65+v)}'
                          f'</button></form> ')
            bases += (f'<form method="post" action="/galleries/{gs["id"]}/use" '
                      f'style="display:inline"><button class="btn save" '
                      f'type="submit">Make the video with this set</button>'
                      f'</form>')

        # TWO LINES PER SHOT, NOT A TABLE ROW. The old layout put the number
        # and a 150-character prompt in the first cell of a <table> with no
        # width on it, so that cell took whatever it wanted and shoved the
        # pictures — the only thing on the page anybody is actually judging —
        # into a strip at the far right. Line one says which shot this is and
        # what it is of; line two is the draws, side by side at a size you can
        # tell apart, which is the comparison the whole stage exists for.
        rows = ""
        for beat in sorted(by_beat):
            cells = ""
            for im in sorted(by_beat[beat], key=lambda r: r["variant"]):
                picked = im["status"] == "chosen"
                letter = chr(65 + im["variant"])
                if picked:
                    action = '<span class="draw-w">ships</span>'
                elif auth.can("generate"):
                    action = (f'<form method="post" action="/galleries/'
                              f'{gs["id"]}/swap/{beat}/{im["variant"]}">'
                              f'<button class="pick" type="submit">use '
                              f'{letter}</button></form>')
                else:
                    action = ""
                cells += (
                    f'<div class="draw{" won" if picked else ""}">'
                    f'<img src="/galleries/image/{im["id"]}" loading="lazy" '
                    f'alt="shot {beat+1} variant {letter}">'
                    f'<div class="draw-f"><span class="draw-v">{letter}</span>'
                    f'{action}</div></div>')
            # The [SHOT=…] tag is a note to the prompt builder about whether
            # this beat draws a person; it is stripped before the model ever
            # sees it, and it is noise to a reader deciding between two
            # pictures. One line, cut at the width the strip actually has.
            prompt = _shot_line(by_beat[beat][0]["prompt"] or "")
            rows += (f'<div class="shot">'
                     f'<div class="shot-h"><span class="shot-n">{beat+1}</span>'
                     f'<span class="shot-p">{_esc(prompt)}</span></div>'
                     f'<div class="draws">{cells}</div></div>')

        blocks += (f'<h2 style="margin-top:22px">{_esc(gs["topic"] or "—")}</h2>'
                   f'<p class="muted">{len(by_beat)} shot(s), '
                   f'{gs["n_variants"]} draw(s) each. Take a base, then swap '
                   f'the shots where the other one came out better — a green '
                   f'border is what ships. Every swap is recorded: the picture '
                   f'you passed over is half of a labelled pair, one per shot '
                   f'rather than one per video.</p>'
                   f'<div style="margin:10px 0">{bases}</div>'
                   f'{rows}')

    if not sets:
        blocks = ('<p class="muted">Nothing waiting. Choose a script on '
                  '<a href="/scripts">Choose a script</a> and two galleries '
                  'for it are drawn here — about forty minutes of the GPU, so '
                  'come back rather than wait.</p>')

    body = f"""
    <a class="back" href="/">← back</a>
    <h2 style="margin-top:14px">Choose the pictures</h2>
    {_flow_bar("/galleries")}
    {_msg_banner()}
    {blocks}
    """
    return _head() + body + PAGE_TAIL


@app.route("/galleries/image/<int:image_id>")
def galleries_image(image_id: int):
    """One gallery still. Served by row id rather than by path, so a filename
    from the query string can never address a file outside the set."""
    auth.require("view")
    try:
        import db_manager as dbm
        row = next((r for s in dbm.gallery_sets(status=None, limit=40)
                    for r in dbm.gallery_images(s["id"])
                    if r["id"] == image_id), None)
    except Exception:
        row = None
    if not row:
        abort(404)
    p = Path(row["path"])
    if not p.exists():
        abort(404)
    return send_from_directory(str(p.parent), p.name, max_age=_IMAGE_MAX_AGE)


@app.route("/galleries/<int:set_id>/base/<int:variant>", methods=["POST"])
def galleries_base(set_id: int, variant: int):
    auth.require("generate")
    import db_manager as dbm
    n = dbm.choose_gallery_base(set_id, variant, by=_whoami())
    if not n:
        return redirect("/galleries?error=" + _urlquote("no such variant"))
    return redirect("/galleries?msg=" + _urlquote(
        f"Taking all {n} shot(s) from {chr(65+variant)} — now swap the ones "
        f"the other draw won."))


@app.route("/galleries/<int:set_id>/swap/<int:beat>/<int:variant>",
           methods=["POST"])
def galleries_swap(set_id: int, beat: int, variant: int):
    auth.require("generate")
    import db_manager as dbm
    if not dbm.swap_gallery_beat(set_id, beat, variant, by=_whoami()):
        return redirect("/galleries?error=" + _urlquote(
            "that draw has no picture for this shot"))
    return redirect("/galleries?msg=" + _urlquote(f"Shot {beat+1} swapped."))


@app.route("/galleries/<int:set_id>/use", methods=["POST"])
def galleries_use(set_id: int):
    """Pictures settled → three reads of the hook, the last thing to choose.

    Refuses a set with a beat nobody picked. clip[i] belongs to beat[i] all the
    way downstream, so a short list does not lose one picture — it slides every
    later one onto the wrong sentence.
    """
    auth.require("generate")
    try:
        import db_manager as dbm
        rows = dbm.chosen_gallery(set_id)
        if not rows:
            return redirect("/galleries?error=" + _urlquote(
                "every shot needs a picked picture first — take a base"))
        gs = next((g for g in dbm.gallery_sets(status=None, limit=40)
                   if g["id"] == set_id), None)
        if not gs:
            return redirect("/galleries?error=" + _urlquote("no such set"))
        if not dbm.decide_gallery_set(set_id, "chosen", by=_whoami()):
            return redirect("/galleries?error=" + _urlquote("already decided"))
        log_path = _launch_voice_takes(script_file=gs["script_file"],
                                       set_id=set_id,
                                       topic=gs["topic"] or "")
    except Exception as e:
        return redirect("/galleries?error=" + _urlquote(f"could not start: {e}"))
    return redirect("/voice?msg=" + _urlquote(
        f'{len(rows)} picture(s) settled. Recording three reads of the hook — '
        f'seconds, not minutes. Log: logs/{log_path.name}'))


# ── One video, one page, five stages ─────────────────────────────────────────
#
# THE MESS THIS REPLACES, in the owner's words. Each choosing stage grew its
# own page, and separately they work — but a person making a video does not
# have four tasks, they have one. /scout, /scripts, /galleries and /voice are
# four places to stand in a queue, none of which knows what the others decided.
#
# So: one page, the current stage on it, and every earlier stage reachable by
# going back. The old pages stay — they are still the right view when you want
# every pending script across every project at once — but this is the one a
# video is actually made on.
#
# REGEN ON EVERY STAGE. The options at each step are samples, not answers, and
# the honest thing to do with a set you do not like is draw another. It rebuilds
# only the stage you are standing on.

# ── how far in, and how much longer ──────────────────────────────────────────
#
# "Reload this page" is not a design, it is an apology. Three of the five
# stages take real time — a minute or two for scripts, seconds for takes, and
# about forty minutes for two galleries — and the page said nothing about any
# of it. You pressed a button, the screen went quiet, and the only way to learn
# anything was to press F5 and guess.
#
# NOTHING NEW HAD TO BE INSTRUMENTED. Every one of those stages fills a table
# as it goes: three candidate rows, thirty-two gallery images, three takes. So
# progress is a row count against a target, and the target is known before the
# work starts. That is exact rather than estimated, and it costs one small
# query.
#
# The ETA is elapsed ÷ done × remaining and nothing cleverer. It is honest
# about being rough because it says "about", and after two or three images it
# is close enough to answer the only question anybody has, which is whether to
# wait or come back.

def _project_progress(p: dict) -> dict:
    """{working, done, total, label, eta_seconds} for the stage in progress."""
    import time as _time
    import db_manager as dbm

    out = {"working": False, "done": 0, "total": 0, "label": "",
           "eta_seconds": None, "starting": False, "stalled": False,
           "quiet_seconds": None, "stage": p.get("stage") or "topic"}
    stage = out["stage"]
    started = None
    try:
        if stage == "script":
            rows = dbm.candidates(project_id=p["id"], limit=20)
            out.update(done=len(rows), total=3, label="scripts written")
            started = rows[0]["created_at"] if rows else None
        elif stage == "gallery":
            sid = p.get("gallery_id")
            sets = [g for g in dbm.gallery_sets(status=None, limit=30)
                    if g.get("candidate_id") == p.get("script_id")]
            if not sid:
                sid = sets[0]["id"] if sets else None
            # created_at comes from the SET whichever way it was found — the
            # elapsed time is what the estimate divides by, and reading it only
            # on the not-yet-chosen path left every in-progress set with no ETA.
            started = next((g["created_at"] for g in sets if g["id"] == sid),
                           None)
            if sid:
                images = dbm.gallery_images(sid)
                # THE TARGET COMES FROM THE PLAN, NOT FROM THE OUTPUT. Both
                # halves of it were inferred from the pictures already drawn,
                # and both were wrong at the moment it mattered most: with
                # nothing drawn yet the target was zero, which reads as
                # finished — so the wizard showed the completed gallery view
                # over an empty table while ComfyUI was still rendering. And
                # counting distinct variants in a half-drawn set answers 1,
                # because the first variant finishes before the second starts.
                row = next((g for g in dbm.gallery_sets(status=None, limit=40)
                            if g["id"] == sid), None)
                variants = int((row or {}).get("n_variants") or 2)
                planned = int((row or {}).get("n_beats") or 0)
                if planned:
                    total = planned * variants
                else:
                    # A set from before n_beats existed, or one whose prompts
                    # are still being planned. Fall back to the widest row seen
                    # so far rather than to zero.
                    beats = len({im["beat_index"] for im in images})
                    total = max(len(images), beats * variants)
                out.update(done=len(images), total=total,
                           label="pictures drawn")
                # MEASURED FROM THE PICTURES, NOT FROM THE SET ROW. See
                # gallery_draw_rate: dividing by the time since the row was
                # written averages in three voice takes and a storyboard call,
                # which is how nine of thirty-eight came to report eleven hours
                # remaining at nineteen seconds a picture.
                rate = dbm.gallery_draw_rate(sid)
                if rate and total > len(images):
                    out["rate_seconds"] = round(rate, 1)
                    out["eta_seconds"] = int(rate * (total - len(images)))
                # STOPPED IS NOT SLOW. Nothing landing for minutes on an
                # unfinished set means the draw died — ComfyUI gone, the
                # subprocess killed, the machine asleep. Quoting an estimate
                # for that is how the owner came to turn ComfyUI OFF trying to
                # unstick it: the screen kept insisting work was in progress.
                quiet = dbm.seconds_since_last_picture(sid)
                if total and len(images) < total and quiet is not None \
                        and quiet > STALLED_AFTER_SECONDS:
                    out["stalled"] = True
                    out["quiet_seconds"] = int(quiet)
                    out["eta_seconds"] = None
                # THE BLIND WINDOW BEFORE THE FIRST PICTURE. gallery_variants
                # writes the set row first, then records three voice takes,
                # then calls the storyboard, and only THEN sets n_beats and
                # starts drawing. For those several minutes n_beats is 0 and
                # no image exists, so total was 0, `working` was False, and the
                # page showed the finished-gallery view with an empty table —
                # while the owner watched ComfyUI churning in another window
                # and asked why the dashboard said nothing.
                #
                # A set row that exists and is still pending IS work in
                # progress. What is unknown at that moment is the size of it,
                # not whether it is happening.
                if not total and (row or {}).get("status") == "pending":
                    out.update(starting=True,
                               label="recording the voice, then planning the "
                                     "shots")
        elif stage == "voice":
            rows = dbm.voice_takes(set_id=p.get("gallery_id"), limit=10)
            out.update(done=len(rows), total=3, label="takes recorded")
            started = rows[0]["created_at"] if rows else None
    except Exception:
        return out

    # `done < total` alone said False for a set with a target it had not
    # started, which is exactly the state a person is looking at when they
    # press the button and go and watch the ComfyUI console.
    out["working"] = (out["total"] > 0 and out["done"] < out["total"]) \
        or bool(out.get("starting"))
    if out.get("eta_seconds") is not None or out.get("stalled"):
        return out          # measured from real timestamps, or not moving
    if out["stage"] == "gallery":
        # No estimate rather than a made-up one. Dividing by the time since
        # the set row was written is what produced "~12h 10m left" for a draw
        # that was doing one picture every nineteen seconds, and the same sum
        # is still wrong for a set drawn before created_at existed.
        return out
    if not out["working"] or not out["done"] or not started:
        return out
    try:
        from datetime import datetime, timezone
        t0 = datetime.fromisoformat(str(started)).replace(tzinfo=timezone.utc)
        elapsed = max(1.0, _time.time() - t0.timestamp())
        per = elapsed / out["done"]
        out["eta_seconds"] = int(per * (out["total"] - out["done"]))
    except Exception:
        pass
    return out


def _eta_words(seconds) -> str:
    if seconds is None:
        return ""
    if seconds < 60:
        return "under a minute left"
    if seconds < 5400:
        return f"about {round(seconds / 60)} minute(s) left"
    return f"about {seconds / 3600:.1f} hour(s) left"


def _working_panel(p: dict, prog: dict, what: str) -> str:
    """The panel a stage shows while its work is still running."""
    done, total = prog["done"], prog["total"]
    pct = int(100 * done / total) if total else 0
    eta = _eta_words(prog.get("eta_seconds"))
    bar = (f'<div style="height:8px;border-radius:999px;background:var(--border);'
           f'overflow:hidden;margin:10px 0">'
           f'<div id="wizard-bar-fill" style="height:100%;width:{pct}%;'
           f'background:var(--accent);transition:width .4s"></div></div>')
    if total:
        counted = (f'<p class="muted" id="wizard-count">{done} of {total} '
                   f'{prog["label"]}{" · " + eta if eta else ""}</p>')
    else:
        # No target yet. Say what IS happening rather than nothing — an empty
        # line under a zero-width bar is indistinguishable from a stuck page,
        # and this phase runs for minutes.
        counted = (f'<p class="muted" id="wizard-count">'
                   f'{_esc(prog.get("label") or "starting")}&hellip;</p>')
    # The wrapper carries the project id and whether work is in flight; the
    # poller below reads both off it. Without it the page is static again and
    # the only way forward is the F5 this replaces.
    return (f'<div id="wizard-progress" data-project="{p["id"]}" '
            f'data-working="1">'
            f'<h2 style="margin-top:22px">{what}</h2>{bar}{counted}'
            f'<p class="muted">This page updates itself — leave it open, or '
            f'come back later. Nothing is lost either way.</p></div>')


@app.route("/api/create/<int:project_id>")
def api_create_progress(project_id: int):
    """What the wizard polls. Cheap: one project read and one count query."""
    auth.require("view")
    import db_manager as dbm
    p = dbm.project(project_id)
    if not p:
        return {"ok": False}, 404
    prog = _project_progress(p)
    prog.update(ok=True, id=project_id, title=p.get("title") or "",
                eta=_eta_words(prog.get("eta_seconds")))
    return prog


# Polls the endpoint above and reloads the page the moment the stage's work
# finishes. Vanilla and inline, like the status bar — this dashboard has no
# build step on purpose, and a page that needs a bundler to tell you it is
# still working is a page that tells you nothing when the bundler is missing.
WIZARD_POLL_JS = """
<script>
(function () {
  var el = document.getElementById('wizard-progress');
  if (!el) return;
  var pid = el.getAttribute('data-project');
  var wasWorking = el.getAttribute('data-working') === '1';
  function tick() {
    fetch('/api/create/' + pid, {headers: {'Accept': 'application/json'}})
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (d) {
        if (!d || !d.ok) return;
        // The stage finished, or moved on: show the real page rather than
        // guessing at what it now contains.
        if (wasWorking && !d.working) { location.reload(); return; }
        var pct = d.total ? Math.round(100 * d.done / d.total) : 0;
        var fill = document.getElementById('wizard-bar-fill');
        if (fill) fill.style.width = pct + '%';
        var txt = document.getElementById('wizard-count');
        if (txt && d.total) {
          txt.textContent = d.done + ' of ' + d.total + ' ' + d.label +
                            (d.eta ? ' · ' + d.eta : '');
        }
      })
      .catch(function () {});
  }
  tick();
  setInterval(tick, 4000);
})();
</script>
"""


_WIZARD_STAGES = (
    ("topic",   "Topic"),
    ("script",  "Script"),
    ("gallery", "Pictures"),
    ("voice",   "Voice"),
    ("render",  "Make it"),
)


def _wizard_bar(project: dict) -> str:
    """Where this project is, and a way back to anything already decided."""
    here = project.get("stage") or "topic"
    order = [s for s, _t in _WIZARD_STAGES]
    at = order.index(here) if here in order else 0
    cells = []
    for i, (stage, title) in enumerate(_WIZARD_STAGES):
        inner = f'<span class="flow-i">{i + 1}</span>{title}'
        if i < at:
            # Earlier stages are links back. Later ones are not decided yet and
            # a link to them would be a link to an empty page.
            cells.append(
                f'<form method="post" action="/create/{project["id"]}/back/'
                f'{stage}" style="display:inline"><button class="flow-step" '
                f'type="submit" title="go back and choose again">{inner}'
                f'</button></form>')
        else:
            state = " here" if i == at else ""
            cells.append(f'<span class="flow-step{state}">{inner}</span>')
    return f'<nav class="flow" aria-label="the five stages">{"".join(cells)}</nav>'


def _wizard_decided(p: dict) -> str:
    bits = []
    if p.get("title"):
        bits.append(f'<strong>{_esc(p["title"])}</strong>')
    for key, label in (("script_id", "script chosen"),
                       ("gallery_id", "pictures chosen"),
                       ("voice_id", "voice chosen")):
        if p.get(key):
            bits.append(label)
    return f'<p class="muted">{" · ".join(bits)}</p>' if bits else ""


def _regen(project_id: int, stage: str, label: str = "Give me another set") -> str:
    return (f'<form method="post" action="/create/{project_id}/regen/{stage}" '
            f'style="display:inline"><button type="submit">&#8635; {label}'
            f'</button></form>')


def _project_spans(p: dict) -> list:
    """The measured shot lengths, from any take recorded for this project.

    Any take: they were all measured against the same script and the same beat
    count, and the takes differ in pace by a few per cent. Which one is chosen
    changes the render's cut points, not whether shot three has room to be read.
    """
    import json as _json
    try:
        import db_manager as dbm
        for t in dbm.voice_takes(set_id=p.get("gallery_id"), limit=10):
            if t.get("spans"):
                return _json.loads(t["spans"])
    except Exception:
        pass
    return []


def _wizard_topic(p: dict) -> str:
    import db_manager as dbm
    opts = dbm.project_topics(p["id"])
    cards = ""
    for o in opts:
        cards += (
            f'<div class="card" style="width:100%;margin-bottom:10px">'
            f'<strong>{_esc(o["title"])}</strong>'
            f'<p class="muted" style="margin:6px 0">{_esc(o["why"] or "")}</p>'
            f'<form method="post" action="/create/{p["id"]}/topic/{o["id"]}">'
            f'<button class="btn save" type="submit">Make this one</button>'
            f'</form></div>')
    if not opts:
        cards = ('<p class="muted">No suggestions yet &mdash; press the button, '
                 'or type the one you already want.</p>')
    return (
        '<h2 style="margin-top:22px">What is it about?</h2>'
        '<p class="muted">Three worth making, each saying why it is here. Or '
        'skip all of it and name the one you want &mdash; a topic you type is '
        'not checked against trends or against what you have already made, '
        'because you said what you want.</p>'
        f'<div style="margin:10px 0">'
        f'<form method="post" action="/create/{p["id"]}/regen/topic" '
        f'style="display:inline"><button class="btn save" type="submit">'
        f'Suggest 3 topics</button></form></div>'
        f'{cards}'
        f'<form method="post" action="/create/{p["id"]}/topic/custom" '
        f'style="display:flex;gap:8px;flex-wrap:wrap;align-items:flex-end;'
        f'margin-top:16px">'
        f'<div style="flex:1;min-width:220px">'
        f'<label for="topic">&hellip;or a topic of your own</label>'
        f'<input class="field" style="margin:6px 0 0" type="text" id="topic" '
        f'name="topic" placeholder="e.g. Bretton Woods, Tulip mania" required>'
        f'</div><button class="btn save" type="submit" style="height:38px">'
        f'Use this</button></form>')


def _wizard_script(p: dict) -> str:
    import db_manager as dbm
    rows = dbm.candidates(project_id=p["id"], status="pending", limit=20)
    title = _esc(p.get("title") or "")
    if not rows:
        prog = _project_progress(p)
        return (
            _working_panel(p, prog,
                           f"Writing three scripts about &ldquo;{title}&rdquo;")
            + '<p class="muted">One per hook style. If nothing arrives, '
              '<a href="/logs">Logs</a> says why.</p>'
            + f'<div style="margin:10px 0">'
              f'{_regen(p["id"], "script", "Try again")}</div>')
    cards = ""
    for c in rows:
        warn = ""
        if not c.get("fact_ok", 1):
            warn = (f'<div class="msg error" style="margin:8px 0">&#9888; the '
                    f'source does not support this: '
                    f'{_esc(c.get("fact_reason") or "unstated")}</div>')
        words = len((c["script"] or "").split())
        cards += (
            f'<div class="card" style="width:100%;margin-bottom:10px">'
            f'<div style="display:flex;justify-content:space-between;gap:12px">'
            f'<strong>{_esc(c["hook"] or "—")}</strong>'
            f'<span class="muted">{_esc(c["hook_style"] or "unpinned")} · '
            f'{c["score"]}/10 · {words}w</span></div>{warn}'
            f'<pre style="white-space:pre-wrap;font-size:13px;margin:8px 0">'
            f'{_esc(c["script"] or "")}</pre>'
            f'<form method="post" action="/create/{p["id"]}/script/{c["id"]}">'
            f'<button class="btn save" type="submit">Make this one</button>'
            f'</form></div>')
    return (
        f'<h2 style="margin-top:22px">Three scripts about '
        f'&ldquo;{title}&rdquo;</h2>'
        '<p class="muted">All three are about the topic you chose; they differ '
        'in how they open. The score is shown, not enforced &mdash; nothing '
        'was withheld for missing a bar, because that is your call. A red '
        'warning means the source does not support a claim in it, which is the '
        'one thing reading it cannot tell you.</p>'
        f'<div style="margin:10px 0">{_regen(p["id"], "script")}</div>{cards}')


def _wizard_gallery(p: dict) -> str:
    import db_manager as dbm
    sid = p.get("gallery_id")
    if not sid:
        sets = [g for g in dbm.gallery_sets(status=None, limit=30)
                if g.get("candidate_id") == p.get("script_id")]
        sid = sets[0]["id"] if sets else None
    if not sid:
        prog = _project_progress(p)
        return (
            _working_panel(p, prog,
                           "Recording the voice, then drawing the pictures")
            + '<p class="muted">The voice is recorded first so the shot '
              'lengths are measured from real audio rather than guessed from '
              'the word count &mdash; which take gets used is the last thing '
              'you choose.</p>'
            + f'<div style="margin:10px 0">'
              f'{_regen(p["id"], "gallery", "Start again")}</div>')
    images = dbm.gallery_images(sid)
    spans = _project_spans(p)
    by_beat: dict = {}
    for im in images:
        by_beat.setdefault(im["beat_index"], []).append(im)
    bases = ""
    for v in sorted({im["variant"] for im in images}):
        bases += (f'<form method="post" action="/galleries/{sid}/base/{v}" '
                  f'style="display:inline"><button type="submit">Take all from '
                  f'{chr(65 + v)}</button></form> ')
    rows = ""
    for beat in sorted(by_beat):
        cells = ""
        for im in sorted(by_beat[beat], key=lambda r: r["variant"]):
            picked = im["status"] == "chosen"
            swap = ""
            if not picked:
                swap = (f'<form method="post" action="/galleries/{sid}/swap/'
                        f'{beat}/{im["variant"]}"><button type="submit">'
                        f'use this one</button></form>')
            cells += (
                f'<td style="vertical-align:top;padding:6px;border:2px solid '
                f'{"var(--ok)" if picked else "transparent"}">'
                f'<img src="/galleries/image/{im["id"]}" style="width:150px;'
                f'max-width:40vw;display:block" alt="shot {beat + 1}">'
                f'<div class="muted" style="font-size:12px">'
                f'{chr(65 + im["variant"])}{" · chosen" if picked else ""}'
                f'</div>{swap}</td>')
        secs = ""
        if beat < len(spans):
            secs = (f'<br><span style="font-size:11px">'
                    f'{spans[beat]["seconds"]:.1f}s</span>')
        prompt = (by_beat[beat][0]["prompt"] or "")[:120]
        rows += (f'<tr><td class="muted" style="vertical-align:top">{beat + 1}'
                 f'{secs}<br><span style="font-size:11px">{_esc(prompt)}'
                 f'</span></td>{cells}</tr>')
    # STILL DRAWING? SHOW IT, AND SHOW WHAT HAS ARRIVED. The set row is written
    # before the first picture, so as soon as drawing starts this branch is the
    # one that renders — and it used to render a half-empty table with nothing
    # to say that more was coming. Forty minutes of that reads as broken.
    prog = _project_progress(p)
    if prog.get("working"):
        # WATCH IT SOMEWHERE THAT IS ONLY THAT. Forty minutes of GPU behind a
        # wizard step with a bar and a sentence is why the owner watched
        # ComfyUI's console in another window instead.
        return (
            _working_panel(p, prog, "Drawing the pictures")
            + f'<p style="margin:14px 0"><a class="btn save" '
              f'style="text-decoration:none;display:inline-block" '
              f'href="/drawing/{sid}">Watch it draw &rarr;</a></p>'
            + '<p class="muted">Every picture appears the moment it lands. '
              'You can leave that page open, or close it &mdash; the drawing '
              'does not depend on it.</p>'
            + f'<div style="margin:10px 0">'
              f'{_regen(p["id"], "gallery", "Start again")}</div>')

    return (
        '<h2 style="margin-top:22px">Which pictures?</h2>'
        '<p class="muted">Two complete draws of the same shots. Take one as a '
        'base, then swap the shots the other won &mdash; a green border is what '
        'ships. The seconds beside each shot are measured from the recorded '
        'voice, not guessed, so you can see which pictures get time to be '
        'read.</p>'
        f'<div style="margin:10px 0">{bases}{_regen(p["id"], "gallery")}</div>'
        f'<div style="overflow-x:auto"><table>{rows}</table></div>'
        f'<form method="post" action="/create/{p["id"]}/gallery/{sid}" '
        f'style="margin-top:14px"><button class="btn save" type="submit">'
        f'These pictures</button></form>')


def _wizard_voice(p: dict) -> str:
    import db_manager as dbm
    takes = dbm.voice_takes(set_id=p.get("gallery_id"), status="pending",
                            limit=10)
    if not takes:
        prog = _project_progress(p)
        return (
            _working_panel(p, prog, "Recording three takes")
            + '<p class="muted">Whole voiceovers, ready to use &mdash; they '
              'differ in pace and weight.</p>'
            + f'<div style="margin:10px 0">'
              f'{_regen(p["id"], "voice", "Try again")}</div>')
    cards = ""
    for t in takes:
        secs = f' &middot; {t["seconds"]:.0f}s' if t.get("seconds") else ""
        cards += (
            f'<div class="card" style="width:100%;margin-bottom:10px">'
            f'<strong>{_esc(t["tone"] or "—")}</strong>'
            f'<span class="muted">{secs}</span>'
            f'<audio controls preload="none" style="display:block;width:100%;'
            f'margin:8px 0" src="/voice/audio/{t["id"]}"></audio>'
            f'<form method="post" action="/create/{p["id"]}/voice/{t["id"]}">'
            f'<button class="btn save" type="submit">Use this take</button>'
            f'</form></div>')
    return (
        '<h2 style="margin-top:22px">Which read?</h2>'
        '<p class="muted">Three complete voiceovers of the same script, '
        'differing in pace and weight. Skip through rather than sitting '
        'through &mdash; what you are judging is audible in ten seconds '
        'anywhere in the file. The tone you pick also sizes this video&rsquo;s '
        'pauses and grades its pictures.</p>'
        f'<div style="margin:10px 0">{_regen(p["id"], "voice")}</div>{cards}')


def _wizard_render(p: dict) -> str:
    return (
        '<h2 style="margin-top:22px">Everything is chosen</h2>'
        '<p class="muted">Nothing below gets regenerated &mdash; the script you '
        'read, the pictures you picked shot by shot and the take you listened '
        'to are what gets made. It lands in the review queue like any other '
        'video.</p>'
        f'{_wizard_decided(p)}'
        f'<form method="post" action="/create/{p["id"]}/render" '
        f'style="margin-top:14px"><button class="btn save" type="submit">'
        f'Make the video</button></form>')


_WIZARD_BODY = {"topic": _wizard_topic, "script": _wizard_script,
                "gallery": _wizard_gallery, "voice": _wizard_voice,
                "render": _wizard_render}


@app.route("/create")
def create_page():
    """One video, one page, five stages."""
    auth.require("view")
    import db_manager as dbm
    pid = request.args.get("project", type=int)
    try:
        open_ones = dbm.projects(status="open", limit=10)
        p = dbm.project(pid) if pid else (open_ones[0] if open_ones else None)
    except Exception as e:
        return _head() + ('<a class="back" href="/">&larr; back</a>'
                          '<h2 style="margin-top:14px">Make a video</h2>'
                          f'<div class="msg error">{_esc(str(e))}</div>'
                          ) + PAGE_TAIL

    if not p:
        body = (
            '<a class="back" href="/">&larr; back</a>'
            '<h2 style="margin-top:14px">Make a video</h2>'
            f'{_msg_banner()}'
            '<p class="muted">Five decisions, one at a time: what it is about, '
            'which script, which pictures, which read, then make it. You can go '
            'back to any decision you have already made and choose again.</p>'
            '<form method="post" action="/create/new" style="margin-top:14px">'
            '<button class="btn save" type="submit">Start a video</button>'
            '</form>'
            f'{_stale_notice()}')
        return _head() + body + PAGE_TAIL

    stage = p.get("stage") or "topic"
    try:
        inner = _WIZARD_BODY.get(stage, _wizard_topic)(p)
    except Exception as e:
        # One stage that cannot render must not cost the way back out of it.
        inner = f'<div class="msg error">{_esc(str(e))}</div>'

    others = ""
    if len(open_ones) > 1:
        links = " &middot; ".join(
            f'<a href="/create?project={o["id"]}">'
            f'{_esc(o["title"] or ("#" + str(o["id"])))}</a>'
            for o in open_ones)
        others = f'<p class="muted">Also open: {links}</p>'

    body = (
        '<a class="back" href="/">&larr; back</a>'
        '<h2 style="margin-top:14px">Make a video</h2>'
        f'{_msg_banner()}{_wizard_bar(p)}{_wizard_decided(p)}{inner}'
        f'<form method="post" action="/create/{p["id"]}/abandon" '
        f'style="margin-top:26px"><button type="submit">Abandon this one'
        f'</button></form>{others}{WIZARD_POLL_JS}')
    return _head() + body + PAGE_TAIL


# ── the wizard's verbs ───────────────────────────────────────────────────────
#
# One rule runs through all of them: choosing at a stage advances to the next,
# and going back to a stage forgets it and everything after it. The second half
# is the one that is easy to skip and impossible to live without — pictures
# drawn for a script you have replaced are not stale, they are pictures of a
# different video.

def _project_or_home(project_id: int):
    import db_manager as dbm
    p = dbm.project(project_id)
    if not p:
        return None, redirect("/create?error=" + _urlquote("no such project"))
    return p, None


@app.route("/create/new", methods=["POST"])
def create_new():
    auth.require("generate")
    import db_manager as dbm
    import research
    try:
        niche = research._load_niche()[1]
    except Exception:
        niche = "money_history"
    try:
        from channel_config import load_channel
        channel = load_channel().id
    except Exception:
        channel = "main_en"
    pid = dbm.new_project(channel=channel, niche=niche, by=_whoami())
    return redirect(f"/create?project={pid}")


def _stale_notice() -> str:
    """Offer the sweep only when there is something to sweep.

    A permanently visible "tidy up" button is a chore the page invents; one
    that appears because twenty-one reads really are waiting from sittings
    that ended days ago is the page telling you something true.

    Counted by asking the sweep itself on a throwaway basis would mean doing
    the work to decide whether to offer it, so this mirrors its rule instead:
    anything pending that is not part of an open project.
    """
    try:
        import db_manager as dbm
        open_ids = {p["id"] for p in dbm.projects(status="open", limit=200)}
        live_cands = {c["id"] for c in dbm.candidates(limit=1000)
                      if c.get("project_id") in open_ids}
        live_sets = {g["id"] for g in dbm.gallery_sets(status=None, limit=500)
                     if g.get("candidate_id") in live_cands}
        stale = sum(1 for c in dbm.candidates(status="pending", limit=500)
                    if c.get("project_id") not in open_ids)
        stale += sum(1 for g in dbm.gallery_sets(status="pending", limit=500)
                     if g["id"] not in live_sets)
        stale += sum(1 for t in dbm.voice_takes(status="pending", limit=500)
                     if t.get("set_id") not in live_sets)
    except Exception:
        return ""
    if not stale:
        return ""
    return (f'<div class="msg" style="margin-top:22px">'
            f'<b>{stale}</b> option(s) are still waiting in the queues from '
            f'sittings you have already finished or abandoned. They make '
            f'/scripts, /galleries and /voice look busier than they are.'
            f'<form method="post" action="/create/tidy" style="margin-top:10px">'
            f'<button type="submit">Clear them out</button></form>'
            f'<div class="muted" style="margin-top:6px">Nothing is deleted '
            f'&mdash; they stop being offered. Anything belonging to a video '
            f'you have open is left alone.</div></div>')


# ── the poller ───────────────────────────────────────────────────────────────
#
# Vanilla and inline, like every other script here: no build step, and it works
# on a phone with nothing but the tailnet.
#
# ONLY WHAT CHANGED IS TOUCHED. Rewriting the whole grid every two seconds
# would restart every image download and make the page flicker — which is the
# "not smooth" the owner is describing. Slots are created once and then only
# filled.
DRAWING_JS = """
<script>
(function () {
  var root = document.getElementById('draw');
  if (!root) return;
  var setId = root.getAttribute('data-set');
  var grid  = document.getElementById('draw-grid');
  var built = false;

  function fmt(s) {
    s = Math.max(0, Math.round(s));
    if (s < 60) return s + 's';
    var m = Math.floor(s / 60);
    if (m < 60) return m + 'm';
    return Math.floor(m / 60) + 'h ' + (m % 60) + 'm';
  }

  function key(sl) { return sl.variant + '_' + sl.beat; }

  function build(d) {
    grid.innerHTML = '';
    d.slots.forEach(function (sl) {
      var cell = document.createElement('div');
      cell.className = 'draw-cell';
      cell.id = 'c' + key(sl);
      cell.innerHTML = '<div class="draw-slot"></div>'
        + '<div class="draw-tag">' + String.fromCharCode(65 + sl.variant)
        + (sl.beat + 1) + '</div>';
      grid.appendChild(cell);
    });
    built = true;
  }

  function render(d) {
    if (!d.ok) return;
    document.getElementById('draw-done').textContent = d.done;
    document.getElementById('draw-total').textContent = d.total || '?';
    var pct = d.total ? Math.round(100 * d.done / d.total) : 0;
    document.getElementById('draw-fill').style.width = pct + '%';

    var sub = document.getElementById('draw-sub');
    if (d.finished) {
      sub.textContent = 'Done — every picture is drawn.';
    } else if (d.stalled) {
      // STOPPED IS NOT SLOW. Quoting an estimate here is what made the owner
      // turn ComfyUI off trying to unstick a draw that had already died.
      root.classList.add('stalled');
      sub.innerHTML = 'Stopped — nothing has arrived for ' + fmt(d.quiet_seconds)
        + '. ' + (d.gpu_up === false
            ? 'ComfyUI is not answering: start it, then draw again.'
            : 'ComfyUI is answering, so the run itself ended — draw again.');
    } else if (!d.total) {
      sub.textContent = 'Recording the voice, then planning the shots…';
    } else if (d.eta_seconds != null) {
      sub.textContent = 'about ' + fmt(d.eta_seconds) + ' left';
    } else {
      sub.textContent = 'drawing…';
    }
    document.getElementById('draw-rate').textContent =
      d.rate_seconds ? d.rate_seconds + 's a picture' : '';

    if (!built && d.slots.length) build(d);
    // Fill only the slots that have arrived and are still empty. Re-setting a
    // src that is already correct makes the browser re-fetch and the picture
    // blink, which across thirty-eight of them looks like a page fighting
    // itself.
    d.slots.forEach(function (sl) {
      if (!sl.image) return;
      var cell = document.getElementById('c' + key(sl));
      if (!cell || cell.getAttribute('data-filled')) return;
      var slot = cell.querySelector('.draw-slot');
      slot.innerHTML = '<img loading="lazy" src="/galleries/image/'
        + sl.image + '" alt="">';
      cell.setAttribute('data-filled', '1');
      cell.className = 'draw-cell in';
    });

    var lg = document.getElementById('draw-log');
    lg.textContent = (d.log || []).join('\\n');

    if (d.finished) {
      clearInterval(timer);
      sub.innerHTML = 'Done &mdash; <a href="/create">choose the pictures &rarr;</a>';
    }
    var again = document.getElementById('draw-again');
    if (again) { again.style.display = d.stalled ? '' : 'none'; }
  }

  function poll() {
    fetch('/api/drawing/' + setId, { headers: { 'Accept': 'application/json' } })
      .then(function (r) { return r.ok ? r.json() : Promise.reject(r.status); })
      .then(render)
      .catch(function () {
        document.getElementById('draw-sub').textContent =
          'Lost the connection — it keeps drawing; this page will catch up.';
      });
  }
  poll();
  // Two seconds. A picture lands every ~19s, so this is responsive without
  // being wasteful, and the payload is one small query plus a log tail.
  var timer = setInterval(poll, 2000);
})();
</script>
"""


# ── the drawing room ─────────────────────────────────────────────────────────
#
# A PAGE THAT IS ONLY THE DRAWING. Forty minutes of GPU is the longest wait in
# this pipeline and it used to happen behind a wizard step that showed a bar
# and a sentence. The owner watched ComfyUI's console in another window
# instead, which is the honest verdict on that: the dashboard was not showing
# the thing he wanted to watch.
#
# So the pictures themselves are the progress. Each one appears the moment it
# lands, in its own slot, and the slots that have not arrived are visible as
# empty frames — so the shape of what is coming is there from the first second
# rather than assembling itself out of nothing.
#
# IT UPDATES IN PLACE. A reloading page loses your scroll position every few
# seconds, which is what "not smooth" means when the thing you are doing is
# watching. One small JSON poll, and only the slots that changed are touched.

@app.route("/drawing/<int:set_id>")
def drawing_page(set_id: int):
    """Watch a gallery being drawn."""
    auth.require("view")
    import db_manager as dbm
    row = next((g for g in dbm.gallery_sets(status=None, limit=200)
                if g["id"] == set_id), None)
    if not row:
        return _head() + ('<a class="back" href="/">&larr; back</a>'
                          '<div class="msg error">No such gallery.</div>'
                          ) + PAGE_TAIL
    # The project that owns this set, so "draw it again" has somewhere to
    # post. A set reaches its project through the chosen script rather than
    # carrying a project id of its own.
    project_id = None
    try:
        cid = row.get("candidate_id")
        if cid:
            cand = next((c for c in dbm.candidates(limit=500)
                         if c["id"] == cid), None)
            project_id = (cand or {}).get("project_id")
    except Exception:
        pass
    again = ""
    if project_id and auth.can("generate"):
        again = (f'<div id="draw-again" style="display:none;margin-bottom:16px">'
                 f'<form method="post" action="/create/{project_id}/regen/gallery">'
                 f'<button class="btn save" type="submit">Draw it again</button>'
                 f'</form><div class="muted" style="margin-top:6px">The '
                 f'pictures already drawn are kept; this starts a fresh pair '
                 f'of draws.</div></div>')
    body = f"""
    <a class="back" href="/create">&larr; back to the video</a>
    <h2 style="margin-top:14px">Drawing &mdash; {_esc(row["topic"] or "")}</h2>
    <div id="draw" data-set="{set_id}">
      <div class="draw-head">
        <div>
          <div class="draw-n"><span id="draw-done">&mdash;</span>
            <span class="draw-of">of <span id="draw-total">&mdash;</span></span></div>
          <div class="muted" id="draw-sub">looking&hellip;</div>
        </div>
        <div class="draw-rate muted" id="draw-rate"></div>
      </div>
      <div class="progress" style="min-width:100%;height:8px;margin:0 0 18px">
        <i id="draw-fill" style="width:0%"></i></div>
      {again}
      <div id="draw-grid" class="draw-grid"></div>
      <pre id="draw-log" class="draw-log"></pre>
    </div>
    {DRAWING_JS}
    """
    return _head() + body + PAGE_TAIL


@app.route("/api/drawing/<int:set_id>")
def api_drawing(set_id: int):
    """Everything the drawing page redraws itself from, in one small payload."""
    auth.require("view")
    import db_manager as dbm
    row = next((g for g in dbm.gallery_sets(status=None, limit=200)
                if g["id"] == set_id), None)
    if not row:
        return {"ok": False}, 404
    images = dbm.gallery_images(set_id)
    variants = int(row.get("n_variants") or 2)
    beats = int(row.get("n_beats") or 0)
    total = beats * variants
    rate = dbm.gallery_draw_rate(set_id)
    done = len(images)
    quiet = dbm.seconds_since_last_picture(set_id)
    finished = bool(total and done >= total)
    # Not moving, and not because it is done. The renderer is what to check —
    # so say that, and say whether it is even answering.
    stalled = bool(total and not finished and quiet is not None
                   and quiet > STALLED_AFTER_SECONDS)
    gpu_up = _comfyui_reachable() if stalled else None
    # THE SLOTS ARE THE POINT. Sending the full grid — arrived and not — lets
    # the page show the shape of the job from the first poll instead of
    # growing a row at a time out of nothing.
    have = {(im["variant"], im["beat_index"]): im["id"] for im in images}
    slots = []
    if beats:
        for v in range(variants):
            for b in range(beats):
                slots.append({"variant": v, "beat": b,
                              "image": have.get((v, b))})
    else:
        # No plan yet: show what has arrived, if anything, and nothing else.
        for im in images:
            slots.append({"variant": im["variant"], "beat": im["beat_index"],
                          "image": im["id"]})
    return {
        "ok": True,
        "done": done,
        "total": total,
        "beats": beats,
        "variants": variants,
        "finished": finished,
        "stalled": stalled,
        "quiet_seconds": int(quiet) if quiet is not None else None,
        "gpu_up": gpu_up,
        "rate_seconds": round(rate, 1) if rate else None,
        "eta_seconds": (int(rate * (total - done))
                        if rate and total > done and not stalled else None),
        "slots": slots,
        "log": _newest_log_lines(6),
        "status": row.get("status") or "",
    }, 200


@app.route("/create/tidy", methods=["POST"])
def create_tidy():
    """Clear the queues of everything not belonging to an open project.

    Retiring on close fixes this going forward and does nothing about what has
    already piled up. This is the sweep for that, and it is safe to press at
    any time: it only ever touches rows whose project is finished or
    abandoned, and rows with no project at all are left alone.
    """
    auth.require("generate")
    import db_manager as dbm
    try:
        n = dbm.retire_all_stale_options()
    except Exception as e:
        return redirect("/create?error=" + _urlquote(f"could not tidy: {e}"))
    if not n:
        return redirect("/create?msg=" + _urlquote(
            "Nothing stale — every option waiting belongs to a project you "
            "still have open."))
    return redirect("/create?msg=" + _urlquote(
        f"{n} option(s) from finished or abandoned projects left the queues. "
        f"Nothing was deleted."))


@app.route("/create/<int:project_id>/abandon", methods=["POST"])
def create_abandon(project_id: int):
    """Abandoned, not deleted. What was drawn and written stays on disk and in
    the tables — it cost real money and real GPU, and a project you gave up on
    is still the record of three scripts somebody read and rejected."""
    auth.require("generate")
    import db_manager as dbm
    dbm.update_project(project_id, status="abandoned")
    # AND IT STOPS ASKING. The stage queues count every pending row in the
    # database with no idea which project it belongs to, so a project's
    # leftovers used to sit in /scripts, /galleries and /voice forever — next
    # to today's, making the badges say seven reads are waiting when one is.
    n = dbm.retire_project_options(project_id)
    tail = f" {n} leftover option(s) left the queues." if n else ""
    return redirect("/create?msg=" + _urlquote(
        f"Abandoned. Nothing was deleted.{tail}"))


@app.route("/create/<int:project_id>/back/<stage>", methods=["POST"])
def create_back(project_id: int, stage: str):
    auth.require("generate")
    import db_manager as dbm
    try:
        dbm.clear_project_from(project_id, stage)
    except ValueError as e:
        return redirect(f"/create?project={project_id}&error=" + _urlquote(str(e)))
    return redirect(f"/create?project={project_id}&msg=" + _urlquote(
        f"Back at the {stage} stage — everything after it is forgotten."))


def _whoami() -> str:
    """The signed-in name, for the decided_by column. "" when nobody is named.

    Empty rather than a placeholder: a NULL in that column honestly means
    "this predates attribution, or auth was off", and inventing "local" or
    "unknown" would make an anonymous decision indistinguishable from one
    somebody actually made. Every read treats a blank as "not recorded".
    """
    return (auth.current_user() or {}).get("name", "") or ""


def _await_new_gallery(script_id, timeout: float = 12.0):
    """The id of the set the launch is about to create, or None.

    gallery_variants writes its set row as its first act, but it is a separate
    process and that takes a second or two to start. Redirecting immediately
    would land on a page for a set that does not exist yet; guessing the next
    id would be wrong the moment two things run at once. So this waits for the
    row to actually appear.

    Bounded, and None on timeout: a launch that failed must send you back to
    the wizard with the buttons on it, not to a room that will never fill.
    """
    if not script_id:
        return None
    import db_manager as dbm
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            for g in dbm.gallery_sets(status="pending", limit=10):
                if g.get("candidate_id") == script_id:
                    return g["id"]
        except Exception:
            pass
        time.sleep(0.4)
    return None


def _replacing(n: int, noun: str) -> str:
    """", replacing the 3 you had" — or nothing at all when there were none.

    Said out loud because the alternative is silent: somebody who waited forty
    minutes for a gallery and pressed the button next to it is entitled to be
    told the old one just went away, rather than discovering it later.
    """
    if not n:
        return ""
    return f", replacing the {n} {noun}{'' if n == 1 else 's'} you had"


@app.route("/create/<int:project_id>/regen/<stage>", methods=["POST"])
def create_regen(project_id: int, stage: str):
    """Draw this stage again. The options are samples, not answers."""
    auth.require("generate")
    import db_manager as dbm
    p, bail = _project_or_home(project_id)
    if bail:
        return bail
    try:
        if stage == "topic":
            import topic_options
            opts = topic_options.suggest(p["niche"] or "money_history", 3)
            n = dbm.save_project_topics(project_id, opts)
            dbm.update_project(project_id, stage="topic")
            return redirect(f"/create?project={project_id}&msg="
                            + _urlquote(f"{n} topic(s) to choose from."))
        # AGAIN MEANS INSTEAD, NOT AS WELL. The old options are retired before
        # the new ones are drawn, so this stage keeps offering one choice of
        # the size it was designed around. Without it a second press left six
        # scripts, or two complete galleries stacked on /galleries, and the
        # stage built to narrow a decision widened it every time it was used.
        if stage == "script":
            if not p.get("title"):
                return redirect(f"/create?project={project_id}&error="
                                + _urlquote("choose a topic first"))
            gone = dbm.superseded_by_regen(project_id, "script")
            log = _launch_candidates(topic=p["title"], proposal_id=None,
                                     channel=p["channel"],
                                     project_id=project_id)
            return redirect(f"/create?project={project_id}&msg=" + _urlquote(
                f"Writing three scripts{_replacing(gone, 'script')} — a minute "
                f"or two. Log: logs/{log.name}"))
        if stage == "gallery":
            if not p.get("script_file"):
                return redirect(f"/create?project={project_id}&error="
                                + _urlquote("choose a script first"))
            gone = dbm.superseded_by_regen(project_id, "gallery")
            _launch_galleries(script_file=p["script_file"],
                              candidate_id=p.get("script_id"),
                              topic=p.get("title") or "")
            # STRAIGHT INTO THE ROOM. The set row is written by the subprocess
            # a moment from now, so this waits for it rather than guessing an
            # id — and falls back to the wizard if it never appears, which is
            # the case where the launch itself failed.
            sid = _await_new_gallery(p.get("script_id"))
            if sid:
                return redirect(f"/drawing/{sid}")
            return redirect(f"/create?project={project_id}&msg=" + _urlquote(
                f"Recording the voice, then drawing two galleries"
                f"{_replacing(gone, 'set')} — about forty minutes."))
        if stage == "voice":
            if not p.get("gallery_id") or not p.get("script_file"):
                return redirect(f"/create?project={project_id}&error="
                                + _urlquote("settle the pictures first"))
            gone = dbm.superseded_by_regen(project_id, "voice")
            log = _launch_voice_takes(script_file=p["script_file"],
                                      set_id=p["gallery_id"],
                                      topic=p.get("title") or "")
            return redirect(f"/create?project={project_id}&msg=" + _urlquote(
                f"Recording three takes{_replacing(gone, 'take')}. "
                f"Log: logs/{log.name}"))
    except Exception as e:
        return redirect(f"/create?project={project_id}&error="
                        + _urlquote(f"could not start: {e}"))
    return redirect(f"/create?project={project_id}&error="
                    + _urlquote(f"nothing to regenerate at {stage!r}"))


@app.route("/create/<int:project_id>/topic/custom", methods=["POST"])
def create_topic_custom(project_id: int):
    """The topic you already wanted. No trending check, no de-duplication."""
    auth.require("generate")
    import db_manager as dbm
    import topic_options
    p, bail = _project_or_home(project_id)
    if bail:
        return bail
    typed = (request.form.get("topic") or "").strip()
    if not typed:
        return redirect(f"/create?project={project_id}&error="
                        + _urlquote("type a topic"))
    got = topic_options.take_topic(typed, p["niche"] or "money_history")
    dbm.update_project(project_id, title=got["title"],
                       topic_source="typed", stage="script")
    return redirect(f"/create/{project_id}/regen/script", code=307)


@app.route("/create/<int:project_id>/topic/<int:topic_id>", methods=["POST"])
def create_topic_choose(project_id: int, topic_id: int):
    auth.require("generate")
    import db_manager as dbm
    got = dbm.choose_project_topic(topic_id)
    if not got:
        return redirect(f"/create?project={project_id}&error="
                        + _urlquote("that one is not pending"))
    dbm.update_project(project_id, title=got["title"],
                       topic_source="suggested", stage="script")
    return redirect(f"/create/{project_id}/regen/script", code=307)


@app.route("/create/<int:project_id>/script/<int:candidate_id>",
           methods=["POST"])
def create_script_choose(project_id: int, candidate_id: int):
    auth.require("generate")
    import db_manager as dbm
    chosen = dbm.choose_candidate(candidate_id, by=_whoami())
    if not chosen:
        return redirect(f"/create?project={project_id}&error="
                        + _urlquote("that script is not pending"))
    out_dir = paths.log_dir() / "chosen_scripts"
    out_dir.mkdir(parents=True, exist_ok=True)
    script_file = out_dir / f"candidate_{candidate_id}.txt"
    script_file.write_text(chosen["script"] or "", encoding="utf-8")
    dbm.update_project(project_id, script_id=candidate_id,
                       script_file=str(script_file), stage="gallery")
    return redirect(f"/create/{project_id}/regen/gallery", code=307)


@app.route("/create/<int:project_id>/gallery/<int:set_id>", methods=["POST"])
def create_gallery_choose(project_id: int, set_id: int):
    auth.require("generate")
    import db_manager as dbm
    rows = dbm.chosen_gallery(set_id)
    if not rows:
        return redirect(f"/create?project={project_id}&error=" + _urlquote(
            "every shot needs a picked picture first — take a base"))
    dbm.decide_gallery_set(set_id, "chosen", by=_whoami())
    dbm.update_project(project_id, gallery_id=set_id, stage="voice")
    return redirect(f"/create?project={project_id}&msg=" + _urlquote(
        f"{len(rows)} picture(s) settled."))


@app.route("/create/<int:project_id>/voice/<int:take_id>", methods=["POST"])
def create_voice_choose(project_id: int, take_id: int):
    auth.require("generate")
    import db_manager as dbm
    take = dbm.choose_voice_take(take_id, by=_whoami())
    if not take:
        return redirect(f"/create?project={project_id}&error="
                        + _urlquote("that take is not pending"))
    dbm.update_project(project_id, voice_id=take_id, stage="render")
    return redirect(f"/create?project={project_id}")


@app.route("/create/<int:project_id>/render", methods=["POST"])
def create_render(project_id: int):
    """The last click. Everything it uses was chosen by a person."""
    auth.require("generate")
    import db_manager as dbm
    p, bail = _project_or_home(project_id)
    if bail:
        return bail
    missing = [name for name, key in (("a topic", "title"),
                                      ("a script", "script_file"),
                                      ("pictures", "gallery_id"),
                                      ("a voice", "voice_id"))
               if not p.get(key)]
    if missing:
        return redirect(f"/create?project={project_id}&error=" + _urlquote(
            f"still needs {', '.join(missing)}"))
    take = next((t for t in dbm.voice_takes(set_id=p["gallery_id"], limit=10)
                 if t["id"] == p["voice_id"]), None)
    try:
        _, log = _launch_run(script_file=p["script_file"], channel=p["channel"],
                             gallery_id=p["gallery_id"],
                             hook_tone=(take or {}).get("tone"))
    except Exception as e:
        return redirect(f"/create?project={project_id}&error="
                        + _urlquote(f"could not start: {e}"))
    dbm.update_project(project_id, status="rendering")
    # Decided, so it stops asking — same reason as abandoning. What was not
    # chosen at each stage is kept as the losing half of the pair; it just
    # stops being offered.
    dbm.retire_project_options(project_id)
    return redirect("/create?msg=" + _urlquote(
        f'Making "{p["title"]}" — it lands in the review queue like any other '
        f'video. Log: logs/{log.name}'))


# ── Choosing the narrator ────────────────────────────────────────────────────

@app.route("/voices")
def voices_page():
    """The same line in every voice. Chosen once, then left alone.

    THE AUDIO TWIN OF /styles, and rare for the same reason. /voice varies the
    tone of one video's opening; this varies the NARRATOR, and a channel whose
    narrator changes every video has no narrator — the voice is the one thing a
    returning viewer recognises before they have read a word.

    One fixed line for every voice, because a voice compared against a
    different sentence than the one before it is not being compared: half of
    what you hear is the writing.
    """
    auth.require("settings")
    import voice_audition as va

    backend = (request.args.get("backend") or "kokoro").strip().lower()
    if backend not in va.BACKENDS:
        backend = "kokoro"
    current = (os.environ.get(va.VOICE_VAR.get(backend, "")) or "").strip()

    out_dir = va.audition_dir() / backend
    cards = ""
    for voice, label in va.catalogue(backend):
        mp3 = out_dir / f"{voice}.mp3"
        here = voice == current
        player = (f'<audio controls preload="none" style="display:block;'
                  f'width:100%;margin:8px 0" '
                  f'src="/voices/audio/{backend}/{voice}"></audio>'
                  if mp3.exists() else
                  '<p class="muted" style="margin:8px 0">not recorded yet — '
                  'use the button below</p>')
        button = ""
        if auth.can("settings") and mp3.exists() and not here:
            button = (f'<form method="post" action="/voices/use" '
                      f'style="display:inline">'
                      f'<input type="hidden" name="backend" value="{backend}">'
                      f'<input type="hidden" name="voice" value="{voice}">'
                      f'<button class="btn save" type="submit">'
                      f'Narrate with this</button></form>')
        cards += (
            f'<div class="card" style="width:100%;margin-bottom:10px;'
            f'border-color:{"var(--accent)" if here else "var(--border)"}">'
            f'<div style="display:flex;justify-content:space-between;gap:12px">'
            f'<strong>{_esc(label)}</strong>'
            f'<span class="muted">{_esc(voice)}'
            f'{" · in use" if here else ""}</span></div>'
            f'{player}<div>{button}</div></div>')

    tabs = ""
    for b in sorted(va.BACKENDS):
        on = " style=\"font-weight:600\"" if b == backend else ""
        tabs += f'<a href="/voices?backend={b}"{on}>{b}</a> '

    record = ""
    if auth.can("settings"):
        record = (f'<form method="post" action="/voices/record" '
                  f'style="margin:10px 0">'
                  f'<input type="hidden" name="backend" value="{backend}">'
                  f'<button type="submit">Record the sheet '
                  f'({len(va.catalogue(backend))} voices)</button></form>')

    body = f"""
    <a class="back" href="/">← back</a>
    <h2 style="margin-top:14px">Choose the narrator</h2>
    {_msg_banner()}
    <p class="muted">One line, every voice, same words:
       “{_esc(va.SAMPLE)}”</p>
    <p class="muted"><strong>Kokoro</strong> is the only backend here that is
       local, free <em>and</em> clear to monetise (Apache&nbsp;2.0). Edge is
       free and good but it is a Microsoft <em>network</em> service written for
       a browser's read-aloud feature. XTTS runs locally and is licensed
       non-commercially. ElevenLabs is cloud and paid.</p>
    <div class="filters">{tabs}</div>
    {record}
    {cards}
    """
    return _head() + body + PAGE_TAIL


@app.route("/voices/audio/<backend>/<voice>")
def voices_audio(backend: str, voice: str):
    """One audition sample. The backend and voice are matched against the
    catalogue rather than used as a path, so nothing in the URL can address a
    file outside the audition directory."""
    auth.require("settings")
    import voice_audition as va
    if voice not in {v for v, _l in va.catalogue(backend)}:
        abort(404)
    d = va.audition_dir() / backend
    name = f"{voice}.mp3"
    if not (d / name).exists():
        abort(404)
    return send_from_directory(str(d), name, max_age=_IMAGE_MAX_AGE)


@app.route("/voices/record", methods=["POST"])
def voices_record():
    """Record the sheet. Inline, not a subprocess.

    Unlike the galleries this is seconds of CPU for a handful of short lines,
    and a person who just clicked the button is looking at the page it fills —
    handing them a redirect and a log file to go and read would be worse than
    the wait.
    """
    auth.require("settings")
    import voice_audition as va
    backend = (request.form.get("backend") or "kokoro").strip().lower()
    if backend not in va.BACKENDS:
        return redirect("/voices?error=" + _urlquote("unknown backend"))
    try:
        done = va.build(backend)
    except Exception as e:
        return redirect(f"/voices?backend={backend}&error="
                        + _urlquote(f"could not record: {e}"))
    if not done:
        return redirect(f"/voices?backend={backend}&error=" + _urlquote(
            "no voice recorded — the [audition] lines in the console say why"))
    return redirect(f"/voices?backend={backend}&msg=" + _urlquote(
        f"Recorded {len(done)} voice(s)."))


@app.route("/voices/use", methods=["POST"])
def voices_use():
    """Save the narrator the way every launch path reads it — same shape as
    /styles/use, because it is the same kind of decision."""
    auth.require("settings")
    import voice_audition as va
    backend = (request.form.get("backend") or "").strip().lower()
    voice = (request.form.get("voice") or "").strip()
    if backend not in va.BACKENDS:
        return redirect("/voices?error=" + _urlquote("unknown backend"))
    if voice not in {v for v, _l in va.catalogue(backend)}:
        return redirect(f"/voices?backend={backend}&error="
                        + _urlquote("unknown voice"))
    var = va.VOICE_VAR[backend]
    values = _load_settings()
    values["RUFUS_TTS"] = backend
    values[var] = voice
    _save_settings(values)
    os.environ["RUFUS_TTS"] = backend
    os.environ[var] = voice
    return redirect(f"/voices?backend={backend}&msg=" + _urlquote(
        f"{voice} narrates from the next run on."))


# ── Choosing how it opens ────────────────────────────────────────────────────

def _tone_reach_note() -> str:
    """What the live voice backend can actually do with a tone.

    A CHOICE BETWEEN THREE READS IS THEATRE IF THE BACKEND RENDERS THEM THE
    SAME, and which backend is live is a runtime fact — RUFUS_TTS, plus
    ElevenLabs falling back to Kokoro on a free-tier library voice, which is
    every run here. Telling somebody that up front is cheaper than letting them
    listen three times for a difference that is not there.
    """
    try:
        import emotional_map
        reach = emotional_map.speaks_tone()
    except Exception:
        return ""
    if "rate, pitch and volume" in reach:
        return ""
    return (f'<div class="msg" style="margin:8px 0">Your voice backend varies '
            f'<strong>{_esc(reach)}</strong>, so these reads differ in pace '
            f'rather than in colour. <code>RUFUS_TTS=edge</code> varies rate, '
            f'pitch and volume — bigger differences, a different voice.</div>')


@app.route("/voice")
def voice_page():
    """Three reads of the opening line. Twenty-four seconds of listening.

    ONLY THE HOOK, because audio is the one thing on this list that cannot be
    skimmed — it plays at one times speed. Three full takes is two and a half
    minutes; three hooks is twenty-four seconds, and if the opening read lands
    the rest follows it.

    The VOICE is not what varies and should not: a channel whose narrator
    changes every video has no narrator. What varies is the tone beat 0 is read
    in — the same lever that sizes its pauses and grades its picture.
    """
    auth.require("view")
    try:
        import db_manager as dbm
        takes = dbm.voice_takes(status="pending", limit=30)
    except Exception as e:
        body = (f'<a class="back" href="/">← back</a><h2 style="margin-top:14px">'
                f'Choose how it opens</h2><div class="msg error">'
                f'{_esc(str(e))}</div>')
        return _head() + body + PAGE_TAIL

    sets: dict = {}
    for t in takes:
        sets.setdefault((t["set_id"], t["topic"]), []).append(t)

    blocks = ""
    for (set_id, topic), rows in sets.items():
        cards = ""
        for t in rows:
            button = ""
            if auth.can("generate"):
                button = (f'<form method="post" action="/voice/{t["id"]}/choose"'
                          f' style="display:inline"><button class="btn save" '
                          f'type="submit">Open like this</button></form>')
            cards += (
                f'<div class="card" style="width:100%;margin-bottom:10px">'
                f'<strong>{_esc(t["tone"] or "—")}</strong>'
                f'<audio controls preload="none" style="display:block;'
                f'width:100%;margin:8px 0" src="/voice/audio/{t["id"]}">'
                f'</audio><div>{button}</div></div>')
        blocks += (f'<h2 style="margin-top:22px">{_esc(topic or "—")}</h2>'
                   f'<p class="muted">“{_esc((rows[0]["text"] or "")[:200])}” — '
                   f'{len(rows)} read(s). The tone you pick sizes this beat\'s '
                   f'pauses and grades its picture too, so it is one decision '
                   f'about what the opening IS rather than three.</p>'
                   f'{_tone_reach_note()}{cards}')

    if not sets:
        blocks = ('<p class="muted">Nothing waiting. Settle the pictures on '
                  '<a href="/galleries">Choose the pictures</a> and three '
                  'reads of the hook are recorded here.</p>')

    body = f"""
    <a class="back" href="/">← back</a>
    <h2 style="margin-top:14px">Choose how it opens</h2>
    {_flow_bar("/voice")}
    {_msg_banner()}
    {blocks}
    """
    return _head() + body + PAGE_TAIL


@app.route("/voice/audio/<int:take_id>")
def voice_audio(take_id: int):
    """One take. Addressed by row id rather than filename, so nothing in a
    query string can reach a file outside the takes directory."""
    auth.require("view")
    try:
        import db_manager as dbm
        row = next((t for t in dbm.voice_takes(limit=200)
                    if t["id"] == take_id), None)
    except Exception:
        row = None
    if not row:
        abort(404)
    p = Path(row["path"])
    if not p.exists():
        abort(404)
    return send_from_directory(str(p.parent), p.name, max_age=_IMAGE_MAX_AGE)


@app.route("/voice/<int:take_id>/choose", methods=["POST"])
def voice_choose(take_id: int):
    """The last choice → the render, with everything a person picked.

    This is where the irreversible expensive step finally happens, and it has
    every human answer behind it: the topic, the script, the pictures shot by
    shot, and now the read. Nothing in the run regenerates any of them.
    """
    auth.require("generate")
    try:
        import db_manager as dbm
        take = dbm.choose_voice_take(take_id, by=_whoami())
        if not take:
            return redirect("/voice?error=" + _urlquote(
                "that read is not pending — already decided?"))
        gs = next((g for g in dbm.gallery_sets(status=None, limit=40)
                   if g["id"] == take["set_id"]), None)
        if not gs:
            return redirect("/voice?error=" + _urlquote(
                "the picture set behind this read is gone"))
        _, log_path = _launch_run(script_file=gs["script_file"],
                                  channel=gs["channel"],
                                  gallery_id=take["set_id"],
                                  hook_tone=take["tone"])
    except Exception as e:
        return redirect("/voice?error=" + _urlquote(f"could not start: {e}"))
    return redirect("/voice?msg=" + _urlquote(
        f'Making it — your script, your pictures, opening in {take["tone"]}. '
        f'It lands in the review queue like any other video. '
        f'Log: logs/{log_path.name}'))


# ── The workflow bench ───────────────────────────────────────────────────────

@app.route("/bench")
def bench_page():
    """One style, MANY workflows, the same scene and the same seed.

    The Style page is the transpose of this one: it renders one fixed scene
    through many STYLES and one workflow. That answers "which look do I want".
    It cannot answer "which workflow draws it best", and that is the question
    the pale-beige gallery raised — a style block that forbids a washed-out
    background twice, obeyed by nobody, is not a wording problem.

    A GRID AND NOT A LIST, because comparing workflows means seeing the SAME
    probe across candidates at once. A list of twenty-four pictures is
    twenty-four pictures; six rows of four is a decision.
    """
    auth.require("settings")
    import workflow_bench as wb

    data = wb.latest()
    cands = wb.candidates()
    busy = [c for c in _channels() if _run_in_progress(c)]

    # Every candidate, whether or not it has been benched — a workflow that is
    # sitting in the folder unvalidated is exactly the thing to know about.
    rows = ""
    for label, path in cands:
        ok, problems = wb.validate(path)
        try:
            import comfy_template
            graph = comfy_template.load_template(path)
            notes = wb.advisories(graph) if graph else []
        except Exception:
            notes = []
        marks = "".join(f'<li class="muted">{_esc(p)}</li>' for p in problems)
        marks += "".join(f'<li class="muted">⚠ {_esc(n)}</li>' for n in notes)
        rows += (f'<tr><td><strong>{_esc(label)}</strong><br>'
                 f'<span class="muted" style="font-size:12px">{_esc(path.name)}'
                 f'</span></td>'
                 f'<td>{"usable" if ok else "<b>unusable</b>"}'
                 + (f'<ul style="margin:4px 0 0;padding-left:16px">{marks}</ul>'
                    if marks else "") + '</td></tr>')

    grid = ""
    if data.get("workflows"):
        usable = [w for w in data["workflows"] if w.get("renders")]
        head = "".join(
            f'<th>{_esc(w["label"])}<br><span class="muted" '
            f'style="font-weight:400;font-size:12px">'
            f'{w.get("passed", 0)}/{len(data.get("probes", []))} clean · '
            f'{w.get("mean_seconds", 0)}s avg</span></th>' for w in usable)
        body = ""
        for probe in data.get("probes", []):
            cells = ""
            for w in usable:
                r = (w.get("renders") or {}).get(probe) or {}
                if r.get("ok") and r.get("file"):
                    name = _urlquote(Path(r["file"]).name)
                    folder = _urlquote(Path(r["file"]).parent.name)
                    verdict = ("" if r.get("gate") == "ok" else
                               f'<div class="muted" style="font-size:12px">'
                               f'{_esc(r.get("gate", ""))}</div>')
                    cells += (f'<td><a href="/bench/file/{folder}/{name}" '
                              f'target="_blank"><img '
                              f'src="/bench/file/{folder}/{name}" loading="lazy" '
                              f'alt="" style="width:100%;border-radius:8px">'
                              f'</a>{verdict}</td>')
                else:
                    cells += '<td class="muted">—</td>'
            body += f'<tr><td><code>{_esc(probe)}</code></td>{cells}</tr>'
        grid = (f'<table><tr><th>probe</th>{head}</tr>{body}</table>'
                f'<p class="muted">Rendered {_esc(data.get("stamp", ""))} at '
                f'{data.get("width", 0)}×{data.get("height", 0)}.</p>')
    else:
        grid = ('<p class="muted">Nothing benched yet. Drop an API export in '
                '<code>config/workflows/</code> and run it — the current '
                'stills workflow is always the first column, so a candidate '
                'is always measured against what ships today.</p>')

    warn = ("" if not busy else
            f"<div class='msg error'>A run is using the GPU "
            f"({_esc(', '.join(busy))}) — benching now would queue behind it.</div>")

    body = f"""
    <a class="back" href="/">← back</a>
    <h2 style="margin-top:14px">Workflow bench</h2>
    {warn}
    {_msg_banner()}
    <p class="muted">The same six scenes, the same seeds, through every
       workflow in <code>config/workflows/</code>. Each probe is a defect this
       channel has actually shipped — a face that came back with the same mild
       smile ten times, a background that came back beige against an explicit
       instruction, two frames that came back as contact sheets. Change ONE
       thing between candidates and name the file after it.</p>
    <table><tr><th>Workflow</th><th>Ready?</th></tr>{rows}</table>
    <form method="post" action="/bench/run" style="margin:14px 0">
      <button class="btn save" type="submit">Render the grid</button>
      <span class="muted" style="margin-left:10px">
        {len(cands)} workflow(s) × {len(__import__("workflow_bench").PROBES)} probes</span>
    </form>
    {grid}
    """
    return _head() + body + PAGE_TAIL


@app.route("/bench/file/<folder>/<name>")
def bench_file(folder: str, name: str):
    auth.require("view")
    import workflow_bench as wb
    data = wb.latest()
    root = Path(data.get("dir") or "") / Path(folder).name
    if not (root / Path(name).name).exists():
        abort(404)
    return send_from_directory(str(root), Path(name).name)


@app.route("/bench/run", methods=["POST"])
def bench_run():
    """Render the grid inline, like the style previews do.

    Same trade as /styles/preview: the dashboard runs threaded=False, so this
    holds the page. It is minutes rather than seconds, so it refuses outright
    while a video run holds the card instead of queueing behind it and looking
    hung.
    """
    auth.require("settings")
    busy = [c for c in _channels() if _run_in_progress(c)]
    if busy:
        return redirect("/bench?error=" + _urlquote(
            f"A run is using the GPU ({', '.join(busy)}). The bench needs the "
            f"same card — try again when it finishes."))
    try:
        import workflow_bench as wb
        data = wb.run()
    except Exception as e:
        return redirect("/bench?error=" + _urlquote(f"Bench failed: {e}"))
    if not data.get("workflows"):
        return redirect("/bench?error=" + _urlquote(
            "Nothing was rendered — check ComfyUI is running and that at "
            "least one export is usable."))
    return redirect("/bench?msg=" + _urlquote(
        f"Benched {len(data['workflows'])} workflow(s)."))


@app.route("/format", methods=["POST"])
def set_format():
    """Persist the format for every launch path, not just this page.

    Written through _save_settings so run.bat, the scheduled task and a bare
    `python scripts/main.py` all see it — the same reason settings_store
    exists. A header button that only changed THIS process would be the
    settings-page-obeyed-by-one-launcher bug wearing a nicer hat.
    """
    auth.require("settings")
    import video_format
    fmt = (request.form.get("format") or "").strip().lower()
    if fmt not in video_format.PROFILES:
        return redirect(request.referrer or "/")
    values = _load_settings()
    values["RUFUS_FORMAT"] = fmt
    _save_settings(values)
    os.environ["RUFUS_FORMAT"] = fmt
    return redirect(request.referrer or "/")


@app.route("/settings/test-notify", methods=["POST"])
def settings_test_notify():
    """Send one real notification, now, with the settings as saved.

    WHY A BUTTON AND NOT A DOCUMENTED CHECK. The only other way to learn
    whether the webhook works is to run a video and wait — twenty-five minutes
    on this hardware — and a webhook URL is exactly the kind of value that is
    wrong in a way nothing reveals until then: a trailing space, a copied
    "Copy Webhook URL" that grabbed the channel link instead, a webhook deleted
    on the Discord side months ago.

    Tests what is ON SCREEN, not what is on disk. The button sits inside the
    settings form, so the fields come with it — someone who pastes a webhook
    and reaches for "test" before "save" is testing the webhook they just
    pasted, which is the only reading of that click that is not a trap.
    Falls back to the saved value for any field left empty.
    """
    auth.require("settings")
    _require_localhost()
    saved = dict(_load_settings())
    for key in SETTINGS_KINDS:
        posted = request.form.get(key, "").strip()
        if posted:
            saved[key] = posted
    old = {k: os.environ.get(k) for k in saved}
    os.environ.update(saved)
    try:
        import importlib
        import notify
        importlib.reload(notify)
        backends = notify.configured()
        if not backends:
            return redirect("/settings?error=" + _urlquote(
                "Nothing to test — no Discord webhook and no ntfy topic is "
                "set. Fill one in and save first."))
        ok = notify.send(
            "Rufus: test notification",
            "If you are reading this, the dashboard can reach you. Finished "
            "videos will arrive here with their score, their hold reason and "
            "the video itself.",
            url=notify._dashboard_url(), priority="normal")
        msg = (f"Sent via {', '.join(backends)} — check the channel."
               if ok else
               f"{', '.join(backends)} configured but the send failed. The "
               f"usual cause is a webhook that was deleted or copied wrong; "
               f"the Logs page has the reason.")
        return redirect(f"/settings?{'msg' if ok else 'error'}=" + _urlquote(msg))
    except Exception as e:
        return redirect("/settings?error=" + _urlquote(f"Test failed: {e}"))
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@app.route("/settings/save", methods=["POST"])
def settings_save():
    """Save the overrides, rejecting a number that is not one.

    VALIDATED HERE AND NOT LATER. A beat count of "24 " or "twenty-four"
    reaches the run as an env var, and the reader that fails to parse it falls
    back to its default silently — so the owner sets a value, sees no error,
    and gets the old behaviour with nothing in the log to explain it."""
    auth.require("settings")
    _require_localhost()
    values, bad = {}, []
    for key, label, kind, _help in SETTINGS_SCHEMA:
        v = request.form.get(key, "").strip()
        if not v:
            continue
        if kind == "number":
            try:
                float(v)
            except ValueError:
                bad.append(f"{label} ({v!r} is not a number)")
                continue
        values[key] = v
    _save_settings(values)
    if bad:
        return redirect("/settings?error=" + _urlquote("Not saved: " + "; ".join(bad)))
    return redirect("/settings?msg=" + _urlquote(f"Saved {len(values)} setting(s)."))


# ── Published by hand ────────────────────────────────────────────────────────

@app.route("/video/<int:video_id>/published", methods=["POST"])
def mark_published(video_id: int):
    """Record that a video is live on YouTube, however it got there.

    THE LOOP THIS CLOSES. Analytics only looks at rows carrying a youtube_id,
    and only the pipeline's own uploader ever set one. The owner published
    several videos by hand — the correct thing to do while nothing
    auto-uploads — and every one was invisible to the whole learning loop: no
    metrics fetched, no views recorded, so feedback_analyzer had no winners to
    learn hooks from, and every quality judgement in this pipeline stayed a
    guess about what works.

    A manual upload is not a lesser kind of publish. Paste the link.
    """
    auth.require("approve")
    _require_localhost()
    raw = request.form.get("youtube", "")
    try:
        ok = db_manager.mark_published(video_id, raw)
    except ValueError as e:
        # The same link pasted onto a second video. Refused rather than
        # accepted, because analytics joins on this column: the duplicate does
        # not just mislabel one row, it credits both with one video's views.
        return redirect(f"/video/{video_id}?error=" + _urlquote(str(e)))
    if not ok:
        return redirect(f"/video/{video_id}?error=" + _urlquote(
            f"Couldn't find a YouTube id in {raw[:60]!r}. Paste the video's "
            f"link or its 11-character id."))
    return redirect(f"/video/{video_id}?msg=" + _urlquote(
        "Recorded as published. Analytics will pick it up on the next fetch "
        "and its views start feeding the hook learning."))


@app.route("/tracking/fetch", methods=["POST"])
def tracking_fetch():
    """Pull view counts, then re-derive what the channel has learned.

    RUN AS A SEPARATE PROCESS, like a video run, for the same reason: this
    Flask app is single-threaded on purpose, and a YouTube round-trip for
    every tracked video would freeze every other page for its duration.

    The FIRST fetch on a machine needs a browser — Google's OAuth consent —
    and analytics_fetcher already says so in as many words when it cannot get
    one. That message goes to the log this writes, which is why the button
    points at the Logs page rather than claiming success.
    """
    auth.require("approve")
    _require_localhost()
    log_dir = paths.log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"analytics_{int(time.time())}.log"
    code = ("import sys; sys.path.insert(0, 'scripts');"
            "import analytics_fetcher, feedback_analyzer;"
            "analytics_fetcher.fetch_analytics();"
            "feedback_analyzer.analyze()")
    try:
        with open(log_path, "wb") as logf:
            subprocess.Popen([sys.executable, "-u", "-c", code], cwd=str(ROOT),
                             stdout=logf, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, env=os.environ.copy())
    except Exception as e:
        return redirect("/tracking?error=" + _urlquote(f"Could not start: {e}"))
    return redirect("/tracking?msg=" + _urlquote(
        f"Fetching. Watch {log_path.name} on the Logs page — the first fetch "
        f"on this machine needs a Google sign-in in a browser, and the log "
        f"says so if it cannot get one."))


def _learnings(channel: str | None = None) -> dict:
    """What the channel has learned from its own view counts, or {}."""
    try:
        import channel_config
        ch = channel_config.load_channel(channel)
        return json.loads(ch.learnings_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _tracking_body():
    """How much of the feedback loop is real, and what is missing from it.

    A youtube_id means a video CAN be tracked; a metrics row means it has
    been. The gap between those two numbers is the honest answer to whether
    the learning loop is closed, and it was invisible before this page — the
    dashboard could show 79 videos and imply a working channel while not one
    of them had a view count attached.
    """
    auth.require("view")
    channel = request.args.get("channel") or None
    try:
        untracked = db_manager.published_without_metrics(channel)
    except Exception as e:
        body = (f'<a class="back" href="/">← back</a><h2 style="margin-top:14px">'
                f'Tracking</h2><div class="msg error">{_esc(str(e))}</div>')
        return body

    videos = _recent_videos(limit=500, channel=channel)
    live = [v for v in videos if v.get("youtube_id")]
    pending = [v for v in videos if not v.get("youtube_id")]

    rows = ""
    for v in untracked:
        rows += (f'<tr><td><a class="row-link" href="/video/{v["id"]}">'
                 f'{_esc((v.get("title") or "—")[:70])}</a></td>'
                 f'<td class="muted">{_esc(v.get("upload_date") or "")}</td>'
                 f'<td class="muted">{v.get("score") or "—"}/10</td>'
                 f'<td><a href="https://youtu.be/{_esc(v["youtube_id"])}" '
                 f'target="_blank" rel="noopener">watch</a></td></tr>')
    untracked_html = (
        f'<table><tr><th>Video</th><th>Published</th><th>Score</th><th></th></tr>'
        f'{rows}</table>' if rows else
        '<p class="muted">Every published video has metrics.</p>')

    note = ""
    if not live:
        note = ('<div class="msg error">No video carries a YouTube id, so the '
                'learning loop has never had any data. If you have published '
                'any by hand, open the video and paste its link — that is the '
                'row it is missing.</div>')

    # feedback_analyzer refuses to draw conclusions from fewer than three
    # measured videos, and saying so beats an empty section that looks broken.
    measured = len(live) - len(untracked)
    learned = _learnings(channel)
    if learned.get("winning_hooks"):
        rows_l = "".join(f"<li>{_esc(h)}</li>" for h in learned["winning_hooks"][:5])
        rows_bad = "".join(f"<li>{_esc(h)}</li>" for h in learned.get("losing_hooks", [])[:5])
        learn_html = (f'<h2>What the views have taught this channel</h2>'
                      f'<div class="grid2"><div><p class="muted">Hooks that '
                      f'performed</p><ul>{rows_l}</ul></div>'
                      f'<div><p class="muted">Hooks that did not</p>'
                      f'<ul>{rows_bad}</ul></div></div>'
                      f'<p class="muted">script_writer feeds these into the '
                      f'hook factory, so this is the loop actually closing.</p>')
    elif measured >= 3:
        learn_html = ('<h2>What the views have taught this channel</h2>'
                      '<p class="muted">Metrics exist but no learnings file yet '
                      '— run a fetch, which re-derives it.</p>')
    else:
        learn_html = (f'<h2>What the views have taught this channel</h2>'
                      f'<p class="muted">Nothing yet. Patterns are drawn from '
                      f'three measured videos at the earliest, and there '
                      f'{"is" if measured == 1 else "are"} {measured}. Until '
                      f'then every quality judgement in the pipeline — the '
                      f'hook scorer, the critic, the score threshold — is a '
                      f'guess about what works rather than a measurement.</p>')

    fetch_btn = ""
    if auth.can("approve"):
        fetch_btn = ('<form method="post" action="/tracking/fetch">'
                     '<button class="btn save" type="submit">'
                     'Fetch view counts now</button></form>')

    # WAITING TO GO LIVE. An upload that went up private with a publishAt is
    # indistinguishable, from every other page here, from one that is private
    # forever — and that difference is the whole question of whether the
    # channel is publishing. It was invisible until there was a column for it.
    try:
        waiting = db_manager.scheduled(channel)
    except Exception:
        waiting = []
    if waiting:
        sched_rows = ""
        for v in waiting:
            sched_rows += (
                f'<tr><td><a class="row-link" href="/video/{v["id"]}">'
                f'{_esc((v.get("title") or v.get("script_hook") or "—")[:70])}'
                f'</a></td>'
                f'<td class="muted">{_esc(v.get("publish_at") or "")} UTC</td>'
                f'<td class="muted">{_esc(v.get("niche") or "")}</td>'
                f'<td>' + (f'<a href="https://youtu.be/{_esc(v["youtube_id"])}" '
                           f'target="_blank" rel="noopener">watch</a>'
                           if v.get("youtube_id") else "") + '</td></tr>')
        sched_html = (f'<h2>Waiting to go live</h2>'
                      f'<p class="muted">Uploaded private with a publish time. '
                      f'YouTube makes these public itself — nothing here has '
                      f'to run at that moment.</p>'
                      f'<table><tr><th>Video</th><th>Goes live</th>'
                      f'<th>Niche</th><th></th></tr>{sched_rows}</table>')
    else:
        sched_html = ""

    body = f"""
    <a class="back" href="/">← back</a>
    <h2 style="margin-top:14px">Tracking</h2>
    {_msg_banner()}
    {note}
    <div class="cards">
      <div class="card"><div class="num">{len(live)}</div>
        <div class="label">published (trackable)</div></div>
      <div class="card"><div class="num">{len(live) - len(untracked)}</div>
        <div class="label">with view counts</div></div>
      <div class="card"><div class="num">{len(untracked)}</div>
        <div class="label">awaiting first fetch</div></div>
      <div class="card"><div class="num">{len(pending)}</div>
        <div class="label">never published</div></div>
    </div>
    <p class="muted">A YouTube id means a video CAN be tracked; a metrics row
       means it has been. Published videos with no metrics are picked up by
       <code>python scripts/analytics_fetcher.py</code> — which is also what
       feeds the hook learning, so until this middle number moves, every
       quality judgement in the pipeline is a guess about what works.</p>
    <div class="actions">{fetch_btn}</div>
    {sched_html}
    {learn_html}
    <h2>Published, not yet measured</h2>
    {untracked_html}
    """
    return body


# ── Advice ───────────────────────────────────────────────────────────────────

def _advice_now() -> tuple[list[dict], dict]:
    """(items, readiness). Empty and "unmeasured" on any failure."""
    try:
        import advisor
        import run_review
        pat = run_review.patterns(limit=30)
        st = _stats()
        cfg = _load_settings()
        return advisor.advise(pat, st, cfg), advisor.readiness(pat, st, cfg)
    except Exception as e:
        print(f"[dashboard] advice unavailable: {e}")
        return [], {"state": "unmeasured", "detail": str(e)}


def _advice_body():
    """What to change before the next video, and a button that changes it.

    THE GAP THIS CLOSES. Insights says what happened; this says what to do
    about it, and then does it. A suggestion the reader has to translate into
    a settings change, on another page, under a name they have to remember, is
    a suggestion most people do not act on — so the ones that map to a setting
    carry the button that sets it.

    Nothing here is a model's opinion. Every line is derived from measurements
    already on disk, which is what makes it checkable rather than plausible.
    """
    auth.require("view")
    items, ready = _advice_now()

    tone = {"needs work": "held", "workable": "pending",
            "good": "ok", "unmeasured": "pending"}.get(ready["state"], "pending")
    header = (f'<div class="card" style="width:100%">'
              f'<span class="badge {tone}">{_esc(ready["state"])}</span> '
              f'<span class="muted">{_esc(ready["detail"])}</span></div>')

    if not items:
        body = (f'<a class="back" href="/">← back</a>'
                f'<h2 style="margin-top:14px">What to change</h2>{header}'
                f'<p class="muted">Nothing to suggest — every measured run is '
                f'inside its thresholds.</p>')
        return body

    cards = ""
    for it in items:
        badge = {"high": "held", "medium": "pending", "low": "ok"}.get(
            it["severity"], "ok")
        apply_btn = ""
        if it.get("setting") and (it.get("value") or it.get("clear_label")):
            label = it.get("clear_label") or \
                f'Set {it["setting"]} = {it["value"]}'
            apply_btn = (
                f'<form method="post" action="/advice/apply" style="margin-top:10px">'
                f'<input type="hidden" name="key" value="{_esc(it["setting"])}">'
                f'<input type="hidden" name="value" value="{_esc(it["value"])}">'
                f'<button class="btn save" type="submit">{_esc(label)}</button></form>')
        cards += (
            f'<div class="card" style="width:100%;margin-bottom:12px">'
            f'<span class="badge {badge}">{_esc(it["severity"])}</span> '
            f'<strong>{_esc(it["title"])}</strong>'
            f'<div class="muted" style="margin:6px 0">{_esc(it["evidence"])}</div>'
            f'<p style="margin:6px 0">{_esc(it["why"])}</p>'
            f'<p style="margin:6px 0"><strong>Do:</strong> {_esc(it["action"])}</p>'
            f'{apply_btn}</div>')

    body = f"""
    <a class="back" href="/">← back</a>
    <h2 style="margin-top:14px">What to change</h2>
    {_msg_banner()}
    {header}
    <p class="muted">Derived from the last runs' own measurements — no model,
       nothing to configure. A finding has to appear in a real share of runs
       before it lands here, because advice that fires every time is advice
       people learn to scroll past.</p>
    {cards}
    """
    return body


@app.route("/advice/apply", methods=["POST"])
def advice_apply():
    """Apply one suggestion, and only one the advisor actually offered.

    The key and value arrive from a form, so they are checked against the
    settings schema rather than trusted — a POST that could write an arbitrary
    key into the settings file would be writing arbitrary environment into
    every future run.
    """
    auth.require("settings")
    _require_localhost()
    key = request.form.get("key", "").strip()
    value = request.form.get("value", "").strip()
    if key not in SETTINGS_KINDS:
        return redirect("/advice?error=" + _urlquote(f"{key} is not a setting."))
    offered = {(i.get("setting"), i.get("value")) for i in _advice_now()[0]}
    if (key, value) not in offered:
        return redirect("/advice?error=" + _urlquote(
            "That suggestion is no longer current — reload and try again."))
    values = dict(_load_settings())
    if value == "":
        # Clearing is a real remedy now — the pipeline's own derived defaults
        # beat most fixed values a page could suggest.
        values.pop(key, None)
        _save_settings(values)
        return redirect("/advice?msg=" + _urlquote(
            f"{key} cleared. The pipeline's own default decides it again."))
    values[key] = value
    _save_settings(values)
    return redirect("/advice?msg=" + _urlquote(
        f"{key} set to {value}. It applies to the next run started here."))


# ── Insights ─────────────────────────────────────────────────────────────────

_SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def _severity_badge(sev: str) -> str:
    cls = {"high": "held", "medium": "pending", "low": "ok"}.get(sev, "ok")
    return f'<span class="badge {cls}">{_esc(sev)}</span>'


def _insights_body():
    """What keeps going wrong, measured rather than remembered.

    THE PROBLEM THIS PAGE IS FOR. Every fix to this pipeline in recent memory
    began with the owner watching a video, noticing something, and pasting a
    log — which works, and needs a person to watch, to remember what the last
    six runs looked like, and to be right about which of twenty things on
    screen is the one that matters. A defect visible in one run is a bad seed;
    the same defect in four runs of six is a code change, and only a record
    kept across runs can tell those apart.

    Everything here is computed by run_review.py from files already on disk —
    no model, no GPU, nothing to configure.
    """
    auth.require("view")
    try:
        import run_review
        data = run_review.patterns(limit=30)
    except Exception as e:
        body = (f'<a class="back" href="/">← back</a>'
                f'<h2 style="margin-top:14px">Insights</h2>'
                f'<div class="msg error">Review data unavailable: {_esc(str(e))}</div>')
        return body

    rows = data.get("rows", [])
    if not rows:
        body = f"""
        <a class="back" href="/">← back</a>
        <h2 style="margin-top:14px">Insights</h2>
        <p class="muted">No runs measured yet. Every finished run writes its
           own review from here on; to measure the ones already on disk, run
           <code>python scripts/run_review.py --all</code> once.</p>
        """
        return body

    # What recurs, worst first.
    recurring = ""
    for r in data.get("recurring", []):
        pct = int(r["share"] * 100)
        bar = ('<div style="height:6px;border-radius:3px;background:var(--border);'
               f'overflow:hidden"><div style="height:6px;width:{pct}%;'
               f'background:var({"--bad" if pct >= 50 else "--warn"})"></div></div>')
        recurring += (f'<tr><td><code>{_esc(r["id"])}</code></td>'
                      f'<td style="width:45%">{bar}</td>'
                      f'<td class="muted">{r["runs"]} of {data["runs_reviewed"]} runs</td></tr>')
    recurring_html = (f'<table><tr><th>Finding</th><th></th><th></th></tr>'
                      f'{recurring}</table>' if recurring else
                      '<p class="muted">Nothing recurring — every measured run '
                      'is inside its thresholds.</p>')

    # Per-run detail, newest first.
    per_run = ""
    for r in rows:
        findings = sorted(r.get("findings", []),
                          key=lambda f: _SEVERITY_ORDER.get(f.get("severity"), 3))
        items = "".join(f'<li>{_severity_badge(f.get("severity","low"))} '
                        f'{_esc(f["text"])}</li>' for f in findings)
        c = r.get("clauses", {})
        d = r.get("dominant_subject", {})
        cuts = r.get("cuts", {})
        summary = (f'{r.get("beats", 0)} pictures · '
                   f'thread {int(c.get("thread_share", 0) * 100)}% · '
                   f'setting {int(c.get("setting_share", 0) * 100)}%')
        if d.get("word"):
            summary += f' · most-named "{_esc(d["word"])}" {int(d.get("share", 0) * 100)}%'
        if cuts.get("longest_hold_s"):
            summary += f' · longest hold {cuts["longest_hold_s"]}s'
        per_run += (
            f'<div class="card" style="width:100%;margin-bottom:10px">'
            f'<strong>{_esc(r.get("run_id", "?"))}</strong>'
            f'<div class="muted" style="margin:4px 0 8px">{summary}</div>'
            + (f'<ul style="margin:0;padding-left:18px">{items}</ul>'
               if items else '<div class="muted">nothing out of range</div>')
            + '</div>')

    body = f"""
    <a class="back" href="/">← back</a>
    <h2 style="margin-top:14px">Insights</h2>
    <p class="muted">Measured from the prompts and keyframes of the last
       {data["runs_reviewed"]} runs. One run's numbers say a video was weak;
       the same finding across most runs is a code change rather than a bad
       seed.</p>
    <h2>What keeps happening</h2>
    {recurring_html}
    <h2>Run by run</h2>
    {per_run}
    """
    return body


# ── Logs ─────────────────────────────────────────────────────────────────────

LOG_TAIL_BYTES = 240_000     # ~2000 lines; a full run log runs to a few hundred KB


def _log_files(limit: int = 40) -> list[dict]:
    """Every run log, newest first.

    Two naming schemes live side by side and both matter: rufus_YYYYMMDD.log is
    the tee'd console log a run.bat run writes, and dashboard_run_<epoch>.log is
    what _launch_run captures for a run started from this page. Reading only one
    of them would hide exactly half the runs from the person trying to work out
    what happened.
    """
    out: list[dict] = []
    for d in (paths.log_dir(),):
        if not d.is_dir():
            continue
        for f in d.glob("*.log"):
            try:
                st = f.stat()
            except OSError:
                continue
            out.append({"name": f.name, "size": st.st_size, "mtime": st.st_mtime,
                        "source": "dashboard" if f.name.startswith("dashboard_run_")
                                  else "console"})
    out.sort(key=lambda r: r["mtime"], reverse=True)
    return out[:limit]


def _read_log(name: str, tail: int = LOG_TAIL_BYTES) -> str:
    """The last `tail` bytes of one log, or "" if it cannot be read.

    Tailed rather than read whole because these grow without bound and the
    interesting part of a failed run is always the end. Path is resolved and
    re-checked against the logs directory — the filename arrives from a query
    string, and `..` in it would otherwise read anything on the disk.
    """
    d = paths.log_dir().resolve()
    f = (d / name).resolve()
    if f.parent != d or not f.is_file():
        return ""
    try:
        with open(f, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - tail))
            raw = fh.read()
    except OSError as e:
        return f"(could not read {name}: {e})"
    text = raw.decode("utf-8", errors="replace")
    if size > tail:
        text = f"… showing the last {tail // 1000}KB of {size // 1000}KB …\n\n" + text
    return text


@app.route("/logs")
def logs_page():
    """Read a run's log without opening a terminal.

    Auto-refreshes while a run is in progress, because the reason to be on this
    page at all is usually that something is running right now and the owner
    wants to see where it has got to.
    """
    auth.require("view")
    files = _log_files()
    name = request.args.get("file") or (files[0]["name"] if files else "")
    running = _run_in_progress("default")

    rows = ""
    for f in files:
        sel = ' style="background:color-mix(in srgb, var(--accent) 12%, transparent)"' if f["name"] == name else ""
        rows += (f'<tr{sel}><td><a class="row-link" href="/logs?file={_urlquote(f["name"])}">'
                 f'{_esc(f["name"])}</a></td>'
                 f'<td class="muted">{f["source"]}</td>'
                 f'<td class="muted">{f["size"] // 1024} KB</td>'
                 f'<td class="muted">{_fmt_ts(f["mtime"])}</td></tr>')
    table = (f'<table><tr><th>File</th><th>From</th><th>Size</th><th>Modified</th></tr>'
             f'{rows}</table>' if rows else
             '<p class="muted">No logs yet — they appear here after the first run.</p>')

    text = _read_log(name) if name else ""
    refresh = ('<meta http-equiv="refresh" content="10">' if running else "")
    note = ('<div class="msg ok">A run is in progress — this page refreshes '
            'every 10 seconds.</div>' if running else "")

    body = f"""
    {refresh}
    <a class="back" href="/">← back</a>
    <h2 style="margin-top:14px">Logs</h2>
    {note}
    {table}
    <h2>{_esc(name) or "—"}</h2>
    <pre style="padding:14px;overflow:auto;max-height:70vh;font-size:12px;
                line-height:1.45;white-space:pre-wrap;word-break:break-word">{_esc(text)}</pre>
    """
    return _head() + body + PAGE_TAIL


@app.route("/gallery")
def gallery():
    """Browsable grid of generated stills across every recent run — the
    per-video detail page already shows one run's keyframes; this is every
    run's, for browsing past visual output rather than reviewing one video
    at a time."""
    images = _gallery_images()
    if not images:
        body = "<a class='back' href='/'>← back</a><p class='muted'>No keyframes saved yet.</p>"
        return _head() + body + PAGE_TAIL
    tiles = ""
    for img in images:
        src = f"/debug/{_esc(img['run_id'])}/{_esc(img['image'])}"
        tiles += (f'<a href="{src}" target="_blank" style="display:inline-block;margin:4px">'
                 f'<img src="{src}?w=240" style="width:120px;height:213px;object-fit:cover;'
                 f'border-radius:6px" loading="lazy" title="{_esc(img["run_id"])}"></a>\n')
    body = f"""
    <a class="back" href="/">← back</a>
    <h2 style="margin-top:14px">Gallery ({len(images)} recent stills)</h2>
    <p class="muted">Every saved keyframe across recent runs, newest first.
       Click one to open full-size.</p>
    <div>{tiles}</div>
    """
    return _head() + body + PAGE_TAIL


@app.route("/trending")
def trending():
    """Browse this week's rising search queries per niche (Google Trends via
    pytrends) before committing a run to one — research.py already uses
    these to auto-pick topics (fetch_trending_wikipedia); this is that same
    signal, made browsable, with a "queue it" button that hands off to the
    existing /request-topic path (same fact-grounding, same gates) rather
    than a separate launch mechanism."""
    import research
    niches = list(research.NICHE_TREND_SEEDS.keys())
    niche = request.args.get("niche") or (niches[0] if niches else None)

    queries: list[str] = []
    error = None
    reason = ""
    if niche:
        try:
            queries, reason = research.trending_queries_with_reason(niche)
        except Exception as e:
            error = str(e)

    niche_links = "".join(
        f'<a href="/trending?niche={_esc(n)}">{_esc(n)}</a> ' for n in niches)

    if error:
        list_html = f"<p class='muted'>Trend lookup failed: {_esc(error)}</p>"
    elif not queries:
        # WHICH of the four, not all four. "pytrends not installed,
        # rate-limited, or nothing rising this week" was three guesses and a
        # shrug: one of them needs a pip command, one clears by itself, and one
        # is not a problem at all — printed identically, so the page could not
        # tell the owner whether to do anything.
        list_html = (f"<p class='muted'>No rising queries for this niche: "
                     f"{_esc(reason)}</p>")
    else:
        items = ""
        for q in queries:
            items += (f'<form method="post" action="/request-topic" style="margin:6px 0">'
                     f'<input type="hidden" name="topic" value="{_esc(q)}">'
                     f'<button type="submit">Queue "{_esc(q)}"</button></form>\n')
        list_html = items

    body = f"""
    <a class="back" href="/">← back</a>
    <h2 style="margin-top:14px">Trending queries — {_esc(niche or '(no niche configured)')}</h2>
    <p class="muted">This week's rising Google Trends searches for the
       niche. Queuing one resolves it to a real Wikipedia article through
       the same /request-topic path as typing a topic by hand — same
       fact-grounding, same gates, nothing skipped.</p>
    <div class="filters">{niche_links}</div>
    {list_html}
    """
    return _head() + body + PAGE_TAIL


@app.route("/request-topic", methods=["POST"])
def request_topic():
    """Kick off a real Rufus run for a topic YOU chose, in the background.
    Never blocks the request (a run takes minutes to hours) and never
    auto-uploads — it lands in the normal pending-review queue like any
    other video, going through every existing gate (fact-check, QC, score)
    exactly the same way. main.py's own per-channel FileLock is what
    actually prevents two overlapping runs of the same channel; this route
    doesn't need to duplicate that check."""
    auth.require("generate")
    topic = request.form.get("topic", "").strip()
    channel = request.form.get("channel", "").strip() or None
    if not topic:
        return _redirect_index(error="topic is required")
    try:
        _, log_path = _launch_run(topic=topic, channel=channel)
        return _redirect_index(
            ok=f'Queued "{topic}" — this can take a while. It will appear in '
               f'the pending list below when done. Log: logs/{log_path.name}')
    except Exception as e:
        return _redirect_index(error=f"failed to start the run: {e}")


def _redirect_index(ok: str = None, error: str = None):
    url = "/"
    if ok:
        url += f"?ok={_urlquote(ok)}"
    elif error:
        url += f"?error={_urlquote(error)}"
    return redirect(url)


# os.environ is process-wide, so two overlapping requests inside _scoped_env
# would interleave their mutations and one upload would go out under the
# other's channel (a real audit finding). This lock is what makes that
# impossible.
#
# IT REPLACES threaded=False, WHICH WAS A MUCH LARGER CLAIM THAN THE PROBLEM.
# "These env mutations must not overlap" was implemented as "nothing anywhere
# in this application may overlap" — so every keyframe thumbnail, every poll of
# /api/status and every page load queued behind a single thread, and the home
# page's ~240 image requests were served strictly one at a time. Over the
# tailnet from a phone that is most of what "slow" meant. One route needed
# mutual exclusion; forty-odd routes were paying for it.
_ENV_LOCK = threading.Lock()


@contextmanager
def _scoped_env(**overrides):
    """Set env vars for the duration of the block, then restore exactly what
    was there before (including "unset" if the key didn't exist).

    Holds _ENV_LOCK for the whole block, so this is safe under a threaded
    server. Mutating os.environ permanently — as a naive assignment would —
    also leaks across every later request in this long-lived process
    (confirmed live: it leaked into an unrelated test suite run in the same
    process), which is what the restore is for.

    The lock is held across the upload itself, not just the assignment. That
    is deliberate: the point is that the environment READ by the upload is the
    one this block set, and releasing early would let a second approval change
    it underneath the first."""
    # SNAPSHOT INSIDE THE LOCK. Reading `prev` first looks harmless and is
    # not: a second approval blocked on acquire() would have already captured
    # the FIRST one's override as "what was there before", and its restore
    # would put that value back instead of unsetting the key. The variable
    # leaks, and the next run inherits a channel nobody chose. Caught by
    # test_two_overlapping_scoped_envs_cannot_interleave.
    _ENV_LOCK.acquire()
    prev = {k: os.environ.get(k) for k in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        # RESTORE INSIDE THE LOCK. Releasing first would let another approval
        # acquire it, apply its own overrides, and then have them overwritten
        # by this block's restore — the exact interleaving the lock exists to
        # prevent, moved four lines later.
        try:
            for k, v in prev.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        finally:
            _ENV_LOCK.release()


def _redirect_detail(video_id: int, ok: str = None, error: str = None):
    url = f"/video/{video_id}"
    if ok:
        url += f"?ok={_urlquote(ok)}"
    elif error:
        url += f"?error={_urlquote(error)}"
    return redirect(url)


@app.route("/video/<int:video_id>")
def video_detail(video_id):
    v = _video_detail(video_id)
    if not v:
        abort(404)

    crit_rows = "".join(
        f"<tr><td>{label}</td><td>{v[key] if v[key] is not None else '—'}</td></tr>"
        for label, key in (
            ("Specificity", "score_specificity"), ("Hook", "score_hook"),
            ("Compression", "score_compression"), ("Loop", "score_loop"),
            ("Human", "score_human"),
        )
    )

    if v["upload_status"] == "approved" and v["youtube_id"]:
        status_html = (f'<span class="badge ok">approved → '
                       f'<a href="https://youtube.com/watch?v={_esc(v["youtube_id"])}" '
                       f'style="color:inherit" target="_blank">watch</a></span>')
    elif v["upload_status"] == "rejected":
        status_html = '<span class="badge held">rejected</span>'
    else:
        status_html = '<span class="badge pending">pending review</span>'
    if v["hold_reason"]:
        status_html += (f'<div class="muted" style="margin-top:6px">auto-gate note: '
                        f'{_esc(v["hold_reason"])}</div>')

    msg_html = _msg_banner()

    # Buttons follow the role: a partner sees Download (their whole reason for
    # being here) but no Approve, because publishing isn't theirs to do.
    buttons = ""
    if v["upload_status"] != "approved" and auth.can("approve"):
        # The confirm names the title, because the editor for it now sits
        # below this button rather than above it. json.dumps builds the JS
        # string literal (quotes, backslashes, newlines) and _esc then makes
        # it safe as an attribute value — hand-escaping one and not the other
        # is how a title with an apostrophe breaks the button entirely.
        going_out = _esc(json.dumps(
            f"Upload \u201c{(v['title'] or v['script_hook'] or 'this video')[:70]}"
            f"\u201d to YouTube now?"))
        buttons += (f'<form method="post" action="/video/{v["id"]}/approve" '
                    f'onsubmit="return confirm({going_out});">'
                    f'<button class="btn approve" type="submit">✓ Approve &amp; Upload</button>'
                    f'</form>')
    if v["upload_status"] != "approved" and auth.can("reject"):
        reject_label = "Un-reject" if v["upload_status"] == "rejected" else "Reject"
        buttons += (f'<form method="post" action="/video/{v["id"]}/reject">'
                    f'<button class="btn reject" type="submit">{reject_label}</button></form>')
    if auth.can("download") and (v["video_file"] and Path(v["video_file"]).exists()):
        buttons += (f'<a class="btn save" style="text-decoration:none;display:inline-block" '
                    f'href="/video/{v["id"]}/download">⬇ Download mp4</a>')
    actions_html = f'<div class="actions">{buttons}</div>' if buttons else ""

    # ALREADY ON YOUTUBE? SAY SO. Publishing by hand is the correct thing to do
    # while nothing auto-uploads, and it left the video invisible to analytics
    # — no youtube_id, so no metrics, so no view counts, so nothing for the
    # hook learning to learn from. One paste fixes that per video.
    published_html = ""
    if auth.can("approve"):
        if v.get("youtube_id"):
            published_html = (
                f'<h2>On YouTube</h2>'
                f'<p class="muted">Tracked as '
                f'<a href="https://youtu.be/{_esc(v["youtube_id"])}" target="_blank" '
                f'rel="noopener">{_esc(v["youtube_id"])}</a>. Its views feed the '
                f'hook learning on the next analytics fetch — see '
                f'<a href="/tracking">Tracking</a>.</p>')
        else:
            published_html = f"""
        <h2>Published this one by hand?</h2>
        <p class="muted">Paste its link and the pipeline can track it. Without
           a YouTube id there are no view counts, and with no view counts every
           quality judgement here stays a guess about what works.</p>
        <form method="post" action="/video/{v['id']}/published"
              style="display:flex;gap:10px;flex-wrap:wrap;align-items:flex-end">
          <div style="flex:1;min-width:260px">
            <label for="yt">YouTube link or id</label>
            <input class="field" style="margin:6px 0 0" type="text" id="yt"
                   name="youtube" required
                   placeholder="https://youtube.com/shorts/... or the 11-character id">
          </div>
          <button class="btn save" type="submit" style="height:38px">
            Mark as published</button>
        </form>
        """

    edit_html = ""
    if v["upload_status"] != "approved" and auth.can("edit"):
        edit_html = f"""
        <h2>Title &amp; description (edit before approving)</h2>
        <form method="post" action="/video/{v['id']}/edit">
          <label for="title">Title</label>
          <input class="field" type="text" id="title" name="title" maxlength="100"
                 value="{_esc(v['title'] or '')}">
          <label for="description">Description</label>
          <textarea class="field" id="description" name="description" rows="6"
                    >{_esc(v['description'] or '')}</textarea>
          <button class="btn save" type="submit">Save</button>
        </form>
        """
    else:
        edit_html = f"""
        <h2>Title &amp; description</h2>
        <p><b>{_esc(v['title'] or '')}</b></p>
        <div class="script">{_esc(v['description'] or '—')}</div>
        """

    # Image-generation prompts (the script→images chain), rendered inline so a
    # reviewer sees exactly what each beat's still was told to draw, next to the
    # still itself — not a pile of files to download one by one.
    img_prompts = _image_prompts(v["run_id"])
    prompts_html = ("<p class='muted'>No per-beat image prompts saved for this "
                    "run yet.</p>")
    if img_prompts:
        cards = ""
        for p in img_prompts:
            thumb = ""
            if p["image"]:
                thumb = (f'<a href="/debug/{_esc(v["run_id"])}/{_esc(p["image"])}" '
                         f'target="_blank"><img src="/debug/{_esc(v["run_id"])}/'
                         f'{_esc(p["image"])}?w=240" loading="lazy" '
                         f'style="width:120px;border-radius:6px;flex:0 0 auto"></a>')
            cards += (
                f'<div style="display:flex;gap:12px;margin:10px 0;align-items:flex-start">'
                f'{thumb}'
                f'<div><b>beat {_esc(p["n"])}</b>'
                f'<div class="script" style="margin-top:4px">{_esc(p["prompt"])}</div>'
                f'</div></div>')
        prompts_html = cards

    assets = _debug_assets(v["run_id"])
    assets_html = "<p class='muted'>No debug artifacts for this run.</p>"
    if assets:
        links = "".join(
            f'<a href="/debug/{_esc(v["run_id"])}/{_esc(a["name"])}" target="_blank">'
            f'{_esc(a["name"])} ({a["size_kb"]}KB)</a>'
            for a in assets
        )
        assets_html = f'<div class="assets">{links}</div>'

    # THE ORDER OF THIS PAGE IS THE ORDER OF THE JOB: hear it, look at the
    # beats, decide. It used to be the order the page was built in — the id
    # line, the status, the score, the five-row criteria table, the approve
    # buttons, the publish form and the title editor all came BEFORE the
    # voiceover and the contact sheet. On a laptop that is a slightly odd
    # page. On a phone, which is where this review actually happens, it is
    # four screens of numbers and forms before the first thing you were going
    # to judge it on, and the buttons are somewhere in the middle of them.
    #
    # So: what it is, then the thing itself, then the decision. Everything
    # that explains or amends the decision — the score breakdown, the critic's
    # reasoning, the rewrite candidate, the seed, the prompts, the artifacts —
    # comes after it, because that is when it gets read.
    #
    # The title editor stays adjacent to the buttons rather than below the
    # fold, and Approve now names the title it is about to publish, so moving
    # the editor down cannot quietly ship a title nobody looked at.
    body = f"""
    <a class="back" href="/">← back</a>
    <h2 style="margin-top:14px">#{v['id']} · {_esc(v['upload_date'])} · {_esc(v['niche'])} · {_esc(v['channel'])}</h2>
    {msg_html}
    <p>{status_html}</p>
    {_preview_block(v['run_id'], v['id'])}
    {actions_html}
    {edit_html}
    <h2>Script</h2>
    <div class="script">{_esc(v['script_full'] or v['script_hook'])}</div>
    {_rewrite_block(v)}
    <h2>Score: <span style="color:{_score_color(v['score'])}">{v['score']}/10</span></h2>
    <p class="muted">{v['attempts_used'] or '?'} attempts, temp {v['final_temperature'] or '?'}</p>
    <table style="max-width:320px">{crit_rows}</table>
    <h2>Why this score (critic reasoning)</h2>
    <div class="script">{_esc(v['score_reasoning'] or '—')}</div>
    <h2>Seed / source</h2>
    <p class="muted">{_esc(v['seed_type'])} · {_esc(v['seed_source'])}</p>
    <div class="script">{_esc(v['seed_content'] or '—')}</div>
    <h2>Image prompts (what each beat was told to draw)</h2>
    {prompts_html}
    {published_html}
    <h2>Debug artifacts (run {_esc(v['run_id'] or '—')})</h2>
    {assets_html}
    """
    return _head() + body + PAGE_TAIL


def _extra_publishers() -> list[tuple[str, object]]:
    """Platforms to cross-post to after YouTube, as (name, upload_fn).

    Opt-in per platform so an unconfigured account never turns a good approval
    into a scary error: RUFUS_PUBLISH_TIKTOK=1 enables TikTok. The module is
    imported lazily because it reads credentials at import time.

    Instagram is deliberately absent: the Graph API needs a Business/Creator
    account AND a PUBLICLY reachable video URL to pull from (it will not accept
    a local file), so it needs hosting decisions this box can't make on its
    own. Add it here once that's settled."""
    out = []
    if os.environ.get("RUFUS_PUBLISH_TIKTOK", "0").strip().lower() in ("1", "true", "yes", "on"):
        try:
            import tiktok_uploader
            out.append(("TikTok", tiktok_uploader.upload))
        except Exception as e:
            print(f"[dashboard] TikTok publisher unavailable ({e})")
    return out


@app.route("/video/<int:video_id>/rewrite", methods=["POST"])
def rewrite_script(video_id: int):
    """Write this video's script again, as a candidate beside the original.

    Launched as a subprocess and not run here: the writer takes tens of
    seconds of API calls, and this dashboard runs threaded=False on purpose
    (see app.run) — doing it inline would freeze every other page for the
    duration.
    """
    auth.require("edit")
    v = _video_detail(video_id)
    if not v:
        abort(404)
    if not (v.get("run_id") or "").strip():
        return _redirect_detail(video_id, error=(
            "this video has no run folder, so there is nowhere to keep a "
            "candidate beside it"))
    try:
        cmd = [sys.executable, str(ROOT / "scripts" / "rewrite.py"), str(video_id)]
        env = os.environ.copy()
        env.update(_load_settings())
        env.setdefault("PYTHONUTF8", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        if v.get("niche"):
            env["RUFUS_NICHE_OVERRIDE"] = str(v["niche"])
        log_dir = paths.log_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        log = log_dir / f"rewrite_{video_id}_{int(time.time())}.log"
        with open(log, "wb") as logf:
            subprocess.Popen(cmd, cwd=str(ROOT), stdout=logf, env=env,
                             stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL)
    except Exception as e:
        return _redirect_detail(video_id, error=f"could not start the rewrite: {e}")
    return _redirect_detail(video_id, ok=(
        "writing another script — refresh in half a minute and it will appear "
        "under the current one"))


@app.route("/video/<int:video_id>/use-script", methods=["POST"])
def use_script(video_id: int):
    """Rebuild the video from the candidate script.

    This is the expensive half and it is a whole new run: voice, stills,
    render. The candidate is handed to main.py with --script so the writer is
    skipped — otherwise the pipeline would write a third script and the choice
    would have meant nothing.
    """
    auth.require("generate")
    v = _video_detail(video_id)
    if not v:
        abort(404)
    import rewrite as rewrite_mod

    cand = rewrite_mod.latest(v.get("run_id") or "")
    if not cand or not (cand.get("script") or "").strip():
        return _redirect_detail(video_id, error="there is no candidate script to use")
    busy = [c for c in _channels() if _run_in_progress(c)]
    if busy:
        return _redirect_detail(video_id, error=(
            f"a run is already in progress ({', '.join(busy)}) — wait for it "
            f"to finish"))

    script_file = DEBUG_ROOT / v["run_id"] / "chosen_script.txt"
    try:
        script_file.write_text(cand["script"].strip() + "\n", encoding="utf-8")
    except OSError as e:
        return _redirect_detail(video_id, error=f"could not stage the script: {e}")

    try:
        proc, log = _launch_run(niche=v.get("niche"), channel=v.get("channel"),
                                script_file=str(script_file))
    except Exception as e:
        return _redirect_detail(video_id, error=f"could not start the run: {e}")
    return redirect("/?ok=" + _urlquote(
        "building a video from the chosen script — it appears in the queue "
        "when it is done"))


@app.route("/video/<int:video_id>/discard-script", methods=["POST"])
def discard_script(video_id: int):
    auth.require("edit")
    v = _video_detail(video_id)
    if not v:
        abort(404)
    import rewrite as rewrite_mod
    rewrite_mod.discard(v.get("run_id") or "")
    return _redirect_detail(video_id, ok="candidate discarded")


def _launch_recut(video_id: int, v: dict):
    """Spawn scripts/recut.py for one video. (proc, log_path).

    Deliberately the same shape as _launch_run: same env layering (saved
    settings on top of the process environment), same UTF-8 defaults for a
    child whose stdout is a file, same log directory. A re-cut that resolved
    the video format or the niche differently from a run would come out the
    wrong shape, and the settings are where that is decided.
    """
    cmd = [sys.executable, str(ROOT / "scripts" / "recut.py"), str(video_id)]
    env = os.environ.copy()
    env.update(_load_settings())
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    # The row's own niche wins over whatever the dashboard is currently set to.
    # Re-cutting last week's money_history video from a machine now pointed at
    # a different niche would fetch the wrong music and grade it wrong.
    if v.get("niche"):
        env["RUFUS_NICHE_OVERRIDE"] = str(v["niche"])
    log_dir = paths.log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"recut_{video_id}_{int(time.time())}.log"
    with open(log_path, "wb") as logf:
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=logf, env=env,
                                stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL)
    return proc, log_path


def _launch_regen(v: dict, beat: int, png: Path, txt: Path, prompt: str,
                  *, save_prompt: bool):
    """Redraw one beat in a subprocess. (proc, log_path).

    Same shape as _launch_recut: sys.executable, saved settings layered over
    this process's environment, UTF-8 defaults for a child whose stdout is a
    file, and a log in logs/.

    The prompt goes through a FILE rather than argv. Beat prompts run to
    several hundred characters, Windows caps a command line at 32,767, and
    quoting a multi-line prompt through cmd is a class of bug worth not having.
    The child deletes it after reading.
    """
    handoff = png.with_name(f"{png.stem}.regen-prompt.txt")
    handoff.write_text(prompt, encoding="utf-8")
    cmd = [sys.executable, str(ROOT / "scripts" / "comfy_client.py"),
           "--regen-beat", "--out", str(png), "--prompt-file", str(handoff)]
    if save_prompt:
        # Only when the human edited it. An unedited prompt is already in the
        # sidecar, and rewriting it would touch the file for no reason.
        cmd += ["--sidecar", str(txt)]
    if v.get("niche"):
        cmd += ["--niche", str(v["niche"])]
    env = os.environ.copy()
    env.update(_load_settings())
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    # The row's own niche wins over whatever the dashboard is set to now —
    # same reasoning as _launch_recut.
    if v.get("niche"):
        env["RUFUS_NICHE_OVERRIDE"] = str(v["niche"])
    log_dir = paths.log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"regen_{v['id']}_{beat:02d}_{int(time.time())}.log"
    try:
        with open(log_path, "wb") as logf:
            proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=logf, env=env,
                                    stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL)
    except Exception:
        handoff.unlink(missing_ok=True)   # nothing will read it now
        raise
    return proc, log_path


def _run_dir(run_id: str):
    """A run's debug folder, or None if the id does not name one.

    The run_id reaches the filesystem from a form field, so it is resolved and
    checked against DEBUG_ROOT rather than trusted — the same shape as
    thumbnail_file and thumbnails_delete.
    """
    if not run_id:
        return None
    folder = (DEBUG_ROOT / run_id).resolve()
    if folder.parent != DEBUG_ROOT.resolve() or not folder.is_dir():
        return None
    return folder


@app.route("/video/<int:video_id>/beat/<int:beat>/regen", methods=["POST"])
def regen_beat(video_id: int, beat: int):
    """Redraw one beat's still, optionally from an edited prompt.

    WHY ONE PICTURE AND NOT THE RUN. A gallery is usually right about most of
    its frames and wrong about one — a contact sheet instead of a scene, a
    figure with no arms, the wrong object. Rebuilding the video for that costs
    thirteen minutes of GPU to change one shot, so it does not get done, and
    the video ships with the bad frame in it.

    The prompt is editable because the prompt is usually what is wrong. It is
    written back to the sidecar so the run's own record stays true to what
    produced the picture.

    The contact sheet is not rebuilt here: review_proxy.contact_sheet already
    regenerates whenever a still is newer than the sheet, so the next page load
    does it.
    """
    auth.require("edit")
    v = _video_detail(video_id)
    if not v:
        abort(404)
    folder = _run_dir(v.get("run_id"))
    if folder is None:
        return _redirect_detail(video_id, error="this run has no saved frames")

    png = folder / f"{beat:02d}.png"
    txt = folder / f"{beat:02d}.txt"
    if not png.exists():
        return _redirect_detail(video_id, error=f"no still {beat:02d}.png in this run")

    import comfy_client
    edited = request.form.get("prompt", "").strip()
    prompt = edited or comfy_client.read_beat_prompt(txt)
    if not prompt:
        return _redirect_detail(video_id, error=(
            f"beat {beat:02d} has no saved prompt, so there is nothing to "
            f"redraw it from — type one in and try again"))

    if _run_in_progress(v.get("channel")):
        return _redirect_detail(video_id, error=(
            "a run is using the GPU right now — wait for it to finish"))

    # NOT ON THIS THREAD. This called render_one_beat() inline, and a GPU
    # render is tens of seconds to minutes — long enough that /healthz cannot
    # answer, the watchdog reads that as death, and it starts a second
    # dashboard into the port this one is still holding. That is the ten-hour
    # outage, reachable by pressing Regen. Its two neighbours, recut_video and
    # rewrite_script, have always spawned subprocesses, and rewrite_script even
    # carries the comment saying why. This is the one that drifted.
    try:
        _proc, log = _launch_regen(v, beat, png, txt, prompt,
                                   save_prompt=bool(edited))
    except Exception as e:
        return _redirect_detail(video_id, error=f"could not start the redraw: {e}")
    return _redirect_detail(video_id, ok=(
        f"redrawing beat {beat:02d} — refresh in a minute, then re-cut to put "
        f"it in the video (log: {log.name})"))


@app.route("/video/<int:video_id>/recut", methods=["POST"])
def recut_video(video_id: int):
    """Rebuild the mp4 from the stills that are on disk right now.

    The voiceover is not regenerated and not re-transcribed, so the word
    timings the cut is built from are the same ones the last render used —
    which is what makes a re-cut safe: it can change which picture is on
    screen, and it cannot move where the cuts are.
    """
    auth.require("edit")
    v = _video_detail(video_id)
    if not v:
        abort(404)
    if _run_in_progress(v.get("channel")):
        return _redirect_detail(video_id, error=(
            "a run is using the GPU right now — wait for it to finish"))
    folder = _run_dir(v.get("run_id"))
    if folder is None:
        return _redirect_detail(video_id, error="this run has no saved frames")
    try:
        proc, log = _launch_recut(video_id, v)
    except Exception as e:
        return _redirect_detail(video_id, error=f"could not start the re-cut: {e}")
    return _redirect_detail(video_id, ok=(
        f"re-cutting — the new file replaces the old one when it finishes "
        f"(log: {log.name})"))


@app.route("/video/<int:video_id>/approve", methods=["POST"])
def approve_video(video_id):
    """The ONLY path that actually uploads a video — see module docstring.
    Reuses the video's own stored channel/niche context (not whatever's
    'active' today) so the right voice/CTA-pool/category apply even if the
    approval happens days after generation.

    Publishing is the one action a partner role must never reach: it puts
    something permanent on the owner's channel."""
    auth.require("approve")
    v = _video_detail(video_id)
    if not v:
        abort(404)
    if v["youtube_id"]:
        return _redirect_detail(video_id, error="already uploaded")
    if v["score"] is None or v["score"] < HARD_MIN_UPLOAD_SCORE:
        shown = "unscored" if v["score"] is None else f"{v['score']}/10"
        return _redirect_detail(
            video_id,
            error=(f"blocked: score {shown} is below the {HARD_MIN_UPLOAD_SCORE}/10 "
                   f"minimum — this video cannot be approved for upload. Reject it "
                   f"or fix the underlying script/QC issue instead."))
    video_file = Path(v["video_file"] or "")
    if not video_file.exists():
        return _redirect_detail(video_id, error=f"video file missing on disk: {video_file}")

    thumb = video_file.with_suffix(".thumb.jpg")
    try:
        import youtube_uploader as yt_mod
        hashtags = yt_mod.NICHE_HASHTAGS.get(v["niche"], ["#Shorts"])
        meta = {
            "title": (v["title"] or v["script_hook"] or "Short")[:100],
            "description": v["description"] or (v["script_full"] or ""),
            "tags": [t.lstrip("#") for t in hashtags],
        }
        env_overrides = {"RUFUS_CHANNEL": v["channel"] or "main_en"}
        if v["niche"]:
            env_overrides["RUFUS_NICHE_OVERRIDE"] = v["niche"]

        with _busy(f"uploading video #{video_id}"), _scoped_env(**env_overrides):
            yt_url, yt_id = yt_mod.upload(video_file, v["script_full"] or "",
                                          thumbnail_path=thumb if thumb.exists() else None,
                                          metadata=meta, source_url=v.get("seed_url") or None,
                                          seed_source=v.get("seed_source"))
    except Exception as e:
        # Upload itself failed — the video did NOT go up, so re-approving is
        # safe. Record it like main.py does so report.py's FAILED count sees
        # dashboard failures too (it used to miss them entirely).
        try:
            db_manager.mark_upload_failed(video_id, str(e))
        except Exception:
            pass
        return _redirect_detail(video_id, error=f"Upload failed (not uploaded, safe to retry): {e}")

    # Cross-post. TikTok has its own uploader module but was never wired to
    # the approve action, so approving only ever reached YouTube. Each extra
    # platform is best-effort and reported in the result message: YouTube
    # already succeeded by this point, so one platform being unconfigured or
    # erroring must not read as "the approval failed".
    extra = []
    for name, fn in _extra_publishers():
        try:
            fn(video_file, v["script_full"] or "")
            extra.append(f"{name} ok")
        except Exception as e:
            extra.append(f"{name} FAILED ({str(e)[:80]})")

    # The upload SUCCEEDED. A DB failure past this point must NOT read as
    # "upload failed" — that would tempt a re-approve and publish a DUPLICATE
    # public video. Separate block, explicit do-not-retry message.
    try:
        db_manager.update_youtube_id(video_id, yt_id)
        db_manager.set_upload_status(video_id, "approved", by=_whoami())
        db_manager.set_publish_at(video_id, getattr(yt_mod, "LAST_PUBLISH_AT", ""))
    except Exception as db_err:
        return _redirect_detail(
            video_id,
            error=(f"UPLOADED OK ({yt_url}) but the status update failed "
                   f"({db_err}). Do NOT re-approve — it's already live. Fix "
                   f"the DB row manually if needed."))
    when = getattr(yt_mod, "LAST_PUBLISH_AT", "")
    msg = (f"Uploaded, live at {when} UTC: {yt_url}" if when
           else f"Uploaded: {yt_url}")
    if extra:
        msg += "  |  " + "; ".join(extra)
    try:
        import notify
        notify.notify_published(
            title=(v["title"] or v["script_hook"] or "Short"),
            youtube_id=yt_id, score=v["score"], niche=v["niche"],
            video_path=video_file,
            by=(getattr(g, "rufus_user", None) or {}).get("name"))
    except Exception:
        pass
    return _redirect_detail(video_id, ok=msg)


@app.route("/video/<int:video_id>/reject", methods=["POST"])
def reject_video(video_id):
    auth.require("reject")
    v = _video_detail(video_id)
    if not v:
        abort(404)
    if v["youtube_id"]:
        return _redirect_detail(video_id, error="already uploaded — can't reject")
    new_status = "pending" if v["upload_status"] == "rejected" else "rejected"
    db_manager.set_upload_status(video_id, new_status, by=_whoami())
    return _redirect_detail(video_id, ok=f"marked {new_status}")


@app.route("/video/<int:video_id>/edit", methods=["POST"])
def edit_video(video_id):
    auth.require("edit")
    v = _video_detail(video_id)
    if not v:
        abort(404)
    title = request.form.get("title", "").strip()
    description = request.form.get("description", "").strip()
    db_manager.update_metadata(video_id, title=title or None, description=description or None)
    return _redirect_detail(video_id, ok="saved")


def _beat_controls(run_id: str, folder, video_id: int) -> str:
    """One row per beat: the frame, its prompt, and a button to redraw it.

    Folded into a <details> on purpose. The preview block exists to be ~1MB on
    a phone on cellular; a grid of every beat with its prompt open would undo
    that for the common case, which is looking at the contact sheet and
    deciding yes or no. This is the drawer you open when the answer is "yes
    except that one".

    The prompt shown is the sidecar's, which is the FINAL prompt — the shot
    description with the style block already on it. Editing it and pressing
    Regen sends exactly what is in the box, so a hand-written prompt is not
    silently given a second helping of the style.
    """
    import comfy_client

    stills = sorted((f for f in folder.glob("*.png") if f.stem.isdigit()),
                    key=lambda f: f.name)
    if not stills:
        return ""

    rows = []
    for png in stills:
        n = png.stem
        prompt = comfy_client.read_beat_prompt(folder / f"{n}.txt")
        rows.append(
            f'<div style="display:flex;gap:10px;padding:8px 0;'
            f'border-top:1px solid var(--border)">'
            f'<img src="/debug/{_esc(run_id)}/{_urlquote(png.name)}?w=120" '
            f'loading="lazy" alt="beat {_esc(n)}" '
            f'style="width:54px;height:96px;object-fit:cover;border-radius:4px;'
            f'flex-shrink:0">'
            f'<form method="post" style="flex:1;min-width:0" '
            f'action="/video/{video_id}/beat/{int(n)}/regen">'
            f'<div class="muted" style="font-size:12px">beat {_esc(n)}</div>'
            f'<textarea name="prompt" rows="3" class="field" '
            f'style="width:100%;font-size:12px;margin:4px 0">'
            f'{_esc(prompt)}</textarea>'
            f'<button class="btn" type="submit" '
            f'style="padding:5px 10px;font-size:12px">Redraw this beat</button>'
            f'</form></div>')

    return (
        f'<details style="margin-top:10px">'
        f'<summary style="cursor:pointer">Redraw a beat '
        f'<span class="muted">({len(stills)} frames — edit a prompt or just '
        f'press redraw for a new roll)</span></summary>'
        f'{"".join(rows)}'
        f'<form method="post" action="/video/{video_id}/recut" '
        f'style="margin-top:12px" onsubmit="return confirm('
        f'\'Re-cut the video from the frames on disk now? The voiceover is '
        f'reused, so the cuts stay where they are.\');">'
        f'<button class="btn save" type="submit">Re-cut the video</button>'
        f'<div class="muted" style="font-size:12px;margin-top:4px">'
        f'Redrawing changes the frame on disk. The mp4 only changes when you '
        f're-cut.</div></form>'
        f'</details>')


def _rewrite_block(v: dict) -> str:
    """The candidate script, beside the one that is live, with both rubrics.

    A COMPARISON AND NOT A MEMORY TEST. The question being asked is "is this
    one better", and answering it by scrolling up to reread the original is
    how you end up approving whichever you read last. Both scores and both
    critic notes are on screen at the same time.
    """
    if not auth.can("edit"):
        return ""
    run_id = (v.get("run_id") or "").strip()
    if not run_id:
        return ""
    try:
        import rewrite as rewrite_mod
        cand = rewrite_mod.latest(run_id)
    except Exception as e:                        # never break the page
        print(f"[dashboard] candidate unavailable ({e})")
        return ""

    ask = (f'<form method="post" action="/video/{v["id"]}/rewrite" '
           f'style="margin:8px 0">'
           f'<button class="btn" type="submit">Write another script</button>'
           f'<span class="muted" style="margin-left:8px;font-size:12px">'
           f'same source, same seed · seconds, no GPU · pressing it again '
           f'gives a different one</span></form>')
    if not cand or not (cand.get("script") or "").strip():
        return f"<h2>Another script</h2>{ask}"

    crits = cand.get("criterion_scores") or {}
    crit_line = " · ".join(f"{k} {crits[k]}/3" for k in sorted(crits)) or ""
    return f"""<h2>Another script</h2>{ask}
    <div class="card" style="padding:12px">
      <div><strong>{_esc(cand.get('score', 0))}/10</strong>
        <span class="muted">· written {_esc(cand.get('written_at', ''))}
        · {_esc(crit_line)}</span></div>
      <div class="script" style="margin-top:8px">{_esc(cand.get('script', ''))}</div>
      <details style="margin-top:6px"><summary class="muted"
        style="cursor:pointer">why this score</summary>
        <div class="script">{_esc(cand.get('reasoning', '') or '—')}</div>
      </details>
      <div class="row" style="margin-top:10px">
        <form method="post" action="/video/{v['id']}/use-script"
              onsubmit="return confirm('Build a new video from this script? '
              + 'That is a full run — voice, pictures, render.');">
          <button class="btn save" type="submit">Use this one</button></form>
        <form method="post" action="/video/{v['id']}/discard-script"
              style="margin-left:8px">
          <button class="btn" type="submit">Discard</button></form>
      </div>
      <div class="muted" style="font-size:12px;margin-top:6px">
        Using it builds a NEW video and leaves this one exactly where it is.
      </div>
    </div>"""


def _preview_block(run_id: str, video_id: int | None = None) -> str:
    """Hear-it-and-see-it review, sized for a phone on cellular.

    Before this, judging a run remotely meant downloading the 15-25MB master
    (as_attachment, so not even a preview) or opening 8-10 debug PNGs at
    1.1-2.7MB each — i.e. the same 20MB, one tap at a time. Both are unusable
    away from the desk, which is why every review of these runs has been done
    by pasting walls of text instead of looking at the video.

    What review actually needs is small: the voiceover (~800KB, and the whole
    question being asked is whether the SCRIPT SOUNDS GOOD ALOUD) and one
    contact sheet of the beats (~300KB). Roughly 1MB for both, versus 20MB.

    preload="none" is load-bearing: without it the browser fetches the audio on
    page load, and the page costs its 800KB whether or not you press play."""
    if not run_id:
        return ""
    folder = (DEBUG_ROOT / run_id).resolve()
    if folder.parent != DEBUG_ROOT.resolve() or not folder.is_dir():
        return ""

    parts = []
    voice = folder / "voiceover.mp3"
    if voice.exists():
        parts.append(
            f'<audio controls preload="none" style="width:100%;max-width:520px" '
            f'src="/debug/{_esc(run_id)}/voiceover.mp3"></audio>'
            f'<div class="muted" style="margin:4px 0 12px">'
            f'voiceover · {voice.stat().st_size // 1024}KB</div>')

    try:
        import review_proxy
        sheet = review_proxy.contact_sheet(folder)
    except Exception as e:                        # never break the page
        print(f"[dashboard] contact sheet unavailable ({e})")
        sheet = None
    if sheet is not None:
        parts.append(
            f'<a href="/debug/{_esc(run_id)}/{_esc(sheet.name)}" target="_blank">'
            f'<img src="/debug/{_esc(run_id)}/{_esc(sheet.name)}" loading="lazy" '
            f'style="width:100%;max-width:760px;border-radius:8px" '
            f'alt="every beat in order"></a>'
            f'<div class="muted" style="margin-top:4px">'
            f'all beats in order · {sheet.stat().st_size // 1024}KB · tap to enlarge</div>')

    if video_id is not None and auth.can("edit"):
        try:
            parts.append(_beat_controls(run_id, folder, video_id))
        except Exception as e:                    # never break the page
            print(f"[dashboard] beat controls unavailable ({e})")

    if not parts:
        return ""
    return "<h2>Preview</h2>" + "".join(parts)


# Downscaled copies of the run keyframes, cached beside them.
#
# WHY. Every gallery tile, every prompt thumbnail and every run preview points
# at the ORIGINAL 1080x1920 png and renders it into a 120px box. The browser
# still downloads the whole thing: one live session's log shows well over a
# hundred of those requests from a few minutes of scrolling, which on this
# channel's keyframes is tens of megabytes to draw a strip of thumbnails. The
# images are already lazy-loaded — that limits WHEN they are fetched, not how
# big they are.
_THUMB_DIR_NAME = ".thumbs"
_THUMB_WIDTHS = (120, 240, 480)     # a fixed set: an open ?w= is a cache bomb

# Rendered stills and generated images are written once and never edited, so
# the browser may keep them indefinitely. Without this every gallery scroll
# re-fetches megabytes it already has — and a regenerated beat still appears,
# because comfy writes a NEW filename rather than overwriting one.
_IMAGE_MAX_AGE = 60 * 60 * 24 * 30


def _thumb_of(folder: Path, filename: str, width: int | None) -> "Path | None":
    """A cached downscale of one keyframe, or None to serve the original.

    None on every uncertainty — an unknown width, a non-image, a Pillow that
    is not installed, a folder that cannot be written to. The original always
    works, so the fast path is an optimisation and never a dependency.
    """
    if not width or width not in _THUMB_WIDTHS:
        return None
    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        return None
    src = (folder / filename).resolve()
    if src.parent != folder or not src.is_file():
        return None
    cache = folder / _THUMB_DIR_NAME
    out = cache / f"{width}_{src.stem}.jpg"
    try:
        if out.is_file() and out.stat().st_mtime >= src.stat().st_mtime:
            return out
        from PIL import Image
        cache.mkdir(exist_ok=True)
        with Image.open(src) as im:
            im = im.convert("RGB")
            im.thumbnail((width, width * 4), Image.LANCZOS)
            im.save(out, "JPEG", quality=82, optimize=True)
        return out
    except Exception:
        return None      # the original is always correct


@app.route("/favicon.ico")
def favicon():
    """A tab icon, so every page load stops 404ing for one.

    Inline SVG rather than a binary in the repo: it is nine lines, it scales,
    and it needs no build step — the same reason the rest of this dashboard
    has no assets directory."""
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
           '<rect width="32" height="32" rx="7" fill="#3b82f6"/>'
           '<path d="M11 8h6.5a5 5 0 0 1 1.2 9.85L22 24h-4l-3-6h-1v6h-3z" '
           'fill="#fff"/></svg>')
    return app.response_class(svg, mimetype="image/svg+xml",
                              headers={"Cache-Control": "public, max-age=604800"})


@app.route("/debug/<run_id>/<path:filename>")
def debug_file(run_id, filename):
    """Read-only static file serving for ONE run's debug folder — the real
    value of remote access (see the FLUX images / hear the voiceover from
    your phone). send_from_directory guards traversal in `filename`, but
    `run_id` is OUR path segment — an audit showed run_id=".." resolved to
    media_library/ itself, serving any rendered (incl. rejected) video. The
    resolve() check pins the folder to a direct child of DEBUG_ROOT."""
    auth.require("download")
    folder = (DEBUG_ROOT / run_id).resolve()
    if folder.parent != DEBUG_ROOT.resolve() or not folder.is_dir():
        abort(404)
    small = _thumb_of(folder, filename, request.args.get("w", type=int))
    if small is not None:
        return send_from_directory(small.parent, small.name,
                                   max_age=_IMAGE_MAX_AGE)
    # The downscaled branch had a cache header and this one did not, so opening
    # a frame at full size re-downloaded it on every single look.
    return send_from_directory(folder, filename, max_age=_IMAGE_MAX_AGE)


@app.route("/video/<int:video_id>/download")
def download_video(video_id):
    """Hand the finished mp4 to the browser as a download.

    The rendered file lives in output/, outside the debug tree /debug/ serves,
    so before this there was no way to get a video off the box except copying
    it at the keyboard. as_attachment is what makes a phone offer "Save to
    Files" / camera roll instead of playing it inline and losing it.
    """
    auth.require("download")
    v = _video_detail(video_id)
    if not v:
        abort(404)
    path = Path(v["video_file"] or "")
    if not path.exists():
        abort(404)
    nice = (v["title"] or v["script_hook"] or f"rufus_{video_id}")[:60]
    safe = "".join(ch if ch.isalnum() or ch in " -_" else "_" for ch in nice).strip() or f"rufus_{video_id}"
    return send_from_directory(path.parent, path.name, as_attachment=True,
                               download_name=f"{safe}.mp4")


@app.errorhandler(401)
def unauthorized(e):
    """Stays a 401 on purpose. Browsers never reach here — _authenticate()
    redirects them to /login by Accept header — so everything that lands on
    this handler is scripted (curl, the watchdog, a phone shortcut), and for
    those a redirect masquerading as success is worse than an honest refusal.
    """
    return ("Not signed in. Open your personal sign-in link "
            "(/?token=…) or send an X-Rufus-Token header.\n"), 401


@app.errorhandler(403)
def forbidden(e):
    """An honest wall, not a mystery. Someone who followed a link their role
    can't use should be told that's what happened, not shown a blank error."""
    who = (getattr(g, "rufus_user", None) or {}).get("role", "your account")
    return _head() + (
        f"<div class='msg error'>Your role ({_esc(who)}) can't use that page. "
        f"Ask the channel owner if you need it.</div>"
        f"<p><a class='back' href='/'>← back to the queue</a></p>"
    ) + PAGE_TAIL, 403


@app.errorhandler(404)
def not_found(e):
    return _head() + "<p>Not found. <a class='back' href='/'>← back</a></p>" + PAGE_TAIL, 404


def _port_taken(host: str, port: int) -> bool:
    """Whether something already holds the port. Best-effort."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.4)
        try:
            return sock.connect_ex((host if host != "0.0.0.0" else "127.0.0.1",
                                    port)) == 0
        except OSError:
            return False


def _answers_healthz(host: str, port: int) -> bool:
    """Whether whatever holds the port is actually SERVING.

    _port_taken() answers a different question — a TCP connect succeeds against
    a listener that has stopped serving, which is exactly the process that kept
    this dashboard down for ten hours. "The port is occupied" and "a dashboard
    is running there" were treated as the same fact, and the startup message
    was written for the second one.
    """
    try:
        r = requests.get(
            f"http://{'127.0.0.1' if host in ('0.0.0.0', '') else host}:{port}"
            f"/healthz", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _venv_python() -> Path:
    name = "python.exe" if os.name == "nt" else "python"
    sub = "Scripts" if os.name == "nt" else "bin"
    return ROOT / ".venv" / sub / name


def _wrong_interpreter() -> str:
    """One line if this process is not the repo's venv python, else "".

    THE HOLE THIS CLOSES. run_dashboard.bat and run_watchdog.bat both refuse to
    fall back to the system interpreter — they exit 9009 and say why, because
    system python has no Flask and no torch. So the guard exists, and it only
    guards the two doors it is nailed to. The python that squatted on port 8765
    for ten hours was AppData/Local/Programs/Python/Python311/python.exe: the
    system one, started by something that was not a launcher, and nothing
    anywhere said a word about it.

    A NOTICE AND NOT A REFUSAL. The .bat files refuse because the venv is
    MISSING, which nothing downstream can recover from. Here it exists and a
    different interpreter was chosen, which may well be deliberate — a second
    dashboard on another port, a debugger, a developer. Refusing would break
    those; saying nothing is how this happened. RUFUS_ALLOW_ANY_PYTHON=1 turns
    it off for anyone who means it.
    """
    if os.environ.get("RUFUS_ALLOW_ANY_PYTHON", "").strip().lower() in (
            "1", "true", "yes", "on"):
        return ""
    venv = _venv_python()
    if not venv.exists():
        return ""      # no venv to be wrong about
    try:
        if Path(sys.executable).resolve() == venv.resolve():
            return ""
    except OSError:
        return ""
    return (f"running on {sys.executable}, not the repo venv at {venv} — "
            f"this interpreter may be missing packages the pipeline needs, and "
            f"no launcher started it")


# ── "I am working, not dead" ────────────────────────────────────────────────

BUSY_MARKER = paths.log_dir() / ".dashboard_busy"


@contextmanager
def _busy(what: str):
    """Declare a long operation, so the watchdog waits instead of killing us.

    THE WATCHDOG CAN NOW END A PROCESS, AND THAT IS ONLY SAFE IF IT CAN TELL
    WORKING FROM DEAD. A YouTube upload of a 25MB file answers nothing for
    minutes; from outside that is indistinguishable from a wedged process
    holding the port, and the wrong guess loses the upload. /healthz cannot
    answer during it and the lock is not visible from another process — a file
    is the only channel the dashboard and the watchdog already share, which is
    the same reasoning run_progress.py is built on.

    Best-effort in both directions: a marker that cannot be written just means
    the watchdog falls back to its other guards, and one left behind by a crash
    ages out (BUSY_MARKER_MAX_S) rather than protecting a corpse forever.
    """
    try:
        BUSY_MARKER.parent.mkdir(parents=True, exist_ok=True)
        BUSY_MARKER.write_text(f"{what}|{time.time()}", encoding="utf-8")
    except OSError:
        pass
    try:
        yield
    finally:
        try:
            BUSY_MARKER.unlink(missing_ok=True)
        except OSError:
            pass


# ── "it came back up" ────────────────────────────────────────────────────────

START_STAMP = paths.log_dir() / ".dashboard_started"


def _announce_start() -> bool:
    """Ping the owner that the dashboard is up, and how long it was not.

    THE DOWNTIME IS THE POINT, and it is the one thing the watchdog's own
    alert cannot tell you. watchdog.py already sends "service restarted" when
    /healthz stops answering, so a crash is covered. What was not covered is
    every other way this process starts — a manual serve.ps1 -Restart, a
    reboot, a task that fired at logon — and, more usefully, the difference
    between them:

        back up after 6s      you just restarted it, carry on
        back up after 4h      it was down all afternoon and nobody knew
        first start           new machine, or the stamp was cleared

    A restart the watchdog handled produces two messages, its and this one.
    That is deliberate rather than sloppy: the pair reads as "it stopped" then
    "it is back, it was gone 65 seconds", and the second half is the half that
    says whether to go and look.

    Never raises and never blocks the server coming up — a notification
    failure must not be the reason the dashboard does not start.
    """
    if os.environ.get("RUFUS_DASHBOARD_NOTIFY", "1").strip().lower() \
            in ("0", "false", "no", "off"):
        return False
    now = time.time()
    gap = None
    try:
        gap = now - float(START_STAMP.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        pass
    try:
        START_STAMP.parent.mkdir(parents=True, exist_ok=True)
        START_STAMP.write_text(str(now), encoding="utf-8")
    except OSError:
        pass

    if gap is None:
        detail = "First start on this machine, or the stamp file was cleared."
    elif gap < 0:
        # A clock that went backwards (NTP correction, a VM resuming). Saying
        # "up 0s ago" would be a lie with a straight face.
        detail = "The previous start is stamped in the future — the system "\
                 "clock moved. Downtime unknown."
    else:
        detail = f"Previous start was {_human_gap(gap)} ago."
    try:
        import notify
        return notify.send("Rufus: dashboard is up", detail)
    except Exception as e:
        print(f"[dashboard] start notification failed ({e})")
        return False


def _human_gap(seconds: float) -> str:
    s = int(seconds)
    if s < 90:
        return f"{s}s"
    if s < 5400:
        return f"{s // 60}m"
    if s < 172800:
        return f"{s // 3600}h"
    return f"{s // 86400}d"


if __name__ == "__main__":
    host = os.environ.get("RUFUS_DASHBOARD_HOST", "127.0.0.1")
    port = int(os.environ.get("RUFUS_DASHBOARD_PORT", "8765"))

    # SAY WHY, BEFORE FLASK SAYS SOMETHING ELSE. run.bat starts a dashboard of
    # its own ("Starting Rufus dashboard..."), so the ordinary way to reach
    # this line is with one already running — and what Flask prints then is
    # WinError 10048 about socket addresses, which does not tell the person
    # reading it that the thing they wanted is already open in another window.
    # run_dashboard.bat exists specifically so a startup failure leaves a
    # readable trace; an unreadable one is only half of that.
    # TWO DIFFERENT FAILURES WORE ONE MESSAGE. "The port is taken" was reported
    # as "a dashboard is already running — open it", and for ten hours that
    # advice pointed at a python that had stopped serving. The health check
    # that disproves it lives one file away and was never asked. So ask it, and
    # give the two states different words AND different exit codes — the
    # watchdog acts on the code, and it cannot act differently on the same
    # number.
    if _port_taken(host, port):
        import port_owner
        who = port_owner.holder(port)
        if _answers_healthz(host, port):
            print(f"[dashboard] port {port} is already in use, and the "
                  f"dashboard there is answering.")
            print(f"[dashboard] Open http://localhost:{port} — that IS this "
                  f"dashboard, and it picked up the latest code when it "
                  f"started.")
            print(f"[dashboard] To run a second one alongside it, set "
                  f"RUFUS_DASHBOARD_PORT to something else.")
            if who:
                print(f"[dashboard] holder: {port_owner.describe(who)}")
            sys.exit(3)
        print(f"[dashboard] port {port} is held by something that is NOT "
              f"answering — opening http://localhost:{port} will not work.")
        print(f"[dashboard] holder: {port_owner.describe(who)}")
        if port_owner.is_rufus_dashboard(who):
            print(f"[dashboard] that is a stale dashboard. End it and start "
                  f"again:")
            print(f"[dashboard]   Stop-Process -Id {who['pid']} -Force"
                  if os.name == "nt" else
                  f"[dashboard]   kill {who['pid']}")
        else:
            print(f"[dashboard] this does not look like a Rufus dashboard, so "
                  f"nothing here will end it for you. Free the port, or set "
                  f"RUFUS_DASHBOARD_PORT to something else.")
        sys.exit(4)

    wrong = _wrong_interpreter()
    if wrong:
        print(f"[dashboard] ⚠ {wrong}")

    db_manager.init_db()
    _announce_start()
    print(f"[dashboard] http://localhost:{port}  (LAN: http://<this PC's IP>:{port})")
    # THREADED, WITH THE ONE THING THAT NEEDED SERIALIZING SERIALIZED DIRECTLY.
    # This was threaded=False, and the reason given was real: approve_video
    # mutates process env through _scoped_env, and two overlapping approvals
    # could interleave RUFUS_CHANNEL and upload a video to the wrong channel.
    #
    # But that is one route. Turning off threading applied its constraint to
    # the whole application, so a page with two hundred keyframe thumbnails
    # served them strictly one at a time, the /api/status poll on every open
    # tab queued behind them, and the review queue on a phone over the tailnet
    # felt broken. _scoped_env now holds _ENV_LOCK, which says exactly what is
    # true — these env mutations must not overlap — instead of saying nothing
    # anywhere may overlap.
    app.run(host=host, port=port, debug=False, threaded=True)
