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
import subprocess
import sys
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
    return PAGE_STYLE + '<header><h1>🎬 Rufus Dashboard</h1></header>\n<main>\n' \
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
        "busy": any(r["running"] for r in runs),
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
         "select:stickman,ink_explainer,flat_vector,ink_woodcut,paper_cut,"
         "chalkboard,retro_print,storybook",
         "Named look from config/styles.json, appended to every image prompt "
         "byte for byte. Leave at (default) to use the niche's own style_suffix. "
         "Render one of each on the Style page before choosing."),
        ("RUFUS_STILLS_DETAIL", "Style override (literal)", "text",
         "A style block written out in full. Beats the preset above — for a "
         "one-off experiment that does not deserve a name yet."),
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
         "Optional. OpenAlex's \"polite pool\" gives higher rate limits to "
         "requests carrying a contact address. Nothing breaks without it."),
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
                channel: str | None = None) -> tuple[subprocess.Popen, Path]:
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
    env = os.environ.copy()
    env.update(_load_settings())
    # THE CHILD'S STDOUT IS A FILE, so Python has no console to ask and falls
    # back to the system ANSI code page — cp1255 here, which has no ✗ and no
    # em-dash. A real run died mid-report on exactly that. The .bat launchers
    # set these; a dashboard-launched run inherits whatever started the
    # dashboard, which is not guaranteed to be one of them.
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"dashboard_run_{int(time.time())}.log"
    with open(log_path, "wb") as logf:
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=logf, env=env,
                               stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    _LAUNCHED[channel or "default"] = proc
    return proc, log_path


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
         "created_at, uploaded_at FROM videos")
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
            "created_at", "uploaded_at"]
    return [dict(zip(cols, r)) for r in rows]


def _video_detail(video_id: int) -> dict | None:
    q = ("SELECT id, upload_date, niche, script_hook, script_full, scene_desc, "
         "seed_type, seed_source, seed_content, seed_url, youtube_id, video_file, score, "
         "run_id, score_specificity, score_hook, score_compression, score_loop, "
         "score_human, attempts_used, final_temperature, score_reasoning, "
         "title, channel, hold_reason, description, upload_status "
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
            "description", "upload_status"]
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
<title>Rufus Dashboard</title>
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
    --radius:  10px;
    --shadow:  0 1px 2px rgba(0,0,0,.28);
  }
  @media (prefers-color-scheme: light) {
    :root {
      --bg: #f6f7f9; --surface: #ffffff; --raised: #ffffff;
      --border: #e3e6ea; --text: #14171c; --dim: #5f6672;
      --shadow: 0 1px 2px rgba(16,24,40,.06), 0 1px 3px rgba(16,24,40,.08);
    }
  }

  * { box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
         Helvetica, Arial, sans-serif; margin: 0; background: var(--bg);
         color: var(--text); -webkit-font-smoothing: antialiased; }
  a { color: var(--accent); }
  :focus-visible { outline: 2px solid var(--accent); outline-offset: 2px;
                   border-radius: 4px; }

  header { position: sticky; top: 0; z-index: 20; padding: 12px 24px;
           background: color-mix(in srgb, var(--bg) 88%, transparent);
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
        border-radius: var(--radius); color: var(--text); }

  .msg { padding: 11px 14px; border-radius: 8px; margin-bottom: 14px;
         font-size: 14px; border: 1px solid transparent; }
  .msg.ok    { background: color-mix(in srgb, var(--ok) 12%, transparent);
               border-color: color-mix(in srgb, var(--ok) 30%, transparent); color: var(--ok); }
  .msg.error { background: color-mix(in srgb, var(--bad) 12%, transparent);
               border-color: color-mix(in srgb, var(--bad) 30%, transparent); color: var(--bad); }

  .actions { margin: 16px 0; display: flex; gap: 10px; flex-wrap: wrap; }
  .btn { border: 1px solid var(--border); background: var(--raised);
         color: var(--text); border-radius: 8px; padding: 10px 18px;
         font-size: 14px; font-weight: 600; cursor: pointer;
         transition: transform .06s ease, filter .12s ease; }
  .btn:hover { filter: brightness(1.08); }
  .btn:active { transform: translateY(1px); }
  .btn.approve { background: var(--ok);     color: #06210f; border-color: transparent; }
  .btn.reject  { background: var(--bad);    color: #2a0a0a; border-color: transparent; }
  .btn.save    { background: var(--accent); color: #06122a; border-color: transparent; }

  .field { display: block; width: 100%; margin: 6px 0 14px; padding: 9px 11px;
           border-radius: 8px; border: 1px solid var(--border);
           background: var(--bg); color: inherit; font-family: inherit;
           font-size: 14px; }
  select { padding: 8px 10px; border-radius: 8px; border: 1px solid var(--border);
           background: var(--bg); color: inherit; font: inherit; }
  label { font-size: 11px; color: var(--dim); text-transform: uppercase;
          letter-spacing: 0.06em; }

  .filters { margin: 12px 0; }
  .filters a { margin-right: 10px; font-size: 13px; text-decoration: none; }
  .back { text-decoration: none; font-size: 14px; }
  .script { white-space: pre-wrap; font-size: 15px; line-height: 1.6;
            background: var(--surface); border: 1px solid var(--border);
            border-radius: var(--radius); padding: 14px; }
  .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  @media (max-width: 760px) { .grid2 { grid-template-columns: 1fr; } }
  .assets a { display: inline-block; margin: 4px 8px 4px 0; font-size: 13px;
              text-decoration: none; }

  .navlink { color: var(--dim); text-decoration: none; font-size: 14px;
             margin-left: 14px; padding: 5px 2px; border-bottom: 2px solid transparent; }
  .navlink:hover { color: var(--accent); border-bottom-color: var(--accent); }

  .orphan { background: var(--surface); border: 1px solid var(--border);
            border-radius: 8px; padding: 12px 14px; margin-bottom: 10px; }

  /* Live status bar — polls /api/status, no page reload */
  #livebar { display: flex; gap: 14px; flex-wrap: wrap; align-items: center;
             background: var(--surface); border: 1px solid var(--border);
             border-radius: var(--radius); padding: 10px 14px;
             margin-bottom: 16px; font-size: 13px; box-shadow: var(--shadow); }
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
               gap: 14px; margin-top: 10px; }
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
    .whoami { margin-left: 0; display: block; margin-top: 8px; }
    /* Tap targets. The review queue is worked from a phone. */
    .btn { padding: 12px 20px; }
    th, td { padding: 12px 10px; }
  }
  .fmt-switch { display:inline-flex; gap:0; margin-left:10px; vertical-align:middle;
                border:1px solid var(--line); border-radius:8px; overflow:hidden }
  .fmt-switch form { margin:0 }
  button.fmt { border:0; background:var(--card); color:var(--muted);
               padding:5px 11px; font-size:13px; cursor:pointer; font-family:inherit }
  button.fmt:hover { color:var(--fg) }
  button.fmt.on { background:var(--accent); color:#fff; font-weight:600 }
  .fmt-badge { margin-left:10px; color:var(--muted); font-size:13px }
  .style-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(230px,1fr));
                gap:14px; margin-top:16px }
  .style-card { border:1px solid var(--line); border-radius:12px; overflow:hidden;
                background:var(--card); display:flex; flex-direction:column }
  .style-card.on { border-color:var(--accent); box-shadow:0 0 0 2px var(--accent) }
  .style-card img { width:100%; aspect-ratio:16/9; object-fit:cover; display:block;
                    background:var(--bg) }
  .style-card .noimg { width:100%; aspect-ratio:16/9; display:flex; align-items:center;
                       justify-content:center; color:var(--muted); font-size:13px;
                       background:var(--bg); text-align:center; padding:10px }
  .style-card .body { padding:10px 12px; display:flex; flex-direction:column; gap:8px;
                      flex:1 }
  .style-card h4 { margin:0; font-size:15px }
  .style-card p { margin:0; font-size:12.5px; color:var(--muted); line-height:1.45;
                  flex:1 }
  .style-card .row { display:flex; gap:8px }
</style></head><body>
"""

# Nav entries gated by permission — a partner never sees Settings or System,
# because a link they can only get a 403 from is worse than no link at all.
NAV_ITEMS = [
    ("/generate",   "▶ Make a video",                     "generate"),
    ("/thumbnails", "🎨 Thumbnails",                      "thumbnail"),
    ("/styles",     "🎨 Style",                           "settings"),
    ("/scout",      "🛰 Scout",                           "view"),
    ("/bench",      "🔬 Workflow bench",                  "settings"),
    ("/failures",   "⚠ Failures &amp; rejected attempts", "view"),
    ("/performance", "📈 Performance",                    "view"),
    ("/trending",   "🔥 Trending",                        "view"),
    ("/gallery",    "🖼 Gallery",                         "view"),
    ("/advice",     "💡 What to change",                  "view"),
    ("/tracking",   "📊 Tracking",                        "view"),
    ("/history",    "🕰 History",                         "view"),
    ("/insights",   "🔬 Insights",                        "view"),
    ("/logs",       "📜 Logs",                            "view"),
    ("/system",     "🖥 System",                          "system"),
    ("/settings",   "⚙ Settings",                         "settings"),
]


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
    if (!active.length) {
      bits.push('<span class="item"><span class="dot on"></span>Idle \\u2014 not making a video</span>');
    } else {
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
    bits.push('<span class="item"><a class="navlink" style="margin:0" href="/">'
              + (q.pending || 0) + ' awaiting review</a></span>');
    el.innerHTML = bits.join('');
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
    links = "".join(f'<a class="navlink" href="{href}">{label}</a>\n'
                    for href, label, perm in NAV_ITEMS if auth.can(perm))
    user = getattr(g, "rufus_user", None) or {}
    who = ""
    if user:
        who = (f'<span class="whoami">{_esc(user.get("name", "?"))}'
               f'<span class="role">{_esc(user.get("role", "?"))}</span>'
               f' · <a class="navlink" href="/logout" style="margin-left:6px">sign out</a></span>')
    return (PAGE_STYLE + '<header><a href="/"><h1>🎬 Rufus Dashboard</h1></a>\n'
            + links + _format_switch() + who + "</header>\n<main>\n"
            + '<div id="livebar"><span class="item">'
              '<span class="dot warn"></span>checking…</span></div>\n'
            + LIVEBAR_JS)


PAGE_TAIL = "</main></body></html>"


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
                preview_cell = (f'<td><a class="row-link" href="/video/{v["id"]}" '
                                f'style="display:flex">{imgs}</a></td>')
            else:
                preview_cell = '<td><span class="muted">—</span></td>'
        went_out = (f'<br><span class="muted">out '
                    f'{_esc(str(v["uploaded_at"]).split(" ")[-1][:5])}</span>'
                    if v.get("uploaded_at") and " " in str(v["uploaded_at"])
                    else "")
        rows += (f'<tr>{preview_cell}<td><a class="row-link" href="/video/{v["id"]}">'
                 f'{_when_cell(v.get("created_at") or v.get("upload_date"))}'
                 f'{went_out}</a></td>'
                 f'<td><a class="row-link" href="/video/{v["id"]}">{_esc(v["niche"])}</a></td>'
                 f'<td><a class="row-link" href="/video/{v["id"]}">{title}</a></td>'
                 f'<td>{score_html}</td><td>{_status_badge(v["upload_status"])}</td></tr>\n')
    preview_th = "<th>Preview</th>" if previews else ""
    return (f"<table><tr>{preview_th}<th>Made</th><th>Niche</th><th>Hook / Title</th>"
            f"<th>Score</th><th>Status</th></tr>{rows}</table>")


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


@app.route("/")
def index():
    channel = request.args.get("channel") or None
    stats   = _stats(channel=channel)
    videos  = _recent_videos(limit=60, channel=channel)
    pending = _recent_videos(limit=60, channel=channel, status="pending")
    rejects = _top_rejections(channel=channel)
    channels = _channels()

    # oldest -> newest for the trend line
    scored = [v["score"] for v in reversed(videos) if v["score"] is not None]

    filt_html = ""
    if channels:
        links = [f'<a href="/">all channels</a>']
        for ch in channels:
            links.append(f'<a href="/?channel={_esc(ch)}">{_esc(ch)}</a>')
        filt_html = f'<div class="filters">{"".join(links)}</div>'

    channel_options = "".join(f'<option value="{_esc(ch)}">{_esc(ch)}</option>' for ch in channels)
    topic_form = f"""
    <h2>🎯 Make a video about a specific topic</h2>
    <p class="muted">Runs in the background (can take a while) — resolved to a
       real Wikipedia article so it's still fact-grounded, then shows up in
       the pending list below like any other video. Never auto-uploads.</p>
    <form method="post" action="/request-topic" style="display:flex;gap:10px;flex-wrap:wrap;align-items:flex-end;margin-bottom:24px">
      <div style="flex:1;min-width:220px">
        <label for="topic">Topic</label>
        <input class="field" style="margin:6px 0 0" type="text" id="topic" name="topic"
               placeholder="e.g. Bretton Woods, Tulip mania..." required>
      </div>
      {f'''<div><label for="channel">Channel</label>
        <select class="field" style="margin:6px 0 0" id="channel" name="channel">
          <option value="">(default)</option>{channel_options}
        </select></div>''' if channels else ""}
      <button class="btn save" type="submit" style="height:38px">Queue it</button>
    </form>
    """

    cards = f"""
    <div class="cards">
      <div class="card"><div class="num">{stats['pending']}</div><div class="label">awaiting review</div></div>
      <div class="card"><div class="num">{stats['uploaded']}</div><div class="label">approved / uploaded</div></div>
      <div class="card"><div class="num">{stats['rejected']}</div><div class="label">rejected</div></div>
      <div class="card"><div class="num">{stats['avg_score']}</div><div class="label">avg score</div></div>
    </div>
    """

    reject_html = ""
    if rejects:
        items = "".join(f"<li>{_esc(r['reason'])} — <b>{r['count']}×</b></li>" for r in rejects)
        reject_html = f"<ul>{items}</ul>"
    else:
        reject_html = "<p class='muted'>No rejected attempts recorded yet.</p>"

    # THE ONE THING TO DO NEXT, above everything else on the page. The front
    # page opened on a topic box, which assumes the answer to "what now" is
    # always "make another video" — and when four of the last six runs share a
    # defect, another video is precisely the wrong move. This is the top
    # finding from /advice, in a line, with a way through to it.
    advice_html = ""
    try:
        items, ready = _advice_now()
        tone = {"needs work": "held", "workable": "pending",
                "good": "ok", "unmeasured": "pending"}.get(ready["state"], "pending")
        top = (f' — <strong>{_esc(items[0]["title"])}</strong>' if items else "")
        more = (f' <span class="muted">and {len(items) - 1} more</span>'
                if len(items) > 1 else "")
        advice_html = (
            f'<div class="card" style="width:100%;margin-bottom:18px">'
            f'<span class="badge {tone}">{_esc(ready["state"])}</span>{top}{more}'
            f'<div class="muted" style="margin-top:6px">'
            f'<a href="/advice">what to change →</a> · '
            f'<a href="/insights">the measurements →</a></div></div>')
    except Exception as e:                       # never break the front page
        print(f"[dashboard] advice summary unavailable: {e}")

    body = f"""
    {_msg_banner()}
    {advice_html}
    {topic_form}
    {filt_html}
    {cards}
    <h2>⏳ Awaiting your review ({len(pending)})</h2>
    {_videos_table(pending, previews=True)}
    <h2>Score trend (oldest → newest)</h2>
    {_sparkline_svg(scored)}
    <div class="grid2">
      <div>
        <h2>All recent videos</h2>
        {_videos_table(videos)}
      </div>
      <div>
        <h2>Most common script rejections</h2>
        {reject_html}
      </div>
    </div>
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


@app.route("/performance")
def performance():
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
    return _head() + body + PAGE_TAIL


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
        when = time.strftime("%d %b %H:%M", time.localtime(img["mtime"]))
        cards += (
            f'<div class="thumbcard">'
            f'<a href="/thumbnails/file/{name}" target="_blank">'
            f'<img src="/thumbnails/file/{name}" loading="lazy" alt=""></a>'
            f'<div class="meta">{_esc(img["prompt"][:90] or img["name"])}<br>'
            f'<a href="/thumbnails/file/{name}?download=1">⬇ Save to phone</a>'
            f' · {img["kb"]}KB · {when}{make_btn}{del_btn}</div></div>')
    gallery = (f'<div class="thumbgrid">{cards}</div>' if cards else
               "<p class='muted'>Nothing generated yet.</p>")

    body = f"""
    <a class="back" href="/">← back</a>
    <h2 style="margin-top:14px">Generate a thumbnail</h2>
    {warn}
    {_msg_banner()}
    <p class="muted">Renders on the owner's RTX 3090 through the same image
       model the videos use. Takes a few seconds — the page waits for it.
       1280×720 is YouTube's thumbnail shape; the other option matches the
       frame the next run renders at.</p>
    <form method="post" action="/thumbnails/generate">
      <label for="tp">Describe the image</label>
      <input class="field" type="text" id="tp" name="prompt" required
             placeholder="a cracked hourglass spilling gold coins across a desk">
      <label for="tshape">Shape</label>
      <select class="field" id="tshape" name="shape">
        <option value="landscape">Landscape {image_gen.THUMB_W}×{image_gen.THUMB_H} (YouTube thumbnail)</option>
        <option value="frame">{image_gen.FRAME_W}×{image_gen.FRAME_H} (video frame)</option>
      </select>
      <button class="btn save" type="submit">Generate</button>
    </form>
    <h2>Generated</h2>
    {gallery}
    """
    return _head() + body + PAGE_TAIL


@app.route("/thumbnails/generate", methods=["POST"])
def thumbnails_generate():
    auth.require("thumbnail")
    import image_gen

    prompt = request.form.get("prompt", "").strip()
    if not prompt:
        return redirect("/thumbnails?error=Describe+the+image+first")
    # A video run owns the GPU for its whole duration, so a thumbnail asked
    # for now would sit in ComfyUI's queue behind it — and because this render
    # is inline on a threaded=False server, that wait freezes the dashboard for
    # everyone. Refuse immediately instead, with the reason.
    busy = [c for c in _channels() if _run_in_progress(c)]
    if busy:
        return redirect("/thumbnails?error=" + _urlquote(
            f"A video run is using the GPU ({', '.join(busy)}). "
            f"Thumbnails have to wait for it to finish — try again shortly."))

    # "frame" is the video's own shape, whatever format the next run is.
    # "portrait" is the value the form used to send and older bookmarks and
    # any open tab still do, so it keeps meaning the same thing.
    want_frame = request.form.get("shape") in ("frame", "portrait")
    w, h = ((image_gen.FRAME_W, image_gen.FRAME_H) if want_frame
            else (image_gen.THUMB_W, image_gen.THUMB_H))
    try:
        path = image_gen.generate_image(prompt, width=w, height=h,
                                        timeout=image_gen.WEB_TIMEOUT)
    except Exception as e:
        return redirect(f"/thumbnails?error={_urlquote(f'Generation failed: {e}')}")
    if path is None:
        return redirect("/thumbnails?error=" + _urlquote(
            "Generation failed — check ComfyUI is running and a stills "
            "workflow is exported to config/stills_api.json"))

    # Mirror it into Discord so the team sees new art without opening the
    # tailnet — best effort, never turns a good render into an error.
    try:
        import notify
        notify.send_file(path, caption=f"🎨 New thumbnail: {prompt[:180]}")
    except Exception:
        pass
    return redirect("/thumbnails?ok=" + _urlquote(f"Generated {path.name}"))


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
    actually getting it into the camera roll."""
    auth.require("download")
    folder = paths.thumbnails_dir().resolve()
    if not folder.is_dir():
        abort(404)
    return send_from_directory(folder, filename,
                               as_attachment=request.args.get("download") == "1")


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
        rows += (
            f'<tr><td>{name}</td><td>{_esc(u.get("role",""))}</td><td>{via}</td>'
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
                f'type="submit">Make this</button></form> '
                f'<form method="post" action="/scout/{p["id"]}/reject" '
                f'style="display:inline"><button type="submit">Not this'
                f'</button></form>')
        cards += (
            f'<div class="card" style="width:100%;margin-bottom:12px">'
            f'<div style="display:flex;justify-content:space-between;gap:12px">'
            f'<strong>{_esc(p["topic"] or "—")}</strong>'
            f'<span class="muted">{p["score"]}/10 · ${p["cost_usd"]:.3f}</span>'
            f'</div>'
            f'<p class="muted" style="margin:6px 0">{_esc(p["evidence"] or "")}</p>'
            f'<details><summary class="muted">the script it wrote</summary>'
            f'<pre style="white-space:pre-wrap;font-size:13px">'
            f'{_esc(p["script"] or "")}</pre></details>'
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
    {_msg_banner()}
    <p class="muted">Watches the channels in <code>config/competitors.json</code>,
       scores every video against <em>its own channel's median</em> — 20k views
       on a channel that averages 3k is the interesting one, not 50k on a
       channel that averages 200k — and proposes what to make. It writes
       scripts and stops there; approving one starts a normal run.</p>
    {cards}
    {seen_html}
    {old_html}
    """
    return _head() + body + PAGE_TAIL


@app.route("/scout/<int:proposal_id>/approve", methods=["POST"])
def scout_approve(proposal_id: int):
    """Approve a proposal → an ordinary run on its topic.

    Reuses the same launch path as /request-topic rather than growing a second
    one: a scout-approved video must go through every gate any other video goes
    through, and the surest way to guarantee that is for it to BE the same
    path.
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
        _, log_path = _launch_run(topic=row["topic"], channel=row["channel"])
    except Exception as e:
        return redirect("/scout?error=" + _urlquote(f"could not start: {e}"))
    return redirect("/scout?msg=" + _urlquote(
        f'Making "{row["topic"]}" — it lands in the review queue like any '
        f'other video. Log: logs/{log_path.name}'))


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
    if not db_manager.mark_published(video_id, raw):
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
    log_dir = ROOT / "logs"
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


@app.route("/tracking")
def tracking_page():
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
        return _head() + body + PAGE_TAIL

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
    return _head() + body + PAGE_TAIL


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


@app.route("/advice")
def advice_page():
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
        return _head() + body + PAGE_TAIL

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
    return _head() + body + PAGE_TAIL


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


@app.route("/insights")
def insights_page():
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
        return _head() + body + PAGE_TAIL

    rows = data.get("rows", [])
    if not rows:
        body = f"""
        <a class="back" href="/">← back</a>
        <h2 style="margin-top:14px">Insights</h2>
        <p class="muted">No runs measured yet. Every finished run writes its
           own review from here on; to measure the ones already on disk, run
           <code>python scripts/run_review.py --all</code> once.</p>
        """
        return _head() + body + PAGE_TAIL

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
    return _head() + body + PAGE_TAIL


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
    for d in (ROOT / "logs",):
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
    d = (ROOT / "logs").resolve()
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


@contextmanager
def _scoped_env(**overrides):
    """Set env vars for the duration of the block, then restore exactly what
    was there before (including "unset" if the key didn't exist). SAFE ONLY
    because app.run() below passes threaded=False — Flask 3.x defaults the
    dev server to threaded=True, under which two overlapping requests would
    interleave these mutations (audit finding: wrong-channel upload). Also,
    mutating os.environ permanently — as a naive assignment would — leaks
    across every later request in this long-lived process (confirmed live:
    it leaked into an unrelated test suite run in the same process)."""
    prev = {k: os.environ.get(k) for k in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


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
        buttons += (f'<form method="post" action="/video/{v["id"]}/approve" '
                    f'onsubmit="return confirm(\'Upload this video to YouTube now?\');">'
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

    body = f"""
    <a class="back" href="/">← back</a>
    <h2 style="margin-top:14px">#{v['id']} · {_esc(v['upload_date'])} · {_esc(v['niche'])} · {_esc(v['channel'])}</h2>
    {msg_html}
    <p>{status_html}</p>
    <p><b>Score: <span style="color:{_score_color(v['score'])}">{v['score']}/10</span></b>
       ({v['attempts_used'] or '?'} attempts, temp {v['final_temperature'] or '?'})</p>
    <table style="max-width:320px">{crit_rows}</table>
    {actions_html}
    {published_html}
    {edit_html}
    {_preview_block(v['run_id'])}
    <h2>Script</h2>
    <div class="script">{_esc(v['script_full'] or v['script_hook'])}</div>
    <h2>Why this score (critic reasoning)</h2>
    <div class="script">{_esc(v['score_reasoning'] or '—')}</div>
    <h2>Seed / source</h2>
    <p class="muted">{_esc(v['seed_type'])} · {_esc(v['seed_source'])}</p>
    <div class="script">{_esc(v['seed_content'] or '—')}</div>
    <h2>Image prompts (what each beat was told to draw)</h2>
    {prompts_html}
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

        with _scoped_env(**env_overrides):
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
        db_manager.set_upload_status(video_id, "approved")
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
    db_manager.set_upload_status(video_id, new_status)
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


def _preview_block(run_id: str) -> str:
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
                                   max_age=60 * 60 * 24 * 30)
    return send_from_directory(folder, filename)


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
    if _port_taken(host, port):
        print(f"[dashboard] port {port} is already in use — a dashboard is "
              f"almost certainly running already.")
        print(f"[dashboard] Open http://localhost:{port} — that IS this "
              f"dashboard, and it picked up the latest code when it started.")
        print(f"[dashboard] If it is stale, close that window (or end the "
              f"python.exe running dashboard.py) and start this again. To run "
              f"a second one alongside it, set RUFUS_DASHBOARD_PORT to "
              f"something else.")
        sys.exit(3)

    db_manager.init_db()
    _announce_start()
    print(f"[dashboard] http://localhost:{port}  (LAN: http://<this PC's IP>:{port})")
    # threaded=False is LOAD-BEARING: approve_video mutates process env via
    # _scoped_env, which is only safe when requests are serialized. Flask 3.x
    # app.run() defaults threaded=True — two overlapping approvals could
    # interleave RUFUS_CHANNEL mutations and upload a video to the WRONG
    # channel. Single-threaded is fine for a 1-2 person review tool.
    app.run(host=host, port=port, debug=False, threaded=False)
