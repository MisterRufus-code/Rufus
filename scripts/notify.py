#!/usr/bin/env python3
"""
notify.py — push a notification to the channel owner's phone.

The approval queue only works if you KNOW something is waiting. A render can
finish at 06:30 while you're asleep; without a ping the video sits in the
dashboard until you happen to look, which defeats the point of scheduling runs
unattended.

Three backends, tried in the order configured. All are optional — with none
configured this module is a no-op that prints a one-line hint, so a missing
notification can never fail a render (same fail-open contract as every other
optional stage).

  ntfy (recommended — free, no account, 2-minute setup)
    1. Install the "ntfy" app (iOS/Android).
    2. Subscribe to a topic name only you know, e.g. rufus-a7f3k9x2.
       The topic IS the secret: anyone who guesses it can send you messages,
       so use something random, not "rufus".
    3. Set RUFUS_NTFY_TOPIC=rufus-a7f3k9x2
    Self-hosting? Point RUFUS_NTFY_SERVER at your own instance.

  Pushover (paid one-off, more reliable delivery, richer formatting)
    Set RUFUS_PUSHOVER_TOKEN (app token) and RUFUS_PUSHOVER_USER (user key).

  Telegram (good if you already run a bot)
    Set RUFUS_TELEGRAM_TOKEN (from @BotFather) and RUFUS_TELEGRAM_CHAT.

Environment:
  RUFUS_NOTIFY            1 (default) — 0 disables all notifications
  RUFUS_NTFY_TOPIC        ntfy topic (the shared secret — make it random)
  RUFUS_NTFY_SERVER       https://ntfy.sh (default)
  RUFUS_PUSHOVER_TOKEN / RUFUS_PUSHOVER_USER
  RUFUS_TELEGRAM_TOKEN / RUFUS_TELEGRAM_CHAT
  RUFUS_DASHBOARD_URL     link included in the notification so the phone can
                          open the review page directly, e.g.
                          http://192.168.1.20:8765 (LAN) or a Tailscale URL
"""

import os

import requests

TIMEOUT = 10


def enabled() -> bool:
    return os.environ.get("RUFUS_NOTIFY", "1").strip().lower() \
        not in ("0", "false", "no", "off")


def _dashboard_url() -> str:
    return (os.environ.get("RUFUS_DASHBOARD_URL") or "").strip().rstrip("/")


def configured() -> list[str]:
    """Which backends have credentials. [] means notifications are inert."""
    out = []
    if os.environ.get("RUFUS_NTFY_TOPIC", "").strip():
        out.append("ntfy")
    if (os.environ.get("RUFUS_PUSHOVER_TOKEN", "").strip()
            and os.environ.get("RUFUS_PUSHOVER_USER", "").strip()):
        out.append("pushover")
    if (os.environ.get("RUFUS_TELEGRAM_TOKEN", "").strip()
            and os.environ.get("RUFUS_TELEGRAM_CHAT", "").strip()):
        out.append("telegram")
    return out


def _send_ntfy(title: str, body: str, url: str, priority: str) -> bool:
    topic  = os.environ.get("RUFUS_NTFY_TOPIC", "").strip()
    server = (os.environ.get("RUFUS_NTFY_SERVER") or "https://ntfy.sh").strip().rstrip("/")
    # ntfy headers must be latin-1 safe — the title carries a video title that
    # can contain an em-dash or emoji, which would raise on encode and lose the
    # notification entirely.
    headers = {
        "Title": title.encode("ascii", "replace").decode("ascii"),
        "Priority": {"high": "high", "normal": "default"}.get(priority, "default"),
        "Tags": "clapper",
    }
    if url:
        headers["Click"] = url
    r = requests.post(f"{server}/{topic}", data=body.encode("utf-8"),
                      headers=headers, timeout=TIMEOUT)
    return r.status_code < 300


def _send_pushover(title: str, body: str, url: str, priority: str) -> bool:
    payload = {
        "token": os.environ.get("RUFUS_PUSHOVER_TOKEN", "").strip(),
        "user":  os.environ.get("RUFUS_PUSHOVER_USER", "").strip(),
        "title": title,
        "message": body,
        "priority": 1 if priority == "high" else 0,
    }
    if url:
        payload["url"] = url
        payload["url_title"] = "Open the review queue"
    r = requests.post("https://api.pushover.net/1/messages.json",
                      data=payload, timeout=TIMEOUT)
    return r.status_code < 300


def _send_telegram(title: str, body: str, url: str, priority: str) -> bool:
    token = os.environ.get("RUFUS_TELEGRAM_TOKEN", "").strip()
    chat  = os.environ.get("RUFUS_TELEGRAM_CHAT", "").strip()
    text  = f"*{title}*\n{body}" + (f"\n{url}" if url else "")
    r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      data={"chat_id": chat, "text": text,
                            "parse_mode": "Markdown",
                            "disable_web_page_preview": True},
                      timeout=TIMEOUT)
    return r.status_code < 300


_BACKENDS = {"ntfy": _send_ntfy, "pushover": _send_pushover, "telegram": _send_telegram}


def send(title: str, body: str, *, url: str | None = None,
         priority: str = "normal") -> bool:
    """Push to every configured backend. True if at least one delivered.

    Never raises — a notification failure must not break a render. Returns
    False (with a printed reason) when disabled, unconfigured, or all sends
    failed, so the caller can log it without special-casing."""
    if not enabled():
        return False
    backends = configured()
    if not backends:
        print("[notify] no backend configured — set RUFUS_NTFY_TOPIC (free, "
              "no account: install the ntfy app, subscribe to a random topic) "
              "to get a phone ping when a video needs approval")
        return False

    link = (url or "").strip() or _dashboard_url()
    delivered = []
    for name in backends:
        try:
            if _BACKENDS[name](title, body, link, priority):
                delivered.append(name)
            else:
                print(f"[notify] {name} rejected the message")
        except Exception as e:
            print(f"[notify] {name} failed ({e})")
    if delivered:
        print(f"[notify] sent via {', '.join(delivered)}")
        return True
    return False


def notify_pending_review(*, title: str, score, niche: str,
                          video_id=None, hold_reason: str | None = None) -> bool:
    """The one that matters: a rendered video is waiting for a human.

    Deep-links straight to that video's page when RUFUS_DASHBOARD_URL is set,
    so approving from a phone is two taps rather than hunting for the row."""
    link = _dashboard_url()
    if link and video_id is not None:
        link = f"{link}/video/{video_id}"
    lines = [f"{niche} · scored {score}/10"]
    if hold_reason:
        lines.append(f"auto-gate note: {hold_reason}")
    lines.append("Approve or reject in the dashboard.")
    return send(f"Rufus: \"{title}\" needs review",
                "\n".join(lines), url=link, priority="high")


def notify_run_failed(reason: str, *, niche: str | None = None,
                      channel: str | None = None) -> bool:
    """A run crashed before reaching the DB save — the orphaned debug folder
    that leaves behind is invisible until someone happens to open /failures.
    This is the only way the owner learns about it in real time, for ANY
    entry point (run_scheduled.bat already alerts on crash for scheduled
    runs specifically, via a hardcoded ntfy curl — this covers every other
    way main.py gets invoked, and every backend notify.py supports)."""
    lines = []
    if niche or channel:
        lines.append(f"{niche or '?'}" + (f" · {channel}" if channel else ""))
    lines.append(reason[-400:])   # truncate — backends cap message length
    link = _dashboard_url()
    if link:
        link = f"{link}/failures"
    return send("Rufus: run CRASHED", "\n".join(lines), url=link, priority="high")
