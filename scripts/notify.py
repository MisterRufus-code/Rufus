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

  Discord (best when more than one person follows the channel's output)
    1. Server Settings → Integrations → Webhooks → New Webhook, pick a channel.
    2. Copy Webhook URL → RUFUS_DISCORD_WEBHOOK
    Unlike the phone-push backends, Discord can carry the ARTIFACT itself:
    send_file() posts the finished mp4 or a generated thumbnail into the
    channel, so the media shows up where the team already talks instead of
    only a link that needs the tailnet to open.

Environment:
  RUFUS_NOTIFY            1 (default) — 0 disables all notifications
  RUFUS_NTFY_TOPIC        ntfy topic (the shared secret — make it random)
  RUFUS_NTFY_SERVER       https://ntfy.sh (default)
  RUFUS_PUSHOVER_TOKEN / RUFUS_PUSHOVER_USER
  RUFUS_TELEGRAM_TOKEN / RUFUS_TELEGRAM_CHAT
  RUFUS_DISCORD_WEBHOOK   full webhook URL
  RUFUS_DISCORD_UPLOAD    1 (default) — 0 to post links only, never the file
  RUFUS_DASHBOARD_URL     link included in the notification so the phone can
                          open the review page directly, e.g.
                          http://192.168.1.20:8765 (LAN) or a Tailscale URL
"""

import os
from pathlib import Path

import requests

TIMEOUT = 10

# Discord rejects an upload over the server's tier limit with a 413 AFTER the
# whole body has been sent. A 40MB Short over a slow uplink would spend a
# minute uploading only to be refused, so anything above the free-tier ceiling
# is linked rather than attached.
DISCORD_MAX_UPLOAD_BYTES = 8 * 1024 * 1024


def enabled() -> bool:
    return os.environ.get("RUFUS_NOTIFY", "1").strip().lower() \
        not in ("0", "false", "no", "off")


def _dashboard_url() -> str:
    """RUFUS_DASHBOARD_URL, falling back to config/dashboard_url.txt — the
    same resolution auth.py uses, so a Discord/ntfy link and a sign-in link
    always point at the same place. See auth._base_url() for why the file
    exists (a `setx` env var doesn't reach a PowerShell window opened before
    it was set; a file does)."""
    env = (os.environ.get("RUFUS_DASHBOARD_URL") or "").strip().rstrip("/")
    if env:
        return env
    try:
        from pathlib import Path
        url_file = Path(__file__).parent.parent / "config" / "dashboard_url.txt"
        return url_file.read_text(encoding="utf-8").strip().rstrip("/")
    except OSError:
        return ""


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
    if _discord_webhook():
        out.append("discord")
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


def _discord_webhook() -> str:
    return (os.environ.get("RUFUS_DISCORD_WEBHOOK") or "").strip()


def _discord_color(priority: str) -> int:
    # Red for the things that need someone now, blue for everything else.
    return 0xEF4444 if priority == "high" else 0x3B82F6


def _send_discord(title: str, body: str, url: str, priority: str) -> bool:
    """Post an embed to the configured webhook.

    An embed rather than plain content so the title, the body and the
    dashboard link stay visually distinct in a busy channel — and so a run
    failure reads as an alert instead of another line of chat.
    """
    hook = _discord_webhook()
    if not hook:
        return False
    embed = {
        "title": title[:256],           # Discord truncates past this and 400s past 6000 total
        "description": body[:4000],
        "color": _discord_color(priority),
    }
    if url:
        embed["url"] = url
        embed["fields"] = [{"name": "Dashboard", "value": url, "inline": False}]
    r = requests.post(hook, json={"embeds": [embed]}, timeout=TIMEOUT)
    return r.status_code < 300


def send_file(path, *, caption: str = "") -> bool:
    """Post an actual file (mp4/png) into the Discord channel. Discord only.

    The phone-push backends can carry a link but not a payload, so this is
    deliberately not part of send()'s fan-out: it's the one thing Discord adds
    that the others structurally cannot. Returns False (never raises) when
    Discord isn't configured, uploads are disabled, the file is missing, or
    it's too big to attach — the caller has already done its real work and a
    failed upload must not turn a good render into an error.
    """
    hook = _discord_webhook()
    if not hook or not enabled():
        return False
    if os.environ.get("RUFUS_DISCORD_UPLOAD", "1").strip().lower() in ("0", "false", "no", "off"):
        return False

    p = Path(path)
    if not p.exists():
        print(f"[notify] discord upload skipped — {p} not found")
        return False

    size = p.stat().st_size
    if size > DISCORD_MAX_UPLOAD_BYTES:
        # Say so rather than failing silently: "why didn't the video appear in
        # Discord" is otherwise a mystery with no trace anywhere.
        print(f"[notify] {p.name} is {size//1024//1024}MB — over Discord's "
              f"{DISCORD_MAX_UPLOAD_BYTES//1024//1024}MB attach limit, posting a link instead")
        return _send_discord(caption or p.name,
                             f"File too large to attach ({size//1024//1024}MB).",
                             _dashboard_url(), "normal")
    try:
        with p.open("rb") as fh:
            r = requests.post(hook,
                              data={"content": caption[:2000]} if caption else None,
                              files={"file": (p.name, fh)}, timeout=120)
        if r.status_code < 300:
            print(f"[notify] posted {p.name} to Discord")
            return True
        print(f"[notify] discord upload rejected (HTTP {r.status_code})")
    except Exception as e:
        print(f"[notify] discord upload failed ({e})")
    return False


_BACKENDS = {"ntfy": _send_ntfy, "pushover": _send_pushover,
             "telegram": _send_telegram, "discord": _send_discord}


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
              "for a phone ping, and/or RUFUS_DISCORD_WEBHOOK to post runs "
              "into a Discord channel")
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


def notify_published(*, title: str, youtube_id: str | None = None,
                     score=None, niche: str | None = None,
                     video_path=None, by: str | None = None) -> bool:
    """A video actually went live — the event the owner most wants mirrored to
    the team, and the one nothing announced before.

    Posts the mp4 into Discord when it fits, so the channel shows what shipped
    rather than just claiming something did."""
    bits = []
    if niche:
        bits.append(niche)
    if score is not None:
        bits.append(f"scored {score}/10")
    if by:
        bits.append(f"approved by {by}")
    link = (f"https://youtu.be/{youtube_id}" if youtube_id else _dashboard_url())
    ok = send(f"Rufus: published \"{title}\"", " · ".join(bits) or "Uploaded.",
              url=link, priority="normal")
    if video_path:
        send_file(video_path, caption=f"**{title}** — {link}" if link else f"**{title}**")
    return ok


def notify_analytics(summary: str, *, rows: int = 0) -> bool:
    """Daily performance digest. Low priority by design — it's a report, not
    an alert, and waking a phone at 10:00 every morning for a number that
    didn't move is how people mute a notification channel entirely."""
    link = _dashboard_url()
    if link:
        link = f"{link}/performance"
    return send(f"Rufus: analytics ({rows} video{'s' if rows != 1 else ''})",
                summary, url=link, priority="normal")


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
