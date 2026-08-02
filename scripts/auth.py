#!/usr/bin/env python3
"""
auth.py — token authentication and roles for the Rufus dashboard.

WHY THIS EXISTS
The dashboard's original guard was `request.remote_addr in ("127.0.0.1","::1")`
— "loopback means it's me at the keyboard." That holds for a PC nobody else can
reach, and it is exactly wrong the moment you run `tailscale serve`: Tailscale
terminates the connection and proxies to 127.0.0.1, so EVERY tailnet visitor
arrives as loopback and passes that check. Sharing the tailnet URL with a
partner would hand them process control (/system) and YouTube publish rights
(/video/<id>/approve) on your channel.

So access is decided by WHO is asking (a token they hold), not WHERE the packet
appears to come from.

ROLES
  owner    — everything: approve/upload, settings, system control, delete.
  partner  — generate videos and thumbnails, view runs, download media.
             Cannot approve/upload, cannot change settings, cannot start or
             kill processes beyond the runs they launch themselves.
  viewer   — read-only: see the queue, watch videos, download. Generates nothing.

BACKWARD COMPATIBILITY
No config/users.json → legacy mode: loopback requests are treated as owner and
non-loopback is refused, i.e. exactly today's behavior. Nothing breaks for a
single-user setup that never creates the file. Creating the file switches the
dashboard into authenticated mode for everyone, loopback included.

SETUP
    python scripts/auth.py init                  # create the file + owner token
    python scripts/auth.py add james --role partner
    python scripts/auth.py list
    python scripts/auth.py revoke james

Each command prints the sign-in URL to hand to that person. The token IS the
credential — anyone holding it has that role, so send it over something private
(Signal/WhatsApp), not a public channel.

Environment:
  RUFUS_AUTH_DISABLED=1   escape hatch: skip all auth (single-user localhost
                          only — never set this while `tailscale serve` is on).
"""

import json
import os
import secrets
import sys
from functools import wraps
from pathlib import Path

from flask import abort, g, redirect, request

ROOT       = Path(__file__).parent.parent
USERS_FILE = ROOT / "config" / "users.json"

COOKIE_NAME = "rufus_auth"
COOKIE_MAX_AGE = 60 * 60 * 24 * 90     # 90 days — a phone shouldn't re-auth weekly

# What each role may do. Checked by require(), which every mutating route calls.
ROLE_PERMISSIONS: dict[str, set[str]] = {
    "owner": {
        "view", "generate", "thumbnail", "download",
        "approve", "reject", "edit", "settings", "system", "cancel",
    },
    # A partner can MAKE things and take them to their phone, but cannot put
    # anything on the channel or touch how the machine is configured. That
    # split is the whole point of having roles here.
    "partner": {"view", "generate", "thumbnail", "download", "edit"},
    "viewer":  {"view", "download"},
}

ROLES = tuple(ROLE_PERMISSIONS)


# ── User store ────────────────────────────────────────────────────────────────

def _load_users() -> list[dict]:
    """Users from config/users.json, or [] when the file is absent/unreadable.

    Never raises: a corrupt users file must not lock you out of your own
    dashboard — it falls back to legacy loopback-owner mode instead.
    """
    try:
        data = json.loads(USERS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return []
    users = data.get("users") if isinstance(data, dict) else data
    return users if isinstance(users, list) else []


def _save_users(users: list[dict]) -> None:
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    USERS_FILE.write_text(json.dumps({"users": users}, indent=2), encoding="utf-8")


def auth_enabled() -> bool:
    """True once at least one user exists. Absent file = legacy mode."""
    if os.environ.get("RUFUS_AUTH_DISABLED", "").strip().lower() in ("1", "true", "yes", "on"):
        return False
    return bool(_load_users())


def user_for_token(token: str) -> dict | None:
    """Constant-time token lookup. Returns the user dict or None.

    compare_digest, not ==, so a token can't be recovered by timing how long a
    wrong guess takes to be rejected.
    """
    if not token:
        return None
    for u in _load_users():
        stored = u.get("token", "")
        if stored and secrets.compare_digest(str(stored), str(token)):
            return u
    return None


def _is_loopback() -> bool:
    return request.remote_addr in ("127.0.0.1", "::1")


# ── Request-time identity ─────────────────────────────────────────────────────

def current_user() -> dict | None:
    """Who is making this request, or None if nobody is authenticated.

    Token sources, in order: explicit ?token= (the shape of a shared sign-in
    link), the X-Rufus-Token header (for scripts/curl), then the cookie that
    ?token= sets so the phone stays signed in.
    """
    if not auth_enabled():
        # Legacy mode: loopback is the owner, anything else is nobody.
        return {"name": "local", "role": "owner"} if _is_loopback() else None

    token = (request.args.get("token", "")
             or request.headers.get("X-Rufus-Token", "")
             or request.cookies.get(COOKIE_NAME, "")).strip()
    return user_for_token(token)


def role() -> str:
    user = getattr(g, "rufus_user", None) or current_user()
    return (user or {}).get("role", "")


def permissions() -> set[str]:
    return ROLE_PERMISSIONS.get(role(), set())


def can(permission: str) -> bool:
    """Template-facing check — hides UI the current user may not use.

    Hiding a button is cosmetic, never the enforcement: every mutating route
    calls require() as well, so a hand-crafted POST is refused the same way.
    """
    return permission in permissions()


def require(permission: str) -> None:
    """Abort the request unless the caller holds `permission`.

    401 when we don't know who they are (their sign-in link is missing or
    revoked), 403 when we do and the answer is simply no — distinct so the
    dashboard can send the first to the login page and show the second an
    honest "your role can't do this."
    """
    user = getattr(g, "rufus_user", None) or current_user()
    if user is None:
        abort(401)
    if permission not in ROLE_PERMISSIONS.get(user.get("role", ""), set()):
        abort(403)


def requires(permission: str):
    """Route decorator form of require()."""
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            require(permission)
            return fn(*args, **kwargs)
        return wrapper
    return decorator


# ── CLI ───────────────────────────────────────────────────────────────────────

def _new_token() -> str:
    return secrets.token_urlsafe(32)


def _base_url() -> str:
    return (os.environ.get("RUFUS_DASHBOARD_URL", "").strip().rstrip("/")
            or f"http://localhost:{os.environ.get('RUFUS_DASHBOARD_PORT', '8765')}")


def _print_signin(user: dict) -> None:
    print(f"\n  {user['name']}  ({user['role']})")
    print(f"  Sign-in link:  {_base_url()}/?token={user['token']}")
    print("  Send this privately — the link IS the password.\n")


def _cmd_init() -> int:
    if _load_users():
        print(f"{USERS_FILE} already has users — use `add` instead.")
        return 1
    owner = {"name": "owner", "role": "owner", "token": _new_token()}
    _save_users([owner])
    print(f"Created {USERS_FILE}")
    print("Auth is now ON for every request, including localhost.")
    _print_signin(owner)
    return 0


def _cmd_add(name: str, role_name: str) -> int:
    if role_name not in ROLE_PERMISSIONS:
        print(f"Unknown role '{role_name}'. Valid: {', '.join(ROLES)}")
        return 1
    users = _load_users()
    if not users:
        print("No users file yet — run `python scripts/auth.py init` first.")
        return 1
    if any(u.get("name") == name for u in users):
        print(f"User '{name}' already exists — `revoke` then `add` to reissue.")
        return 1
    user = {"name": name, "role": role_name, "token": _new_token()}
    users.append(user)
    _save_users(users)
    print(f"Added '{name}' as {role_name}.")
    _print_signin(user)
    return 0


def _cmd_list() -> int:
    users = _load_users()
    if not users:
        print("No users — auth is OFF (legacy loopback-owner mode).")
        return 0
    print(f"{len(users)} user(s) in {USERS_FILE}:")
    for u in users:
        perms = ", ".join(sorted(ROLE_PERMISSIONS.get(u.get("role", ""), set()))) or "(unknown role)"
        print(f"  {u.get('name'):<16} {u.get('role'):<10} {perms}")
    return 0


def _cmd_revoke(name: str) -> int:
    users = _load_users()
    remaining = [u for u in users if u.get("name") != name]
    if len(remaining) == len(users):
        print(f"No user named '{name}'.")
        return 1
    if not any(u.get("role") == "owner" for u in remaining):
        print("Refusing — that's the last owner; you'd lock yourself out.")
        return 1
    _save_users(remaining)
    print(f"Revoked '{name}'. Their link stops working immediately.")
    return 0


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1
    cmd = argv[0]
    if cmd == "init":
        return _cmd_init()
    if cmd == "add":
        if len(argv) < 2:
            print("usage: auth.py add <name> [--role owner|partner|viewer]")
            return 1
        role_name = "partner"
        if "--role" in argv:
            i = argv.index("--role")
            if i + 1 < len(argv):
                role_name = argv[i + 1]
        return _cmd_add(argv[1], role_name)
    if cmd == "list":
        return _cmd_list()
    if cmd == "revoke":
        if len(argv) < 2:
            print("usage: auth.py revoke <name>")
            return 1
        return _cmd_revoke(argv[1])
    print(f"Unknown command '{cmd}'. Try: init | add | list | revoke")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
