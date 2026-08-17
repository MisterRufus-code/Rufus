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
    python scripts/auth.py add james --role partner --google james@gmail.com
    python scripts/auth.py list
    python scripts/auth.py link james             # reprint a link without rotating the token
    python scripts/auth.py rotate owner           # NEW token, same user — for a leaked link
    python scripts/auth.py revoke james

Each command prints the sign-in URL to hand to that person. The token IS the
credential — anyone holding it has that role, so send it over something private
(Signal/WhatsApp), not a public channel.

All of this is also available from inside the dashboard itself, for an owner,
at /settings/users — add/revoke without a terminal. --google (or the
dashboard form's Google email field) additionally lets that person sign in
with their Google account instead of holding a link; see google_oauth_config()
below and the README's "Google Sign-In" section for the one-time setup.

Environment:
  RUFUS_AUTH_DISABLED=1   escape hatch: skip all auth (single-user localhost
                          only — never set this while `tailscale serve` is on).
"""

import json
import os
import secrets
import sys
import time
from functools import wraps
from pathlib import Path

from flask import abort, g, redirect, request


class AuthError(ValueError):
    """A problem with a user-management request that should be shown to a
    human (bad role, duplicate name, unconfigured Google client) — a distinct
    type from a bare ValueError so callers can catch precisely this without
    swallowing an unrelated bug."""

ROOT       = Path(__file__).parent.parent
USERS_FILE = ROOT / "config" / "users.json"

COOKIE_NAME = "rufus_auth"
COOKIE_MAX_AGE = 60 * 60 * 24 * 90     # 90 days — a phone shouldn't re-auth weekly

# What each role may do. Checked by require(), which every mutating route calls.
ROLE_PERMISSIONS: dict[str, set[str]] = {
    "owner": {
        "view", "generate", "thumbnail", "download",
        "approve", "reject", "edit", "settings", "system", "cancel",
        "manage_users",   # can add/revoke OTHER users — owner-only, obviously
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


# ── User management (shared by the CLI and the dashboard's /settings/users) ──
# One implementation of the actual rules — valid role, no duplicate name,
# never leave the dashboard with zero owners — so the CLI and the web form
# can't drift into checking different things.

def add_user(name: str, role_name: str, *, google_email: str | None = None) -> dict:
    """Create a user and return their record (including the generated token).

    Raises AuthError with a human-readable reason on any problem — empty
    name, unknown role, or a name already taken.
    """
    name = (name or "").strip()
    if not name:
        raise AuthError("Name can't be empty.")
    if role_name not in ROLE_PERMISSIONS:
        raise AuthError(f"Unknown role '{role_name}'. Valid: {', '.join(ROLES)}")
    users = _load_users()
    if any(u.get("name") == name for u in users):
        raise AuthError(f"User '{name}' already exists.")
    user = {"name": name, "role": role_name, "token": _new_token()}
    if google_email:
        # Stored lowercase so a later case-different login (Gmail addresses
        # are case-insensitive) still matches in find_user_by_email().
        user["google_email"] = google_email.strip().lower()
    users.append(user)
    _save_users(users)
    return user


def rotate_token(name: str) -> dict | None:
    """Issue a NEW token for an existing user, keeping their name and role.

    THE PATH THAT DID NOT EXIST. A sign-in link is the credential — it gets
    pasted into a chat, screenshotted, or read out of a config file, and then
    it has to be replaced. `_cmd_add`'s own error text said how: revoke, then
    add. For every user but one that works. For the LAST OWNER it cannot:
    revoke_user refuses them by design, so the person whose link is most
    valuable to leak was the one person who could not replace theirs without
    hand-editing config/users.json.

    Rotating in place sidesteps that entirely — the owner never stops being
    an owner, so the guard never has anything to refuse. Old link dead, new
    link printed, one command.
    """
    users = _load_users()
    for u in users:
        if u.get("name") == name:
            u["token"] = _new_token()
            _save_users(users)
            return u
    return None


def revoke_user(name: str) -> str:
    """Remove a user. Returns 'ok', 'not_found', or 'last_owner' (refused —
    would leave nobody who can sign back in as owner)."""
    users = _load_users()
    remaining = [u for u in users if u.get("name") != name]
    if len(remaining) == len(users):
        return "not_found"
    if not any(u.get("role") == "owner" for u in remaining):
        return "last_owner"
    _save_users(remaining)
    return "ok"


def find_user_by_email(email: str) -> dict | None:
    """Match a verified Google account email to a user record, or None.

    Case-insensitive: Google addresses are effectively case-insensitive, and
    a casing mismatch shouldn't be able to lock someone out of an account
    the owner clearly intended to grant them."""
    email = (email or "").strip().lower()
    if not email:
        return None
    for u in _load_users():
        if (u.get("google_email") or "").strip().lower() == email:
            return u
    return None


# ── Google Sign-In (optional alternative to a shared token link) ────────────
# A SEPARATE OAuth client from the one youtube_uploader.py uses: that one is
# registered in Google Cloud Console as a Desktop app (loopback redirect,
# config/client_secrets.json); this needs a Web application client (a fixed
# https redirect URI matching the tailnet domain) — Google does not let the
# two share one registration. See README "Google Sign-In" for the one-time
# console setup.
#
# What this buys over the token link: identity is vouched for by Google
# instead of by possession of a bearer string, so there's nothing to leak in
# a screenshot or a forwarded chat message. What it does NOT do: grant access
# to anyone with a Google account — find_user_by_email() only recognizes an
# email the owner explicitly added, so an unrecognized sign-in is refused the
# same as a wrong token.

GOOGLE_OAUTH_FILE = ROOT / "config" / "google_oauth.json"

_OAUTH_STATE_TTL = 600   # seconds — long enough for a slow phone to finish the redirect
_pending_oauth_states: dict[str, float] = {}


def google_oauth_config() -> dict | None:
    """{"client_id", "client_secret"} from config/google_oauth.json, or None
    if the file is missing/incomplete — Google sign-in is simply absent from
    the login page in that case, not an error."""
    try:
        data = json.loads(GOOGLE_OAUTH_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    if not data.get("client_id") or not data.get("client_secret"):
        return None
    return data


def google_oauth_enabled() -> bool:
    return google_oauth_config() is not None


def google_redirect_uri() -> str:
    """Must byte-for-byte match an 'Authorized redirect URI' registered on
    the Google Cloud OAuth client, or Google refuses the whole flow with
    redirect_uri_mismatch. Derived from the same _base_url() the sign-in
    links use, so the two can't drift apart independently."""
    return f"{_base_url()}/auth/google/callback"


def _prune_oauth_states() -> None:
    cutoff = time.time() - _OAUTH_STATE_TTL
    for k in [k for k, t in _pending_oauth_states.items() if t < cutoff]:
        _pending_oauth_states.pop(k, None)


def new_oauth_state() -> str:
    """A one-time CSRF token for the Google redirect round-trip. Stored
    in-process (fine here: the dashboard is a single Flask process,
    threaded=False, never multiple workers) rather than in a cookie, so
    there's no session/secret-key machinery to add just for this."""
    _prune_oauth_states()
    state = secrets.token_urlsafe(24)
    _pending_oauth_states[state] = time.time()
    return state


def consume_oauth_state(state: str) -> bool:
    """True exactly once per state issued by new_oauth_state(). Popping
    rather than merely checking membership means a captured callback URL
    can't be replayed to sign in a second time."""
    _prune_oauth_states()
    return bool(state) and _pending_oauth_states.pop(state, None) is not None


def build_google_flow():
    """A google_auth_oauthlib Flow for the WEB authorization-code flow (not
    InstalledAppFlow — that variant is for a desktop app's own loopback
    redirect, which is what youtube_uploader.py uses; this dashboard is
    itself the web server receiving the redirect). Imported lazily — same
    reason as youtube_uploader.py's Google imports: this module must stay
    importable with no Google auth stack installed when Google sign-in isn't
    configured at all."""
    from google_auth_oauthlib.flow import Flow

    cfg = google_oauth_config()
    if cfg is None:
        raise AuthError(f"Google sign-in isn't configured ({GOOGLE_OAUTH_FILE} missing).")
    client_config = {"web": {
        "client_id": cfg["client_id"],
        "client_secret": cfg["client_secret"],
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
    }}
    return Flow.from_client_config(
        client_config,
        scopes=["openid", "https://www.googleapis.com/auth/userinfo.email"],
        redirect_uri=google_redirect_uri(),
    )


def verify_google_id_token(id_token_jwt: str) -> dict:
    """Decode + verify a Google ID token, returning its claims.

    The verification (signature against Google's published keys, issuer,
    audience == our client_id, expiry) is what makes the email claim
    trustworthy — skipping it would let anyone hand the callback a
    self-signed 'email' and be waved in as whoever they typed."""
    from google.auth.transport import requests as google_requests
    from google.oauth2 import id_token as google_id_token

    cfg = google_oauth_config()
    if cfg is None:
        raise AuthError("Google sign-in isn't configured.")
    return google_id_token.verify_oauth2_token(
        id_token_jwt, google_requests.Request(), cfg["client_id"])


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


DASHBOARD_URL_FILE = ROOT / "config" / "dashboard_url.txt"


def _base_url() -> str:
    """Where to send someone to sign in.

    Checked in order: RUFUS_DASHBOARD_URL (works within the process that set
    it), then config/dashboard_url.txt (serve.ps1 writes the tailnet https
    URL here after `tailscale serve` runs, since a NEW PowerShell window
    doesn't inherit an env var `setx` set in a previous one — a file survives
    that where an env var doesn't), then localhost as the last resort for a
    single-PC setup with no remote access configured at all.
    """
    env = os.environ.get("RUFUS_DASHBOARD_URL", "").strip().rstrip("/")
    if env:
        return env
    try:
        saved = DASHBOARD_URL_FILE.read_text(encoding="utf-8").strip().rstrip("/")
        if saved:
            return saved
    except OSError:
        pass
    return f"http://localhost:{os.environ.get('RUFUS_DASHBOARD_PORT', '8765')}"


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


def _cmd_add(name: str, role_name: str, google_email: str | None = None) -> int:
    if not _load_users():
        print("No users file yet — run `python scripts/auth.py init` first.")
        return 1
    try:
        user = add_user(name, role_name, google_email=google_email)
    except AuthError as e:
        msg = str(e)
        if "already exists" in msg:
            msg += (f" `link {name}` reprints their existing sign-in link, "
                    f"or `rotate {name}` issues a new token and kills the old "
                    f"one — which is what you want for a leaked link, and "
                    f"works for the last owner, who cannot be revoked.")
        print(msg)
        return 1
    print(f"Added '{name}' as {role_name}.")
    _print_signin(user)
    if google_email:
        print(f"  Google sign-in: {google_email} (once you've set up "
              f"config/google_oauth.json — see README)")
    return 0


def _cmd_link(name: str) -> int:
    """Reprint an existing user's sign-in link without rotating their token.

    Needed whenever _base_url() changes after the fact — e.g. `serve.ps1
    -Tailscale` ran for the first time after users already existed, so their
    original link (printed against localhost) was wrong even though the
    token itself was always fine. `revoke` + `add` would work too, but it
    invalidates a link that might already be saved on someone's phone for no
    reason."""
    for u in _load_users():
        if u.get("name") == name:
            _print_signin(u)
            return 0
    print(f"No user named '{name}'.")
    return 1


def _cmd_list() -> int:
    users = _load_users()
    if not users:
        print("No users — auth is OFF (legacy loopback-owner mode).")
        return 0
    print(f"{len(users)} user(s) in {USERS_FILE}:")
    for u in users:
        perms = ", ".join(sorted(ROLE_PERMISSIONS.get(u.get("role", ""), set()))) or "(unknown role)"
        via = f"google:{u['google_email']}" if u.get("google_email") else "token link"
        print(f"  {u.get('name'):<16} {u.get('role'):<10} {via:<28} {perms}")
    return 0


def _cmd_rotate(name: str) -> int:
    user = rotate_token(name)
    if user is None:
        print(f"No user named '{name}'.")
        return 1
    print(f"Rotated '{name}'. Their OLD link stopped working just now.")
    _print_signin(user)
    return 0


def _cmd_revoke(name: str) -> int:
    status = revoke_user(name)
    if status == "not_found":
        print(f"No user named '{name}'.")
        return 1
    if status == "last_owner":
        print("Refusing — that's the last owner; you'd lock yourself out.")
        return 1
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
            print("usage: auth.py add <name> [--role owner|partner|viewer] [--google email]")
            return 1
        role_name = "partner"
        if "--role" in argv:
            i = argv.index("--role")
            if i + 1 < len(argv):
                role_name = argv[i + 1]
        google_email = None
        if "--google" in argv:
            i = argv.index("--google")
            if i + 1 < len(argv):
                google_email = argv[i + 1]
        return _cmd_add(argv[1], role_name, google_email)
    if cmd == "list":
        return _cmd_list()
    if cmd == "link":
        if len(argv) < 2:
            print("usage: auth.py link <name>")
            return 1
        return _cmd_link(argv[1])
    if cmd == "rotate":
        if len(argv) < 2:
            print("usage: auth.py rotate <name>")
            return 1
        return _cmd_rotate(argv[1])
    if cmd == "revoke":
        if len(argv) < 2:
            print("usage: auth.py revoke <name>")
            return 1
        return _cmd_revoke(argv[1])
    print(f"Unknown command '{cmd}'. Try: init | add | list | link | rotate "
          f"| revoke")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
