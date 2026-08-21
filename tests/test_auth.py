"""Tests for auth.py + the dashboard's role gates.

The thing under test is a security boundary, so these assert the NEGATIVE
cases hardest: a partner must not be able to publish to the owner's YouTube
channel, and an anonymous request must not reach anything at all — including
over a connection that LOOKS like loopback, which is exactly what every
`tailscale serve` request looks like.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import auth
import dashboard
import db_manager

OWNER_TOKEN   = "owner-token-for-tests"
PARTNER_TOKEN = "partner-token-for-tests"
VIEWER_TOKEN  = "viewer-token-for-tests"


@pytest.fixture
def users_file(tmp_path, monkeypatch):
    """A real users.json, so auth reads the same path production does."""
    f = tmp_path / "users.json"
    f.write_text(json.dumps({"users": [
        {"name": "dani",  "role": "owner",   "token": OWNER_TOKEN},
        {"name": "james", "role": "partner", "token": PARTNER_TOKEN},
        {"name": "guest", "role": "viewer",  "token": VIEWER_TOKEN},
    ]}))
    monkeypatch.setattr(auth, "USERS_FILE", f)
    monkeypatch.delenv("RUFUS_AUTH_DISABLED", raising=False)
    return f


@pytest.fixture
def client(tmp_path, monkeypatch, users_file):
    monkeypatch.setattr(db_manager, "DB_FILE", tmp_path / "test.db")
    db_manager.init_db()
    monkeypatch.setattr(dashboard, "DEBUG_ROOT", tmp_path / "debug")
    dashboard.app.config["TESTING"] = True
    with dashboard.app.test_client() as c:
        yield c


def _seed_pending() -> int:
    db_manager.save_video(niche="finance", script_hook="A pending hook",
                          scene_desc="s", video_file="pending.mp4",
                          score=9, run_id="run_pending", upload_status="pending")
    return 1


# ── The store itself ──────────────────────────────────────────────────────────

def test_auth_disabled_without_users_file(tmp_path, monkeypatch):
    monkeypatch.setattr(auth, "USERS_FILE", tmp_path / "nope.json")
    monkeypatch.delenv("RUFUS_AUTH_DISABLED", raising=False)
    assert auth.auth_enabled() is False


def test_auth_enabled_once_users_exist(users_file):
    assert auth.auth_enabled() is True


def test_unknown_token_resolves_to_nobody(users_file):
    assert auth.user_for_token("not-a-real-token") is None
    assert auth.user_for_token("") is None


def test_known_tokens_resolve_to_their_roles(users_file):
    assert auth.user_for_token(OWNER_TOKEN)["role"] == "owner"
    assert auth.user_for_token(PARTNER_TOKEN)["role"] == "partner"


def test_corrupt_users_file_falls_back_rather_than_crashing(tmp_path, monkeypatch):
    bad = tmp_path / "users.json"
    bad.write_text("{ this is not json")
    monkeypatch.setattr(auth, "USERS_FILE", bad)
    assert auth._load_users() == []


def test_partner_permissions_exclude_publishing():
    partner = auth.ROLE_PERMISSIONS["partner"]
    assert "generate" in partner and "thumbnail" in partner and "download" in partner
    for forbidden in ("approve", "settings", "system", "cancel"):
        assert forbidden not in partner


def test_viewer_cannot_generate():
    assert "generate" not in auth.ROLE_PERMISSIONS["viewer"]
    assert "view" in auth.ROLE_PERMISSIONS["viewer"]


# ── Request-level enforcement ─────────────────────────────────────────────────

def test_anonymous_request_is_refused_even_from_loopback(client):
    """The whole reason auth exists: `tailscale serve` proxies via 127.0.0.1,
    so a loopback source address is NOT evidence the owner is asking."""
    r = client.get("/", headers={"Accept": "text/html"})
    assert r.status_code == 302
    assert "/login" in r.headers["Location"]


def test_anonymous_api_style_request_gets_401(client):
    assert client.get("/").status_code == 401


def test_login_page_is_reachable_without_a_token(client):
    assert client.get("/login").status_code == 401   # renders the form, unauthenticated


def test_healthz_needs_no_token(client):
    r = client.get("/healthz")
    assert r.status_code == 200 and r.get_json()["ok"] is True


def test_owner_token_opens_the_queue(client):
    assert client.get(f"/?token={OWNER_TOKEN}").status_code == 200


def test_partner_token_opens_the_queue(client):
    assert client.get(f"/?token={PARTNER_TOKEN}").status_code == 200


def test_token_is_persisted_as_a_cookie(client):
    r = client.get(f"/?token={PARTNER_TOKEN}")
    assert r.status_code == 200
    # Subsequent request carries no token in the URL and must still work.
    assert client.get("/").status_code == 200


def test_cookie_is_httponly_and_samesite_strict(client):
    r = client.get(f"/?token={PARTNER_TOKEN}")
    cookie = r.headers.get("Set-Cookie", "")
    assert "HttpOnly" in cookie
    assert "SameSite=Strict" in cookie


def test_logout_clears_access(client):
    client.get(f"/?token={PARTNER_TOKEN}")
    client.get("/logout")
    assert client.get("/").status_code == 401


# ── The gates that matter ─────────────────────────────────────────────────────

def test_partner_cannot_approve_an_upload(client, monkeypatch):
    """The single most important assertion in this file."""
    vid = _seed_pending()
    called = []
    import youtube_uploader
    monkeypatch.setattr(youtube_uploader, "upload",
                        lambda *a, **k: called.append(a) or ("url", "id"))
    client.get(f"/?token={PARTNER_TOKEN}")
    r = client.post(f"/video/{vid}/approve")
    assert r.status_code == 403
    assert called == [], "partner reached the YouTube upload path"


def test_partner_cannot_open_settings_or_system(client):
    client.get(f"/?token={PARTNER_TOKEN}")
    assert client.get("/settings").status_code == 403
    assert client.get("/system").status_code == 403


def test_partner_cannot_save_settings(client):
    client.get(f"/?token={PARTNER_TOKEN}")
    assert client.post("/settings/save", data={}).status_code == 403


def test_partner_cannot_cancel_a_run(client):
    client.get(f"/?token={PARTNER_TOKEN}")
    assert client.post("/system/cancel", data={}).status_code == 403


def test_owner_can_open_settings_and_system(client):
    client.get(f"/?token={OWNER_TOKEN}")
    assert client.get("/settings").status_code == 200
    assert client.get("/system").status_code == 200


def test_partner_can_reach_the_generate_page(client):
    client.get(f"/?token={PARTNER_TOKEN}")
    assert client.get("/generate").status_code == 200


def test_viewer_cannot_reach_the_generate_page(client):
    client.get(f"/?token={VIEWER_TOKEN}")
    assert client.get("/generate").status_code == 403


def test_viewer_cannot_start_a_run(client, monkeypatch):
    launched = []
    monkeypatch.setattr(dashboard, "_launch_run",
                        lambda **k: launched.append(k) or (None, Path("x.log")))
    client.get(f"/?token={VIEWER_TOKEN}")
    assert client.post("/system/run", data={}).status_code == 403
    assert launched == []


def test_partner_can_start_a_run(client, monkeypatch):
    launched = []
    monkeypatch.setattr(dashboard, "_launch_run",
                        lambda **k: launched.append(k) or (None, Path("x.log")))
    monkeypatch.setattr(dashboard, "_run_in_progress", lambda cid: False)
    client.get(f"/?token={PARTNER_TOKEN}")
    r = client.post("/system/run", data={"topic": "test topic"})
    assert r.status_code == 302
    assert launched and launched[0]["topic"] == "test topic"


def test_partner_can_reach_thumbnails(client):
    client.get(f"/?token={PARTNER_TOKEN}")
    assert client.get("/thumbnails").status_code == 200


def test_viewer_cannot_generate_a_thumbnail(client, monkeypatch):
    import image_gen
    made = []
    monkeypatch.setattr(image_gen, "generate_image",
                        lambda *a, **k: made.append(a) or Path("x.png"))
    client.get(f"/?token={VIEWER_TOKEN}")
    assert client.post("/thumbnails/generate", data={"prompt": "x"}).status_code == 403
    assert made == []


def test_nav_hides_owner_only_links_from_a_partner(client):
    client.get(f"/?token={PARTNER_TOKEN}")
    html = client.get("/").get_data(as_text=True)
    assert "/settings" not in html
    assert "/generate" in html


def test_nav_shows_owner_everything(client):
    client.get(f"/?token={OWNER_TOKEN}")
    html = client.get("/").get_data(as_text=True)
    assert "/settings" in html and "/system" in html


def test_approve_still_works_for_the_owner(client, monkeypatch):
    """The gates must not have broken the owner's own happy path."""
    vid = _seed_pending()
    import youtube_uploader
    monkeypatch.setattr(youtube_uploader, "upload",
                        lambda *a, **k: ("https://youtu.be/xyz", "xyz"))
    monkeypatch.setattr(Path, "exists", lambda self: True)
    client.get(f"/?token={OWNER_TOKEN}")
    r = client.post(f"/video/{vid}/approve")
    assert r.status_code == 302
    assert "error" not in r.headers["Location"]


# ── Dashboard URL resolution (the tailscale-link bug) ─────────────────────────

def test_base_url_falls_back_to_localhost_with_nothing_configured(monkeypatch, tmp_path):
    monkeypatch.delenv("RUFUS_DASHBOARD_URL", raising=False)
    monkeypatch.setattr(auth, "DASHBOARD_URL_FILE", tmp_path / "nope.txt")
    assert auth._base_url() == "http://localhost:8765"


def test_base_url_prefers_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv("RUFUS_DASHBOARD_URL", "https://from-env.example")
    monkeypatch.setattr(auth, "DASHBOARD_URL_FILE", tmp_path / "nope.txt")
    assert auth._base_url() == "https://from-env.example"


def test_base_url_reads_the_saved_tailnet_url(monkeypatch, tmp_path):
    """This is the actual bug: serve.ps1 -Tailscale creates a tailnet URL that
    a NEW `python scripts\\auth.py add ...` invocation (a separate process)
    has no way to know about via env var alone — the file is what carries it
    across process/terminal boundaries."""
    monkeypatch.delenv("RUFUS_DASHBOARD_URL", raising=False)
    f = tmp_path / "dashboard_url.txt"
    f.write_text("https://rufus.tail635959.ts.net/\n")
    monkeypatch.setattr(auth, "DASHBOARD_URL_FILE", f)
    assert auth._base_url() == "https://rufus.tail635959.ts.net"


def test_cmd_link_reprints_without_rotating_the_token(users_file, capsys):
    rc = auth._cmd_link("james")
    out = capsys.readouterr().out
    assert rc == 0
    assert PARTNER_TOKEN in out
    # The token itself must be unchanged afterward.
    assert auth.user_for_token(PARTNER_TOKEN)["name"] == "james"


def test_cmd_link_reports_unknown_user(users_file, capsys):
    rc = auth._cmd_link("nobody")
    assert rc == 1
    assert "No user named" in capsys.readouterr().out


# ── User management functions (shared by CLI and /settings/users) ────────────

def test_add_user_creates_and_returns_the_record(users_file):
    user = auth.add_user("newperson", "viewer")
    assert user["name"] == "newperson" and user["role"] == "viewer" and user["token"]
    assert auth.user_for_token(user["token"])["name"] == "newperson"


def test_add_user_rejects_duplicate_name(users_file):
    with pytest.raises(auth.AuthError, match="already exists"):
        auth.add_user("james", "viewer")


def test_add_user_rejects_unknown_role(users_file):
    with pytest.raises(auth.AuthError, match="Unknown role"):
        auth.add_user("someone", "superadmin")


def test_add_user_rejects_empty_name(users_file):
    with pytest.raises(auth.AuthError, match="empty"):
        auth.add_user("   ", "viewer")


def test_add_user_stores_google_email_lowercased(users_file):
    user = auth.add_user("newperson", "partner", google_email="James@Gmail.com")
    assert user["google_email"] == "james@gmail.com"


def test_revoke_user_removes_them(users_file):
    assert auth.revoke_user("james") == "ok"
    assert auth.user_for_token(PARTNER_TOKEN) is None


def test_revoke_user_reports_not_found(users_file):
    assert auth.revoke_user("nobody") == "not_found"


def test_revoke_user_refuses_to_remove_the_last_owner(users_file):
    auth.revoke_user("james")
    auth.revoke_user("guest")
    assert auth.revoke_user("dani") == "last_owner"
    assert auth.user_for_token(OWNER_TOKEN) is not None   # unchanged


def test_find_user_by_email_matches_case_insensitively(users_file):
    auth.add_user("newperson", "partner", google_email="James@Gmail.com")
    assert auth.find_user_by_email("james@GMAIL.com")["name"] == "newperson"


def test_find_user_by_email_returns_none_for_unrecognized_address(users_file):
    assert auth.find_user_by_email("stranger@example.com") is None


def test_find_user_by_email_returns_none_for_empty_input(users_file):
    assert auth.find_user_by_email("") is None
    assert auth.find_user_by_email(None) is None


# ── Google OAuth config / state (no live network) ────────────────────────────

def test_google_oauth_disabled_without_config_file(tmp_path, monkeypatch):
    monkeypatch.setattr(auth, "GOOGLE_OAUTH_FILE", tmp_path / "nope.json")
    assert auth.google_oauth_enabled() is False


def test_google_oauth_enabled_once_configured(tmp_path, monkeypatch):
    f = tmp_path / "google_oauth.json"
    f.write_text(json.dumps({"client_id": "x", "client_secret": "y"}))
    monkeypatch.setattr(auth, "GOOGLE_OAUTH_FILE", f)
    assert auth.google_oauth_enabled() is True


def test_google_oauth_disabled_when_incomplete(tmp_path, monkeypatch):
    f = tmp_path / "google_oauth.json"
    f.write_text(json.dumps({"client_id": "x"}))   # no client_secret
    monkeypatch.setattr(auth, "GOOGLE_OAUTH_FILE", f)
    assert auth.google_oauth_enabled() is False


def test_oauth_state_is_consumed_exactly_once():
    state = auth.new_oauth_state()
    assert auth.consume_oauth_state(state) is True
    assert auth.consume_oauth_state(state) is False, "state was replayable"


def test_unknown_oauth_state_is_rejected():
    assert auth.consume_oauth_state("made-up-state") is False


def test_empty_oauth_state_is_rejected():
    assert auth.consume_oauth_state("") is False


def test_expired_oauth_state_is_rejected(monkeypatch):
    monkeypatch.setattr(auth, "_OAUTH_STATE_TTL", 0)
    state = auth.new_oauth_state()
    assert auth.consume_oauth_state(state) is False


def test_google_redirect_uri_matches_base_url(monkeypatch, tmp_path):
    monkeypatch.setenv("RUFUS_DASHBOARD_URL", "https://rufus.example.ts.net")
    assert auth.google_redirect_uri() == "https://rufus.example.ts.net/auth/google/callback"


def test_build_google_flow_raises_when_unconfigured(tmp_path, monkeypatch):
    monkeypatch.setattr(auth, "GOOGLE_OAUTH_FILE", tmp_path / "nope.json")
    with pytest.raises(auth.AuthError):
        auth.build_google_flow()


# ── Dashboard: /settings/users permission gating ──────────────────────────────

def test_partner_cannot_manage_users(client):
    client.get(f"/?token={PARTNER_TOKEN}")
    assert client.get("/settings/users").status_code == 403
    assert client.post("/settings/users/add", data={"name": "x", "role": "viewer"}).status_code == 403
    assert client.post("/settings/users/revoke", data={"name": "guest"}).status_code == 403


def test_owner_can_add_and_revoke_a_user_through_the_dashboard(client):
    client.get(f"/?token={OWNER_TOKEN}")
    r = client.post("/settings/users/add", data={"name": "newpartner", "role": "partner"})
    assert r.status_code == 302
    assert auth.user_for_token != None  # sanity
    users = auth._load_users()
    assert any(u["name"] == "newpartner" for u in users)

    r2 = client.post("/settings/users/revoke", data={"name": "newpartner"})
    assert r2.status_code == 302
    users_after = auth._load_users()
    assert not any(u["name"] == "newpartner" for u in users_after)


def test_owner_cannot_revoke_themselves_as_last_owner(client):
    client.get(f"/?token={OWNER_TOKEN}")
    client.post("/settings/users/revoke", data={"name": "james"})
    client.post("/settings/users/revoke", data={"name": "guest"})
    r = client.post("/settings/users/revoke", data={"name": "dani"})
    assert "error" in r.headers["Location"]
    assert auth.user_for_token(OWNER_TOKEN) is not None


def test_dashboard_add_user_rejects_bad_role(client):
    client.get(f"/?token={OWNER_TOKEN}")
    r = client.post("/settings/users/add", data={"name": "x", "role": "superadmin"})
    assert "error" in r.headers["Location"]


# ── Dashboard: Google login routes ────────────────────────────────────────────

def test_google_start_404s_when_not_configured(client):
    assert client.get("/auth/google/start").status_code == 404


def test_google_callback_404s_when_not_configured(client):
    assert client.get("/auth/google/callback").status_code == 404


def test_google_start_redirects_to_google_when_configured(client, tmp_path, monkeypatch):
    f = tmp_path / "google_oauth.json"
    f.write_text(json.dumps({"client_id": "cid", "client_secret": "secret"}))
    monkeypatch.setattr(auth, "GOOGLE_OAUTH_FILE", f)
    r = client.get("/auth/google/start")
    assert r.status_code == 302
    assert "accounts.google.com" in r.headers["Location"]


def test_google_callback_rejects_bad_state(client, tmp_path, monkeypatch):
    f = tmp_path / "google_oauth.json"
    f.write_text(json.dumps({"client_id": "cid", "client_secret": "secret"}))
    monkeypatch.setattr(auth, "GOOGLE_OAUTH_FILE", f)
    r = client.get("/auth/google/callback?state=bogus&code=abc")
    assert r.status_code == 302 and "/login" in r.headers["Location"]


def test_google_callback_rejects_denied_consent(client, tmp_path, monkeypatch):
    f = tmp_path / "google_oauth.json"
    f.write_text(json.dumps({"client_id": "cid", "client_secret": "secret"}))
    monkeypatch.setattr(auth, "GOOGLE_OAUTH_FILE", f)
    r = client.get("/auth/google/callback?error=access_denied")
    assert r.status_code == 302 and "/login" in r.headers["Location"]


def test_google_callback_signs_in_a_recognized_email(client, tmp_path, monkeypatch):
    f = tmp_path / "google_oauth.json"
    f.write_text(json.dumps({"client_id": "cid", "client_secret": "secret"}))
    monkeypatch.setattr(auth, "GOOGLE_OAUTH_FILE", f)
    auth.add_user("googleuser", "partner", google_email="partner@example.com")

    state = auth.new_oauth_state()

    class FakeCreds:
        id_token = "fake-jwt"

    class FakeFlow:
        credentials = FakeCreds()
        def fetch_token(self, code):
            pass

    monkeypatch.setattr(auth, "build_google_flow", lambda: FakeFlow())
    monkeypatch.setattr(auth, "verify_google_id_token",
                        lambda jwt: {"email": "partner@example.com", "email_verified": True})

    r = client.get(f"/auth/google/callback?state={state}&code=anything")
    assert r.status_code == 302 and r.headers["Location"] == "/"
    # The session cookie now authenticates as that user.
    assert client.get("/generate").status_code == 200


def test_google_callback_refuses_an_unrecognized_email(client, tmp_path, monkeypatch):
    f = tmp_path / "google_oauth.json"
    f.write_text(json.dumps({"client_id": "cid", "client_secret": "secret"}))
    monkeypatch.setattr(auth, "GOOGLE_OAUTH_FILE", f)
    state = auth.new_oauth_state()

    class FakeCreds:
        id_token = "fake-jwt"

    class FakeFlow:
        credentials = FakeCreds()
        def fetch_token(self, code):
            pass

    monkeypatch.setattr(auth, "build_google_flow", lambda: FakeFlow())
    monkeypatch.setattr(auth, "verify_google_id_token",
                        lambda jwt: {"email": "stranger@example.com", "email_verified": True})

    r = client.get(f"/auth/google/callback?state={state}&code=anything")
    assert r.status_code == 302 and "/login" in r.headers["Location"]
    assert client.get("/generate").status_code == 401   # never signed in


def test_google_callback_refuses_unverified_email(client, tmp_path, monkeypatch):
    f = tmp_path / "google_oauth.json"
    f.write_text(json.dumps({"client_id": "cid", "client_secret": "secret"}))
    monkeypatch.setattr(auth, "GOOGLE_OAUTH_FILE", f)
    auth.add_user("googleuser", "partner", google_email="partner@example.com")
    state = auth.new_oauth_state()

    class FakeCreds:
        id_token = "fake-jwt"

    class FakeFlow:
        credentials = FakeCreds()
        def fetch_token(self, code):
            pass

    monkeypatch.setattr(auth, "build_google_flow", lambda: FakeFlow())
    monkeypatch.setattr(auth, "verify_google_id_token",
                        lambda jwt: {"email": "partner@example.com", "email_verified": False})

    r = client.get(f"/auth/google/callback?state={state}&code=anything")
    assert r.status_code == 302 and "/login" in r.headers["Location"]


# ── replacing a leaked link ──────────────────────────────────────────────────
#
# THE LINK IS THE PASSWORD, and passwords get out — pasted into a chat,
# screenshotted, read aloud out of a config file. So replacing one has to be
# possible, and until `rotate` existed it was not, for exactly the person whose
# link is worth the most.
#
# The documented path was revoke-then-add: `_cmd_add` said so in its own error
# text. That path runs straight into revoke_user's last-owner guard, which is
# correct and should stay — refusing to delete the only owner is what stops you
# locking yourself out. But it meant the owner's answer to "my link leaked" was
# to hand-edit config/users.json.
#
# Rotating in place never touches the guard: the owner does not stop being an
# owner for even one instant, so there is nothing to refuse.

def test_rotating_issues_a_new_token_and_kills_the_old_one(users_file):
    user = auth.rotate_token("dani")
    assert user is not None
    assert user["token"] != OWNER_TOKEN
    assert auth.user_for_token(OWNER_TOKEN) is None, "the leaked link still works"
    assert auth.user_for_token(user["token"])["name"] == "dani"


def test_rotating_keeps_the_name_and_the_role(users_file):
    """A rotation is not a demotion. If this dropped the role the owner would
    rotate a leaked link and lock themselves out of /settings — the same
    failure the last-owner guard exists to prevent, arrived at sideways."""
    user = auth.rotate_token("dani")
    assert user["name"] == "dani"
    assert user["role"] == "owner"


def test_the_last_owner_can_rotate_even_though_they_cannot_be_revoked(users_file):
    """The whole reason this exists. Both statements must hold at once."""
    assert auth.revoke_user("dani") == "last_owner"
    assert auth.rotate_token("dani") is not None


def test_rotating_leaves_everyone_else_alone(users_file):
    auth.rotate_token("dani")
    assert auth.user_for_token(PARTNER_TOKEN)["name"] == "james"
    assert auth.user_for_token(VIEWER_TOKEN)["name"] == "guest"


def test_rotating_an_unknown_name_is_none_not_a_new_user(users_file):
    assert auth.rotate_token("nobody") is None
    assert len(auth._load_users()) == 3


def test_the_cli_offers_rotate(users_file, capsys):
    assert auth.main(["rotate", "dani"]) == 0
    out = capsys.readouterr().out
    assert "Sign-in link:" in out
    assert OWNER_TOKEN not in out
    auth.main(["nonsense"])
    assert "rotate" in capsys.readouterr().out, "an undiscoverable command"


# ── the setting that mailed the owner's token to a chat channel ──────────────
#
# The leak had a mechanism, and it was a UI one. auth.py prints a sign-in link
# and says to save it. The dashboard offers a text setting called "Dashboard
# URL" described as "where this dashboard is reachable from your phone". The
# saved link is exactly that, so it gets pasted in — with ?token=<owner> on it.
#
# Nothing stripped it, so notify.py deep-linked every Discord and ntfy alert
# with the owner's credential several times a day, and auth.py appended a
# SECOND ?token= when printing a rotated link, producing a URL that carried the
# old token and did not work.

def test_a_token_in_the_dashboard_url_is_stripped(monkeypatch, capsys):
    monkeypatch.setenv("RUFUS_DASHBOARD_URL",
                       "https://rufus.tail635959.ts.net/?token=leaked")
    auth._URL_CREDENTIAL_WARNED = False
    assert auth._base_url() == "https://rufus.tail635959.ts.net"
    said = capsys.readouterr().out
    assert "TOKEN" in said
    assert "rotate" in said, "it has to say the leaked one is still live"


def test_a_rotated_link_is_not_two_tokens_deep(monkeypatch, users_file):
    """The bug as the owner would have met it: rotate, copy the printed link,
    and find it does not sign you in."""
    monkeypatch.setenv("RUFUS_DASHBOARD_URL",
                       "https://rufus.tail635959.ts.net/?token=old")
    auth._URL_CREDENTIAL_WARNED = False
    user = auth.rotate_token("dani")
    link = f"{auth._base_url()}/?token={user['token']}"
    assert link.count("token=") == 1, link
    assert "old" not in link


def test_a_clean_url_is_left_exactly_alone(monkeypatch, capsys):
    monkeypatch.setenv("RUFUS_DASHBOARD_URL", "https://rufus.tail635959.ts.net")
    auth._URL_CREDENTIAL_WARNED = False
    assert auth._base_url() == "https://rufus.tail635959.ts.net"
    assert capsys.readouterr().out == "", "a check that fires on a correct setup is noise"


def test_the_warning_is_said_once_not_per_call(monkeypatch, capsys):
    """notify calls this per notification. A per-call print would bury it."""
    monkeypatch.setenv("RUFUS_DASHBOARD_URL", "https://x.ts.net/?token=t")
    auth._URL_CREDENTIAL_WARNED = False
    for _ in range(5):
        auth._base_url()
    assert capsys.readouterr().out.count("RUFUS_DASHBOARD_URL") == 1


def test_notify_strips_it_the_same_way_auth_does(monkeypatch):
    """A HAND-COPY, because notify.py imports os/Path/requests and nothing
    else on purpose — importing auth would drag flask into every notification.
    Two copies of a rule need a test asserting they still agree, or the one
    that drifts is the one nobody notices is wrong."""
    import notify
    for raw in ("https://a.ts.net/?token=x", "https://a.ts.net#frag",
                "https://a.ts.net/", "https://a.ts.net", "", "   "):
        auth._URL_CREDENTIAL_WARNED = True      # silence both
        notify._URL_CREDENTIAL_WARNED = True
        assert (auth._base_url_without_credentials(raw, "auth")
                == notify._base_url_without_credentials(raw, "notify")), raw


# ── deleting a generated image ───────────────────────────────────────────────
#
# WHY DELETE IS NOT COSMETIC HERE. /make's "Pick a look" gallery drops a stored
# prompt into the topic field on click, so every image ever generated stays a
# live one-click suggestion for what the next video should be ABOUT. A test
# render or a bad idea kept offering itself forever and the only way to stop
# one was to reach the filesystem — which a partner on a phone cannot do, and
# the owner should not have to.
#
# The permission is owner-only even though `thumbnail` (make one) is shared
# with partner: making costs GPU seconds and is undone by making another,
# deleting is neither.

@pytest.fixture
def thumbs(tmp_path, monkeypatch):
    import paths
    d = tmp_path / "thumbs"
    d.mkdir()
    (d / "a.png").write_bytes(b"x" * 2048)
    (d / "a.txt").write_text("PROMPT: a cracked hourglass\nSEED: 1\n")
    monkeypatch.setattr(paths, "thumbnails_dir", lambda: d)
    return d


def test_owner_can_delete_a_generated_image(client, thumbs):
    client.get(f"/?token={OWNER_TOKEN}")
    r = client.post("/thumbnails/delete", data={"name": "a.png"})
    assert r.status_code == 302
    assert not (thumbs / "a.png").exists()


def test_the_prompt_sidecar_goes_with_the_picture(client, thumbs):
    """Leaving the .txt keeps the prompt in the gallery's metadata with no
    image to explain it — worse than having both or neither."""
    client.get(f"/?token={OWNER_TOKEN}")
    client.post("/thumbnails/delete", data={"name": "a.png"})
    assert not (thumbs / "a.txt").exists()


def test_a_partner_cannot_delete_the_owners_images(client, thumbs):
    client.get(f"/?token={PARTNER_TOKEN}")
    r = client.post("/thumbnails/delete", data={"name": "a.png"})
    assert r.status_code in (401, 403)
    assert (thumbs / "a.png").exists(), "a partner deleted the owner's file"


def test_deleting_something_that_is_not_in_the_listing_is_refused(client, thumbs):
    """The posted name reaches the filesystem, and unlike make-video this
    operation is destructive."""
    client.get(f"/?token={OWNER_TOKEN}")
    for attack in ("../../etc/passwd", "..\\..\\config\\users.json", "nope.png"):
        r = client.post("/thumbnails/delete", data={"name": attack})
        assert r.status_code == 302, attack
        assert "error" in r.headers["Location"], attack
    assert (thumbs / "a.png").exists()


def test_delete_is_an_owner_permission(users_file):
    assert "delete" in auth.ROLE_PERMISSIONS["owner"]
    assert "delete" not in auth.ROLE_PERMISSIONS["partner"]
    assert "delete" not in auth.ROLE_PERMISSIONS["viewer"]


# ── when a video was made, and when it went out ──────────────────────────────
#
# `upload_date` is date('now') — no time — and it carried two different
# meanings depending on how a video reached YouTube. The pipeline's uploader
# never touched it, so for those rows it was the day the video was GENERATED;
# mark_published overwrote it with today, so for a hand-published row it was
# the day it went LIVE. One column, two meanings, no way to tell which you
# were looking at, and never an hour.

def test_a_new_video_records_the_minute_it_was_made(client, tmp_path):
    vid = db_manager.save_video(niche="finance", script_hook="h", scene_desc="s",
                                video_file="a.mp4", score=8)
    row = next(r for r in db_manager.history() if r["id"] == vid)
    assert len(row["created_at"]) == len("2026-08-18 03:51:05")
    assert row["uploaded_at"] is None, "not uploaded yet — it must not claim a time"


def test_uploading_stamps_a_separate_time(client):
    vid = db_manager.save_video(niche="finance", script_hook="h", scene_desc="s",
                                video_file="a.mp4", score=8)
    db_manager.update_youtube_id(vid, "dQw4w9WgXcQ")
    row = next(r for r in db_manager.history() if r["id"] == vid)
    assert row["uploaded_at"], "the upload time was not recorded"
    assert row["created_at"], "and it must not have eaten the creation time"


def test_a_hand_published_video_keeps_both_times(client):
    """mark_published rewrites upload_date by design. It must not also
    overwrite created_at, or publishing a video retroactively changes when it
    was made."""
    vid = db_manager.save_video(niche="finance", script_hook="h", scene_desc="s",
                                video_file="a.mp4", score=8)
    made = next(r for r in db_manager.history() if r["id"] == vid)["created_at"]
    db_manager.mark_published(vid, "https://youtu.be/dQw4w9WgXcQ")
    row = next(r for r in db_manager.history() if r["id"] == vid)
    assert row["created_at"] == made
    assert row["uploaded_at"]
    assert row["youtube_id"] == "dQw4w9WgXcQ"


def test_history_is_newest_first_and_shows_what_is_still_waiting(client):
    a = db_manager.save_video(niche="f", script_hook="first", scene_desc="s",
                              video_file="a.mp4", score=8)
    b = db_manager.save_video(niche="f", script_hook="second", scene_desc="s",
                              video_file="b.mp4", score=8)
    ids = [r["id"] for r in db_manager.history()]
    assert ids.index(b) < ids.index(a), "newest must come first"
    pending = [r for r in db_manager.history() if r["uploaded_at"] is None]
    assert {a, b} <= {r["id"] for r in pending}


def test_the_history_page_renders(client):
    db_manager.save_video(niche="finance", script_hook="a hook", scene_desc="s",
                          video_file="a.mp4", score=8)
    client.get(f"/?token={OWNER_TOKEN}")
    r = client.get("/history")
    assert r.status_code == 200
    assert b"a hook" in r.data


def test_a_viewer_can_read_the_history(client):
    """It is a record, not a control. A partner asking when their video went
    out should not need the owner's link."""
    client.get(f"/?token={VIEWER_TOKEN}")
    assert client.get("/history").status_code == 200


# ── "the dashboard is up" ────────────────────────────────────────────────────
#
# watchdog.py already pings when /healthz stops answering, so a crash-restart
# is covered. Every other way this process starts was silent: a manual
# serve.ps1 -Restart, a reboot, the logon task. The owner asked to be told.
#
# The downtime is the part worth sending. "Back up after 6s" is a restart you
# did; "back up after 4h" is an afternoon nobody was watching, and only the
# starting process can know the difference — the watchdog's own alert cannot.

def _stamp(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, "START_STAMP", tmp_path / ".started")
    monkeypatch.delenv("RUFUS_DASHBOARD_NOTIFY", raising=False)
    sent = []
    import notify
    monkeypatch.setattr(notify, "send",
                        lambda title, body, **kw: sent.append((title, body)) or True)
    return sent


def test_the_first_start_says_it_is_the_first(tmp_path, monkeypatch):
    sent = _stamp(tmp_path, monkeypatch)
    assert dashboard._announce_start() is True
    assert "First start" in sent[0][1]


def test_the_second_start_reports_how_long_it_was_down(tmp_path, monkeypatch):
    import time as _t
    sent = _stamp(tmp_path, monkeypatch)
    (tmp_path / ".started").write_text(str(_t.time() - 7200), encoding="utf-8")
    dashboard._announce_start()
    assert "2h ago" in sent[0][1], sent[0][1]


def test_the_stamp_is_rewritten_so_the_next_gap_is_measured_from_now(tmp_path, monkeypatch):
    import time as _t
    _stamp(tmp_path, monkeypatch)
    dashboard._announce_start()
    written = float((tmp_path / ".started").read_text(encoding="utf-8"))
    assert abs(written - _t.time()) < 5


def test_a_clock_that_went_backwards_does_not_invent_a_downtime(tmp_path, monkeypatch):
    """NTP corrections and resumed VMs move the clock. "up 0s ago" would be a
    lie told with a straight face."""
    import time as _t
    sent = _stamp(tmp_path, monkeypatch)
    (tmp_path / ".started").write_text(str(_t.time() + 9999), encoding="utf-8")
    dashboard._announce_start()
    assert "clock moved" in sent[0][1]
    assert "Downtime unknown" in sent[0][1]


def test_a_corrupt_stamp_is_treated_as_no_stamp(tmp_path, monkeypatch):
    sent = _stamp(tmp_path, monkeypatch)
    (tmp_path / ".started").write_text("not a number", encoding="utf-8")
    assert dashboard._announce_start() is True
    assert "First start" in sent[0][1]


def test_it_can_be_switched_off(tmp_path, monkeypatch):
    sent = _stamp(tmp_path, monkeypatch)
    monkeypatch.setenv("RUFUS_DASHBOARD_NOTIFY", "0")
    assert dashboard._announce_start() is False
    assert sent == []


def test_a_broken_notifier_does_not_stop_the_dashboard_starting(tmp_path, monkeypatch, capsys):
    """This runs on the path to app.run(). A raise here is a dashboard that
    does not come up because the phone ping failed."""
    monkeypatch.setattr(dashboard, "START_STAMP", tmp_path / ".started")
    monkeypatch.delenv("RUFUS_DASHBOARD_NOTIFY", raising=False)
    import notify
    def _boom(*a, **k):
        raise RuntimeError("no network")
    monkeypatch.setattr(notify, "send", _boom)
    assert dashboard._announce_start() is False
    assert "start notification failed" in capsys.readouterr().out


def test_an_unwritable_stamp_still_sends(tmp_path, monkeypatch):
    """A read-only logs dir is a reason to lose the downtime number, not a
    reason to lose the alert."""
    sent = _stamp(tmp_path, monkeypatch)
    monkeypatch.setattr(dashboard, "START_STAMP",
                        tmp_path / "nope" / "deeper" / ".started")
    monkeypatch.setattr(dashboard.Path, "mkdir",
                        lambda self, **kw: (_ for _ in ()).throw(OSError("ro")))
    assert dashboard._announce_start() is True
    assert sent


@pytest.mark.parametrize("secs,want", [
    (5, "5s"), (89, "89s"), (90, "1m"), (3600, "60m"),
    (5400, "1h"), (172800, "2d"),
])
def test_the_gap_reads_like_a_person_wrote_it(secs, want):
    assert dashboard._human_gap(secs) == want


# ── the time on the front page, not only in History ──────────────────────────
#
# History is the record you go and look at. The front page is the one that is
# already open, and "which of these ran this morning" was unanswerable there:
# it printed upload_date, a bare day, for every row.
#
# Both pages format the stamp through _when_cell for the ordinary reason —
# two hand-copies of the same formatting is how one of them ends up showing
# seconds, or a padded midnight, after somebody edits the other.

def test_the_front_page_row_carries_the_time(client):
    db_manager.save_video(niche="finance", script_hook="a fresh hook",
                          scene_desc="s", video_file="a.mp4", score=8)
    client.get(f"/?token={OWNER_TOKEN}")
    page = client.get("/").data.decode("utf-8", "replace")
    row = db_manager.history()[0]
    assert row["created_at"][:10] in page
    assert row["created_at"][11:16] in page, "the hour and minute never rendered"


def test_both_pages_format_a_stamp_identically():
    stamp = "2026-08-18 03:51:05"
    assert "2026-08-18" in dashboard._when_cell(stamp)
    assert "03:51" in dashboard._when_cell(stamp)
    assert "05" not in dashboard._when_cell(stamp).split("03:51")[1], \
        "seconds are noise in a table"


def test_a_row_with_no_time_says_so_rather_than_showing_midnight(client):
    """The 19 videos published before created_at existed genuinely have no
    hour. Padding them to 00:00 would read as 'uploaded at midnight'."""
    out = dashboard._when_cell("2026-08-15")
    assert "no time" in out
    assert "00:00" not in out


def test_an_empty_stamp_is_a_dash_not_a_crash():
    for empty in (None, "", "   "):
        assert "&mdash;" in dashboard._when_cell(empty) or \
               "no time" in dashboard._when_cell(empty), repr(empty)


# ── the grouped nav still hides what a partner cannot use ────────────────────
#
# Sixteen flat links became four groups. Hiding a link was always cosmetic —
# every route enforces its own permission besides — but a group HEADING over
# an empty column is worse than either: it advertises a page and then refuses
# to show it.

def test_a_partner_sees_no_link_they_cannot_use(client):
    """The grouping must not smuggle a link back in. A partner keeps Logs,
    which is why the System group survives for them — but not Settings, the
    bench, or System itself."""
    import re
    client.get(f"/?token={PARTNER_TOKEN}")
    header = client.get("/").data.decode("utf-8", "replace") \
        .split("<header>", 1)[1].split("</header>", 1)[0]
    hrefs = set(re.findall(r'navlink" href="([^"]+)"', header))
    assert {"/settings", "/system", "/bench", "/styles"} & hrefs == set()
    assert "/gallery" in hrefs


def test_an_owner_still_reaches_every_registered_page(client):
    """The grouping must not lose a page on the way. A route that exists and
    is linked from nowhere is worse than one that was never written."""
    client.get(f"/?token={OWNER_TOKEN}")
    page = client.get("/").data.decode("utf-8", "replace")
    for href, _label, _perm in dashboard.NAV_ITEMS:
        assert f'href="{href}"' in page, href


def test_an_empty_group_is_omitted_rather_than_rendered_bare(client):
    """A heading over an empty column advertises pages and then refuses them.

    A VIEWER is the case that empties one: Make holds generate, thumbnail and
    settings permissions and a viewer has none of the three. A partner is not
    the test — they keep `view`, so System still contains Logs.
    """
    import re
    client.get(f"/?token={VIEWER_TOKEN}")
    header = client.get("/").data.decode("utf-8", "replace") \
        .split("<header>", 1)[1].split("</header>", 1)[0]
    groups = re.findall(r'navgroup-t">([^<]+)<', header)
    assert "Make" not in groups
    assert "Review" in groups, "a viewer can still review"
