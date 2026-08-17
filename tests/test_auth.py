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
