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
