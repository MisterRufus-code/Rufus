"""Tests for live run status: the progress file, /api/status, and the
gallery-to-video flow.

The load-bearing property throughout is that observability must never be able
to break a render — run_progress is called from inside the pipeline, so every
one of its functions has to swallow its own failures.
"""

import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import auth
import dashboard
import db_manager
import paths
import run_progress

OWNER_TOKEN   = "owner-token-live"
PARTNER_TOKEN = "partner-token-live"
VIEWER_TOKEN  = "viewer-token-live"


@pytest.fixture
def progress_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "log_dir", lambda: tmp_path)
    run_progress._current = {}          # each test starts with no in-flight run
    return tmp_path / "progress"


@pytest.fixture
def users_file(tmp_path, monkeypatch):
    f = tmp_path / "users.json"
    f.write_text(json.dumps({"users": [
        {"name": "dani",  "role": "owner",   "token": OWNER_TOKEN},
        {"name": "james", "role": "partner", "token": PARTNER_TOKEN},
        {"name": "guest", "role": "viewer",  "token": VIEWER_TOKEN},
    ]}), encoding="utf-8")
    monkeypatch.setattr(auth, "USERS_FILE", f)
    monkeypatch.delenv("RUFUS_AUTH_DISABLED", raising=False)
    return f


@pytest.fixture
def client(tmp_path, monkeypatch, users_file, progress_dir):
    monkeypatch.setattr(db_manager, "DB_FILE", tmp_path / "test.db")
    db_manager.init_db()
    monkeypatch.setattr(dashboard, "DEBUG_ROOT", tmp_path / "debug")
    monkeypatch.setattr(dashboard, "_comfyui_reachable", lambda: True)
    dashboard.app.config["TESTING"] = True
    with dashboard.app.test_client() as c:
        yield c


# ── run_progress: the file itself ────────────────────────────────────────────

def test_read_returns_none_before_any_run(progress_dir):
    assert run_progress.read("main_en") is None


def test_read_all_is_empty_before_any_run(progress_dir):
    assert run_progress.read_all() == []


def test_begin_then_read_round_trips(progress_dir):
    run_progress.begin("main_en", run_id="r1", niche="finance", topic="Bretton Woods")
    got = run_progress.read("main_en")
    assert got["channel"] == "main_en"
    assert got["niche"] == "finance"
    assert got["topic"] == "Bretton Woods"
    assert got["status"] == "running"
    assert got["step"] == 0


def test_update_advances_the_step(progress_dir):
    run_progress.begin("main_en")
    run_progress.update(4, "writing the script")
    got = run_progress.read("main_en")
    assert got["step"] == 4 and got["label"] == "writing the script"
    assert got["total"] == run_progress.TOTAL_STEPS


def test_finish_marks_the_run_done(progress_dir):
    run_progress.begin("main_en")
    run_progress.update(7, "queued for review")
    run_progress.finish("done")
    got = run_progress.read("main_en")
    assert got["status"] == "done" and "ended_at" in got


def test_finish_records_a_failure_reason(progress_dir):
    run_progress.begin("main_en")
    run_progress.finish("failed", "ComfyUI went away")
    got = run_progress.read("main_en")
    assert got["status"] == "failed" and "ComfyUI" in got["detail"]


def test_update_without_begin_is_a_noop_not_a_crash(progress_dir):
    """An odd entry point that never called begin() must not take the
    pipeline down on a status write."""
    run_progress._current = {}
    run_progress.update(3, "whatever")      # must not raise
    assert run_progress.read("main_en") is None


def test_finish_without_begin_is_a_noop(progress_dir):
    run_progress._current = {}
    run_progress.finish("failed", "x")      # called from a finally block
    assert run_progress.read("main_en") is None


def test_progress_never_raises_when_the_directory_is_unwritable(monkeypatch):
    """The contract: a video must never fail because a status file couldn't
    be written."""
    monkeypatch.setattr(paths, "log_dir",
                        lambda: (_ for _ in ()).throw(OSError("disk gone")))
    run_progress.begin("main_en")           # must not raise
    run_progress.update(1, "x")
    run_progress.finish("done")


def test_corrupt_progress_file_reads_as_none(progress_dir):
    progress_dir.mkdir(parents=True, exist_ok=True)
    (progress_dir / "main_en.json").write_text("{ not json", encoding="utf-8")
    assert run_progress.read("main_en") is None


def test_channel_id_cannot_escape_the_progress_directory(progress_dir):
    """The id builds a filesystem path, so path separators are neutralised
    rather than trusted."""
    run_progress.begin("../../etc/passwd")
    written = list(progress_dir.glob("*.json"))
    assert len(written) == 1
    assert ".." not in written[0].name and "/" not in written[0].name


def test_a_quiet_running_file_is_reported_stale(progress_dir):
    run_progress.begin("main_en")
    run_progress.update(5, "rendering")
    # Rewind updated_at past the staleness window: the process died without
    # ever calling finish().
    p = progress_dir / "main_en.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    data["updated_at"] = time.time() - (run_progress.STALE_AFTER_SECONDS + 60)
    p.write_text(json.dumps(data), encoding="utf-8")
    assert run_progress.read("main_en")["stale"] is True


def test_a_finished_run_is_never_stale(progress_dir):
    run_progress.begin("main_en")
    run_progress.finish("done")
    p = progress_dir / "main_en.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    data["updated_at"] = time.time() - (run_progress.STALE_AFTER_SECONDS + 60)
    p.write_text(json.dumps(data), encoding="utf-8")
    assert run_progress.read("main_en")["stale"] is False


def test_read_all_sorts_newest_activity_first(progress_dir):
    run_progress.begin("channel_a")
    run_progress.finish("done")
    time.sleep(0.01)
    run_progress.begin("channel_b")
    names = [r["channel"] for r in run_progress.read_all()]
    assert names[0] == "channel_b"


def test_clear_forgets_a_channel(progress_dir):
    run_progress.begin("main_en")
    run_progress.clear("main_en")
    assert run_progress.read("main_en") is None


# ── /api/status ──────────────────────────────────────────────────────────────

def test_status_requires_authentication(client):
    """Run topics and niches are not public."""
    assert client.get("/api/status").status_code == 401


def test_viewer_can_read_status(client):
    client.get(f"/?token={VIEWER_TOKEN}")
    assert client.get("/api/status").status_code == 200


def test_status_reports_the_machine_is_up(client):
    client.get(f"/?token={OWNER_TOKEN}")
    d = client.get("/api/status").get_json()
    assert d["ok"] is True
    assert d["uptime_seconds"] >= 0
    assert d["comfyui"] is True


def test_status_reports_idle_when_nothing_is_running(client, monkeypatch):
    monkeypatch.setattr(dashboard, "_run_in_progress", lambda cid: False)
    client.get(f"/?token={OWNER_TOKEN}")
    d = client.get("/api/status").get_json()
    assert d["busy"] is False
    assert all(r["running"] is False for r in d["runs"])


def test_status_reports_the_current_step_of_a_live_run(client, monkeypatch):
    monkeypatch.setattr(dashboard, "_channels", lambda: ["main_en"])
    monkeypatch.setattr(dashboard, "_run_in_progress", lambda cid: True)
    run_progress.begin("main_en", niche="finance")
    run_progress.update(4, "writing the script")

    client.get(f"/?token={OWNER_TOKEN}")
    d = client.get("/api/status").get_json()
    assert d["busy"] is True
    run = d["runs"][0]
    assert run["step"] == 4 and run["label"] == "writing the script"
    assert run["total"] == run_progress.TOTAL_STEPS


def test_status_flags_a_held_lock_whose_progress_went_quiet(client, monkeypatch, progress_dir):
    """Lock held but nothing writing = the run almost certainly died. Worth
    surfacing rather than showing a progress bar frozen at step 5 forever."""
    monkeypatch.setattr(dashboard, "_channels", lambda: ["main_en"])
    monkeypatch.setattr(dashboard, "_run_in_progress", lambda cid: True)
    run_progress.begin("main_en")
    run_progress.update(5, "rendering")
    p = progress_dir / "main_en.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    data["updated_at"] = time.time() - (run_progress.STALE_AFTER_SECONDS + 60)
    p.write_text(json.dumps(data), encoding="utf-8")

    client.get(f"/?token={OWNER_TOKEN}")
    assert client.get("/api/status").get_json()["runs"][0]["stale"] is True


def test_status_includes_the_review_queue_count(client):
    db_manager.save_video(niche="finance", script_hook="Pending one", scene_desc="s",
                          video_file="a.mp4", score=9, run_id="r1",
                          upload_status="pending")
    client.get(f"/?token={OWNER_TOKEN}")
    assert client.get("/api/status").get_json()["queue"]["pending"] == 1


def test_status_survives_a_missing_progress_file(client, monkeypatch):
    """A run started before this feature existed has no progress file; the
    lock still says it's running and the API must not break."""
    monkeypatch.setattr(dashboard, "_channels", lambda: ["main_en"])
    monkeypatch.setattr(dashboard, "_run_in_progress", lambda cid: True)
    client.get(f"/?token={OWNER_TOKEN}")
    run = client.get("/api/status").get_json()["runs"][0]
    assert run["running"] is True and run["step"] == 0


def test_every_page_carries_the_live_bar(client):
    client.get(f"/?token={OWNER_TOKEN}")
    html = client.get("/").get_data(as_text=True)
    assert 'id="livebar"' in html and "/api/status" in html


# ── Queue previews ───────────────────────────────────────────────────────────

def test_pending_queue_shows_keyframe_previews(client, tmp_path):
    run_dir = tmp_path / "debug" / "run_prev"
    run_dir.mkdir(parents=True)
    (run_dir / "01.png").write_bytes(b"\x89PNG")
    (run_dir / "02.png").write_bytes(b"\x89PNG")
    db_manager.save_video(niche="finance", script_hook="Needs review", scene_desc="s",
                          video_file="a.mp4", score=9, run_id="run_prev",
                          upload_status="pending")
    client.get(f"/?token={OWNER_TOKEN}")
    # The queue moved to /review in the minimal home rebuild. The previews
    # exist for the same reason as ever: approving a video is a judgement
    # about how it LOOKS, and making that call otherwise means opening every
    # row one at a time.
    html = client.get("/review").get_data(as_text=True)
    assert "/debug/run_prev/01.png" in html


def test_run_keyframes_is_empty_for_an_unknown_run(client):
    assert dashboard._run_keyframes("no-such-run") == []
    assert dashboard._run_keyframes(None) == []


def test_run_keyframes_is_capped(client, tmp_path):
    run_dir = tmp_path / "debug" / "many"
    run_dir.mkdir(parents=True)
    for i in range(9):
        (run_dir / f"0{i}.png").write_bytes(b"x")
    assert len(dashboard._run_keyframes("many", limit=4)) == 4


# ── Gallery → video ──────────────────────────────────────────────────────────

def _seed_gallery_image(monkeypatch, prompt="a cracked hourglass spilling coins"):
    import image_gen
    monkeypatch.setattr(image_gen, "recent_images",
                        lambda limit=40: [{"name": "a.png", "prompt": prompt,
                                           "mtime": 0, "kb": 10}])


def test_make_video_from_an_image_starts_a_run_with_its_prompt(client, monkeypatch):
    _seed_gallery_image(monkeypatch)
    launched = []
    monkeypatch.setattr(dashboard, "_launch_run",
                        lambda **k: launched.append(k) or (None, Path("x.log")))
    monkeypatch.setattr(dashboard, "_run_in_progress", lambda cid: False)

    client.get(f"/?token={OWNER_TOKEN}")
    r = client.post("/thumbnails/make-video", data={"name": "a.png"})
    assert r.status_code == 302
    assert launched and launched[0]["topic"] == "a cracked hourglass spilling coins"


def test_partner_may_make_a_video_from_an_image(client, monkeypatch):
    _seed_gallery_image(monkeypatch)
    launched = []
    monkeypatch.setattr(dashboard, "_launch_run",
                        lambda **k: launched.append(k) or (None, Path("x.log")))
    monkeypatch.setattr(dashboard, "_run_in_progress", lambda cid: False)
    client.get(f"/?token={PARTNER_TOKEN}")
    assert client.post("/thumbnails/make-video", data={"name": "a.png"}).status_code == 302
    assert launched


def test_viewer_may_not_make_a_video_from_an_image(client, monkeypatch):
    _seed_gallery_image(monkeypatch)
    launched = []
    monkeypatch.setattr(dashboard, "_launch_run",
                        lambda **k: launched.append(k) or (None, Path("x.log")))
    client.get(f"/?token={VIEWER_TOKEN}")
    assert client.post("/thumbnails/make-video", data={"name": "a.png"}).status_code == 403
    assert launched == []


def test_make_video_rejects_an_unknown_image(client, monkeypatch):
    _seed_gallery_image(monkeypatch)
    launched = []
    monkeypatch.setattr(dashboard, "_launch_run",
                        lambda **k: launched.append(k) or (None, Path("x.log")))
    client.get(f"/?token={OWNER_TOKEN}")
    r = client.post("/thumbnails/make-video", data={"name": "../../secret.png"})
    assert "error" in r.headers["Location"]
    assert launched == [], "a posted filename reached the launcher unvalidated"


def test_make_video_refuses_when_a_run_is_already_going(client, monkeypatch):
    _seed_gallery_image(monkeypatch)
    launched = []
    monkeypatch.setattr(dashboard, "_launch_run",
                        lambda **k: launched.append(k) or (None, Path("x.log")))
    monkeypatch.setattr(dashboard, "_channels", lambda: ["main_en"])
    monkeypatch.setattr(dashboard, "_run_in_progress", lambda cid: True)
    client.get(f"/?token={OWNER_TOKEN}")
    r = client.post("/thumbnails/make-video", data={"name": "a.png"})
    assert "error" in r.headers["Location"]
    assert launched == []


def test_make_video_refuses_an_image_with_no_saved_prompt(client, monkeypatch):
    _seed_gallery_image(monkeypatch, prompt="")
    launched = []
    monkeypatch.setattr(dashboard, "_launch_run",
                        lambda **k: launched.append(k) or (None, Path("x.log")))
    monkeypatch.setattr(dashboard, "_run_in_progress", lambda cid: False)
    client.get(f"/?token={OWNER_TOKEN}")
    r = client.post("/thumbnails/make-video", data={"name": "a.png"})
    assert "error" in r.headers["Location"]
    assert launched == []


# ── /generate upgrade: niche dropdown + embedded "pick a look" gallery ──────
# Per clarified intent: a collaborator with dashboard access still couldn't
# build a genuinely CUSTOM video (a specific niche, a specific visual style)
# — /generate had a free-text niche field (silently no-ops on a typo) and no
# way to pick a look without leaving the page. This is the fix, not an
# approve/publish permission change — that boundary is untouched.

def test_generate_page_lists_real_niches_in_a_dropdown(client):
    client.get(f"/?token={OWNER_TOKEN}")
    html = client.get("/generate").get_data(as_text=True)
    assert '<select class="field" id="gen-niche"' in html
    # At least one real niche id from config/niches.json must be offered.
    import dashboard
    niches = dashboard._available_niches()
    assert niches, "config/niches.json produced no niches — check the fixture repo state"
    assert f'value="{niches[0]}"' in html


def test_available_niches_handles_a_missing_file(monkeypatch, tmp_path):
    import dashboard
    monkeypatch.setattr(dashboard, "ROOT", tmp_path)   # no config/niches.json here
    assert dashboard._available_niches() == []


def test_generate_page_shows_pick_a_look_gallery(client, monkeypatch):
    import image_gen
    monkeypatch.setattr(image_gen, "recent_images",
                        lambda limit=40: [{"name": "a.png", "prompt": "a cracked hourglass",
                                           "mtime": 0, "kb": 10}])
    client.get(f"/?token={OWNER_TOKEN}")
    html = client.get("/generate").get_data(as_text=True)
    assert "pick-look" in html
    assert "a cracked hourglass" in html
    assert "/thumbnails/file/a.png" in html


def test_generate_gallery_skips_images_with_no_saved_prompt(client, monkeypatch):
    """An image with no prompt has nothing to seed a topic with — showing it
    here would just be a dead click."""
    import image_gen
    monkeypatch.setattr(image_gen, "recent_images",
                        lambda limit=40: [{"name": "a.png", "prompt": "",
                                           "mtime": 0, "kb": 10}])
    client.get(f"/?token={OWNER_TOKEN}")
    html = client.get("/generate").get_data(as_text=True)
    assert "pick-look" not in html


def test_generate_gallery_prompt_is_html_escaped_in_the_data_attribute(client, monkeypatch):
    """Regression guard for the actual bug caught while building this: a
    prompt containing an apostrophe must not break out of the data-prompt
    attribute or (worse) close an inline JS string early. html.escape() on
    the attribute value, read back via .dataset in JS — never concatenated
    into a JS string literal in the markup at all."""
    import image_gen
    monkeypatch.setattr(image_gen, "recent_images",
                        lambda limit=40: [{"name": "a.png",
                                           "prompt": "it's <b>a</b> test \"quote\"",
                                           "mtime": 0, "kb": 10}])
    client.get(f"/?token={OWNER_TOKEN}")
    html = client.get("/generate").get_data(as_text=True)
    # The raw prompt must never appear unescaped inside the attribute.
    assert 'data-prompt="it\'s' not in html
    assert "&#39;" in html or "&#x27;" in html   # apostrophe escaped
    assert "&lt;b&gt;" in html                    # angle brackets escaped
    assert "&quot;" in html                       # double quote escaped
    # No inline onclick string literal was built from the prompt at all.
    assert "onclick=" not in html


def test_generate_page_still_requires_generate_permission(client):
    """The actual bug found while building this: the @app.route decorator
    landed on the wrong function (a helper, not generate_page) after a
    refactor, so a viewer got 200 instead of 403. Locks in the fix."""
    client.get(f"/?token={VIEWER_TOKEN}")
    assert client.get("/generate").status_code == 403
