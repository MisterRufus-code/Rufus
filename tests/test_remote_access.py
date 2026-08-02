"""Tests for the remote-server additions: image generation, the Discord
notify backend, and the analytics digest.

Nothing here touches the network or a GPU — ComfyUI, the webhook and the
YouTube client are all stubbed. What's asserted is the wiring and the
fail-open contract: every one of these is an optional layer, and none of them
may turn a working render into an error when it's unconfigured or broken.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import image_gen
import notify


# ── image_gen ────────────────────────────────────────────────────────────────

def test_landscape_default_is_youtubes_thumbnail_shape():
    assert (image_gen.THUMB_W, image_gen.THUMB_H) == (1280, 720)


def test_apply_image_dims_sets_width_and_height():
    graph = {"1": {"class_type": "EmptyLatentImage",
                   "inputs": {"width": 1080, "height": 1920}}}
    image_gen._apply_image_dims(graph, 1280, 720)
    assert graph["1"]["inputs"] == {"width": 1280, "height": 720}


def test_apply_image_dims_skips_video_nodes():
    """A node carrying `length`/`duration` is sizing a CLIP, not a still —
    resizing it here would quietly reshape a video graph."""
    graph = {
        "1": {"class_type": "EmptyLatentImage", "inputs": {"width": 1080, "height": 1920}},
        "2": {"class_type": "EmptyHunyuanLatentVideo",
              "inputs": {"width": 720, "height": 1280, "length": 49}},
        "3": {"class_type": "LTXV", "inputs": {"width": 704, "height": 1216, "duration": 5}},
    }
    image_gen._apply_image_dims(graph, 1280, 720)
    assert graph["1"]["inputs"]["width"] == 1280
    assert graph["2"]["inputs"] == {"width": 720, "height": 1280, "length": 49}
    assert graph["3"]["inputs"] == {"width": 704, "height": 1216, "duration": 5}


def test_apply_image_dims_ignores_malformed_nodes():
    graph = {"1": {"class_type": "X", "inputs": None}, "2": {"no_inputs": True}}
    image_gen._apply_image_dims(graph, 100, 100)   # must not raise


def test_slugify_makes_a_readable_filename():
    assert image_gen._slugify("A cracked hourglass, spilling coins!") \
        .startswith("a_cracked_hourglass")


def test_slugify_survives_a_prompt_with_no_usable_characters():
    assert image_gen._slugify("!!!???") == "image"


def test_generate_image_returns_none_when_comfy_is_down(monkeypatch):
    """A Flask route calls this — it must report failure, never raise."""
    import comfy_client
    monkeypatch.setattr(comfy_client, "is_available", lambda: False)
    assert image_gen.generate_image("anything") is None


def test_generate_image_returns_none_without_a_stills_template(monkeypatch):
    import comfy_client
    monkeypatch.setattr(comfy_client, "is_available", lambda: True)
    monkeypatch.setattr(comfy_client, "_stills_template", lambda: None)
    assert image_gen.generate_image("anything") is None


def test_generate_image_writes_png_and_prompt_sidecar(tmp_path, monkeypatch):
    import comfy_client
    monkeypatch.setattr(comfy_client, "is_available", lambda: True)
    monkeypatch.setattr(comfy_client, "_stills_template",
                        lambda: {"1": {"class_type": "EmptyLatentImage",
                                       "inputs": {"width": 1, "height": 1,
                                                  "text": "RUFUS_PROMPT"}}})
    monkeypatch.setattr(comfy_client, "_submit", lambda g, c: "pid-1")
    monkeypatch.setattr(comfy_client, "_await_image", lambda pid: b"\x89PNG-bytes")

    out = tmp_path / "thumb.png"
    result = image_gen.generate_image("gold coins", out, seed=7)

    assert result == out and out.read_bytes() == b"\x89PNG-bytes"
    sidecar = out.with_suffix(".txt").read_text(encoding="utf-8")
    assert "PROMPT: gold coins" in sidecar and "SEED: 7" in sidecar


def test_generate_image_returns_none_when_render_produces_nothing(tmp_path, monkeypatch):
    import comfy_client
    monkeypatch.setattr(comfy_client, "is_available", lambda: True)
    monkeypatch.setattr(comfy_client, "_stills_template",
                        lambda: {"1": {"class_type": "X", "inputs": {"t": "RUFUS_PROMPT"}}})
    monkeypatch.setattr(comfy_client, "_submit", lambda g, c: "pid-1")
    monkeypatch.setattr(comfy_client, "_await_image", lambda pid: None)
    assert image_gen.generate_image("x", tmp_path / "a.png") is None


def test_recent_images_is_empty_when_nothing_generated(tmp_path, monkeypatch):
    import paths
    monkeypatch.setattr(paths, "thumbnails_dir", lambda: tmp_path / "nope")
    assert image_gen.recent_images() == []


def test_recent_images_reads_the_prompt_sidecar(tmp_path, monkeypatch):
    import paths
    monkeypatch.setattr(paths, "thumbnails_dir", lambda: tmp_path)
    (tmp_path / "a.png").write_bytes(b"x" * 2048)
    (tmp_path / "a.txt").write_text("PROMPT: a golden hourglass\nSEED: 1\n")
    got = image_gen.recent_images()
    assert len(got) == 1 and got[0]["prompt"] == "a golden hourglass"


# ── Discord backend ──────────────────────────────────────────────────────────

def test_discord_not_configured_by_default(monkeypatch):
    monkeypatch.delenv("RUFUS_DISCORD_WEBHOOK", raising=False)
    assert "discord" not in notify.configured()


def test_discord_appears_once_a_webhook_is_set(monkeypatch):
    monkeypatch.setenv("RUFUS_DISCORD_WEBHOOK", "https://discord.com/api/webhooks/x/y")
    assert "discord" in notify.configured()


def test_send_discord_posts_an_embed(monkeypatch):
    sent = {}

    def fake_post(url, **kw):
        sent["url"] = url
        sent["json"] = kw.get("json")

        class R:
            status_code = 204
        return R()

    monkeypatch.setenv("RUFUS_DISCORD_WEBHOOK", "https://discord.com/api/webhooks/x/y")
    monkeypatch.setattr(notify.requests, "post", fake_post)
    assert notify._send_discord("Title", "Body", "http://dash", "high") is True
    assert sent["json"]["embeds"][0]["title"] == "Title"
    assert sent["json"]["embeds"][0]["color"] == 0xEF4444   # high priority = red


def test_send_file_is_a_noop_without_discord(monkeypatch, tmp_path):
    monkeypatch.delenv("RUFUS_DISCORD_WEBHOOK", raising=False)
    f = tmp_path / "v.mp4"
    f.write_bytes(b"x")
    assert notify.send_file(f) is False


def test_send_file_skips_a_missing_file(monkeypatch, tmp_path):
    monkeypatch.setenv("RUFUS_DISCORD_WEBHOOK", "https://discord.com/api/webhooks/x/y")
    assert notify.send_file(tmp_path / "gone.mp4") is False


def test_send_file_uploads_when_small_enough(monkeypatch, tmp_path):
    posted = {}

    def fake_post(url, **kw):
        posted["files"] = kw.get("files")

        class R:
            status_code = 200
        return R()

    monkeypatch.setenv("RUFUS_DISCORD_WEBHOOK", "https://discord.com/api/webhooks/x/y")
    monkeypatch.setattr(notify.requests, "post", fake_post)
    f = tmp_path / "v.mp4"
    f.write_bytes(b"x" * 1024)
    assert notify.send_file(f, caption="hi") is True
    assert posted["files"]["file"][0] == "v.mp4"


def test_send_file_links_instead_of_uploading_an_oversized_file(monkeypatch, tmp_path):
    """Discord rejects an over-limit upload only AFTER the whole body is sent,
    so a big Short must not be pushed up the wire just to be refused."""
    calls = []
    monkeypatch.setenv("RUFUS_DISCORD_WEBHOOK", "https://discord.com/api/webhooks/x/y")
    monkeypatch.setattr(notify, "_send_discord",
                        lambda *a, **k: calls.append(a) or True)
    big = tmp_path / "big.mp4"
    big.write_bytes(b"0" * (notify.DISCORD_MAX_UPLOAD_BYTES + 1))
    assert notify.send_file(big, caption="big one") is True
    assert calls, "should have fallen back to posting a link"


def test_send_file_respects_the_upload_kill_switch(monkeypatch, tmp_path):
    monkeypatch.setenv("RUFUS_DISCORD_WEBHOOK", "https://discord.com/api/webhooks/x/y")
    monkeypatch.setenv("RUFUS_DISCORD_UPLOAD", "0")
    f = tmp_path / "v.mp4"
    f.write_bytes(b"x")
    assert notify.send_file(f) is False


def test_send_file_never_raises_when_the_post_blows_up(monkeypatch, tmp_path):
    def boom(*a, **k):
        raise RuntimeError("network gone")

    monkeypatch.setenv("RUFUS_DISCORD_WEBHOOK", "https://discord.com/api/webhooks/x/y")
    monkeypatch.setattr(notify.requests, "post", boom)
    f = tmp_path / "v.mp4"
    f.write_bytes(b"x")
    assert notify.send_file(f) is False


def test_notify_published_survives_a_dead_backend(monkeypatch):
    monkeypatch.delenv("RUFUS_DISCORD_WEBHOOK", raising=False)
    monkeypatch.delenv("RUFUS_NTFY_TOPIC", raising=False)
    monkeypatch.delenv("RUFUS_PUSHOVER_TOKEN", raising=False)
    monkeypatch.delenv("RUFUS_TELEGRAM_TOKEN", raising=False)
    assert notify.notify_published(title="A video", youtube_id="abc") is False


# ── analytics digest ─────────────────────────────────────────────────────────

def test_digest_summarizes_totals_and_top_videos(monkeypatch):
    import analytics_fetcher
    captured = {}
    monkeypatch.setattr(notify, "notify_analytics",
                        lambda summary, rows=0: captured.update(summary=summary, rows=rows) or True)
    analytics_fetcher._post_digest([
        {"youtube_id": "a", "views": 100, "watch_pct": 40.0, "title": "Low one"},
        {"youtube_id": "b", "views": 900, "watch_pct": 60.0, "title": "Big one"},
    ])
    assert "1,000 views" in captured["summary"]
    assert captured["rows"] == 2
    # Highest-viewed first — the digest is scanned, not read.
    assert captured["summary"].index("Big one") < captured["summary"].index("Low one")


def test_digest_failure_never_propagates(monkeypatch):
    import analytics_fetcher

    def boom(*a, **k):
        raise RuntimeError("notify is down")

    monkeypatch.setattr(notify, "notify_analytics", boom)
    analytics_fetcher._post_digest([{"youtube_id": "a", "views": 1,
                                     "watch_pct": 1.0, "title": "t"}])


def test_digest_is_skipped_when_nothing_was_fetched(monkeypatch):
    import analytics_fetcher
    called = []
    monkeypatch.setattr(analytics_fetcher, "_post_digest", lambda rows: called.append(rows))
    monkeypatch.setattr(analytics_fetcher, "list_channels", lambda: [])
    analytics_fetcher.fetch_analytics()
    assert called == []
