#!/usr/bin/env python3
"""
comfy_client.py — ComfyUI stills generator for Rufus.

Generates photoreal portrait images via a local ComfyUI server, then reuses
sd_client's PIL/FFmpeg helpers to upscale → crop 1080×1920 → Ken Burns mp4.
Produces the same deliverable as sd_client / hyperframes — a list of .mp4
paths — so the rest of the pipeline is untouched.

MODEL: exclusively via config/stills_api.json — a user-exported ComfyUI API
workflow (see comfy_template.py). This is Rufus's ONLY stills engine here;
there is deliberately no built-in fallback model, because the built-in engine
this file used to fall back to (FLUX.1-dev) is non-commercial-licensed and
this pipeline is monetized. The default/documented model is Z-Image-Turbo
(Alibaba Tongyi, Apache 2.0, fully commercial-safe) — see README for the
exact ComfyUI setup. Any other Apache/MIT/commercial-safe image model works
the same way: run its workflow once, set the positive prompt to RUFUS_PROMPT,
Export (API) → config/stills_api.json.

Requirements: ComfyUI running with --listen, and config/stills_api.json in
place (see README's "Swappable stills model" section). With no template
exported, this backend has nothing to render and generate_clips() returns []
so main.py falls through to sd/diffusers/pexels.

Environment:
  COMFY_HOST    (default: http://localhost:8188)
  SD_CLIPS      (clip count — shared with sd_client)
  RUFUS_DEBUG=1 (also print verbose per-clip progress; keyframes + prompts
                are ALWAYS kept under media_library/debug/<stamp>/, on every
                run, regardless of this flag — see _housekeeping's retention)

Usage:
  RUFUS_VIDEO_SOURCE=comfy python scripts/main.py --skip-upload
  python scripts/comfy_client.py "modern luxury kitchen, golden hour, wide angle"
"""

import json
import os
import random
import time
import uuid
from pathlib import Path

import paths

import requests

# Reuse the proven, dependency-free (PIL/FFmpeg-only) helpers from sd_client so
# animation and perceptual dedup behave identically across backends. One-way
# import — sd_client never imports comfy_client.
from sd_client import (
    _animate_to_clip,
    _avg_hash,
    _hamming,
    DUP_THRESHOLD,
    MAX_DUP_RETRIES,
    OUT_W,
    OUT_H,
)


def _fit_to_portrait(img_bytes: bytes, out_path: Path) -> bool:
    """Cover-resize a stills-model frame to exactly 1080×1920, preserving composition.

    832×1472 (0.5652) vs 1080×1920 (0.5625) are near-identical aspect ratios, so
    we Lanczos-scale to just cover the target and trim the ~0.5% sliver. This
    keeps ~99% of the frame as composed — unlike sd_client's fixed 2× upscale
    + center-crop, which at this generation size would discard 35% of the image.
    """
    from PIL import Image
    import io as _io

    img  = Image.open(_io.BytesIO(img_bytes)).convert("RGB")
    w, h = img.size
    scale = max(OUT_W / w, OUT_H / h)
    new_w, new_h = round(w * scale), round(h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)

    left = (new_w - OUT_W) // 2
    top  = (new_h - OUT_H) // 2
    img  = img.crop((left, top, left + OUT_W, top + OUT_H))

    img.save(str(out_path), format="PNG", optimize=False)
    return out_path.exists() and out_path.stat().st_size > 20_000

POLL_INTERVAL = 1.5    # seconds between /history polls
GEN_TIMEOUT   = 300    # max seconds to wait for one image (~20-30s on a 3090)
GEN_ERROR_BACKOFF = 3.0  # pause before resubmitting after a submit/generation failure

# Cross-run visual freshness: perceptual hashes of images accepted in RECENT
# runs are persisted and pre-seeded into the dup check, so a new image that
# merely LOOKS like one from a previous video triggers a regen retry — the
# prompt-level DO-NOT-REPEAT list (main._freshness_block) catches repeated
# ideas, this catches repeated pixels. Disable with RUFUS_FRESH_IMAGES=0.
FRESH_HASH_FILE = Path(__file__).parent.parent / "config" / "recent_image_hashes.json"
FRESH_HASH_CAP  = 120   # ~12 runs of history — enough to stop déjà vu, small file


def _fresh_images_enabled() -> bool:
    return os.environ.get("RUFUS_FRESH_IMAGES", "1").strip().lower() \
        not in ("0", "false", "no", "off")


def _load_prior_hashes() -> list[int]:
    try:
        data = json.loads(FRESH_HASH_FILE.read_text())
        return [int(h) for h in data.get("hashes", [])][-FRESH_HASH_CAP:]
    except (OSError, json.JSONDecodeError, ValueError):
        return []


def _save_hashes(hashes: list[int]) -> None:
    try:
        FRESH_HASH_FILE.parent.mkdir(parents=True, exist_ok=True)
        FRESH_HASH_FILE.write_text(
            json.dumps({"hashes": hashes[-FRESH_HASH_CAP:]}))
    except OSError as e:
        print(f"[comfy] couldn't save image-hash history: {e}")


def _host() -> str:
    return os.environ.get("COMFY_HOST", "http://localhost:8188").rstrip("/")


def is_available() -> bool:
    """Return True if a ComfyUI server is up and responsive."""
    try:
        r = requests.get(f"{_host()}/system_stats", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def _parse_checkpoint_list(obj_info: dict) -> list[str]:
    """Extract loadable checkpoint names from /object_info/CheckpointLoaderSimple.

    Shape: {"CheckpointLoaderSimple": {"input": {"required":
            {"ckpt_name": [["a.safetensors", ...], ...]}}}}
    Returns [] on any shape mismatch — callers treat that as "can't verify".
    """
    try:
        names = obj_info["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
        return [n for n in names if isinstance(n, str)] if isinstance(names, list) else []
    except (KeyError, IndexError, TypeError):
        return []


def list_checkpoints() -> list[str]:
    """Checkpoint names the running ComfyUI can actually load. [] on any failure."""
    try:
        r = requests.get(f"{_host()}/object_info/CheckpointLoaderSimple", timeout=10)
        r.raise_for_status()
        return _parse_checkpoint_list(r.json())
    except Exception:
        return []


# THE STILLS MODEL — swap it without touching code. Same template pattern as
# hunyuan_client: replay a user-exported, verified ComfyUI graph instead of
# blind-wiring each new model's node stack. Drop in ANY commercial-safe image
# model this way — Z-Image-Turbo (recommended, Apache 2.0), Qwen-Image, etc.:
# run its ComfyUI workflow once at portrait ~832×1472, set the positive prompt
# text to RUFUS_PROMPT, Export (API) → config/stills_api.json.
#
# Deliberately NO built-in fallback model here. This used to fall back to a
# hardcoded FLUX.1-dev graph on any failure — removed because FLUX.1-dev is
# non-commercial-licensed and this pipeline is monetized; a "safety net" that
# silently renders a non-commercial model into the final video isn't safe at
# all. If the template is missing or a render fails, generate_clips() returns
# [] and main.py falls through to sd/diffusers/pexels instead. Opt the
# template out with RUFUS_STILLS_TEMPLATE=0.
#   config/stills_api.json  — the current/primary image model (model-agnostic)
#   config/flux2_api.json   — kept as a fallback FILENAME for back-compat
#                             (historical name from FLUX.2 testing; holds
#                             whatever commercial-safe model you export to it)
STILLS_TEMPLATE = Path(__file__).parent.parent / "config" / "stills_api.json"
FLUX2_TEMPLATE  = Path(__file__).parent.parent / "config" / "flux2_api.json"


def _stills_template() -> dict | None:
    """The exported image workflow to use for every still, or None. Honors the
    new RUFUS_STILLS_TEMPLATE kill-switch and the legacy RUFUS_FLUX2 one."""
    off = ("0", "false", "no", "off")
    if os.environ.get("RUFUS_STILLS_TEMPLATE", "1").strip().lower() in off:
        return None
    if os.environ.get("RUFUS_FLUX2", "1").strip().lower() in off:
        return None
    import comfy_template
    for path in (STILLS_TEMPLATE, FLUX2_TEMPLATE):
        tpl = comfy_template.load_template(path)
        if tpl is not None and comfy_template.has_placeholder(tpl):
            return tpl
    return None


# Back-compat alias — older references / tests still call _flux2_template().
_flux2_template = _stills_template


# ── Detail / realism direction appended to every stills prompt ────────────────
# NOT the SD1.5 idiom. sd_client's QUALITY_SUFFIX stacks booru-style tokens
# ("8k, masterpiece, ultra-detailed") because Realistic Vision v5.1 was trained
# on caption text full of exactly those tags, so they land as real signal there.
#
# The stills model here (Z-Image-Turbo by default) encodes prompts with
# Qwen3-4B — a modern LLM text encoder trained on long, descriptive natural-
# language captions. Keyword spam is out-of-distribution for it: "8k,
# masterpiece" contributes little and can crowd out the actual subject tokens.
# What DOES drive fine detail in an LLM-encoder model is concrete physical
# description — real optics, a named light behaviour, and micro-surface facts
# the model can actually render. So this reads like a photographer's note, not
# a tag dump.
#
# Tune or replace wholesale with RUFUS_STILLS_DETAIL; set it empty to disable.
DEFAULT_DETAIL_SUFFIX = (
    "Shot on a full-frame camera with an 85mm f/1.4 prime at close range, "
    "shallow depth of field: the subject is tack-sharp with visible focus "
    "falloff, the background dissolving into soft round bokeh. Motivated "
    "directional key light rakes across the surface at a low angle so every "
    "raised and recessed detail casts its own micro-shadow, with a soft fill "
    "opening the darker side and real gradient falloff — never flat lighting. "
    "Extreme surface fidelity, legible down to the smallest scale: individual "
    "material grain and pores, hairline scratches and scuffs, worn and "
    "rounded edges, dust motes and lint caught in the light, fingerprints and "
    "smudges, tarnish and patina pooling in recesses, fibres standing off cut "
    "paper, the weave and loose threads of fabric, condensation beading, "
    "chipped paint, oxidation, the faint irregularity of anything handmade. "
    "Nothing is pristine or computer-clean — every surface carries the "
    "evidence of having existed and been handled. Fine natural film grain, "
    "true-to-life colour response, natural chromatic falloff toward the frame "
    "edges, no digital over-sharpening, no plastic smoothing, no HDR halos. "
    "Documentary photojournalism captured on a real camera."
)


def _detail_suffix() -> str:
    return os.environ.get("RUFUS_STILLS_DETAIL", DEFAULT_DETAIL_SUFFIX).strip()


def _with_detail(prompt: str) -> str:
    """Append the detail/realism direction, unless it's disabled or the prompt
    already carries its own photographic direction (a niche style_suffix, or a
    hand-written --topic prompt, shouldn't get a second contradictory one)."""
    tail = _detail_suffix()
    if not tail:
        return prompt
    low = prompt.lower()
    if "f/1.4" in low or "depth of field" in low:
        return prompt
    return f"{prompt.rstrip().rstrip('.')}. {tail}"


def _render_image(prompt: str, seed: int, client_id: str) -> bytes | None:
    """Render one still via config/stills_api.json → raw PNG bytes, or None.

    No built-in fallback model — see the module docstring for why (licensing).
    Returns None if no template is exported, or if the render itself fails;
    the caller's own retry loop (generate_clips' MAX_DUP_RETRIES) and
    beat-alignment reuse already handle a transient failure, same as any
    other clip-generation error."""
    tpl = _stills_template()
    if tpl is None:
        return None
    import comfy_template
    g = comfy_template.prepare(tpl, prompt=prompt, seed=seed,
                               save_prefix="rufus_stills")
    pid = _submit(g, client_id)
    if not pid:
        return None
    return _await_image(pid)


def _free_comfy_memory() -> None:
    """Ask ComfyUI to unload models + free VRAM/RAM (its /free endpoint).
    Used at the stills→motion phase boundary: a 24GB card can't hold the
    stills model and a 14B/8B video model together, and letting ComfyUI
    evict lazily under pressure is exactly the documented RAM-leak/
    degradation pattern. Best effort — an older ComfyUI without /free just
    ignores this."""
    try:
        r = requests.post(f"{_host()}/free",
                          json={"unload_models": True, "free_memory": True},
                          timeout=30)
        if r.status_code == 200:
            print("[comfy] freed ComfyUI model memory (stills done — loading motion model)")
    except Exception:
        pass


def _submit(graph: dict, client_id: str) -> str | None:
    """POST a workflow to /prompt. Returns prompt_id or None."""
    try:
        r = requests.post(f"{_host()}/prompt",
                          json={"prompt": graph, "client_id": client_id},
                          timeout=30)
        r.raise_for_status()
        return r.json().get("prompt_id")
    except Exception as e:
        print(f"[comfy] submit failed: {e}")
        return None


def _await_image(prompt_id: str) -> bytes | None:
    """Poll /history/<id> until the SaveImage node reports an output, then fetch
    the PNG via /view. Returns raw bytes or None on timeout/failure."""
    deadline = time.time() + GEN_TIMEOUT
    while time.time() < deadline:
        try:
            r = requests.get(f"{_host()}/history/{prompt_id}", timeout=15)
            r.raise_for_status()
            hist = r.json()
        except Exception:
            time.sleep(POLL_INTERVAL)
            continue

        entry = hist.get(prompt_id)
        if entry:
            outputs = entry.get("outputs", {})
            for node in outputs.values():
                images = node.get("images") or []
                if images:
                    img = images[0]
                    try:
                        vr = requests.get(
                            f"{_host()}/view",
                            params={"filename": img.get("filename", ""),
                                    "subfolder": img.get("subfolder", ""),
                                    "type": img.get("type", "output")},
                            timeout=30,
                        )
                        vr.raise_for_status()
                        return vr.content
                    except Exception as e:
                        print(f"[comfy] image fetch failed: {e}")
                        return None
            # entry exists but no images → generation errored in ComfyUI
            if "status" in entry and entry["status"].get("status_str") == "error":
                print("[comfy] ComfyUI reported a generation error for this prompt")
                return None
        time.sleep(POLL_INTERVAL)

    print(f"[comfy] timed out after {GEN_TIMEOUT}s waiting for image")
    return None


def generate_clips(queries: list[str], n: int = 4,
                   clip_duration: float = 8.0) -> list[Path]:
    """Generate one Ken Burns clip per query via ComfyUI, in order.

    Pipeline per clip:
      query → stills model (config/stills_api.json) 832×1472 → Lanczos 2× →
      crop 1080×1920 → Ken Burns mp4

    Matches sd_client.generate_clips' contract: returns a list of 1080×1920 mp4
    Paths (one per query), or [] if ComfyUI is not running / no stills template
    is exported / all images fail, so main.py can fall through to the next
    backend.
    """
    if not is_available():
        print(f"[comfy] ComfyUI not running at {_host()} — start it with --listen, "
              f"or set COMFY_HOST. Falling back.")
        return []

    if _stills_template() is None:
        print("[comfy] no stills model configured — export a ComfyUI image "
              "workflow (Z-Image-Turbo recommended, Apache 2.0/commercial-safe) "
              "to config/stills_api.json. See README's 'Swappable stills model' "
              "section. Falling back.")
        return []

    # Image-to-video: animate each still into real motion instead of the
    # Ken Burns zoom, via an ORDERED engine chain resolved once per run —
    # Wan 2.2 (best temporal consistency, takes a motion prompt) → SVD →
    # Ken Burns. Any per-image failure walks down the chain, so a clip is
    # never lost to a fancier engine.
    motion_engines: list[tuple[str, object]] = []
    try:
        import svd_client
        _stills_only_reason = ("RUFUS_STILLS_ONLY=1 forces images-only"
                               if svd_client._stills_only() else None)
    except Exception:
        _stills_only_reason = None
    try:
        import wan_client
        if wan_client.enabled():
            wan_ok, wan_why = wan_client.ready()
            print(f"[comfy] motion wan 2.2: {'ON' if wan_ok else 'off'} — {wan_why}")
            if wan_ok:
                motion_engines.append(("wan", wan_client.animate_image))
        else:
            print(f"[comfy] motion wan 2.2: off — disabled "
                  f"({_stills_only_reason or 'RUFUS_WAN=0'})")
    except Exception as e:
        print(f"[comfy] wan unavailable ({e})")
    try:
        import hunyuan_client
        if hunyuan_client.enabled():
            hy_ok, hy_why = hunyuan_client.ready()
            print(f"[comfy] motion hunyuan 1.5: {'ON' if hy_ok else 'off'} — {hy_why}")
            if hy_ok:
                # After Wan on purpose: Wan keeps non-face shots (proven
                # quality), and its face-skip returns False fast — so face
                # shots land here instead of falling to static Ken Burns.
                motion_engines.append(("hunyuan", hunyuan_client.animate_image))
        else:
            print(f"[comfy] motion hunyuan 1.5: off — disabled "
                  f"({_stills_only_reason or 'RUFUS_HUNYUAN=0'})")
    except Exception as e:
        print(f"[comfy] hunyuan unavailable ({e})")
    try:
        import svd_client
        if svd_client.img2vid_enabled():
            svd_engine, svd_why = svd_client.resolve_engine()
            print(f"[comfy] motion svd: "
                  f"{'ON via ' + svd_engine if svd_engine else 'off'} — {svd_why}")
            if svd_engine:
                def _svd_animate(png, clip, duration=8.0, idx=0, prompt="",
                                 _engine=svd_engine):
                    return svd_client.animate_image(png, clip, duration=duration,
                                                    idx=idx, engine=_engine,
                                                    prompt=prompt)
                motion_engines.append(("svd", _svd_animate))
    except Exception as e:
        print(f"[comfy] img2vid unavailable ({e}) — Ken Burns only")

    tmp_dir = paths.media_root() / "temp" / "comfy"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    prompts = list(queries or ["cinematic establishing shot"])
    if len(prompts) < n:
        base = prompts[:]
        while len(prompts) < n:
            prompts.append(base[len(prompts) % len(base)] + ", different angle, wider shot")
    prompts = prompts[:n]
    prompts = [_with_detail(p) for p in prompts]

    # pid in the stamp: with per-channel locks two channels may run
    # concurrently, and two runs starting the same second would otherwise
    # collide on identical {stamp}_{i}.png/.mp4 temp names in the shared dir.
    stamp        = f"{int(time.time())}_{os.getpid()}"
    client_id    = uuid.uuid4().hex
    master_seed  = random.randint(1, 2_000_000_000)
    # Seed the dup check with prior runs' hashes so cross-run look-alikes
    # regen too, not just within-run ones. n_prior marks where history ends.
    accepted_hashes: list[int] = _load_prior_hashes() if _fresh_images_enabled() else []
    n_prior = len(accepted_hashes)
    if n_prior:
        print(f"[comfy] freshness: {n_prior} image hash(es) from recent runs loaded")
    clips: list[Path] = []
    print(f"[comfy] stills: config/stills_api.json  base_seed={master_seed}")

    # Every run keeps its keyframes + prompts, not just RUFUS_DEBUG=1 runs —
    # the quality-review workflow needs every image logged, not a sampled
    # subset a reviewer has to remember to opt into. Prefer the shared run id
    # main.py sets (RUFUS_DEBUG_RUN_ID) so this run's images land in the SAME
    # folder as its script/voiceover instead of a folder named after this
    # stage's own timestamp. Falls back to the temp-file stamp when
    # comfy_client runs standalone (its __main__).
    debug_name = os.environ.get("RUFUS_DEBUG_RUN_ID") or str(stamp)
    debug_dir = paths.debug_root() / debug_name
    debug_dir.mkdir(parents=True, exist_ok=True)
    print(f"[comfy] keeping keyframes in {debug_dir}")

    # ── Phase 1: generate every still (the stills model stays loaded the whole time) ────
    stills: list[tuple[int, Path, str]] = []   # (beat index, png path, prompt)
    for i, prompt in enumerate(prompts):
        print(f"[comfy] {i+1}/{len(prompts)}: {prompt}")
        png_path = tmp_dir / f"{stamp}_{i}.png"
        accepted = False

        for retry in range(MAX_DUP_RETRIES + 1):
            # %(2**31) keeps the seed in range for any backend; offset per clip/retry.
            seed  = (master_seed + i + 1000 * retry) % (2**31 - 1)
            img_bytes = _render_image(prompt, seed, client_id)
            if not img_bytes:
                # A hard generation error (vs. a plain duplicate) is often a
                # transient GPU/model-loading hiccup on the ComfyUI side —
                # a short pause before resubmitting gives it a chance to clear
                # instead of hammering the same broken state 3x back-to-back.
                time.sleep(GEN_ERROR_BACKOFF)
                continue

            if not _fit_to_portrait(img_bytes, png_path):  # → exactly 1080×1920
                continue

            h = _avg_hash(png_path)
            is_dup = (h is not None and accepted_hashes
                      and min(_hamming(h, p) for p in accepted_hashes) < DUP_THRESHOLD)
            if is_dup and retry < MAX_DUP_RETRIES:
                print(f"[comfy] dup on clip {i+1} → regen (retry {retry+1})")
                continue
            if is_dup:
                print(f"[comfy] clip {i+1} still near-dup after retries — keeping")
            if h is not None:
                accepted_hashes.append(h)
            accepted = True
            break

        if not accepted:
            # BEAT ALIGNMENT: skipping would shift every LATER clip one beat
            # earlier than its narration (clip lists are positional — the
            # renderer cuts assume clip[i] ↔ beat[i]). Reuse the previous
            # accepted still instead: a repeated image with a different
            # Ken Burns/motion treatment is far less damaging than every
            # subsequent image narrating the wrong sentence.
            if stills:
                print(f"[comfy] no usable image for clip {i+1} — reusing "
                      f"previous still to keep images aligned with narration")
                png_path.write_bytes(stills[-1][1].read_bytes())
            else:
                print(f"[comfy] no usable image for clip {i+1} — skipping")
                continue

        if debug_dir is not None:
            try:
                (debug_dir / f"{i+1:02d}.png").write_bytes(png_path.read_bytes())
                (debug_dir / f"{i+1:02d}.txt").write_text(
                    f"FLUX PROMPT:\n{prompt}\n", encoding="utf-8")
            except Exception as e:
                print(f"[comfy] debug-save failed for clip {i+1}: {e}")

        stills.append((i, png_path, prompt))

    # ── Phase 2: animate every still ────────────────────────────────────────
    # Two phases instead of image→animate per clip: interleaving forced
    # ComfyUI to swap FLUX in and out for EVERY clip (10 model swaps per
    # video on a 24GB card that can't hold both), thrashing RAM/VRAM and —
    # per a ComfyUI-reliability audit — degrading until every clip silently
    # fell through to Ken Burns. Batching stills first means exactly ONE
    # switch, with an explicit /free between so the motion model loads into
    # a clean card.
    if motion_engines and stills:
        _free_comfy_memory()

    motion_log: list[dict] = []
    for i, png_path, prompt in stills:
        clip_path = tmp_dir / f"{stamp}_{i}.mp4"
        made_via = None
        tried: list[str] = []
        for eng_name, animate in motion_engines:
            t0 = time.time()
            okd = animate(png_path, clip_path, duration=clip_duration, idx=i,
                          prompt=prompt)
            secs = time.time() - t0
            # Record what the engine ACTUALLY used, not what we assume it did:
            # the motion prompt is derived inside the engine and the settings
            # come from env + the exported template, so neither is knowable
            # here. Engines publish both via LAST_CALL/settings().
            rec = {"beat": i + 1, "engine": eng_name, "ok": bool(okd),
                   "seconds": round(secs, 1)}
            try:
                mod = {"hunyuan": "hunyuan_client", "wan": "wan_client"}.get(eng_name)
                if mod:
                    rec.update(__import__(mod).LAST_CALL)
            except Exception:
                pass
            motion_log.append(rec)
            tried.append(f"{eng_name} {'ok' if okd else 'failed'} in {secs:.0f}s")
            if okd:
                made_via = eng_name
                break
            print(f"[comfy] {eng_name} failed for clip {i+1} — trying next engine")
        made = made_via is not None or _animate_to_clip(png_path, clip_path,
                                                        duration=clip_duration, idx=i)
        if made_via is None and motion_engines:
            motion_log.append({"beat": i + 1, "engine": "kenburns", "ok": bool(made),
                               "note": "all motion engines declined/failed"})
        if made:
            clips.append(clip_path)
            print(f"[comfy] clip {i+1} ready"
                  + (f" ({made_via} motion)" if made_via else " (Ken Burns)"))
        else:
            print(f"[comfy] animation failed for clip {i+1} — later images may "
                  f"drift ahead of narration")
        png_path.unlink(missing_ok=True)

    if motion_log:
        try:
            paths.write_run_report(debug_name, motion=motion_log)
        except Exception as e:
            print(f"[comfy] motion-report write skipped (non-fatal): {e}")

    if _fresh_images_enabled() and len(accepted_hashes) > n_prior:
        _save_hashes(accepted_hashes)
    print(f"[comfy] {len(clips)}/{len(prompts)} clips ready")
    return clips


if __name__ == "__main__":
    import sys
    qs = sys.argv[1:] or ["modern luxury kitchen interior, golden hour light, wide angle",
                          "sunlit living room, floor to ceiling windows, city view"]
    for p in generate_clips(qs, n=len(qs)):
        print(f"CLIP={p}")
