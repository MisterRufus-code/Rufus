#!/usr/bin/env python3
"""
comfy_client.py — ComfyUI + FLUX.1-dev image generator for Rufus.

Generates photoreal portrait images via a local ComfyUI server (FLUX.1-dev),
then reuses sd_client's PIL/FFmpeg helpers to upscale → crop 1080×1920 → Ken
Burns mp4. Produces the same deliverable as sd_client / hyperframes — a list of
.mp4 paths — so the rest of the pipeline is untouched.

Why this exists: with a 24GB GPU (RTX 3090) FLUX.1-dev is the best free local
image model. ComfyUI exposes a headless HTTP API (POST /prompt, poll /history,
GET /view) — no websocket dependency needed.

Requirements (Windows 11 / Linux):
  ComfyUI running with --listen, FLUX checkpoint in models/checkpoints/:
    flux1-dev-fp8.safetensors   (single-file fp8 build — fits 24GB comfortably)

Environment:
  COMFY_HOST    (default: http://localhost:8188)
  COMFY_MODEL   (default: flux1-dev-fp8.safetensors)
  COMFY_STEPS   (default: 20)
  SD_CLIPS      (clip count — shared with sd_client)
  RUFUS_DEBUG=1 (keep keyframes + prompts under media_library/debug/<stamp>/)

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

import requests

# Reuse the proven, dependency-free (PIL/FFmpeg-only) helpers from sd_client so
# animation, perceptual dedup, and the negative prompt behave identically
# across backends. One-way import — sd_client never imports comfy_client.
from sd_client import (
    _animate_to_clip,
    _avg_hash,
    _hamming,
    DUP_THRESHOLD,
    MAX_DUP_RETRIES,
    NEGATIVE_PROMPT,
    OUT_W,
    OUT_H,
)


def _fit_to_portrait(img_bytes: bytes, out_path: Path) -> bool:
    """Cover-resize a FLUX frame to exactly 1080×1920, preserving composition.

    832×1472 (0.5652) vs 1080×1920 (0.5625) are near-identical aspect ratios, so
    we Lanczos-scale to just cover the target and trim the ~0.5% sliver. This
    keeps ~99% of the frame FLUX composed — unlike sd_client's fixed 2× upscale
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

# FLUX likes ~1MP. 832×1472 is ÷16 on both axes, ~9:16, then we upscale+crop to
# exactly 1080×1920. (Generating native 1080×1920 wastes VRAM/time for no gain.)
GEN_W, GEN_H = 832, 1472

POLL_INTERVAL = 1.5    # seconds between /history polls
GEN_TIMEOUT   = 300    # max seconds to wait for one image (FLUX ~20-30s on a 3090)
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


# ── Optional face restoration ─────────────────────────────────────────────────
# FLUX (like any diffusion model) renders human faces worst — the prompt steering
# in main.py avoids the hardest close-ups, and this is the belt-and-suspenders:
# if a face-restoration custom node is installed in ComfyUI, route each FLUX still
# through it (CodeFormer / GFPGAN) to clean up eyes/teeth/skin before animation.
#
# STRICTLY OPTIONAL and fail-safe: with no node installed, or RUFUS_FACE_RESTORE=0,
# or ANY failure of the restore graph, we render the plain graph you already run —
# restoration can only ever ADD quality, never reduce reliability or clip yield.
#
#   RUFUS_FACE_RESTORE        auto (default: use a restore node if one is installed)
#                             | 0 to force off | 1 = same as auto
#   RUFUS_FACE_RESTORE_MODEL  restore weights filename (default GFPGANv1.4.pth)
#   RUFUS_FACE_FIDELITY       0..1, higher = truer to the original (default 0.5)


def _has_node(class_type: str) -> bool:
    """True if the running ComfyUI has a node of this class_type installed."""
    try:
        r = requests.get(f"{_host()}/object_info/{class_type}", timeout=10)
        if r.status_code != 200:
            return False
        return bool(r.json().get(class_type))
    except Exception:
        return False


def _face_restore_setting() -> str:
    return os.environ.get("RUFUS_FACE_RESTORE", "auto").strip().lower()


def resolve_face_restore() -> dict | None:
    """Pick an installed face-restoration node, or None to skip it entirely.

    Prefers the dedicated facerestore_cf pack (FaceRestoreCFWithModel), then the
    ReActor pack (ReActorRestoreFace). Returns a descriptor dict consumed by
    _apply_face_restore, or None when disabled or no supported node is present."""
    if _face_restore_setting() in ("0", "false", "no", "off"):
        return None
    model    = os.environ.get("RUFUS_FACE_RESTORE_MODEL", "GFPGANv1.4.pth")
    fidelity = float(os.environ.get("RUFUS_FACE_FIDELITY", "0.5"))
    if _has_node("FaceRestoreCFWithModel") and _has_node("FaceRestoreModelLoader"):
        return {"kind": "facerestore_cf", "model": model, "fidelity": fidelity}
    if _has_node("ReActorRestoreFace"):
        return {"kind": "reactor", "model": model, "fidelity": fidelity}
    return None


def _apply_face_restore(graph: dict, restore: dict) -> dict:
    """Insert a face-restoration node between VAEDecode ('8') and SaveImage ('9'),
    re-pointing SaveImage at the restored image. Mutates and returns graph.

    A restore node with no face in the frame passes the image through unchanged,
    so it's safe on object-only beats (coins, documents, streets)."""
    if restore["kind"] == "facerestore_cf":
        graph["20"] = {"class_type": "FaceRestoreModelLoader",
                       "inputs": {"model_name": restore["model"]}}
        graph["21"] = {"class_type": "FaceRestoreCFWithModel",
                       "inputs": {
                           "facerestore_model": ["20", 0],
                           "image": ["8", 0],
                           "facedetection": "retinaface_resnet50",
                           "codeformer_fidelity": restore["fidelity"],
                       }}
        graph["9"]["inputs"]["images"] = ["21", 0]
    elif restore["kind"] == "reactor":
        graph["21"] = {"class_type": "ReActorRestoreFace",
                       "inputs": {
                           "image": ["8", 0],
                           "facedetection": "retinaface_resnet50",
                           "model": restore["model"],
                           "visibility": 1.0,
                           "codeformer_weight": restore["fidelity"],
                       }}
        graph["9"]["inputs"]["images"] = ["21", 0]
    return graph


def _build_flux_graph_face(prompt: str, seed: int, model: str, steps: int,
                           restore: dict) -> dict:
    """Plain FLUX graph plus a face-restoration tail."""
    return _apply_face_restore(_build_flux_graph(prompt, seed, model, steps), restore)


# FLUX.2 stills via a user-exported API workflow (same template pattern as
# hunyuan_client, same rationale: replay the channel owner's verified graph
# instead of blind-wiring FLUX.2's new node stack). Setup: run ComfyUI's
# FLUX.2 template once at portrait 832×1472, set the positive prompt text to
# RUFUS_PROMPT, Export (API) → config/flux2_api.json. Any failure falls back
# to the proven FLUX.1 graph with the same seed — an upgrade can never cost
# a clip. Opt out with RUFUS_FLUX2=0.
FLUX2_TEMPLATE = Path(__file__).parent.parent / "config" / "flux2_api.json"


def _flux2_template() -> dict | None:
    if os.environ.get("RUFUS_FLUX2", "1").strip().lower() in ("0", "false", "no", "off"):
        return None
    import comfy_template
    tpl = comfy_template.load_template(FLUX2_TEMPLATE)
    if tpl is None or not comfy_template.has_placeholder(tpl):
        return None
    return tpl


def _render_image(prompt: str, seed: int, model: str, steps: int,
                  client_id: str, restore: dict | None) -> tuple[bytes | None, bool]:
    """Render one FLUX still → raw PNG bytes (or None). Returns (bytes, used_restore).

    Engine order: FLUX.2 template (if exported) → face-restore FLUX.1 graph
    (if configured) → plain FLUX.1 graph, all with the SAME seed — a fancier
    path failing can never cost a clip."""
    tpl = _flux2_template()
    if tpl is not None:
        import comfy_template
        g = comfy_template.prepare(tpl, prompt=prompt, seed=seed,
                                   save_prefix="rufus_flux2")
        pid = _submit(g, client_id)
        if pid:
            img = _await_image(pid)
            if img:
                return img, False
        print(f"[comfy] FLUX.2 template render failed — falling back to FLUX.1 (seed={seed})")
    if restore:
        pid = _submit(_build_flux_graph_face(prompt, seed, model, steps, restore), client_id)
        if pid:
            img = _await_image(pid)
            if img:
                return img, True
        print(f"[comfy] face-restore render failed — falling back to plain (seed={seed})")
    pid = _submit(_build_flux_graph(prompt, seed, model, steps), client_id)
    if not pid:
        return None, False
    return _await_image(pid), False


def _build_flux_graph(prompt: str, seed: int, model: str, steps: int) -> dict:
    """A minimal FLUX.1-dev txt2img graph (no custom nodes required).

    CheckpointLoaderSimple → CLIPTextEncode(+/-) → FluxGuidance → KSampler
    → VAEDecode → SaveImage. Mirrors the tutorial workflow but strips the WAS
    style-CSV nodes (which need extra installs) and injects our own prompt.
    """
    return {
        "4": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": model}},
        "6": {"class_type": "CLIPTextEncode",
              "inputs": {"text": prompt, "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode",
              "inputs": {"text": NEGATIVE_PROMPT, "clip": ["4", 1]}},
        "15": {"class_type": "FluxGuidance",
               "inputs": {"guidance": 3.5, "conditioning": ["6", 0]}},
        "10": {"class_type": "EmptySD3LatentImage",
               "inputs": {"width": GEN_W, "height": GEN_H, "batch_size": 1}},
        "3": {"class_type": "KSampler",
              "inputs": {
                  "seed": seed, "steps": steps, "cfg": 1.0,
                  "sampler_name": "dpmpp_2m", "scheduler": "sgm_uniform",
                  "denoise": 1.0,
                  "model": ["4", 0], "positive": ["15", 0],
                  "negative": ["7", 0], "latent_image": ["10", 0],
              }},
        "8": {"class_type": "VAEDecode",
              "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage",
              "inputs": {"filename_prefix": "rufus", "images": ["8", 0]}},
    }


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
                   clip_duration: float = 8.0,
                   niche_cfg: dict | None = None) -> list[Path]:
    """Generate one Ken Burns clip per query via ComfyUI + FLUX.1-dev, in order.

    Pipeline per clip:
      query → FLUX 832×1472 → Lanczos 2× → crop 1080×1920 → Ken Burns mp4

    Matches sd_client.generate_clips' contract: returns a list of 1080×1920 mp4
    Paths (one per query), or [] if ComfyUI is not running / all images fail, so
    main.py can fall through to the next backend.
    """
    if not is_available():
        print(f"[comfy] ComfyUI not running at {_host()} — start it with --listen, "
              f"or set COMFY_HOST. Falling back.")
        return []

    model = (os.environ.get("COMFY_MODEL")
             or (niche_cfg or {}).get("comfy_model")
             or "flux1-dev-fp8.safetensors")
    steps = int(os.environ.get("COMFY_STEPS", "32"))

    # Preflight the checkpoint: server-up but model-missing is the classic
    # first-run failure. Catch it here with the exact fix instead of submitting
    # a batch of doomed jobs. Empty list = endpoint unavailable → can't verify,
    # proceed (fail-open).
    available = list_checkpoints()
    if available and model not in available and _flux2_template() is None:
        # (With a FLUX.2 template exported, the FLUX.1 checkpoint is only the
        # fallback engine — don't bail the whole run over it.)
        print(f"[comfy] checkpoint '{model}' not loadable by ComfyUI.")
        print(f"[comfy] it sees: {', '.join(available[:5])}")
        print(f"[comfy] put the file in ComfyUI\\models\\checkpoints\\ (not unet/), "
              f"or set COMFY_MODEL to one of the names above. Falling back.")
        return []

    # Image-to-video: animate each FLUX still into real motion instead of the
    # Ken Burns zoom, via an ORDERED engine chain resolved once per run —
    # Wan 2.2 (best temporal consistency, takes a motion prompt) → SVD →
    # Ken Burns. Any per-image failure walks down the chain, so a clip is
    # never lost to a fancier engine.
    motion_engines: list[tuple[str, object]] = []
    try:
        import wan_client
        if wan_client.enabled():
            wan_ok, wan_why = wan_client.ready()
            print(f"[comfy] motion wan 2.2: {'ON' if wan_ok else 'off'} — {wan_why}")
            if wan_ok:
                motion_engines.append(("wan", wan_client.animate_image))
        else:
            print("[comfy] motion wan 2.2: off — disabled (RUFUS_WAN=0)")
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
            print("[comfy] motion hunyuan 1.5: off — disabled (RUFUS_HUNYUAN=0)")
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

    # Face restoration (optional): clean up FLUX faces if a restore node is
    # installed. Fail-safe — any problem falls back to the plain graph per image.
    restore = None
    try:
        restore = resolve_face_restore()
        if restore:
            print(f"[comfy] face restore: ON via {restore['kind']} ({restore['model']})")
        elif _face_restore_setting() not in ("0", "false", "no", "off"):
            print("[comfy] face restore: off (no restore node found in ComfyUI — "
                  "install one to enable, see README)")
    except Exception as e:
        print(f"[comfy] face restore unavailable ({e}) — plain render")

    tmp_dir = Path(__file__).parent.parent / "media_library" / "temp" / "comfy"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    prompts = list(queries or ["cinematic establishing shot"])
    if len(prompts) < n:
        base = prompts[:]
        while len(prompts) < n:
            prompts.append(base[len(prompts) % len(base)] + ", different angle, wider shot")
    prompts = prompts[:n]

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
    if _flux2_template() is not None:
        print(f"[comfy] stills: FLUX.2 template (config/flux2_api.json) — "
              f"FLUX.1 '{model}' is the fallback")
    print(f"[comfy] FLUX model={model} steps={steps} base_seed={master_seed}")

    debug_dir = None
    if os.environ.get("RUFUS_DEBUG"):
        # Prefer the shared run id main.py sets (RUFUS_DEBUG_RUN_ID) so this
        # run's images land in the SAME folder as its script/voiceover instead
        # of a folder named after this stage's own timestamp. Falls back to
        # the temp-file stamp when comfy_client runs standalone (its __main__).
        debug_name = os.environ.get("RUFUS_DEBUG_RUN_ID") or str(stamp)
        debug_dir = Path(__file__).parent.parent / "media_library" / "debug" / debug_name
        debug_dir.mkdir(parents=True, exist_ok=True)
        print(f"[comfy] DEBUG on — keeping keyframes in {debug_dir}")

    for i, prompt in enumerate(prompts):
        print(f"[comfy] {i+1}/{len(prompts)}: {prompt[:70]}")
        png_path = tmp_dir / f"{stamp}_{i}.png"
        accepted = False

        for retry in range(MAX_DUP_RETRIES + 1):
            # %(2**31) keeps the seed in range for any backend; offset per clip/retry.
            seed  = (master_seed + i + 1000 * retry) % (2**31 - 1)
            # Face-restore graph if configured, else plain — with an automatic
            # same-seed fallback to plain inside _render_image, so restoration
            # never costs a clip.
            img_bytes, _ = _render_image(prompt, seed, model, steps, client_id, restore)
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
            print(f"[comfy] no usable image for clip {i+1} — skipping")
            continue

        if debug_dir is not None:
            try:
                (debug_dir / f"{i+1:02d}.png").write_bytes(png_path.read_bytes())
                (debug_dir / f"{i+1:02d}.txt").write_text(
                    f"FLUX PROMPT:\n{prompt}\n", encoding="utf-8")
            except Exception as e:
                print(f"[comfy] debug-save failed for clip {i+1}: {e}")

        clip_path = tmp_dir / f"{stamp}_{i}.mp4"
        made_via = None
        for eng_name, animate in motion_engines:
            if animate(png_path, clip_path, duration=clip_duration, idx=i,
                       prompt=prompt):
                made_via = eng_name
                break
            print(f"[comfy] {eng_name} failed for clip {i+1} — trying next engine")
        made = made_via is not None or _animate_to_clip(png_path, clip_path,
                                                        duration=clip_duration, idx=i)
        if made:
            clips.append(clip_path)
            print(f"[comfy] clip {i+1} ready"
                  + (f" ({made_via} motion)" if made_via else " (Ken Burns)"))
        else:
            print(f"[comfy] animation failed for clip {i+1}")
        png_path.unlink(missing_ok=True)

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
