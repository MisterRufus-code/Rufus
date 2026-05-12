"""
Feature extraction pipeline using CLIP, BLIP, and YOLOv8.

CLIP  → 512-dim semantic embedding (primary retrieval vector)
BLIP  → natural language caption
YOLO  → detected objects + confidence scores
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Lazy model singletons — loaded once, reused across calls
# ---------------------------------------------------------------------------

_clip_model = None
_clip_preprocess = None
_blip_processor = None
_blip_model = None
_yolo_model = None
_device: Optional[torch.device] = None


def _get_device() -> torch.device:
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device


def _load_clip():
    global _clip_model, _clip_preprocess
    if _clip_model is None:
        import clip
        _clip_model, _clip_preprocess = clip.load("ViT-B/32", device=_get_device())
        _clip_model.eval()
    return _clip_model, _clip_preprocess


def _load_blip():
    global _blip_processor, _blip_model
    if _blip_model is None:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        _blip_processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        _blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(_get_device())
        _blip_model.eval()
    return _blip_processor, _blip_model


def _load_yolo():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        _yolo_model = YOLO("yolov8n.pt")
    return _yolo_model


# ---------------------------------------------------------------------------
# Per-asset feature extraction
# ---------------------------------------------------------------------------

@dataclass
class AssetFeatures:
    asset_id: str                        # sha256 of file path
    path: str
    asset_type: str                      # "video" | "image"
    clip_embedding: list[float] = field(default_factory=list)   # 512-dim
    caption: str = ""
    detected_objects: list[str] = field(default_factory=list)
    dominant_colors: list[str] = field(default_factory=list)    # hex strings
    motion_score: float = 0.0           # 0-1, video only
    duration_seconds: float = 0.0       # video only
    frame_count: int = 0                # video only


def _asset_id(path: Path) -> str:
    return hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:16]


def _extract_dominant_colors(image: Image.Image, k: int = 5) -> list[str]:
    """k-means on pixel colors → hex strings."""
    from sklearn.cluster import MiniBatchKMeans
    img_small = image.resize((64, 64)).convert("RGB")
    pixels = np.array(img_small).reshape(-1, 3).astype(np.float32)
    km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init="auto")
    km.fit(pixels)
    centers = km.cluster_centers_.astype(int)
    return [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in centers]


def _clip_embed_image(image: Image.Image) -> list[float]:
    model, preprocess = _load_clip()
    tensor = preprocess(image).unsqueeze(0).to(_get_device())
    with torch.no_grad():
        embedding = model.encode_image(tensor)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.squeeze().cpu().numpy().tolist()


def _blip_caption(image: Image.Image) -> str:
    processor, model = _load_blip()
    inputs = processor(image, return_tensors="pt").to(_get_device())
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=60)
    return processor.decode(out[0], skip_special_tokens=True)


def _yolo_objects(image: Image.Image) -> list[str]:
    yolo = _load_yolo()
    results = yolo(np.array(image), verbose=False)
    names = yolo.names
    objects = []
    for r in results:
        for cls_id, conf in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            if conf >= 0.4:
                objects.append(names[int(cls_id)])
    return list(set(objects))


def _video_thumbnail(video_path: Path) -> tuple[Image.Image, float, float]:
    """Extract a representative frame, duration, motion score."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps else 0.0

    # Sample frames for motion estimation
    samples = min(10, frame_count)
    step = max(1, frame_count // samples)
    frames = []
    for i in range(samples):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()

    motion_score = 0.0
    if len(frames) > 1:
        diffs = [
            np.mean(np.abs(frames[i].astype(float) - frames[i - 1].astype(float)))
            for i in range(1, len(frames))
        ]
        motion_score = float(np.clip(np.mean(diffs) / 50.0, 0.0, 1.0))

    # Use middle frame as thumbnail
    mid_frame_idx = frame_count // 2
    cap2 = cv2.VideoCapture(str(video_path))
    cap2.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_idx)
    ret, frame = cap2.read()
    cap2.release()

    if ret:
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    else:
        pil_img = Image.new("RGB", (320, 180), color=(30, 30, 30))

    return pil_img, duration, motion_score


def extract_features(path: Path, asset_type: str) -> AssetFeatures:
    """Run the full CLIP + BLIP + YOLO extraction pipeline on one asset."""
    features = AssetFeatures(
        asset_id=_asset_id(path),
        path=str(path),
        asset_type=asset_type,
    )

    if asset_type == "video":
        image, features.duration_seconds, features.motion_score = _video_thumbnail(path)
        features.frame_count = int(features.duration_seconds * 30)
    else:
        image = Image.open(path).convert("RGB")

    features.clip_embedding = _clip_embed_image(image)
    features.caption = _blip_caption(image)
    features.detected_objects = _yolo_objects(image)
    features.dominant_colors = _extract_dominant_colors(image)

    return features


def embed_text(text: str) -> list[float]:
    """Embed a text string with CLIP (same space as image embeddings)."""
    import clip
    model, _ = _load_clip()
    tokens = clip.tokenize([text], truncate=True).to(_get_device())
    with torch.no_grad():
        embedding = model.encode_text(tokens)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.squeeze().cpu().numpy().tolist()
