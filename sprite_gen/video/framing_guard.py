# SPDX-License-Identifier: Apache-2.0
"""First-frame subject scale / reframing guard for generated video clips.

Video models (notably `magi_2`) occasionally zoom into the start still on frame 0
(e.g. 1.19x–1.49x), pushing the subject's feet into the bottom edge before any
motion begins. Checking frame 0 against the reference canvas immediately after
extraction avoids wasting expensive full-clip chroma keying on a zoomed take.
"""

from __future__ import annotations

import io
import shutil
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image

from sprite_gen._deps import np
from sprite_gen.frames.cutout import KEY_TARGETS
from sprite_gen.frames.extract import (
    _border_key_candidate_field,
    _key_channel_split,
    _painted_key_leans,
    is_border_key_candidate,
)

DEFAULT_MAX_SCALE_DRIFT = 0.15
BOTTOM_MARGIN_MIN = 0.03
BOTTOM_CLIP_SCALE_MIN = 1.08
EDGE_BAND_PX = 2
BG_CHANNEL_TOLERANCE = 36
BG_EUCLIDEAN_TOLERANCE = 48.0
MIN_SUBJECT_HEIGHT_RATIO = 0.02


def _detect_chroma_key(image: Image.Image, key: str = "auto") -> str | None:
    """Return `green` | `magenta` when the image's corners match a chroma key, else None."""
    if key == "white":
        return None
    if image.mode in ("RGBA", "LA"):
        rgba = image.convert("RGBA")
        w, h = rgba.size
        if all(rgba.getpixel(p)[3] == 0 for p in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1))):
            return None
    rgb = image.convert("RGB")
    w, h = rgb.size
    corners = [rgb.getpixel(p) for p in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1))]
    if key in KEY_TARGETS:
        return key if all(is_border_key_candidate(c, KEY_TARGETS[key]) for c in corners) else None
    if key == "auto":
        for kind, target in KEY_TARGETS.items():
            if all(is_border_key_candidate(c, target) for c in corners):
                return kind
    return None


def _subject_vertical_metrics(image: Image.Image, kind: str) -> dict[str, Any] | None:
    """Measure the vertical extent and edge contact of non-background pixels on a chroma key."""
    rgb = image.convert("RGB")
    arr = np.asarray(rgb, dtype=np.int16)
    h, w = arr.shape[:2]
    if h == 0 or w == 0:
        return None
    corners = np.array([arr[0, 0], arr[0, w - 1], arr[h - 1, 0], arr[h - 1, w - 1]], dtype=np.int16)
    bg = np.median(corners, axis=0).astype(np.int16)
    diff = np.abs(arr - bg)
    max_diff = diff.max(axis=2)
    euclid = np.sqrt((diff.astype(np.float32) ** 2).sum(axis=2))
    near_bg = (max_diff <= BG_CHANNEL_TOLERANCE) & (euclid <= BG_EUCLIDEAN_TOLERANCE)

    keyed_channels, unkeyed_channels = _key_channel_split(KEY_TARGETS[kind])
    painted = (int(bg[0]), int(bg[1]), int(bg[2]))
    lean = _painted_key_leans(painted, keyed_channels, unkeyed_channels)
    keyish = _border_key_candidate_field(arr.astype(np.int32), keyed_channels, unkeyed_channels, lean=lean)

    fg = ~(near_bg | keyish)
    min_row_px = max(1, int(w * 0.002))
    rows = np.flatnonzero(fg.sum(axis=1) >= min_row_px)
    if rows.size == 0:
        return None
    top_y = int(rows[0])
    bottom_y = int(rows[-1])
    return {
        "top_y": top_y,
        "bottom_y": bottom_y,
        "height_ratio": (bottom_y - top_y + 1) / float(h),
        "top_margin": top_y / float(h),
        "bottom_margin": (h - 1 - bottom_y) / float(h),
        "touches_top": top_y <= EDGE_BAND_PX - 1,
        "touches_bottom": bottom_y >= h - EDGE_BAND_PX,
    }


def compare_subject_framing(
    ref_im: Image.Image,
    frame0_im: Image.Image,
    *,
    key: str = "auto",
    max_scale_drift: float = DEFAULT_MAX_SCALE_DRIFT,
) -> dict[str, Any]:
    """Compare subject vertical scale and bottom/top contact between reference and clip frame 0."""
    kind = _detect_chroma_key(ref_im, key)
    if kind is None:
        return {"checked": False, "passed": True, "reason": "non-chroma reference"}

    ref_m = _subject_vertical_metrics(ref_im, kind)
    if ref_m is None or ref_m["height_ratio"] < MIN_SUBJECT_HEIGHT_RATIO:
        return {"checked": False, "passed": True, "reason": "no subject detected in reference"}

    clip_m = _subject_vertical_metrics(frame0_im, kind)
    if clip_m is None or clip_m["height_ratio"] < MIN_SUBJECT_HEIGHT_RATIO:
        return {
            "checked": True,
            "passed": False,
            "key": kind,
            "scale_ratio": 0.0,
            "ref_height_ratio": round(ref_m["height_ratio"], 4),
            "clip_height_ratio": 0.0,
            "ref_touches_bottom": bool(ref_m["touches_bottom"]),
            "clip_touches_bottom": False,
            "clip_touches_top": False,
            "max_scale_drift": max_scale_drift,
            "reason": "no subject detected in clip first frame",
        }

    scale_ratio = clip_m["height_ratio"] / ref_m["height_ratio"]
    drift_failed = abs(scale_ratio - 1.0) > max_scale_drift
    new_bottom_clip = (
        not ref_m["touches_bottom"]
        and ref_m["bottom_margin"] >= BOTTOM_MARGIN_MIN
        and clip_m["touches_bottom"]
        and scale_ratio > BOTTOM_CLIP_SCALE_MIN
    )
    reasons: list[str] = []
    if drift_failed:
        direction = "zoomed in" if scale_ratio > 1.0 else "shrunk"
        reasons.append(
            f"first frame {direction} to {scale_ratio:.2f}x reference height "
            f"(drift {abs(scale_ratio - 1.0):.1%} > {max_scale_drift:.0%})"
        )
    if new_bottom_clip:
        reasons.append(
            f"first frame clips the bottom edge at {scale_ratio:.2f}x scale "
            f"(reference bottom margin was {ref_m['bottom_margin']:.1%})"
        )
    passed = not reasons
    return {
        "checked": True,
        "passed": passed,
        "key": kind,
        "scale_ratio": round(scale_ratio, 4),
        "ref_height_ratio": round(ref_m["height_ratio"], 4),
        "clip_height_ratio": round(clip_m["height_ratio"], 4),
        "ref_touches_bottom": bool(ref_m["touches_bottom"]),
        "clip_touches_bottom": bool(clip_m["touches_bottom"]),
        "clip_touches_top": bool(clip_m["touches_top"]),
        "max_scale_drift": max_scale_drift,
        "reason": "; ".join(reasons) if reasons else None,
    }


def extract_first_frame(clip: Path) -> Image.Image:
    """Decode only frame 0 of `clip` via ffmpeg into an RGB Pillow image."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found on PATH")
    proc = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(clip), "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
        capture_output=True,
    )
    if proc.returncode != 0 or not proc.stdout:
        detail = proc.stderr.decode("utf-8", errors="replace").strip()[:200]
        raise RuntimeError(f"ffmpeg could not extract frame 0 from {clip}: {detail or 'empty output'}")
    with Image.open(io.BytesIO(proc.stdout)) as im:
        return im.convert("RGB")


def check_first_frame_scale(
    clip: Path,
    reference: Path,
    *,
    key: str = "auto",
    max_scale_drift: float = DEFAULT_MAX_SCALE_DRIFT,
) -> dict[str, Any]:
    """Extract frame 0 of `clip` and verify its subject framing against `reference`."""
    frame0 = extract_first_frame(clip)
    with Image.open(reference) as ref_im:
        return compare_subject_framing(ref_im, frame0, key=key, max_scale_drift=max_scale_drift)
