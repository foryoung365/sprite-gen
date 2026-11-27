# SPDX-License-Identifier: Apache-2.0
"""`sprite-gen video-color-lock` — OKLab height-banded chroma lock for keyed video sprite frames.

Image-to-video models drift colours from frame to frame (boots shifting olive to maroon, a moving
hand picking up a pink tint from the chroma key). This pass operates on straight-alpha RGBA frames
and adjusts OKLab chroma ``(a, b)`` only — never lightness ``L`` and never alpha — preserving
shading, outlines and anti-aliased edges:

1. Master colours by height: the master's solid interior is partitioned into 5 overlapping height
   bands (feet to head), each learning up to 12 OKLab colour centres via deterministic weighted
   k-means++.
2. Matching: every visible pixel matches the nearest master centre in its own band and adjacent
   bands; pixels lock fully within ``0.6 * MATCH`` and smoothly fade out by ``MATCH``.
3. Local lock: per-colour chroma offsets are averaged locally (box filter of radius 3% of body
   height) and removed, pulling drifted parts back without flattening fine texture.
4. Soft snap: pixels within ``SNAP`` of their centre are pulled softly toward it in chroma, with a
   gain cap ``max(0.02, 0.5 * chroma)`` so neutrals never turn chromatic.
5. Temporal chroma smoothing: still pixels (small source OKLab delta and alpha delta) keep 50% of
   the previous output frame's chroma, removing shimmer without ghosting moving limbs. Cyclic
   actions warm up from the last frame.

Adapted from 0x0funky/agent-sprite-forge (MIT, commit 25203d03900b46b444bdd72a75768c33d1895ad6).
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from sprite_gen._deps import np
from sprite_gen.spec.runio import atomic_write_text
from sprite_gen.util.console import utf8_streams

COLOR_LOCK_MODES = ("off", "on")
COLOR_LOCK_DEFAULT = "off"
REPORT_KIND = "sprite-gen-video-color-lock-report"

BANDS = 5
BAND_COLOURS = 12
BAND_OVERLAP = 0.5
MASTER_SAMPLES = 20000
L_WEIGHT = 1.0
CHROMA_GAIN = 0.02
MEASURE_L_WEIGHT = 0.5
MATCH = 0.12
SMOOTH = 0.03
SNAP = 0.06
SNAP_STRENGTH = 0.35
STILL_DE = 0.04
STILL_ALPHA = 13
TEMPORAL_KEEP = 0.5
FLIP_AB = 0.03
FLIP_L = 0.05
MEASURE_MATCH = 0.10
REGION_PRIOR = 2.0

FOREIGN_AB = 0.06
FOREIGN_CHROMA = 0.05
FOREIGN_ALPHA = 160
DRIFT_MIN_SHARE = 0.02

_CHUNK_PAIRS = 1 << 21
_KMEANS_BIN_SHIFT = 3
_KMEANS_MAX_SAMPLES = 32768
_KMEANS_MAX_ITER = 30

_SRGB_TO_LMS = np.array(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ],
    dtype=np.float32,
).T
_LMS_TO_OKLAB = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ],
    dtype=np.float32,
).T


class ColourLockError(ValueError):
    """User-facing refusal for invalid colour-lock inputs."""


# --------------------------------------------------------------------------- OKLab & k-means++


def to_oklab(rgb: Any) -> np.ndarray:
    """Convert 8-bit sRGB (``... x 3``) to OKLab (Ottosson 2020), float32."""
    c = np.asarray(rgb, dtype=np.float32) / np.float32(255.0)
    linear = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)
    lms = linear @ _SRGB_TO_LMS
    return np.cbrt(lms) @ _LMS_TO_OKLAB


def from_oklab(lab: Any) -> np.ndarray:
    """Convert OKLab (``... x 3``) back to 8-bit sRGB uint8 (round half-up, gamut clipped)."""
    arr = np.asarray(lab, dtype=np.float64)
    big_l, a, b = arr[..., 0], arr[..., 1], arr[..., 2]
    l_ = (big_l + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m_ = (big_l - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s_ = (big_l - 0.0894841775 * a - 1.2914855480 * b) ** 3
    linear = np.stack(
        [
            4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_,
            -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_,
            -0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_,
        ],
        axis=-1,
    )
    linear = np.clip(linear, 0.0, 1.0)
    srgb = np.where(linear <= 0.0031308, linear * 12.92, 1.055 * np.power(linear, 1.0 / 2.4) - 0.055)
    return np.floor(np.clip(srgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def _pack_rgb(rgb: np.ndarray) -> np.ndarray:
    channels = np.asarray(rgb, dtype=np.uint32)
    return (channels[..., 0] << 16) | (channels[..., 1] << 8) | channels[..., 2]


def _unpack_rgb(keys: np.ndarray) -> np.ndarray:
    packed = np.asarray(keys, dtype=np.uint32)
    return np.stack([(packed >> 16) & 255, (packed >> 8) & 255, packed & 255], axis=-1).astype(np.uint8)


def _squared_distances(points: np.ndarray, centres: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nearest = np.zeros(len(points), dtype=np.int64)
    distance = np.zeros(len(points), dtype=np.float64)
    centre_norms = (centres ** 2).sum(axis=1)
    step = max(1, _CHUNK_PAIRS // max(1, len(centres)))
    for start in range(0, len(points), step):
        block = points[start : start + step]
        squared = (block ** 2).sum(axis=1)[:, None] - 2.0 * (block @ centres.T) + centre_norms[None, :]
        np.maximum(squared, 0.0, out=squared)
        best = squared.argmin(axis=1)
        nearest[start : start + step] = best
        distance[start : start + step] = squared[np.arange(len(block)), best]
    return nearest, distance


def _bin_colours(lab: np.ndarray, weights: np.ndarray, colours: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bins = _pack_rgb(colours >> _KMEANS_BIN_SHIFT)
    _, member = np.unique(bins, return_inverse=True)
    member = member.reshape(-1)
    mass = np.bincount(member, weights=weights)
    means = np.stack([np.bincount(member, weights=weights * lab[:, axis]) for axis in range(3)], axis=1)
    return means / mass[:, None], mass


def _kmeans_oklab(
    points: np.ndarray,
    weights: np.ndarray,
    count: int,
    rng: np.random.Generator,
    max_iter: int = _KMEANS_MAX_ITER,
) -> np.ndarray:
    distance = np.full(len(points), np.inf, dtype=np.float64)
    centres: list[np.ndarray] = []
    for _ in range(count):
        score = weights if not centres else weights * distance
        total = float(score.sum())
        if not total > 0.0:
            break
        chosen = int(rng.choice(len(points), p=score / total))
        centres.append(points[chosen].copy())
        distance = np.minimum(distance, ((points - points[chosen]) ** 2).sum(axis=1))
    learned = np.array(centres, dtype=np.float64).reshape(-1, 3)
    if not len(learned):
        return learned
    previous: np.ndarray | None = None
    for _ in range(max_iter):
        assignment, distance = _squared_distances(points, learned)
        if previous is not None and np.array_equal(assignment, previous):
            break
        previous = assignment
        mass = np.bincount(assignment, weights=weights, minlength=len(learned))
        sums = np.stack(
            [np.bincount(assignment, weights=weights * points[:, axis], minlength=len(learned)) for axis in range(3)],
            axis=1,
        )
        filled = mass > 0
        learned[filled] = sums[filled] / mass[filled, None]
        spread = weights * distance
        for empty in np.flatnonzero(~filled):
            far = int(spread.argmax())
            if not spread[far] > 0.0:
                break
            learned[empty] = points[far]
            spread[far] = 0.0
    return learned


def _learn_band_centres(samples_rgb: np.ndarray, per_band: int, seed: int) -> np.ndarray:
    """Learn up to ``per_band`` OKLab centres from ``(N, 3)`` uint8 sRGB samples."""
    keys, inverse = np.unique(_pack_rgb(samples_rgb.reshape(-1, 3)), return_inverse=True)
    unique = _unpack_rgb(keys)
    weights = np.bincount(inverse.reshape(-1), minlength=len(keys)).astype(np.float64)
    if len(unique) <= per_band:
        chosen_rgb = unique
    else:
        lab = to_oklab(unique).astype(np.float64)
        points, point_weights = lab, weights
        if len(points) > _KMEANS_MAX_SAMPLES:
            points, point_weights = _bin_colours(lab, weights, unique)
        centres = _kmeans_oklab(points, point_weights, per_band, np.random.default_rng(seed))
        chosen_rgb = from_oklab(centres).reshape(-1, 3)
    kept = np.unique(_pack_rgb(chosen_rgb), return_index=True)[1]
    chosen_rgb = chosen_rgb[np.sort(kept)]
    chosen_lab = to_oklab(chosen_rgb).astype(np.float64)
    if len(chosen_lab) > 1:
        hue = np.arctan2(chosen_lab[:, 2], chosen_lab[:, 1])
        order = np.lexsort((hue, chosen_lab[:, 0]))
        chosen_lab = chosen_lab[order]
    return chosen_lab


# --------------------------------------------------------------------------- body bands & master palette


def _to_rgba_array(image: Any) -> np.ndarray:
    """Return a fresh ``(H, W, 4)`` uint8 RGBA array from a Path, PIL image, or ndarray."""
    if isinstance(image, (str, Path)):
        with Image.open(Path(image)) as im:
            return np.array(im.convert("RGBA"), dtype=np.uint8, copy=True)
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGBA"), dtype=np.uint8, copy=True)
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ColourLockError(f"expected an RGB(A) image, got shape {array.shape}")
    if array.shape[2] == 3:
        alpha = np.full(array.shape[:2] + (1,), 255, dtype=np.uint8)
        return np.concatenate([array.astype(np.uint8), alpha], axis=2)
    return np.array(array, dtype=np.uint8, copy=True)


def body_rows(alpha: np.ndarray) -> tuple[float, float] | None:
    """Return ``(top, feet)`` rows of the solid body (alpha >= 128; rows with >= 2 px), or None."""
    solid = np.asarray(alpha) >= 128
    counts = solid.sum(axis=1)
    rows = np.flatnonzero(counts >= 2)
    if rows.size == 0:
        rows = np.flatnonzero(counts > 0)
        if rows.size == 0:
            return None
    return float(rows[0]), float(rows[-1] + 1)


def height_map(shape: tuple[int, int], rows: tuple[float, float]) -> np.ndarray:
    """Per-pixel normalized height: 0 at feet line, 1 at top of body (clipped to -0.2..1.2)."""
    top, feet = rows
    span = max(1.0, feet - top)
    y = np.arange(shape[0], dtype=np.float64) + 0.5
    return np.broadcast_to(np.clip((feet - y) / span, -0.2, 1.2)[:, None], shape)


def _band_of(height: np.ndarray) -> np.ndarray:
    return np.clip(np.floor(height * BANDS), 0, BANDS - 1).astype(np.int64)


@dataclass(frozen=True)
class MasterColours:
    """Design colours of the master still: OKLab centres and the height band each was learned in."""

    lab: np.ndarray  # (K, 3) float64
    band: np.ndarray  # (K,) int64

    def __len__(self) -> int:
        return len(self.lab)


def master_colours(master: Any, *, per_band: int = BAND_COLOURS, seed: int = 0) -> MasterColours:
    """Learn OKLab colours per height band from a keyed master's eroded solid interior.

    An opaque master (all alpha == 255) or empty master raises ``ColourLockError``.
    """
    if per_band < 1:
        raise ColourLockError(f"per_band must be >= 1, got {per_band!r}")
    pixels = _to_rgba_array(master)
    alpha = pixels[..., 3]
    if (alpha == 255).all():
        raise ColourLockError("the colour-lock master has no transparency; pass a keyed RGBA master")
    rows = body_rows(alpha)
    if rows is None:
        raise ColourLockError("the colour-lock master is empty")

    solid = alpha >= 250
    interior = solid.copy()
    interior[1:] &= solid[:-1]
    interior[:-1] &= solid[1:]
    interior[:, 1:] &= solid[:, :-1]
    interior[:, :-1] &= solid[:, 1:]
    if int(interior.sum()) < 64:
        interior = alpha >= 128

    heights = height_map(alpha.shape, rows)
    rng = np.random.default_rng(seed)
    labs: list[np.ndarray] = []
    bands: list[np.ndarray] = []
    for band in range(BANDS):
        low = -1.0 if band == 0 else (band - BAND_OVERLAP) / BANDS
        high = 2.0 if band == BANDS - 1 else (band + 1 + BAND_OVERLAP) / BANDS
        chosen = interior & (heights >= low) & (heights < high)
        samples = pixels[..., :3][chosen]
        if len(samples) < 8:
            continue
        if len(samples) > MASTER_SAMPLES:
            samples = samples[np.sort(rng.choice(len(samples), MASTER_SAMPLES, replace=False))]
        centres = _learn_band_centres(samples, int(per_band), seed=seed)
        labs.append(centres)
        bands.append(np.full(len(centres), band, dtype=np.int64))
    if not labs:
        raise ColourLockError("the colour-lock master has too few solid pixels")
    return MasterColours(np.concatenate(labs), np.concatenate(bands))


def _as_master(value: Any) -> MasterColours:
    if isinstance(value, MasterColours):
        return value
    if isinstance(value, (str, Path, Image.Image)):
        return master_colours(value)
    arr = np.asarray(value)
    if arr.ndim == 3 and arr.shape[2] in (3, 4):
        return master_colours(arr)
    lab = np.asarray(value, dtype=np.float64).reshape(-1, 3)
    return MasterColours(np.repeat(lab, BANDS, axis=0), np.tile(np.arange(BANDS, dtype=np.int64), len(lab)))


def _assign(
    lab: np.ndarray,
    band: np.ndarray,
    master: MasterColours,
    l_weight: float = L_WEIGHT,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest master colour among the pixel's band and adjacent bands (lightness weighted)."""
    weights = np.array([l_weight, 1.0, 1.0], dtype=np.float64)
    best = np.zeros(len(lab), dtype=np.int64)
    best_d = np.full(len(lab), np.inf, dtype=np.float64)
    for start in range(0, len(master), 16):
        block = master.lab[start : start + 16]
        allowed = np.abs(master.band[start : start + 16][None, :] - band[:, None]) <= 1
        distance = (((lab[:, None, :] - block[None]) ** 2) * weights).sum(axis=-1)
        distance = np.where(allowed, distance, np.inf)
        index = distance.argmin(axis=1)
        value = distance[np.arange(len(lab)), index]
        better = value < best_d
        best = np.where(better, index + start, best)
        best_d = np.where(better, value, best_d)
    return best, np.sqrt(best_d)


def _box_sum(values: np.ndarray, radius: int) -> np.ndarray:
    """Separable box sum over ``(2r + 1)^2`` windows with zero padding (pure numpy)."""
    if radius <= 0:
        return values
    out = values
    for axis in (0, 1):
        pad_spec = [(radius + 1, radius) if a == axis else (0, 0) for a in range(out.ndim)]
        padded = np.pad(out, pad_spec)
        summed = np.cumsum(padded, axis=axis)
        size = out.shape[axis]
        upper = np.take(summed, np.arange(2 * radius + 1, 2 * radius + 1 + size), axis=axis)
        lower = np.take(summed, np.arange(0, size), axis=axis)
        out = upper - lower
    return out


# --------------------------------------------------------------------------- lock engine


def _region_offsets(
    shape: tuple[int, int],
    visible: np.ndarray,
    index: np.ndarray,
    membership: np.ndarray,
    offset: np.ndarray,
    radius: int,
) -> np.ndarray:
    """Local mean chroma offset per visible pixel over same-colour matches within ``radius`` px."""
    height, width = shape
    ys, xs = np.nonzero(visible)
    smooth = np.zeros((len(ys), 2), dtype=np.float64)
    for colour in np.unique(index[membership > 0]):
        own = np.flatnonzero(index == colour)
        members = own[membership[own] > 0]
        y0 = max(0, int(ys[members].min()) - radius)
        y1 = min(height, int(ys[members].max()) + radius + 1)
        x0 = max(0, int(xs[members].min()) - radius)
        x1 = min(width, int(xs[members].max()) + radius + 1)
        weight = np.zeros((y1 - y0, x1 - x0), dtype=np.float64)
        field = np.zeros((y1 - y0, x1 - x0, 2), dtype=np.float64)
        weight[ys[members] - y0, xs[members] - x0] = membership[members]
        field[ys[members] - y0, xs[members] - x0] = offset[members] * membership[members, None]
        total = _box_sum(weight, radius)
        summed = _box_sum(field, radius)
        inside = own[(ys[own] >= y0) & (ys[own] < y1) & (xs[own] >= x0) & (xs[own] < x1)]
        at_y, at_x = ys[inside] - y0, xs[inside] - x0
        smooth[inside] = summed[at_y, at_x] / (total[at_y, at_x] + REGION_PRIOR)[:, None]
    return smooth


def _lock_one(frame: np.ndarray, master: MasterColours, strength: float) -> tuple[np.ndarray, dict[str, Any]]:
    """Local region drift removal and soft chroma snap for a single RGBA frame (returns OKLab)."""
    lab = to_oklab(frame[..., :3]).astype(np.float64)
    alpha = frame[..., 3]
    visible = alpha > 0
    stats: dict[str, Any] = {"matched": 0, "mean_shift": 0.0, "max_shift": 0.0}
    rows = body_rows(alpha)
    if rows is None or strength <= 0.0:
        return lab, stats

    heights = height_map(alpha.shape, rows)
    points = lab[visible]
    index, distance = _assign(points, _band_of(heights[visible]), master)
    own = np.clip((MATCH - distance) / (0.4 * MATCH), 0.0, 1.0)
    membership = own * (alpha[visible] >= 128)
    offset = points[:, 1:] - master.lab[index, 1:]
    radius = max(1, int(round(SMOOTH * (rows[1] - rows[0]))))
    smooth = _region_offsets(alpha.shape, visible, index, membership, offset, radius)

    shifted = points.copy()
    shifted[:, 1:] -= strength * own[:, None] * smooth
    weights = np.array([L_WEIGHT, 1.0, 1.0], dtype=np.float64)
    after = np.sqrt((((shifted - master.lab[index]) ** 2) * weights).sum(axis=-1))
    pull = strength * SNAP_STRENGTH * np.clip(1.0 - after / SNAP, 0.0, 1.0)
    shifted[:, 1:] += pull[:, None] * (master.lab[index, 1:] - shifted[:, 1:])

    chroma = np.sqrt((points[:, 1:] ** 2).sum(axis=-1))
    gain = np.sqrt((shifted[:, 1:] ** 2).sum(axis=-1)) - chroma
    allowed = np.maximum(CHROMA_GAIN, 0.5 * chroma)
    scale = np.where(gain > allowed, allowed / np.maximum(gain, 1e-12), 1.0)
    shifted[:, 1:] = points[:, 1:] + scale[:, None] * (shifted[:, 1:] - points[:, 1:])

    moved = np.sqrt(((shifted[:, 1:] - points[:, 1:]) ** 2).sum(axis=-1))
    lab[visible] = shifted
    stats = {
        "matched": int((membership > 0).sum()),
        "mean_shift": float(moved.mean()) if moved.size else 0.0,
        "max_shift": float(moved.max()) if moved.size else 0.0,
    }
    return lab, stats


def lock_frames(
    frames: Sequence[Any],
    master: Any,
    *,
    strength: float = 1.0,
    loop: bool = False,
    temporal: bool = True,
) -> tuple[list[Any], dict[str, Any]]:
    """Lock straight-alpha RGBA frames to ``master`` colours in OKLab chroma only.

    Accepts Pillow ``Image.Image`` objects or ``(H, W, 4)`` uint8 arrays and returns a new list of
    the same element type without mutating inputs. Lightness ``L`` and alpha are never modified.
    """
    mc = _as_master(master)
    if not frames:
        return [], {"frames": 0, "master_colours": int(len(mc)), "strength": float(strength), "loop": bool(loop)}
    strength_f = float(strength)
    if not (math.isfinite(strength_f) and 0.0 <= strength_f <= 1.0):
        raise ColourLockError(f"colour-lock strength must be in 0..1, got {strength!r}")

    as_pil = isinstance(frames[0], Image.Image)
    arrays = [_to_rgba_array(frame) for frame in frames]
    shape = arrays[0].shape
    if any(arr.shape != shape for arr in arrays):
        raise ColourLockError("colour-lock frames must share one canvas size")

    if strength_f == 0.0:
        out_unchanged: list[Any] = [Image.fromarray(arr, "RGBA") if as_pil else arr.copy() for arr in arrays]
        return out_unchanged, {
            "frames": len(arrays),
            "master_colours": int(len(mc)),
            "strength": 0.0,
            "mean_shift": 0.0,
            "max_shift": 0.0,
            "temporal_held_px": 0,
            "temporal": bool(temporal),
            "loop": bool(loop),
        }

    locked: list[np.ndarray] = []
    per_frame: list[dict[str, Any]] = []
    for arr in arrays:
        lab, frame_stats = _lock_one(arr, mc, strength_f)
        locked.append(lab)
        per_frame.append(frame_stats)

    held = 0
    if temporal and len(arrays) > 1:
        sources = [to_oklab(arr[..., :3]).astype(np.float64) for arr in arrays]
        alphas = [arr[..., 3].astype(np.int16) for arr in arrays]
        previous: tuple[int, np.ndarray] | None = (len(arrays) - 1, locked[-1].copy()) if loop else None
        keep = TEMPORAL_KEEP * strength_f
        for idx in range(len(arrays)):
            if previous is not None:
                prev_idx, prev_out = previous
                still = (
                    (((sources[idx] - sources[prev_idx]) ** 2).sum(axis=-1) < STILL_DE ** 2)
                    & (np.abs(alphas[idx] - alphas[prev_idx]) < STILL_ALPHA)
                    & (alphas[idx] > 0)
                    & (alphas[prev_idx] > 0)
                )
                if still.any():
                    current = locked[idx]
                    current[..., 1:][still] += keep * (prev_out[..., 1:][still] - current[..., 1:][still])
                    held += int(still.sum())
            previous = (idx, locked[idx])

    out: list[Any] = []
    for arr, lab in zip(arrays, locked):
        result = arr.copy()
        visible = result[..., 3] > 0
        if visible.any():
            result[..., :3][visible] = from_oklab(lab[visible])
        out.append(Image.fromarray(result, "RGBA") if as_pil else result)

    stats = {
        "frames": len(arrays),
        "master_colours": int(len(mc)),
        "strength": strength_f,
        "mean_shift": round(float(np.mean([item["mean_shift"] for item in per_frame])), 5),
        "max_shift": round(float(max(item["max_shift"] for item in per_frame)), 5),
        "temporal_held_px": held,
        "temporal": bool(temporal),
        "loop": bool(loop),
    }
    return out, stats


# --------------------------------------------------------------------------- measurement


def measure(
    frames: Sequence[Any],
    master: Any,
    *,
    loop: bool = False,
    min_share: float = 0.005,
) -> dict[str, Any]:
    """Measure colour flicker and drift of RGBA frames against ``master`` colours."""
    mc = _as_master(master)
    labs: list[np.ndarray] = []
    solids: list[np.ndarray] = []
    regions: list[tuple[np.ndarray, np.ndarray]] = []

    for frame in frames:
        arr = _to_rgba_array(frame)
        lab = to_oklab(arr[..., :3]).astype(np.float64)
        solid = arr[..., 3] >= 128
        index = np.full(solid.shape, -1, dtype=np.int64)
        distance = np.full(solid.shape, np.inf, dtype=np.float64)
        rows = body_rows(arr[..., 3])
        if rows is not None and solid.any():
            heights = height_map(solid.shape, rows)
            index[solid], distance[solid] = _assign(lab[solid], _band_of(heights[solid]), mc, MEASURE_L_WEIGHT)
        labs.append(lab)
        solids.append(solid)
        regions.append((index, distance))

    pairs = [(i - 1, i) for i in range(1, len(frames))]
    if loop and len(frames) > 2:
        pairs.append((len(frames) - 1, 0))

    flips: list[int] = []
    for first, second in pairs:
        both = solids[first] & solids[second]
        d_l = np.abs(labs[second][..., 0] - labs[first][..., 0])
        d_ab = np.sqrt(((labs[second][..., 1:] - labs[first][..., 1:]) ** 2).sum(axis=-1))
        flips.append(int((both & (d_l < FLIP_L) & (d_ab > FLIP_AB)).sum()))

    k = len(mc)
    means = np.full((len(frames), k, 2), np.nan, dtype=np.float64)
    shares = np.zeros((len(frames), k), dtype=np.float64)
    foreign: list[float] = []
    for f, ((index, distance), lab, solid) in enumerate(zip(regions, labs, solids)):
        total = max(1, int(solid.sum()))
        member = solid & (distance < MEASURE_MATCH)
        foreign.append(float((solid & (distance >= 2.0 * MEASURE_MATCH)).sum()) / total)
        counts = np.bincount(index[member], minlength=k)
        shares[f] = counts / total
        for channel in (1, 2):
            sums = np.bincount(index[member], weights=lab[..., channel][member], minlength=k)
            means[f, :, channel - 1] = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)

    stable = (shares >= min_share).mean(axis=0) >= 0.5 if len(frames) else np.zeros(k, dtype=bool)
    spreads: dict[int, float] = {}
    for region in np.flatnonzero(stable):
        values = means[:, region][~np.isnan(means[:, region]).any(axis=1)]
        if len(values) >= 2:
            spreads[int(region)] = float(np.sqrt(((values - values.mean(axis=0)) ** 2).sum(axis=1).mean()))

    worst = sorted(spreads.items(), key=lambda item: -item[1])[:5]
    return {
        "frames": len(frames),
        "pairs": len(pairs),
        "hue_flips_per_pair": round(float(np.mean(flips)) if flips else 0.0, 2),
        "regions": len(spreads),
        "region_spread_mean": round(float(np.mean(list(spreads.values()))) if spreads else 0.0, 5),
        "region_spread_max": round(float(max(spreads.values())) if spreads else 0.0, 5),
        "worst_regions": [
            {
                "region": region,
                "band": int(mc.band[region]),
                "master_lab": [round(float(v), 4) for v in mc.lab[region]],
                "spread": round(spread, 5),
            }
            for region, spread in worst
        ],
        "foreign_share": round(float(np.mean(foreign)) if foreign else 0.0, 5),
    }


def foreign_drift(frame: Any, master: Any) -> dict[str, float]:
    """Measure foreign-hue share and worst region chroma drift for one RGBA frame."""
    mc = _as_master(master)
    arr = _to_rgba_array(frame)
    solid = arr[..., 3] >= FOREIGN_ALPHA
    rows = body_rows(arr[..., 3])
    if rows is None or not solid.any():
        return {"foreign": 0.0, "drift": 0.0}

    lab = to_oklab(arr[..., :3][solid]).astype(np.float64)
    bands = _band_of(height_map(solid.shape, rows)[solid])
    total = len(lab)
    nearest_ab = np.full(total, np.inf, dtype=np.float64)
    for start in range(0, len(mc), 16):
        block = mc.lab[start : start + 16, 1:]
        allowed = np.abs(mc.band[start : start + 16][None, :] - bands[:, None]) <= 1
        distance = np.sqrt(((lab[:, None, 1:] - block[None]) ** 2).sum(axis=-1))
        nearest_ab = np.minimum(nearest_ab, np.where(allowed, distance, np.inf).min(axis=1))
    chroma = np.sqrt((lab[:, 1:] ** 2).sum(axis=-1))
    foreign = float(((chroma > FOREIGN_CHROMA) & (nearest_ab >= FOREIGN_AB)).sum()) / total

    index, distance = _assign(lab, bands, mc, MEASURE_L_WEIGHT)
    member = distance < MEASURE_MATCH
    counts = np.bincount(index[member], minlength=len(mc))
    drift = 0.0
    big = counts >= max(3, DRIFT_MIN_SHARE * total)
    if big.any():
        sums_a = np.bincount(index[member], weights=lab[member, 1] - mc.lab[index[member], 1], minlength=len(mc))
        sums_b = np.bincount(index[member], weights=lab[member, 2] - mc.lab[index[member], 2], minlength=len(mc))
        drift = float(np.sqrt((sums_a[big] / counts[big]) ** 2 + (sums_b[big] / counts[big]) ** 2).max())
    return {"foreign": round(foreign, 5), "drift": round(drift, 5)}


# --------------------------------------------------------------------------- CLI


def run_color_lock(
    frames_dir: Path,
    *,
    master: Path | None = None,
    out_dir: Path | None = None,
    strength: float = 1.0,
    loop: bool = False,
    measure_only: bool = False,
    report_path: Path | None = None,
) -> dict[str, Any]:
    """Run OKLab banded colour lock (or measurement only) over a folder of keyed PNG frames."""
    src_dir = frames_dir.expanduser().resolve()
    if not src_dir.is_dir():
        raise SystemExit(f"video-color-lock: --frames-dir not found: {src_dir}")
    files = sorted(p for p in src_dir.glob("*.png") if p.is_file())
    if not files:
        raise SystemExit(f"video-color-lock: no PNG frames found in {src_dir}")
    if not measure_only and out_dir is None:
        raise SystemExit("video-color-lock: --out-dir is required unless --measure-only is set")
    if not (math.isfinite(strength) and 0.0 <= strength <= 1.0):
        raise SystemExit(f"video-color-lock: --strength must be in 0..1, got {strength!r}")

    master_path = master.expanduser().resolve() if master is not None else files[0]
    if not master_path.is_file():
        raise SystemExit(f"video-color-lock: --master not found: {master_path}")

    rgba_frames = [_to_rgba_array(path) for path in files]
    try:
        mc = master_colours(master_path)
    except ColourLockError as exc:
        raise SystemExit(f"video-color-lock: {exc}") from None

    before = measure(rgba_frames, mc, loop=loop)
    payload: dict[str, Any] = {
        "kind": REPORT_KIND,
        "frames_dir": str(src_dir),
        "master": str(master_path),
        "master_colours": int(len(mc)),
        "loop": bool(loop),
        "measure_only": bool(measure_only),
        "strength": float(strength),
        "before": before,
    }

    if not measure_only:
        assert out_dir is not None
        dst_dir = out_dir.expanduser().resolve()
        if dst_dir == src_dir:
            raise SystemExit("video-color-lock: --out-dir must differ from --frames-dir to avoid overwriting inputs")
        try:
            locked, lock_stats = lock_frames(rgba_frames, mc, strength=strength, loop=loop)
        except ColourLockError as exc:
            raise SystemExit(f"video-color-lock: {exc}") from None
        after = measure(locked, mc, loop=loop)
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src_path, arr in zip(files, locked):
            Image.fromarray(arr, "RGBA").save(dst_dir / src_path.name)
        payload.update(
            {
                "out_dir": str(dst_dir),
                "applied": True,
                "stats": lock_stats,
                "after": after,
            }
        )
        target = (report_path or (dst_dir / "color-lock.report.json")).expanduser().resolve()
        atomic_write_text(target, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        payload["report"] = str(target)
    elif report_path is not None:
        target = report_path.expanduser().resolve()
        atomic_write_text(target, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        payload["report"] = str(target)

    return payload


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--frames-dir", required=True, type=Path, help="folder of keyed straight-alpha RGBA PNG frames")
    parser.add_argument("--master", type=Path, default=None, help="keyed RGBA master still (default: first frame in --frames-dir)")
    parser.add_argument("--out-dir", type=Path, default=None, help="output folder for locked RGBA PNG frames (required unless --measure-only)")
    parser.add_argument("--strength", type=float, default=1.0, help="lock strength in 0..1 (default 1.0)")
    parser.add_argument("--loop", action="store_true", help="treat sequence as a loop (warm up temporal smoothing from last frame and measure wrap pair)")
    parser.add_argument("--measure-only", action="store_true", help="measure hue flips, region spread and foreign hue share without writing frames")
    parser.add_argument("--report", type=Path, default=None, help="optional JSON report path (default <out-dir>/color-lock.report.json when writing)")


def run(**kwargs: object) -> int:
    payload = run_color_lock(
        Path(str(kwargs["frames_dir"])),
        master=kwargs.get("master"),  # type: ignore[arg-type]
        out_dir=kwargs.get("out_dir"),  # type: ignore[arg-type]
        strength=float(kwargs.get("strength") if kwargs.get("strength") is not None else 1.0),
        loop=bool(kwargs.get("loop")),
        measure_only=bool(kwargs.get("measure_only")),
        report_path=kwargs.get("report"),  # type: ignore[arg-type]
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    utf8_streams()
    parser = argparse.ArgumentParser(prog="sprite-gen video-color-lock", description=__doc__)
    add_arguments(parser)
    return run(**vars(parser.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
