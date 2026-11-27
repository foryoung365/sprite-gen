# SPDX-License-Identifier: Apache-2.0
"""Action timing compaction for attack clips with long mid-action holds or dead idle padding.

4-second pinned video clips (such as `magi_2`'s minimum 4 s duration) often perform a
single melee strike, freeze at the impact pose for 1–2 seconds, then recover and stand
still until the end of the clip. This module detects the active windup/strike/recovery
phases and dead hold plateau, then resamples the original crisp frames into a punchy
~0.8–1.1 s loop at the source fps while preserving exact start/end closure.
"""

from __future__ import annotations

from typing import Any

from PIL import Image

from sprite_gen._deps import np

ACTION_RETIME_MODES = ("off", "auto", "punch")
ACTION_RETIME_DEFAULT = "auto"
ACTION_RETIME_STATES = frozenset({"attack"})
ACTION_RETIME_CYCLES = frozenset({"pinned", "one-shot", "once-return"})

THUMB_SIZE = (160, 90)
MIN_FRAMES = 12
AUTO_MIN_DURATION_S = 3.2
AUTO_MIN_HOLD_S = 0.35
AUTO_MIN_IDLE_S = 0.60
HOLD_MIN_RUN_S = 0.25
HOLD_DIST_FRACTION = 0.65
HOLD_STEP_FRACTION = 0.15
HOLD_POSE_DRIFT_FRACTION = 0.10
HOLD_BRIDGE_MAX_S = 0.16
ACTIVE_DIST_FRACTION = 0.08
TARGET_STRIKE_S = 0.32
TARGET_HITSTOP_S = 0.11
TARGET_RECOVERY_S = 0.28
MAX_REST_FRAMES = 2


def _thumb_features(frames: list[Image.Image]) -> np.ndarray:
    """Premultiplied RGBA float32 thumbnails (N, H*W*4) in [0, 1]."""
    vecs = []
    for im in frames:
        small = im.convert("RGBA").resize(THUMB_SIZE, Image.Resampling.BILINEAR)
        arr = np.asarray(small, dtype=np.float32) / 255.0
        rgb = arr[..., :3] * arr[..., 3:4]
        vecs.append(np.concatenate([rgb, arr[..., 3:4]], axis=-1).reshape(-1))
    return np.stack(vecs)


def _motion_curves(frames: list[Image.Image]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute thumbnail features, per-frame departure from frame 0 (`dist`), and inter-frame step (`step`)."""
    feats = _thumb_features(frames)
    dist = np.abs(feats - feats[0]).mean(axis=1)
    step = np.zeros(len(frames), dtype=np.float32)
    if len(frames) > 1:
        step[1:] = np.abs(feats[1:] - feats[:-1]).mean(axis=1)
    return feats, dist, step


def _tighten_hold_plateau(
    feats: np.ndarray,
    lo: int,
    hi: int,
    *,
    peak_dist: float,
    min_frames: int,
    max_gap: int,
) -> tuple[int, int] | None:
    """Trim leading/trailing motion (e.g. a blade snapping shut while the body is planted) off a candidate hold."""
    if hi - lo + 1 < min_frames:
        return None
    settled_start = lo + (hi - lo) // 2
    settled_pose = feats[settled_start : hi + 1].mean(axis=0)
    pose_diff = np.abs(feats[lo : hi + 1] - settled_pose).mean(axis=1)
    pose_tol = max(HOLD_POSE_DRIFT_FRACTION * peak_dist, 1e-4)
    runs: list[tuple[int, int]] = []
    run_start: int | None = None
    for offset, d in enumerate(pose_diff):
        idx = lo + offset
        if float(d) <= pose_tol and run_start is None:
            run_start = idx
        elif float(d) > pose_tol and run_start is not None:
            runs.append((run_start, idx - 1))
            run_start = None
    if run_start is not None:
        runs.append((run_start, hi))
    if not runs:
        return None
    # Bridge brief micro-twitches (1–max_gap frames) inside an otherwise settled hold.
    merged: list[tuple[int, int]] = [runs[0]]
    for r_lo, r_hi in runs[1:]:
        prev_lo, prev_hi = merged[-1]
        if (r_lo - prev_hi - 1) <= max_gap:
            merged[-1] = (prev_lo, r_hi)
        else:
            merged.append((r_lo, r_hi))
    long_runs = [r for r in merged if (r[1] - r[0] + 1) >= min_frames]
    if not long_runs:
        return None
    return max(long_runs, key=lambda r: r[1] - r[0])


def _find_hold_plateau(
    feats: np.ndarray,
    dist: np.ndarray,
    step: np.ndarray,
    *,
    start: int,
    end: int,
    peak_dist: float,
    max_step: float,
    fps: float,
) -> tuple[int, int] | None:
    """Find the longest high-excursion frozen plateau inside [start, end]."""
    dist_floor = HOLD_DIST_FRACTION * peak_dist
    step_ceil = HOLD_STEP_FRACTION * max_step
    runs: list[tuple[int, int]] = []
    run_start: int | None = None
    for idx in range(start, end + 1):
        is_hold = float(dist[idx]) >= dist_floor and float(step[idx]) <= step_ceil
        if is_hold and run_start is None:
            run_start = idx
        elif not is_hold and run_start is not None:
            runs.append((run_start, idx - 1))
            run_start = None
    if run_start is not None:
        runs.append((run_start, end))
    if not runs:
        return None

    # Merge adjacent plateaus separated only by a brief micro-twitch while still extended.
    max_gap = max(2, round(HOLD_BRIDGE_MAX_S * fps))
    merged: list[tuple[int, int]] = [runs[0]]
    for lo, hi in runs[1:]:
        prev_lo, prev_hi = merged[-1]
        gap = lo - prev_hi - 1
        if gap <= max_gap and all(float(dist[k]) >= dist_floor for k in range(prev_hi + 1, lo)):
            merged[-1] = (prev_lo, hi)
        else:
            merged.append((lo, hi))

    min_frames = max(3, round(HOLD_MIN_RUN_S * fps))
    tightened = [
        t for lo, hi in merged
        if (t := _tighten_hold_plateau(feats, lo, hi, peak_dist=peak_dist, min_frames=min_frames, max_gap=max_gap)) is not None
    ]
    if not tightened:
        return None
    return max(tightened, key=lambda r: r[1] - r[0])


def _sample_segment(start: int, end: int, count: int, *, curve: str = "linear") -> list[int]:
    """Sample `count` monotonic source indices across [start, end] inclusive."""
    if start > end or count <= 0:
        return []
    span = end - start + 1
    if span <= count:
        return list(range(start, end + 1))
    if count == 1:
        return [start]
    out: list[int] = []
    for k in range(count):
        u = k / float(count - 1)
        if curve == "ease-in":
            t = u ** 0.85
        elif curve == "ease-out":
            t = 1.0 - (1.0 - u) ** 1.15
        else:
            t = u
        idx = start + round(t * (end - start))
        if not out or idx > out[-1]:
            out.append(idx)
    return out


def _unapplied(mode: str, n: int, fps: float, why: str) -> dict[str, Any]:
    dur = round(n / fps, 3) if fps > 0 else 0.0
    return {
        "mode": mode,
        "applied": False,
        "why": why,
        "frames_before": n,
        "frames_after": n,
        "duration_before": dur,
        "duration_after": dur,
        "dead_hold_trimmed_s": 0.0,
    }


def _select_indices(
    n: int,
    *,
    fps: float,
    cycle_mode: str,
    active_start: int,
    peak_idx: int,
    active_end: int,
    hold: tuple[int, int] | None,
) -> tuple[list[int], int]:
    """Assemble monotonic source frame indices for lead, strike, hitstop, recovery and tail."""
    indices: list[int] = [0]
    if active_start >= 2:
        lead_second = min(active_start - 1, max(1, active_start // 2))
        if lead_second > 0:
            indices.append(lead_second)

    if hold is not None:
        hold_start, hold_end = hold
        strike_end = max(active_start, hold_start - 1)
        strike_n = max(4, round(TARGET_STRIKE_S * fps))
        indices.extend(_sample_segment(active_start, strike_end, strike_n, curve="ease-in"))

        hitstop_n = max(2, round(TARGET_HITSTOP_S * fps))
        anchor = (hold_start + hold_end) // 2
        hs_lo = max(hold_start, anchor - hitstop_n // 2)
        hs_hi = min(hold_end, hs_lo + hitstop_n - 1)
        hs_lo = max(hold_start, hs_hi - hitstop_n + 1)
        hitstop_indices = _sample_segment(hs_lo, hs_hi, hitstop_n, curve="linear")
        indices.extend(hitstop_indices)
        kept_hold_frames = len(hitstop_indices)
        rec_start = min(active_end, hold_end + 1)
    else:
        strike_end = max(active_start, peak_idx)
        strike_n = max(4, round(TARGET_STRIKE_S * fps))
        indices.extend(_sample_segment(active_start, strike_end, strike_n, curve="ease-in"))

        hitstop_n = max(2, round(TARGET_HITSTOP_S * fps))
        hs_lo = max(active_start, peak_idx)
        hs_hi = min(active_end, peak_idx + hitstop_n - 1)
        indices.extend(_sample_segment(hs_lo, hs_hi, hitstop_n, curve="linear"))
        kept_hold_frames = 0
        rec_start = min(active_end, hs_hi + 1)

    if rec_start <= active_end:
        rec_n = max(4, round(TARGET_RECOVERY_S * fps))
        indices.extend(_sample_segment(rec_start, active_end, rec_n, curve="ease-out"))

    tail_candidates: list[int] = []
    if active_end + 1 < n:
        tail_candidates.append(active_end + 1)
    if cycle_mode in ("once-return", "one-shot") and n - 1 > active_end + 1:
        tail_candidates.append(n - 1)
    elif active_end + 3 < n:
        tail_candidates.append(active_end + 3)
    indices.extend(tail_candidates[:MAX_REST_FRAMES])

    # Deduplicate while preserving strictly increasing order.
    deduped: list[int] = []
    for idx in indices:
        if 0 <= idx < n and (not deduped or idx > deduped[-1]):
            deduped.append(idx)
    return deduped, kept_hold_frames


def retime_action(
    frames: list[Image.Image],
    *,
    fps: float,
    state: str,
    cycle_mode: str,
    mode: str = ACTION_RETIME_DEFAULT,
) -> tuple[list[Image.Image], dict[str, Any]]:
    """Trim dead mid-action holds and idle padding from an attack cycle."""
    if mode not in ACTION_RETIME_MODES:
        raise ValueError(f"unknown action_retime mode {mode!r}; expected one of {', '.join(ACTION_RETIME_MODES)}")
    n = len(frames)
    if mode == "off":
        return frames, _unapplied(mode, n, fps, "turned off (--action-retime off)")
    if n < MIN_FRAMES:
        return frames, _unapplied(mode, n, fps, f"fewer than {MIN_FRAMES} frames")
    state_norm = (state or "").strip().lower()
    if state_norm not in ACTION_RETIME_STATES:
        return frames, _unapplied(mode, n, fps, f"state {state_norm!r} is not retimed")
    if cycle_mode not in ACTION_RETIME_CYCLES:
        return frames, _unapplied(mode, n, fps, f"cycle mode {cycle_mode!r} is not retimed")
    if fps <= 0:
        return frames, _unapplied(mode, n, fps, "non-positive fps")

    feats, dist, step = _motion_curves(frames)
    peak_idx = int(np.argmax(dist))
    peak_dist = float(dist[peak_idx])
    max_step = float(step.max())
    if peak_dist <= 1e-5 or max_step <= 1e-6:
        return frames, _unapplied(mode, n, fps, "no clear action excursion")

    active_thresh = max(ACTIVE_DIST_FRACTION * peak_dist, 1e-4)
    active = np.flatnonzero(dist >= active_thresh)
    if active.size < 4:
        return frames, _unapplied(mode, n, fps, "active excursion too short")
    active_start = int(active[0])
    active_end = int(active[-1])
    if active_end - active_start < 4:
        return frames, _unapplied(mode, n, fps, "active span too narrow")

    hold = _find_hold_plateau(
        feats, dist, step, start=active_start, end=active_end, peak_dist=peak_dist, max_step=max_step, fps=fps
    )
    hold_frames = (hold[1] - hold[0] + 1) if hold is not None else 0
    hold_duration_s = hold_frames / fps
    idle_frames = active_start + max(0, (n - 1) - active_end)
    idle_duration_s = idle_frames / fps
    total_duration_s = n / fps

    if mode == "auto":
        if total_duration_s < AUTO_MIN_DURATION_S:
            return frames, _unapplied(
                mode, n, fps, f"duration {total_duration_s:.2f} s < {AUTO_MIN_DURATION_S:.1f} s auto threshold"
            )
        if hold_duration_s < AUTO_MIN_HOLD_S and idle_duration_s < AUTO_MIN_IDLE_S:
            return frames, _unapplied(
                mode,
                n,
                fps,
                f"no long dead hold ({hold_duration_s:.2f} s) or excess idle ({idle_duration_s:.2f} s)",
            )

    indices, kept_hold = _select_indices(
        n,
        fps=fps,
        cycle_mode=cycle_mode,
        active_start=active_start,
        peak_idx=peak_idx,
        active_end=active_end,
        hold=hold,
    )
    if len(indices) >= n or len(indices) < 6:
        return frames, _unapplied(mode, n, fps, "clip is already compact")

    out_frames = [frames[idx] for idx in indices]
    dead_hold_trimmed_s = max(0.0, (hold_frames - kept_hold) / fps) if hold is not None else 0.0
    reasons: list[str] = []
    if hold is not None:
        reasons.append(f"trimmed {dead_hold_trimmed_s:.2f} s mid-action hold")
    if idle_duration_s >= 0.2:
        reasons.append(f"trimmed excess rest ({idle_duration_s:.2f} s)")
    if not reasons:
        reasons.append("compacted to punchy attack timing")

    report: dict[str, Any] = {
        "mode": mode,
        "applied": True,
        "why": "; ".join(reasons),
        "frames_before": n,
        "frames_after": len(out_frames),
        "duration_before": round(total_duration_s, 3),
        "duration_after": round(len(out_frames) / fps, 3),
        "dead_hold_trimmed_s": round(dead_hold_trimmed_s, 3),
        "source_indices": indices,
        "phases": {
            "active_start": active_start,
            "peak": peak_idx,
            "hold": list(hold) if hold is not None else None,
            "active_end": active_end,
        },
    }
    return out_frames, report
