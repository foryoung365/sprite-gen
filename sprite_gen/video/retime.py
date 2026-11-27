# SPDX-License-Identifier: Apache-2.0
"""Action timing compaction for attack clips with long mid-action holds or dead idle padding.

4-second pinned video clips (such as `magi_2`'s minimum 4 s duration) often perform a
single melee strike, freeze at the impact pose for 1–2 seconds, then recover and stand
still until the end of the clip. This module detects the active windup/strike/recovery
phases and dead hold plateau (or accepts an explicit `--hit-frame` and `--hit-stop-ms`),
resamples the original crisp frames into a punchy ~0.8–1.1 s sequence at the source fps
while preserving monotonic forward recovery and emitting a structured gameplay event
timeline (`in`, `hit`, `cancel`, `end`) on the 60 Hz tick grid.
"""

from __future__ import annotations

from typing import Any, Sequence

from PIL import Image

from sprite_gen._deps import np

ACTION_RETIME_MODES = ("off", "auto", "punch")
ACTION_RETIME_DEFAULT = "auto"
ACTION_RETIME_STATES = frozenset({"attack"})
ACTION_RETIME_CYCLES = frozenset({"pinned", "one-shot", "once-return"})

TICK_HZ = 60
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
        t
        for lo, hi in merged
        if (
            t := _tighten_hold_plateau(
                feats, lo, hi, peak_dist=peak_dist, min_frames=min_frames, max_gap=max_gap
            )
        )
        is not None
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


def _ms_to_tick_60hz(at_ms: int) -> int:
    return (2 * int(at_ms) * TICK_HZ + 1000) // 2000


def _frame_to_ms(frame_idx: int, fps: float) -> int:
    return int(round(frame_idx * 1000.0 / fps))


def assert_monotonic_recovery(indices: Sequence[int], hit_output_frame: int) -> None:
    """Assert attack recovery never plays source frames backwards (strictly non-decreasing)."""
    start = max(0, int(hit_output_frame))
    for pos in range(start, len(indices) - 1):
        if indices[pos + 1] < indices[pos]:
            raise ValueError(
                f"attack recovery must not reverse: output frames {pos}->{pos + 1} "
                f"go backwards from source {indices[pos]} to {indices[pos + 1]}"
            )


def build_events(
    *,
    frame_count: int,
    fps: float,
    hit_output_frame: int,
    cancel_output_frame: int,
) -> list[dict[str, Any]]:
    """Build structured gameplay timeline events (`in`, `hit`, `cancel`, `end`) on the 60 Hz tick grid."""
    duration_ms = _frame_to_ms(frame_count, fps)
    last_frame = max(0, frame_count - 1)
    hit_f = max(0, min(last_frame, int(hit_output_frame)))
    cancel_f = max(hit_f, min(last_frame, int(cancel_output_frame)))
    hit_ms = _frame_to_ms(hit_f, fps)
    cancel_ms = _frame_to_ms(cancel_f, fps)
    return [
        {"name": "in", "frame": 0, "at_ms": 0, "tick_60hz": 0},
        {"name": "hit", "frame": hit_f, "at_ms": hit_ms, "tick_60hz": _ms_to_tick_60hz(hit_ms)},
        {"name": "cancel", "frame": cancel_f, "at_ms": cancel_ms, "tick_60hz": _ms_to_tick_60hz(cancel_ms)},
        {"name": "end", "frame": last_frame, "at_ms": duration_ms, "tick_60hz": _ms_to_tick_60hz(duration_ms)},
    ]


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


def _append_strict(out: list[int], items: Sequence[int], n: int) -> None:
    for idx in items:
        if 0 <= idx < n and (not out or idx > out[-1]):
            out.append(idx)


def _append_nondecreasing(out: list[int], items: Sequence[int], n: int) -> None:
    for idx in items:
        if 0 <= idx < n and (not out or idx >= out[-1]):
            out.append(idx)


def _select_indices(
    n: int,
    *,
    fps: float,
    cycle_mode: str,
    active_start: int,
    peak_idx: int,
    active_end: int,
    hold: tuple[int, int] | None,
    hit_frame: int | None = None,
    hit_stop_ms: int | None = None,
) -> dict[str, Any]:
    """Assemble monotonic source frame indices for lead, strike, hitstop, recovery and tail."""
    indices: list[int] = [0]
    if active_start >= 2:
        lead_second = min(active_start - 1, max(1, active_start // 2))
        if lead_second > 0:
            indices.append(lead_second)

    strike_n = max(4, round(TARGET_STRIKE_S * fps))
    if hit_stop_ms is None:
        hitstop_n = max(2, round(TARGET_HITSTOP_S * fps))
    elif hit_stop_ms == 0:
        hitstop_n = 0
    else:
        hitstop_n = max(1, round((hit_stop_ms / 1000.0) * fps))

    # Preserve exact legacy auto selection when neither hit_frame nor hit_stop_ms is overridden.
    if hit_frame is None and hit_stop_ms is None:
        if hold is not None:
            hold_start, hold_end = hold
            strike_end = max(active_start, hold_start - 1)
            _append_strict(indices, _sample_segment(active_start, strike_end, strike_n, curve="ease-in"), n)

            anchor = (hold_start + hold_end) // 2
            hs_lo = max(hold_start, anchor - hitstop_n // 2)
            hs_hi = min(hold_end, hs_lo + hitstop_n - 1)
            hs_lo = max(hold_start, hs_hi - hitstop_n + 1)
            hitstop_indices = _sample_segment(hs_lo, hs_hi, hitstop_n, curve="linear")
            hit_output_frame = len(indices)
            _append_strict(indices, hitstop_indices, n)
            hit_output_frame = min(hit_output_frame, len(indices) - 1)
            hit_source_frame = indices[hit_output_frame]
            kept_hold_frames = len(hitstop_indices)
            actual_hitstop_frames = len(hitstop_indices)
            rec_start = min(active_end, hold_end + 1)
        else:
            strike_end = max(active_start, peak_idx)
            _append_strict(indices, _sample_segment(active_start, strike_end, strike_n, curve="ease-in"), n)

            hs_lo = max(active_start, peak_idx)
            hs_hi = min(active_end, peak_idx + hitstop_n - 1)
            before_hs = len(indices)
            _append_strict(indices, _sample_segment(hs_lo, hs_hi, hitstop_n, curve="linear"), n)
            hit_source_frame = peak_idx
            hit_output_frame = (
                indices.index(peak_idx) if peak_idx in indices else min(before_hs, len(indices) - 1)
            )
            kept_hold_frames = 0
            actual_hitstop_frames = max(1, len(indices) - hit_output_frame)
            rec_start = min(active_end, hs_hi + 1)
    else:
        # Explicit hit_frame and/or explicit hit_stop_ms.
        near_margin = max(2, round(0.15 * fps))
        if hit_frame is not None:
            hit_src = int(hit_frame)
            hold_aligned = hold is not None and (hold[0] - near_margin) <= hit_src <= hold[1]
        else:
            hold_aligned = hold is not None
            hit_src = ((hold[0] + hold[1]) // 2) if hold is not None else peak_idx

        if hit_src > active_start:
            _append_strict(indices, _sample_segment(active_start, hit_src - 1, strike_n, curve="ease-in"), n)

        hit_output_frame = len(indices)
        if hitstop_n == 0:
            _append_nondecreasing(indices, [hit_src], n)
            kept_hold_frames = 1 if hold_aligned else 0
            actual_hitstop_frames = 0
            rec_start = (hold[1] + 1) if (hold_aligned and hold is not None) else (hit_src + 1)
        elif hold_aligned and hold is not None:
            hold_start, hold_end = hold
            hs_lo = hit_src
            hs_hi = min(hold_end, hs_lo + hitstop_n - 1)
            hs_seq = list(range(hs_lo, hs_hi + 1)) if hs_hi >= hs_lo else [hit_src]
            if len(hs_seq) < hitstop_n:
                hs_seq.extend([hs_seq[-1]] * (hitstop_n - len(hs_seq)))
            _append_nondecreasing(indices, hs_seq, n)
            kept_hold_frames = len(hs_seq)
            actual_hitstop_frames = len(hs_seq)
            rec_start = hold_end + 1
        else:
            hs_seq = [hit_src] * hitstop_n
            _append_nondecreasing(indices, hs_seq, n)
            kept_hold_frames = 0
            actual_hitstop_frames = len(hs_seq)
            rec_start = hit_src + 1
        hit_output_frame = min(hit_output_frame, len(indices) - 1)
        hit_source_frame = hit_src

    rec_out_start = len(indices)
    if rec_start <= active_end:
        rec_n = max(4, round(TARGET_RECOVERY_S * fps))
        _append_strict(indices, _sample_segment(rec_start, active_end, rec_n, curve="ease-out"), n)
    rec_out_end = len(indices) - 1

    tail_candidates: list[int] = []
    if active_end + 1 < n:
        tail_candidates.append(active_end + 1)
    if cycle_mode in ("once-return", "one-shot") and n - 1 > active_end + 1:
        tail_candidates.append(n - 1)
    elif active_end + 3 < n:
        tail_candidates.append(active_end + 3)
    _append_strict(indices, tail_candidates[:MAX_REST_FRAMES], n)

    if rec_out_end >= rec_out_start:
        cancel_output_frame = (rec_out_start + rec_out_end) // 2
    else:
        cancel_output_frame = min(len(indices) - 1, hit_output_frame + max(1, actual_hitstop_frames))

    return {
        "indices": indices,
        "kept_hold_frames": kept_hold_frames,
        "hit_source_frame": int(hit_source_frame),
        "hit_output_frame": int(hit_output_frame),
        "hitstop_frames": int(actual_hitstop_frames),
        "cancel_output_frame": int(cancel_output_frame),
    }


def retime_action(
    frames: list[Image.Image],
    *,
    fps: float,
    state: str,
    cycle_mode: str,
    mode: str = ACTION_RETIME_DEFAULT,
    hit_frame: int | None = None,
    hit_stop_ms: int | None = None,
) -> tuple[list[Image.Image], dict[str, Any]]:
    """Trim dead mid-action holds and idle padding from an attack cycle and emit gameplay events."""
    if mode not in ACTION_RETIME_MODES:
        raise ValueError(f"unknown action_retime mode {mode!r}; expected one of {', '.join(ACTION_RETIME_MODES)}")
    n = len(frames)
    if hit_frame is not None and not (0 <= int(hit_frame) < n):
        raise ValueError(f"hit_frame {hit_frame} is outside 0..{max(0, n - 1)}")
    if hit_stop_ms is not None and int(hit_stop_ms) < 0:
        raise ValueError(f"hit_stop_ms must be >= 0, got {hit_stop_ms!r}")

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
    if hit_frame is None and (peak_dist <= 1e-5 or max_step <= 1e-6):
        return frames, _unapplied(mode, n, fps, "no clear action excursion")

    active_thresh = max(ACTIVE_DIST_FRACTION * peak_dist, 1e-4)
    active = np.flatnonzero(dist >= active_thresh)
    if active.size < 4:
        if hit_frame is None:
            return frames, _unapplied(mode, n, fps, "active excursion too short")
        active_start = max(1, int(hit_frame) - max(3, round(0.25 * fps)))
        active_end = min(n - 1, int(hit_frame) + max(3, round(0.25 * fps)))
    else:
        active_start = int(active[0])
        active_end = int(active[-1])
        if hit_frame is not None:
            active_start = min(active_start, int(hit_frame))
            active_end = max(active_end, int(hit_frame))
    if active_end - active_start < 4 and hit_frame is None:
        return frames, _unapplied(mode, n, fps, "active span too narrow")

    hold = (
        _find_hold_plateau(
            feats, dist, step, start=active_start, end=active_end, peak_dist=peak_dist, max_step=max_step, fps=fps
        )
        if peak_dist > 1e-5 and max_step > 1e-6
        else None
    )
    hold_frames = (hold[1] - hold[0] + 1) if hold is not None else 0
    hold_duration_s = hold_frames / fps
    idle_frames = active_start + max(0, (n - 1) - active_end)
    idle_duration_s = idle_frames / fps
    total_duration_s = n / fps

    if mode == "auto" and hit_frame is None:
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

    selected = _select_indices(
        n,
        fps=fps,
        cycle_mode=cycle_mode,
        active_start=active_start,
        peak_idx=peak_idx,
        active_end=active_end,
        hold=hold,
        hit_frame=hit_frame,
        hit_stop_ms=hit_stop_ms,
    )
    indices: list[int] = selected["indices"]
    if len(indices) >= n or len(indices) < 6:
        return frames, _unapplied(mode, n, fps, "clip is already compact")

    hit_output_frame: int = selected["hit_output_frame"]
    assert_monotonic_recovery(indices, hit_output_frame)

    out_frames = [frames[idx] for idx in indices]
    kept_hold = selected["kept_hold_frames"]
    dead_hold_trimmed_s = max(0.0, (hold_frames - kept_hold) / fps) if hold is not None else 0.0
    actual_hit_stop_ms = 0 if hit_stop_ms == 0 else _frame_to_ms(selected["hitstop_frames"], fps)
    duration_ms = _frame_to_ms(len(out_frames), fps)
    events = build_events(
        frame_count=len(out_frames),
        fps=fps,
        hit_output_frame=hit_output_frame,
        cancel_output_frame=selected["cancel_output_frame"],
    )

    reasons: list[str] = []
    if hit_frame is not None:
        reasons.append(f"explicit hit frame {int(hit_frame)}")
    if hold is not None and dead_hold_trimmed_s > 0:
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
        "duration_ms": duration_ms,
        "dead_hold_trimmed_s": round(dead_hold_trimmed_s, 3),
        "hit_source": "explicit" if hit_frame is not None else "auto",
        "hit_source_frame": selected["hit_source_frame"],
        "hit_output_frame": hit_output_frame,
        "hit_stop_ms": actual_hit_stop_ms,
        "events": events,
        "source_indices": indices,
        "phases": {
            "active_start": active_start,
            "peak": peak_idx,
            "hold": list(hold) if hold is not None else None,
            "active_end": active_end,
        },
    }
    return out_frames, report
