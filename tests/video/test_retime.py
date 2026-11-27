# SPDX-License-Identifier: Apache-2.0
"""Offline unit tests for attack hold compaction & action retiming (`retime` & `video-loop`)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from sprite_gen.video import loop as loop_mod
from sprite_gen.video import retime as retime_mod


def _make_attack_frames(
    *,
    n_lead: int = 3,
    n_windup: int = 12,
    n_strike: int = 6,
    n_hold: int = 42,
    n_recover: int = 20,
    n_tail: int = 22,
    size: tuple[int, int] = (160, 90),
    twitch_in_hold: bool = False,
) -> list[Image.Image]:
    """Synthetic RGBA attack: idle -> windup -> fast strike -> dead hold -> recovery -> tail stand."""
    frames: list[Image.Image] = []

    def draw(arm_reach: float, *, body_shift: int = 0) -> Image.Image:
        im = Image.new("RGBA", size, (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        # Torso stays anchored
        d.rectangle((48 + body_shift, 24, 76 + body_shift, 80), fill=(180, 60, 50, 255))
        # Weapon / striking arm extends forward proportionally to arm_reach in [0, 1]
        reach_px = round(arm_reach * 62)
        d.rectangle((72, 36, 78 + reach_px, 50), fill=(230, 210, 90, 255))
        if arm_reach > 0.7:
            d.rectangle((74 + reach_px, 28, 86 + reach_px, 58), fill=(255, 255, 255, 255))
        return im

    for _ in range(n_lead):
        frames.append(draw(0.0))
    for k in range(n_windup):
        frames.append(draw(0.25 * (k + 1) / max(1, n_windup)))
    for k in range(n_strike):
        u = (k + 1) / max(1, n_strike)
        frames.append(draw(0.25 + 0.75 * u))
    for k in range(n_hold):
        if twitch_in_hold and k in (10, 11):
            frames.append(draw(0.95, body_shift=1))
        else:
            frames.append(draw(1.0))
    for k in range(n_recover):
        u = (k + 1) / max(1, n_recover)
        frames.append(draw(max(0.0, 1.0 - u)))
    for _ in range(n_tail):
        frames.append(draw(0.0))
    return frames


def test_retime_action_compacts_long_attack_with_dead_hold_and_twitch() -> None:
    frames = _make_attack_frames(twitch_in_hold=True)  # 105 frames @ 25 fps = 4.20 s
    out, rep = retime_mod.retime_action(frames, fps=25.0, state="attack", cycle_mode="pinned", mode="auto")
    assert rep["applied"] is True
    assert rep["frames_before"] == len(frames)
    assert 16 <= rep["frames_after"] <= 34
    assert 0.65 <= rep["duration_after"] <= 1.35
    assert rep["dead_hold_trimmed_s"] >= 1.2
    assert len(out) == rep["frames_after"]
    # Monotonic source indices starting at 0
    indices = rep["source_indices"]
    assert indices[0] == 0
    assert indices == sorted(set(indices))


def test_retime_action_skips_short_clips_non_attacks_and_off_mode() -> None:
    short = _make_attack_frames(n_lead=2, n_windup=8, n_strike=5, n_hold=6, n_recover=10, n_tail=2)  # 33 frames @ 24fps = 1.375s
    out_short, rep_short = retime_mod.retime_action(short, fps=24.0, state="attack", cycle_mode="pinned", mode="auto")
    assert rep_short["applied"] is False and len(out_short) == len(short)

    long_frames = _make_attack_frames()
    out_off, rep_off = retime_mod.retime_action(long_frames, fps=25.0, state="attack", cycle_mode="pinned", mode="off")
    assert rep_off["applied"] is False and out_off is long_frames

    out_walk, rep_walk = retime_mod.retime_action(long_frames, fps=25.0, state="walk", cycle_mode="pinned", mode="auto")
    assert rep_walk["applied"] is False and out_walk is long_frames

    out_per, rep_per = retime_mod.retime_action(long_frames, fps=25.0, state="attack", cycle_mode="periodic", mode="auto")
    assert rep_per["applied"] is False and out_per is long_frames


def test_retime_action_punch_mode_compacts_shorter_attack() -> None:
    short = _make_attack_frames(n_lead=4, n_windup=12, n_strike=6, n_hold=12, n_recover=14, n_tail=4)  # 52 frames @ 25fps = 2.08s
    out, rep = retime_mod.retime_action(short, fps=25.0, state="attack", cycle_mode="pinned", mode="punch")
    assert rep["applied"] is True
    assert rep["frames_after"] < len(short)
    assert rep["duration_after"] <= 1.25


def test_run_loop_applies_action_retime_and_records_report(tmp_path: Path) -> None:
    keyed = tmp_path / "keyed"
    keyed.mkdir()
    frames = _make_attack_frames()
    # Add closing pin frame matching frame 0
    frames.append(frames[0].copy())
    for k, im in enumerate(frames):
        im.save(keyed / f"frame-{k:04d}.png")

    rep_auto = loop_mod.run_loop(
        keyed, tmp_path / "loop_auto", fps=25.0, state="attack", min_len=None, max_len=None,
        n_out=None, seam_max=loop_mod.SEAM_RATIO_MAX, name="atk", report_path=None,
        cycle_mode="pinned", action_retime="auto", fill_held="off", repair="off",
    )
    assert rep_auto["action_retime"]["applied"] is True
    assert rep_auto["action_retime"]["frames_before"] == len(frames) - 1
    assert rep_auto["action_retime"]["frames_after"] < 35
    assert rep_auto["cycle_seconds"] < 1.4
    assert len(list((tmp_path / "loop_auto" / "cycle").glob("frame-*.png"))) == rep_auto["action_retime"]["frames_after"]
    strip_meta = json.loads((tmp_path / "loop_auto" / "atk.strip.json").read_text())
    assert strip_meta["action_retime"]["applied"] is True
    assert strip_meta["frames"] == rep_auto["action_retime"]["frames_after"]

    rep_off = loop_mod.run_loop(
        keyed, tmp_path / "loop_off", fps=25.0, state="attack", min_len=None, max_len=None,
        n_out=None, seam_max=loop_mod.SEAM_RATIO_MAX, name="atk", report_path=None,
        cycle_mode="pinned", action_retime="off", fill_held="off", repair="off",
    )
    assert rep_off["action_retime"]["applied"] is False
    assert len(list((tmp_path / "loop_off" / "cycle").glob("frame-*.png"))) == len(frames) - 1


def test_hold_plateau_tightens_past_secondary_weapon_motion() -> None:
    """If the body plants first while a weapon/prop finishes closing before freezing, the hold starts after the close."""
    def body_frame(shift_x: int) -> Image.Image:
        im = Image.new("RGBA", (160, 90), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.rectangle((20 + shift_x, 22, 56 + shift_x, 82), fill=(180, 60, 50, 255))
        return im

    frames: list[Image.Image] = []
    # Lead rest (0..7)
    for _ in range(8):
        frames.append(body_frame(0))
    # Big body lunge (8..19)
    for k in range(12):
        frames.append(body_frame(round(50 * (k + 1) / 12)))
    # Secondary weapon snap while body is planted (20..29): smaller per-frame step than lunge, distinct from final pose
    for k in range(10):
        im = body_frame(50)
        draw = ImageDraw.Draw(im)
        y_tip = 10 + round(50 * (k + 1) / 10)
        draw.rectangle((108, y_tip, 148, y_tip + 14), fill=(245, 245, 100, 255))
        frames.append(im)
    # Frozen closed-blade hold (30..64)
    closed = frames[-1].copy()
    for _ in range(35):
        frames.append(closed.copy())
    # Recovery (65..84) + tail rest (85..104)
    for k in range(20):
        frames.append(body_frame(50 - round(50 * (k + 1) / 20)))
    for _ in range(20):
        frames.append(body_frame(0))

    out, rep = retime_mod.retime_action(frames, fps=25.0, state="attack", cycle_mode="pinned", mode="auto")
    assert rep["applied"] is True
    hold_lo, hold_hi = rep["phases"]["hold"]
    assert hold_lo >= 28, f"hold started too early and swallowed secondary weapon motion: {rep['phases']}"
    assert hold_hi >= 60
    assert any(22 <= idx <= 29 for idx in rep["source_indices"])
    assert len(out) < 35


def test_retime_action_emits_events_and_explicit_hit_frame_and_hit_stop() -> None:
    frames = _make_attack_frames(twitch_in_hold=True)  # 105 frames @ 25 fps

    # Auto mode emits structured events and auto hit metadata
    out_auto, rep_auto = retime_mod.retime_action(frames, fps=25.0, state="attack", cycle_mode="pinned", mode="auto")
    assert rep_auto["applied"] is True
    assert rep_auto["hit_source"] == "auto"
    assert isinstance(rep_auto["hit_source_frame"], int)
    assert isinstance(rep_auto["hit_output_frame"], int)
    assert rep_auto["hit_stop_ms"] > 0
    assert rep_auto["duration_ms"] == round(len(out_auto) * 1000.0 / 25.0)
    names = [e["name"] for e in rep_auto["events"]]
    assert names == ["in", "hit", "cancel", "end"]
    assert rep_auto["events"][0] == {"name": "in", "frame": 0, "at_ms": 0, "tick_60hz": 0}
    assert rep_auto["events"][1]["frame"] == rep_auto["hit_output_frame"]
    assert rep_auto["events"][1]["frame"] <= rep_auto["events"][2]["frame"] <= rep_auto["events"][3]["frame"]
    assert rep_auto["events"][3]["at_ms"] == rep_auto["duration_ms"]

    # Explicit hit_frame=31 and hit_stop_ms=120 (3 frames @ 25 fps)
    out_exp, rep_exp = retime_mod.retime_action(
        frames, fps=25.0, state="attack", cycle_mode="pinned", mode="auto", hit_frame=31, hit_stop_ms=120
    )
    assert rep_exp["applied"] is True
    assert rep_exp["hit_source"] == "explicit"
    assert rep_exp["hit_source_frame"] == 31
    assert rep_exp["hit_stop_ms"] == 120
    assert rep_exp["source_indices"][rep_exp["hit_output_frame"]] == 31
    assert rep_exp["events"][1]["frame"] == rep_exp["hit_output_frame"]
    # Recovery is strictly non-decreasing
    rec_slice = rep_exp["source_indices"][rep_exp["hit_output_frame"] :]
    assert all(rec_slice[k + 1] >= rec_slice[k] for k in range(len(rec_slice) - 1))

    # hit_stop_ms=0 removes extra hold frames
    out_zero, rep_zero = retime_mod.retime_action(
        frames, fps=25.0, state="attack", cycle_mode="pinned", mode="auto", hit_frame=31, hit_stop_ms=0
    )
    assert rep_zero["applied"] is True
    assert rep_zero["hit_stop_ms"] == 0
    assert len(out_zero) < len(out_exp)


def test_assert_monotonic_recovery_rejects_reverse_indices() -> None:
    retime_mod.assert_monotonic_recovery([0, 10, 20, 31, 31, 45, 60], hit_output_frame=3)
    with pytest.raises(ValueError, match="must not reverse"):
        retime_mod.assert_monotonic_recovery([0, 10, 20, 31, 28, 45], hit_output_frame=3)


def test_run_loop_threads_hit_frame_hit_stop_and_color_lock_into_strip_meta(tmp_path: Path) -> None:
    keyed = tmp_path / "keyed"
    keyed.mkdir()
    frames = _make_attack_frames()
    frames.append(frames[0].copy())
    for k, im in enumerate(frames):
        im.save(keyed / f"frame-{k:04d}.png")

    rep = loop_mod.run_loop(
        keyed, tmp_path / "loop_exp", fps=25.0, state="attack", min_len=None, max_len=None,
        n_out=None, seam_max=loop_mod.SEAM_RATIO_MAX, name="atk", report_path=None,
        cycle_mode="pinned", action_retime="auto", hit_frame=25, hit_stop_ms=120,
        color_lock="on", color_lock_strength=0.8, fill_held="off", repair="off",
    )
    assert rep["action_retime"]["hit_source"] == "explicit"
    assert rep["action_retime"]["hit_source_frame"] == 25
    assert rep["action_retime"]["hit_stop_ms"] == 120
    assert [e["name"] for e in rep["action_retime"]["events"]] == ["in", "hit", "cancel", "end"]
    assert rep["color_lock"]["applied"] is True
    assert rep["color_lock"]["loop"] is False

    strip_meta = json.loads((tmp_path / "loop_exp" / "atk.strip.json").read_text())
    assert strip_meta["action_retime"]["hit_source"] == "explicit"
    assert strip_meta["action_retime"]["hit_source_frame"] == 25
    assert strip_meta["action_retime"]["hit_stop_ms"] == 120
    assert [e["name"] for e in strip_meta["action_retime"]["events"]] == ["in", "hit", "cancel", "end"]
    assert strip_meta["color_lock"]["applied"] is True
    assert strip_meta["color_lock"]["strength"] == 0.8
