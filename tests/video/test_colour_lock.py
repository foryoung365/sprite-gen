# SPDX-License-Identifier: Apache-2.0
"""Offline unit tests for OKLab height-banded colour lock (`colour_lock`, CLI & `video-loop`)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from sprite_gen._deps import np
from sprite_gen.video import colour_lock as cl
from sprite_gen.video import loop as loop_mod


def _make_sprite(
    *,
    boot_rgb: tuple[int, int, int] = (132, 76, 44),
    coat_rgb: tuple[int, int, int] = (62, 118, 168),
    head_rgb: tuple[int, int, int] = (224, 186, 152),
    neutral_rgb: tuple[int, int, int] = (128, 128, 128),
    size: tuple[int, int] = (96, 120),
) -> np.ndarray:
    """Synthetic RGBA character with distinct height bands (head, coat, neutral trim, boots)."""
    im = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(im)
    # Head (top band)
    draw.rectangle((34, 12, 62, 36), fill=(*head_rgb, 255))
    # Coat (mid bands)
    draw.rectangle((28, 37, 68, 74), fill=(*coat_rgb, 255))
    # Neutral grey belt
    draw.rectangle((28, 75, 68, 84), fill=(*neutral_rgb, 255))
    # Boots (bottom band)
    draw.rectangle((30, 85, 45, 110), fill=(*boot_rgb, 255))
    draw.rectangle((51, 85, 66, 110), fill=(*boot_rgb, 255))
    return np.asarray(im, dtype=np.uint8)


def test_oklab_roundtrip_is_exact_for_sampled_srgb() -> None:
    rng = np.random.default_rng(42)
    samples = rng.integers(0, 256, size=(2000, 3), dtype=np.uint8)
    corners = np.array([[0, 0, 0], [255, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 0, 255]], dtype=np.uint8)
    rgb = np.concatenate([corners, samples], axis=0)
    rebuilt = cl.from_oklab(cl.to_oklab(rgb))
    assert np.array_equal(rebuilt, rgb)


def test_master_colours_refuses_opaque_and_empty_inputs() -> None:
    opaque = np.full((64, 64, 4), 255, dtype=np.uint8)
    with pytest.raises(cl.ColourLockError, match="no transparency"):
        cl.master_colours(opaque)

    empty = np.zeros((64, 64, 4), dtype=np.uint8)
    with pytest.raises(cl.ColourLockError, match="empty"):
        cl.master_colours(empty)


def test_lock_frames_reduces_drift_preserves_alpha_lightness_and_neutrals() -> None:
    master_arr = _make_sprite()
    mc = cl.master_colours(master_arr)
    assert len(mc) >= 5

    # Alternating drifted boot/coat tints across frames (olive / maroon boot drift)
    frames = [
        master_arr.copy(),
        _make_sprite(boot_rgb=(116, 92, 42), coat_rgb=(54, 124, 152)),
        _make_sprite(boot_rgb=(148, 66, 56), coat_rgb=(74, 110, 178)),
        _make_sprite(boot_rgb=(118, 90, 44), coat_rgb=(56, 122, 156)),
        _make_sprite(boot_rgb=(146, 68, 54), coat_rgb=(72, 112, 176)),
        master_arr.copy(),
    ]
    originals = [f.copy() for f in frames]

    before = cl.measure(frames, mc, loop=False)
    assert {"hue_flips_per_pair", "region_spread_mean", "region_spread_max", "foreign_share"} <= set(before)
    assert before["hue_flips_per_pair"] > 0
    assert before["region_spread_mean"] > 0

    locked, stats = cl.lock_frames(frames, mc, strength=1.0, loop=False)
    after = cl.measure(locked, mc, loop=False)

    # Inputs were not mutated
    for orig, cur in zip(originals, frames):
        assert np.array_equal(orig, cur)

    # Flicker & spread dropped substantially
    assert after["hue_flips_per_pair"] < before["hue_flips_per_pair"]
    assert after["region_spread_mean"] < before["region_spread_mean"]
    assert after["region_spread_max"] < before["region_spread_max"]
    assert stats["frames"] == len(frames)

    # Alpha is byte-identical and OKLab lightness L stays preserved (within 8-bit sRGB rounding)
    for src, dst in zip(originals, locked):
        assert np.array_equal(src[..., 3], dst[..., 3])
        vis = src[..., 3] > 0
        l_before = cl.to_oklab(src[..., :3][vis])[..., 0]
        l_after = cl.to_oklab(dst[..., :3][vis])[..., 0]
        assert float(np.abs(l_after - l_before).max()) < 0.025
        # Neutral grey belt never gains high chroma
        belt_lab = cl.to_oklab(dst[76:83, 30:66, :3])
        belt_chroma = np.sqrt(belt_lab[..., 1] ** 2 + belt_lab[..., 2] ** 2)
        assert float(belt_chroma.max()) < 0.025


def test_lock_frames_strength_zero_and_pil_roundtrip_and_loop_warmup() -> None:
    master_pil = Image.fromarray(_make_sprite(), "RGBA")
    frames_pil = [
        Image.fromarray(_make_sprite(boot_rgb=(120, 88, 42)), "RGBA"),
        Image.fromarray(_make_sprite(boot_rgb=(142, 70, 52)), "RGBA"),
        Image.fromarray(_make_sprite(boot_rgb=(132, 76, 44)), "RGBA"),
    ]
    unchanged, stats0 = cl.lock_frames(frames_pil, master_pil, strength=0.0, loop=True)
    assert all(isinstance(im, Image.Image) for im in unchanged)
    assert stats0["mean_shift"] == 0.0
    for a, b in zip(frames_pil, unchanged):
        assert np.array_equal(np.asarray(a), np.asarray(b))

    locked_loop, stats_loop = cl.lock_frames(frames_pil, master_pil, strength=1.0, loop=True)
    locked_once, stats_once = cl.lock_frames(frames_pil, master_pil, strength=1.0, loop=False)
    assert all(isinstance(im, Image.Image) for im in locked_loop)
    assert stats_loop["loop"] is True and stats_once["loop"] is False
    # Loop mode warms up frame 0 from last frame, holding more temporal pixels
    assert stats_loop["temporal_held_px"] > stats_once["temporal_held_px"]


def test_cli_video_color_lock_measure_only_and_write(tmp_path: Path, capsys) -> None:
    src_dir = tmp_path / "keyed"
    src_dir.mkdir()
    frames = [
        _make_sprite(),
        _make_sprite(boot_rgb=(114, 94, 40)),
        _make_sprite(boot_rgb=(150, 64, 58)),
        _make_sprite(),
    ]
    for idx, arr in enumerate(frames):
        Image.fromarray(arr, "RGBA").save(src_dir / f"frame-{idx:04d}.png")

    # --measure-only works without --out-dir
    report_m = tmp_path / "measure.json"
    assert cl.main(["--frames-dir", str(src_dir), "--measure-only", "--report", str(report_m)]) == 0
    out_m = json.loads(capsys.readouterr().out)
    assert out_m["measure_only"] is True
    assert out_m["before"]["hue_flips_per_pair"] > 0
    assert report_m.is_file()

    # Without --measure-only, missing --out-dir is refused
    with pytest.raises(SystemExit, match="--out-dir is required"):
        cl.main(["--frames-dir", str(src_dir)])

    # Full run writes locked frames and report
    out_dir = tmp_path / "locked"
    assert cl.main(["--frames-dir", str(src_dir), "--out-dir", str(out_dir), "--strength", "1.0"]) == 0
    out_w = json.loads(capsys.readouterr().out)
    assert out_w["applied"] is True
    assert out_w["after"]["hue_flips_per_pair"] < out_w["before"]["hue_flips_per_pair"]
    assert len(list(out_dir.glob("frame-*.png"))) == len(frames)
    assert (out_dir / "color-lock.report.json").is_file()


def test_video_loop_color_lock_off_by_default_and_on_records_stats(tmp_path: Path) -> None:
    keyed = tmp_path / "keyed"
    keyed.mkdir()
    base = _make_sprite()
    seq = [
        base,
        _make_sprite(boot_rgb=(116, 92, 42)),
        _make_sprite(boot_rgb=(148, 66, 56)),
        _make_sprite(boot_rgb=(118, 90, 44)),
        _make_sprite(boot_rgb=(146, 68, 54)),
        _make_sprite(boot_rgb=(122, 86, 44)),
        _make_sprite(boot_rgb=(140, 72, 50)),
        base,
    ]
    for idx, arr in enumerate(seq):
        Image.fromarray(arr, "RGBA").save(keyed / f"frame-{idx:04d}.png")

    rep_off = loop_mod.run_loop(
        keyed, tmp_path / "loop_off", fps=24.0, state="idle", min_len=None, max_len=None,
        n_out=None, seam_max=loop_mod.SEAM_RATIO_MAX, name="idle", report_path=None,
        cycle_mode="pinned", fill_held="off", repair="off",
    )
    assert "color_lock" not in rep_off
    strip_off = json.loads((tmp_path / "loop_off" / "idle.strip.json").read_text())
    assert "color_lock" not in strip_off

    rep_on = loop_mod.run_loop(
        keyed, tmp_path / "loop_on", fps=24.0, state="idle", min_len=None, max_len=None,
        n_out=None, seam_max=loop_mod.SEAM_RATIO_MAX, name="idle", report_path=None,
        cycle_mode="pinned", fill_held="off", repair="off",
        color_lock="on", color_lock_strength=0.9,
    )
    assert rep_on["color_lock"]["applied"] is True
    assert rep_on["color_lock"]["strength"] == 0.9
    assert rep_on["color_lock"]["loop"] is True
    assert rep_on["color_lock"]["after"]["hue_flips_per_pair"] <= rep_on["color_lock"]["before"]["hue_flips_per_pair"]
    strip_on = json.loads((tmp_path / "loop_on" / "idle.strip.json").read_text())
    assert strip_on["color_lock"]["applied"] is True
    assert strip_on["color_lock"]["strength"] == 0.9
    assert {"before", "after"} <= set(strip_on["color_lock"])
