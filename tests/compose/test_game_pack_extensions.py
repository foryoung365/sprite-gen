# SPDX-License-Identifier: Apache-2.0
"""Tests for game-pack large-pack limits, max_source_pixels, lossy WebP, --no-png, and variants."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from sprite_gen.compose import game_pack
from sprite_gen.spec import assets


def write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def gradient_strip(tmp_path, size=(64, 48), frames=2):
    rng = np.random.default_rng(42)
    arr = rng.integers(0, 256, size=(size[1], size[0] * frames, 4), dtype=np.uint8)
    ys, xs = np.mgrid[0:size[1], 0:size[0] * frames]
    mask = (xs % size[0] > 4) & (xs % size[0] < size[0] - 4) & (ys > 4) & (ys < size[1] - 4)
    arr[~mask] = 0
    arr[mask, 3] = 220
    sheet = Image.fromarray(arr, mode="RGBA")
    sheet.save(tmp_path / "hero.strip.png")
    return write_json(tmp_path / "hero.strip.json", {
        "frames": frames,
        "w": size[0],
        "h": size[1],
        "delay_ms": 80,
        "anchor": [size[0] // 2, size[1] - 2],
    })


def request_v1(tmp_path, entries):
    return write_json(tmp_path / "game.json", {"version": 1, "assets": entries})


def test_max_source_pixels_param_and_env_override_in_assets(tmp_path, monkeypatch):
    still = tmp_path / "still.png"
    Image.new("RGBA", (10, 10), (10, 20, 30, 255)).save(still)
    strip_json = gradient_strip(tmp_path, size=(10, 10), frames=2)

    with pytest.raises(ValueError, match="exceeds the supported 99 pixels"):
        assets.load_asset(still, max_source_pixels=99)
    assert assets.load_asset(still, max_source_pixels=100).size == (10, 10)

    monkeypatch.setenv("SPRITE_GEN_MAX_SOURCE_PIXELS", "150")
    with pytest.raises(ValueError, match="exceeds the supported 150 pixels"):
        assets.load_asset_states(strip_json)
    # Explicit parameter takes precedence over environment variable.
    loaded = assets.load_asset_states(strip_json, max_source_pixels=200)
    assert loaded[None].size == (10, 10)

    with pytest.raises(ValueError, match="exceeds the supported 199 pixels"):
        assets.load_asset_states(strip_json, max_source_pixels=199)


@pytest.mark.parametrize("bad_value", [0, -1, True, False, 1.5])
def test_invalid_max_source_pixels_is_rejected(tmp_path, bad_value):
    still = tmp_path / "still.png"
    Image.new("RGBA", (4, 4)).save(still)
    with pytest.raises(ValueError, match="positive integer"):
        assets.load_asset(still, max_source_pixels=bad_value)
    with pytest.raises(ValueError, match="positive integer"):
        assets.load_asset_states(still, max_source_pixels=bad_value)


@pytest.mark.parametrize("bad_env", ["0", "-10", "abc"])
def test_invalid_max_source_pixels_env_is_rejected(tmp_path, monkeypatch, bad_env):
    still = tmp_path / "still.png"
    Image.new("RGBA", (4, 4)).save(still)
    monkeypatch.setenv("SPRITE_GEN_MAX_SOURCE_PIXELS", bad_env)
    with pytest.raises(ValueError, match="positive integer"):
        assets.load_asset(still)


def test_oversized_source_above_67mp_and_pillow_guard_works_when_configured(tmp_path):
    # 9500 x 9500 = 90,250,000 pixels: exceeds both MAX_SOURCE_PIXELS (67.1M) and Pillow's 89.4M guard.
    import warnings

    huge = tmp_path / "huge.png"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Image.DecompressionBombWarning)
        Image.new("L", (9500, 9500), 128).save(huge, compress_level=1)
        req = request_v1(tmp_path, [{"id": "huge", "source": "huge.png", "display": {"height": 16}, "resample": "nearest"}])

        with pytest.raises(ValueError, match="exceeds the supported"):
            game_pack.pack(req, tmp_path / "blocked")
    assert not (tmp_path / "blocked").exists()

    out = tmp_path / "allowed"
    result = game_pack.pack(req, out, max_source_pixels=100_000_000)
    assert result["animations"]["huge"]["size"] == {"w": 16, "h": 16}

    small = tmp_path / "small.png"
    Image.new("RGBA", (20, 20), (10, 20, 30, 255)).save(small)
    small_req = write_json(tmp_path / "small_game.json", {
        "version": 1,
        "assets": [{"id": "small", "source": "small.png", "display": {"height": 8}}],
    })
    blocked_cli = subprocess.run(
        [
            sys.executable, "-m", "sprite_gen.cli", "game-pack",
            "--request", str(small_req),
            "--out-dir", str(tmp_path / "cli-blocked"),
            "--max-source-pixels", "399",
        ],
        capture_output=True,
        encoding="utf-8",
    )
    assert blocked_cli.returncode != 0
    assert "exceeds the supported 399 pixels" in blocked_cli.stderr

    cli_out = tmp_path / "allowed-cli"
    proc = subprocess.run(
        [
            sys.executable, "-m", "sprite_gen.cli", "game-pack",
            "--request", str(small_req),
            "--out-dir", str(cli_out),
            "--max-source-pixels", "400",
        ],
        capture_output=True,
        encoding="utf-8",
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["ok"] is True


def test_allow_large_and_unlimited_max_bytes(tmp_path):
    gradient_strip(tmp_path)
    req = request_v1(tmp_path, [{"id": "hero", "source": "hero.strip.json", "display": {"height": 48}}])

    # Without allow_large, >50 MB and 0 are still refused.
    for bad in (0, 50_000_001):
        with pytest.raises(ValueError):
            game_pack.pack(req, tmp_path / "bad", max_bytes=bad)
        assert not (tmp_path / "bad").exists()

    # With allow_large=True, >50 MB cap is lifted and 0 means unlimited.
    out_capped = tmp_path / "large-capped"
    game_pack.pack(req, out_capped, max_bytes=80_000_000, allow_large=True)
    assert (out_capped / "game-pack.json").is_file()

    out_unlim = tmp_path / "unlimited"
    proc = subprocess.run(
        [
            sys.executable, "-m", "sprite_gen.cli", "game-pack",
            "--request", str(req),
            "--out-dir", str(out_unlim),
            "--allow-large",
            "--max-bytes", "0",
        ],
        capture_output=True,
        encoding="utf-8",
    )
    assert proc.returncode == 0, proc.stderr
    assert (out_unlim / "game-pack.json").is_file()

    # Budget enforcement still applies when allow_large=True with a positive cap.
    with pytest.raises(ValueError, match="bytes"):
        game_pack.pack(req, tmp_path / "too-small", max_bytes=10, allow_large=True)
    assert not (tmp_path / "too-small").exists()


def test_allow_large_unlimited_works_with_audio(tmp_path):
    gradient_strip(tmp_path, size=(16, 16), frames=1)
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    files = {}
    for fmt in ("ogg", "m4a"):
        data = b"synthetic-" + fmt.encode()
        (audio_dir / f"sfx.{fmt}").write_bytes(data)
        files[fmt] = {
            "path": f"sfx.{fmt}",
            "duration_seconds": 1,
            "lufs": -18,
            "true_peak_dbfs": -2,
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    write_json(audio_dir / "sound.json", {
        "kind": "sprite-gen-sound",
        "version": 2,
        "event": "combat.hit",
        "audio_kind": "sfx",
        "loop": False,
        "duration_seconds": 1,
        "target_lufs": -18,
        "files": files,
        "recipe": {"processing_version": 4},
        "normalization": {"pcm_lufs": -18, "pcm_true_peak_dbtp": -2},
        "playback": {"version": 2, "sample_rate": 48000, "start_frame": 0, "end_frame": 48000, "overlap_frames": 0},
    })
    req = write_json(tmp_path / "game.json", {
        "version": 1,
        "assets": [{"id": "hero", "source": "hero.strip.json", "display": {"height": 16}}],
        "audio": [{"id": "hit", "source": "audio/sound.json"}],
    })
    out = tmp_path / "out"
    result = game_pack.pack(req, out, max_bytes=0, allow_large=True, webp_variants="lossless,q90")
    assert "audio" in result
    q90 = json.loads((out / "game-pack_q90.json").read_text(encoding="utf-8"))
    assert q90["audio"] == result["audio"]


def test_lossy_webp_quality_and_no_png_flags(tmp_path):
    gradient_strip(tmp_path, size=(128, 96), frames=2)
    req = request_v1(tmp_path, [{"id": "hero", "source": "hero.strip.json", "display": {"height": 96}}])

    out_lossless = tmp_path / "lossless"
    game_pack.pack(req, out_lossless)

    out_lossy = tmp_path / "lossy"
    result_lossy = game_pack.pack(req, out_lossy, webp_quality=90, webp_alpha_quality=95)
    assert "webp_variant" not in result_lossy
    assert (out_lossy / "atlas-000.png").is_file()
    assert (out_lossy / "atlas-000.webp").stat().st_size < (out_lossless / "atlas-000.webp").stat().st_size

    out_no_png = tmp_path / "no-png"
    proc = subprocess.run(
        [
            sys.executable, "-m", "sprite_gen.cli", "game-pack",
            "--request", str(req),
            "--out-dir", str(out_no_png),
            "--webp-quality", "80",
            "--webp-alpha-quality", "90",
            "--skip-png",
        ],
        capture_output=True,
        encoding="utf-8",
    )
    assert proc.returncode == 0, proc.stderr
    assert sorted(p.name for p in out_no_png.iterdir()) == ["atlas-000.webp", "game-pack.json"]
    manifest = json.loads((out_no_png / "game-pack.json").read_text(encoding="utf-8"))
    tex = manifest["textures"][0]
    assert tex["image"] == "atlas-000.webp"
    assert tex["webp"] == "atlas-000.webp"
    assert tex["meta"]["image"] == "atlas-000.webp"


def test_webp_variants_emit_matching_files_and_manifests(tmp_path):
    gradient_strip(tmp_path, size=(64, 64), frames=3)
    req = request_v1(tmp_path, [{"id": "hero", "source": "hero.strip.json", "display": {"height": 64}}])

    # Multipage + lossless + q90 + q80
    out = tmp_path / "variants"
    result = game_pack.pack(req, out, page_size=70, webp_variants="lossless,q90,q80")
    assert len(result["textures"]) == 3
    names = {p.name for p in out.iterdir()}
    expected = {"game-pack.json", "game-pack_q90.json", "game-pack_q80.json"}
    for idx in range(3):
        expected.update({
            f"atlas-{idx:03d}.png",
            f"atlas-{idx:03d}.webp",
            f"atlas-{idx:03d}_q90.webp",
            f"atlas-{idx:03d}_q80.webp",
        })
    assert names == expected
    assert not (out / "game-pack_lossless.json").exists()

    q90 = json.loads((out / "game-pack_q90.json").read_text(encoding="utf-8"))
    assert q90["webp_variant"] == {
        "name": "q90",
        "lossless": False,
        "quality": 90,
        "alpha_quality": 95,
        "method": 6,
    }
    assert q90["animations"] == result["animations"]
    for idx, tex in enumerate(q90["textures"]):
        assert tex["image"] == f"atlas-{idx:03d}.png"
        assert tex["meta"]["image"] == f"atlas-{idx:03d}.png"
        assert tex["webp"] == f"atlas-{idx:03d}_q90.webp"
        assert tex["frames"] == result["textures"][idx]["frames"]

    q80 = json.loads((out / "game-pack_q80.json").read_text(encoding="utf-8"))
    assert q80["webp_variant"] == {
        "name": "q80",
        "lossless": False,
        "quality": 80,
        "alpha_quality": 90,
        "method": 6,
    }

    # Repeated invocation with identical inputs succeeds idempotently.
    game_pack.pack(req, out, page_size=70, webp_variants="lossless,q90,q80")

    # --no-png combined with --webp-variants points image and meta.image to variant WebP too.
    out_nopng = tmp_path / "variants-nopng"
    game_pack.pack(req, out_nopng, no_png=True, webp_variants="q75")
    assert sorted(p.name for p in out_nopng.iterdir()) == [
        "atlas-000.webp",
        "atlas-000_q75.webp",
        "game-pack.json",
        "game-pack_q75.json",
    ]
    q75 = json.loads((out_nopng / "game-pack_q75.json").read_text(encoding="utf-8"))
    assert q75["webp_variant"] == {
        "name": "q75",
        "lossless": False,
        "quality": 75,
        "alpha_quality": 85,
        "method": 6,
    }
    assert q75["textures"][0]["image"] == "atlas-000_q75.webp"
    assert q75["textures"][0]["meta"]["image"] == "atlas-000_q75.webp"
    assert q75["textures"][0]["webp"] == "atlas-000_q75.webp"


def test_webp_variants_budget_is_checked_before_publish(tmp_path):
    gradient_strip(tmp_path, size=(64, 64), frames=1)
    req = request_v1(tmp_path, [{"id": "hero", "source": "hero.strip.json", "display": {"height": 64}}])
    base_out = tmp_path / "base"
    game_pack.pack(req, base_out)
    base_total = sum(p.stat().st_size for p in base_out.iterdir())

    # Fits without variants, exceeds when q90+q80 variants are included.
    with pytest.raises(ValueError, match="bytes"):
        game_pack.pack(req, tmp_path / "over-budget", max_bytes=base_total + 10, webp_variants="q90,q80")
    assert not (tmp_path / "over-budget").exists()


@pytest.mark.parametrize("kwargs,match", [
    ({"webp_quality": 0}, "webp quality"),
    ({"webp_quality": 101}, "webp quality"),
    ({"webp_quality": "bad"}, "webp quality"),
    ({"webp_quality": True}, "webp quality"),
    ({"webp_alpha_quality": 0}, "webp alpha quality"),
    ({"webp_alpha_quality": 101}, "webp alpha quality"),
    ({"webp_alpha_quality": True}, "webp alpha quality"),
    ({"webp_variants": ""}, "empty"),
    ({"webp_variants": "q90,"}, "empty"),
    ({"webp_variants": "jpg"}, "unsupported webp variant"),
    ({"webp_variants": "q0"}, "unsupported webp variant"),
    ({"webp_variants": "q101"}, "unsupported webp variant"),
    ({"webp_quality": 90, "webp_variants": "lossless,q80"}, "requires --webp-quality lossless"),
    ({"allow_large": "yes"}, "bool"),
    ({"no_png": "yes"}, "bool"),
    ({"max_bytes": -1, "allow_large": True}, "non-negative"),
    ({"max_bytes": True, "allow_large": True}, "finite number"),
])
def test_invalid_extension_arguments_fail_cleanly(tmp_path, kwargs, match):
    gradient_strip(tmp_path, size=(8, 8), frames=1)
    req = request_v1(tmp_path, [{"id": "hero", "source": "hero.strip.json", "display": {"height": 8}}])
    with pytest.raises(ValueError, match=match):
        game_pack.pack(req, tmp_path / "bad", **kwargs)
    assert not (tmp_path / "bad").exists()
