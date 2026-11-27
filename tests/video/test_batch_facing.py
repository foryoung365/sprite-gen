# SPDX-License-Identifier: Apache-2.0
"""Run the batch wiring with a real canvas and an offline video runner."""
import json
from pathlib import Path

import pytest

from sprite_gen.video import batch

FIXTURE = Path(__file__).parents[1] / "fixtures" / "facing" / "left.png"


@pytest.fixture
def offline(monkeypatch):
    monkeypatch.setattr(batch.facing_mod.vision, "grok_inspect", lambda *a, **kw: ("left", {}))
    monkeypatch.setattr(batch.frames_mod, "run_frames", lambda clip, out, **kw: {
        "fps": 24, "frames": 10, "alpha_zero_pct_min": 50, "alpha_zero_pct_max": 60,
        "keyed_dir": str(out)})
    monkeypatch.setattr(batch.loop_mod, "run_loop", lambda *a, **kw: {
        "cycle": {"length": 10, "period_global": 10, "ratio": 0.2},
        "resampled_seam_ratio": 0.2, "held_fill": {"cells": 10, "poses": 10, "held_runs": [], "made": [], "before": {"cells": 10, "poses": 10, "held_runs": []}}, "n_out": 10, "gif": {"file": "loop.gif"},
        "webp": {"file": "loop.webp"}, "strip": {"path": "strip.png"}})
    monkeypatch.setattr(batch, "_staggered_start", lambda gap: None)


# The side canvas and prompt take the facing; front and back are always canvassed facing right.
@pytest.mark.parametrize("direction,facing,options", [
    ("side", "left", {}),
    ("side", "left", {"facing_fix": "mirror"}),
    ("side", "right", {}),
    ("side", "right", {"facing_fix": "mirror"}),
    ("front", "left", {}),
], ids=["side-left", "side-left-mirror", "side-right", "side-right-mirror", "front-left"])
def test_set_facing_reaches_both_real_canvas_and_prompt(tmp_path, offline, facing, direction, options):
    prompts = []
    source_bytes = FIXTURE.read_bytes()
    def video(image, prompt, out, report, **kw):
        prompts.append(prompt)
        out.write_bytes(b"test-video")
        report.write_text(json.dumps({"prompt": prompt}))
        return 0
    result = batch.run_set(bases={direction: FIXTURE}, states=["attack"], root=tmp_path,
                           character=None, duration=3, resolution="720p", key="green",
                           concurrency=1, force=False, gap=0, facing=facing, video_runner=video, **options)
    assert result["ok"] == 1 and result["facing"] == facing
    assert FIXTURE.read_bytes() == source_bytes
    canvas = json.loads((tmp_path / f"{direction}-attack" / "canvas.report.json").read_text())
    if direction == "side":
        assert f"facing {facing}" in prompts[0]
        assert result["items"][0]["facing"]["action"] == ("mirror" if facing == "right" and options else "none")
        if not options:
            assert (tmp_path / "side.facing.png").read_bytes() == source_bytes
            assert result["items"][0]["facing"]["final_direction_source"] == "observation"
        behind = round(canvas["canvas"][0] * canvas["trail"])
        assert canvas["facing"] == facing
        assert canvas["offset"][0] == (behind if facing == "right" else canvas["canvas"][0] - canvas["still"][0] - behind)
    else:
        assert "facing" not in result["items"][0]
        assert prompts[0] == batch.build_prompt(direction, "attack", None, facing="right")
        assert canvas["facing"] == "right"
        assert canvas["offset"][0] == round(canvas["canvas"][0] * canvas["trail"])


def test_changed_facing_cannot_reuse_an_opposite_prompt_clip(tmp_path, offline, monkeypatch):
    calls = []
    def video(image, prompt, out, report, **kw):
        calls.append(1)
        out.write_bytes(b"cached")
        report.write_text(json.dumps({"prompt": prompt}))
        return 0
    options = dict(item="side-attack", direction="side", state="attack", base=FIXTURE,
                   root=tmp_path, character=None, duration=3, resolution="720p", key="green",
                   force=False, gap=0, video_runner=video)
    result = batch.run_item(**options)
    assert result["ok"]
    item = tmp_path / "side-attack"
    previous_canvas = (item / "canvas.png").read_bytes()
    def no_vision(*a, **kw):
        pytest.fail("cached clip must not trigger another inspection")
    monkeypatch.setattr(batch.facing_mod.vision, "grok_inspect", no_vision)
    result = batch.run_item(**options, facing="left")
    assert not result["ok"] and "--force" in result["error"]
    assert (item / "canvas.png").read_bytes() == previous_canvas
    result = batch.run_item(**options)
    assert result["ok"] and result["clip"]["reused"] and len(calls) == 1
    result = batch.run_item(**options, facing_fix="mirror")
    assert not result["ok"] and "--force" in result["error"]


def test_cli_threads_facing_into_set(monkeypatch, tmp_path):
    seen = []
    def run_set(**kw):
        seen.append(kw)
        return {"failed": []}
    monkeypatch.setattr(batch, "run_set", run_set)
    assert batch.main(["--base", f"side={FIXTURE}", "--states", "attack", "--out-dir", str(tmp_path),
                       "--facing", "left"]) == 0
    assert seen[0]["facing"] == "left" and seen[0]["facing_fix"] == "none"


def test_parallel_states_share_one_inspection_of_the_side_still(tmp_path, offline, monkeypatch):
    calls = []
    def inspect(*a, **kw):
        calls.append(1)
        return "left", {}
    monkeypatch.setattr(batch.facing_mod.vision, "grok_inspect", inspect)
    def video(image, prompt, out, report, **kw):
        out.write_bytes(b"test-video")
        report.write_text(json.dumps({"prompt": prompt}))
        return 0
    source = FIXTURE.with_name("right.png")
    result = batch.run_set(bases={"side": source}, states=["idle", "attack", "walk"], root=tmp_path,
                           character=None, duration=3, resolution="720p", key="green",
                           concurrency=3, force=False, gap=0, video_runner=video)
    assert result["ok"] == 3 and len(calls) == 1
    assert len({item["facing"]["out"] for item in result["items"]}) == 1
    assert (tmp_path / "side.facing.png").read_bytes() == source.read_bytes()
    assert all(item["facing"]["action"] == "none" and item["facing"]["requested"] == "right"
               for item in result["items"])
    assert result["facing_fix"] == "none"


def test_tool_defaults_to_record_only(monkeypatch, tmp_path):
    seen = []
    def run_set(**kwargs):
        seen.append(kwargs)
        return {"failed": []}
    monkeypatch.setattr(batch, "run_set", run_set)
    assert batch.run(base=[f"side={FIXTURE}"], out_dir=tmp_path) == 0
    assert seen[0]["facing_fix"] == "none"


def test_magi2_batch_auto_routes_magenta_canvas_bottom_pad_and_frames_key(tmp_path, monkeypatch):
    """Pinned Higgsfield items routed to magi_2 automatically rekey canvas to magenta with bottom_pad=0.14
    and pass key='magenta' to video-frames, while threading --action-retime to video-loop."""
    from PIL import Image, ImageDraw
    from sprite_gen.gen import higgsfield_video

    monkeypatch.delenv(higgsfield_video.PROVIDER_ENV, raising=False)
    monkeypatch.setenv(higgsfield_video.MODEL_ENV, "magi_2")
    monkeypatch.setattr(batch, "_facing_vision", lambda: None)
    monkeypatch.setattr(batch.facing_mod.vision, "grok_inspect", lambda *a, **kw: ("left", {}))
    monkeypatch.setattr(batch, "_staggered_start", lambda gap: None)

    green_still = tmp_path / "green_side.png"
    im = Image.new("RGB", (120, 160), (0, 255, 0))
    ImageDraw.Draw(im).rectangle((36, 32, 84, 150), fill=(180, 60, 45))
    im.save(green_still)

    seen_frames: list[dict] = []
    seen_loop: list[dict] = []

    def fake_frames(clip, out, **kw):
        seen_frames.append(kw)
        return {"fps": 25.0, "frames": 106, "alpha_zero_pct_min": 80.0, "alpha_zero_pct_max": 90.0, "keyed_dir": str(out)}

    def fake_loop(frames_dir, out, **kw):
        seen_loop.append(kw)
        return {
            "cycle": {"kind": "pinned", "length": 105, "period_global": None, "ratio": 0.1},
            "resampled_seam_ratio": 0.1,
            "held_fill": {"cells": 24, "poses": 24, "held_runs": [], "made": [], "before": {"cells": 24, "poses": 24, "held_runs": []}},
            "action_retime": {"mode": kw["action_retime"], "applied": True, "why": "trimmed", "frames_before": 105, "frames_after": 24,
                              "duration_before": 4.2, "duration_after": 0.96, "dead_hold_trimmed_s": 1.6},
            "n_out": 24, "gif": {"file": "loop.gif"}, "webp": {"file": "loop.webp"}, "strip": {"path": "strip.png"},
        }

    def fake_video(image, prompt, out, report, **kw):
        out.write_bytes(b"test-video")
        report.write_text(json.dumps({"prompt": prompt}))
        return 0

    monkeypatch.setattr(batch.frames_mod, "run_frames", fake_frames)
    monkeypatch.setattr(batch.loop_mod, "run_loop", fake_loop)

    res = batch.run_set(
        bases={"side": green_still}, states=["attack"], root=tmp_path / "set", character=None, duration=None,
        resolution="720p", key="green", concurrency=1, force=False, gap=0.0, facing="left",
        video_runner=fake_video, action_retime="punch",
    )
    assert res["ok"] == 1
    canvas_rep = json.loads((tmp_path / "set" / "side-attack" / "canvas.report.json").read_text())
    assert canvas_rep["key"] == "magenta"
    assert canvas_rep["rekeyed_from"] == "green"
    assert canvas_rep["bottom_pad"] == batch.MAGI2_BOTTOM_PAD
    assert seen_frames[0]["key"] == "magenta"
    assert seen_loop[0]["action_retime"] == "punch"
    assert res["items"][0]["loop"]["action_retime"]["applied"] is True
    assert res["items"][0]["loop"]["action_retime"]["mode"] == "punch"


def test_uses_magi2_respects_provider_and_model_env(monkeypatch):
    from sprite_gen.gen import higgsfield_video

    monkeypatch.delenv(higgsfield_video.PROVIDER_ENV, raising=False)
    monkeypatch.delenv(higgsfield_video.MODEL_ENV, raising=False)
    assert batch.uses_magi2(pins=True) is True
    assert batch.uses_magi2(pins=False) is False
    assert batch.uses_magi2(pins=False, end_frame=Path("end.png")) is True

    monkeypatch.setenv(higgsfield_video.MODEL_ENV, "wan3_0")
    assert batch.uses_magi2(pins=True) is False

    monkeypatch.setenv(higgsfield_video.MODEL_ENV, "magi_2")
    assert batch.uses_magi2(pins=False) is True

    monkeypatch.setenv(higgsfield_video.PROVIDER_ENV, "grok")
    assert batch.uses_magi2(pins=True) is False


def test_batch_threads_fix_clauses_hit_stop_and_color_lock(tmp_path, offline, monkeypatch):
    seen_prompts: list[str] = []
    seen_loop: list[dict] = []

    def fake_video(image, prompt, out, report, **kw):
        seen_prompts.append(prompt)
        out.write_bytes(b"test-video")
        report.write_text(json.dumps({"prompt": prompt}))
        return 0

    def fake_loop(*a, **kw):
        seen_loop.append(kw)
        return {
            "cycle": {"length": 10, "period_global": 10, "ratio": 0.2},
            "resampled_seam_ratio": 0.2,
            "held_fill": {"cells": 10, "poses": 10, "held_runs": [], "made": [], "before": {"cells": 10, "poses": 10, "held_runs": []}},
            "n_out": 10, "gif": {"file": "loop.gif"}, "webp": {"file": "loop.webp"}, "strip": {"path": "strip.png"},
        }

    monkeypatch.setattr(batch.loop_mod, "run_loop", fake_loop)

    res = batch.run_set(
        bases={"side": FIXTURE}, states=["walk", "attack"], root=tmp_path / "set_fix",
        character=None, duration=3, resolution="720p", key="magenta", concurrency=1,
        force=False, gap=0, facing="left", video_runner=fake_video,
        fix=["camera-locked", "feet-planted"], hit_stop_ms=120, color_lock="on", color_lock_strength=0.75,
    )
    assert res["ok"] == 2
    assert res["fix"] == ["camera-locked", "feet-planted"]
    walk_item = next(it for it in res["items"] if it["state"] == "walk")
    atk_item = next(it for it in res["items"] if it["state"] == "attack")

    # walk automatically rewrites feet-planted -> feet-ground-line; attack keeps feet-planted
    assert walk_item["fix"]["ids"] == ["camera-locked", "feet-ground-line"]
    assert walk_item["fix"]["rewritten"][0]["to"] == "feet-ground-line"
    assert atk_item["fix"]["ids"] == ["camera-locked", "feet-planted"]
    assert atk_item["fix"]["rewritten"] == []
    assert any("Every step lands on the same ground line" in p and "Both feet stay planted" not in p for p in seen_prompts)
    assert any("Both feet stay planted" in p for p in seen_prompts)

    # hit_stop_ms and color_lock options threaded to loop_mod.run_loop
    assert all(kw["hit_stop_ms"] == 120 for kw in seen_loop)
    assert all(kw["color_lock"] == "on" and kw["color_lock_strength"] == 0.75 for kw in seen_loop)

    # CLI argument wiring check
    cli_seen: list[dict] = []
    monkeypatch.setattr(batch, "run_set", lambda **kw: (cli_seen.append(kw), {"failed": []})[1])
    assert batch.main([
        "--base", f"side={FIXTURE}", "--states", "attack", "--out-dir", str(tmp_path / "cli_out"),
        "--fix", "camera-locked", "--fix", "inside-frame",
        "--hit-stop-ms", "100", "--color-lock", "on", "--color-lock-strength", "0.85",
    ]) == 0
    assert cli_seen[0]["fix"] == ["camera-locked", "inside-frame"]
    assert cli_seen[0]["hit_stop_ms"] == 100
    assert cli_seen[0]["color_lock"] == "on"
    assert cli_seen[0]["color_lock_strength"] == 0.85
