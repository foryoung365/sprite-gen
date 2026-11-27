# SPDX-License-Identifier: Apache-2.0
"""Offline unit tests for first-frame scale / reframing guard (`framing_guard` & `video-frames`)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from sprite_gen.video import framing_guard, frames as frames_mod

MAGENTA = (255, 0, 255)
GREEN = (0, 255, 0)
IVORY = (248, 247, 242)
SUBJECT = (190, 60, 45)


def _sprite_frame(
    size: tuple[int, int] = (320, 180),
    *,
    bg: tuple[int, int, int] = MAGENTA,
    top: int = 45,
    bottom: int = 145,
    left: int = 130,
    right: int = 190,
) -> Image.Image:
    im = Image.new("RGB", size, bg)
    ImageDraw.Draw(im).rectangle((left, top, right, bottom), fill=SUBJECT)
    return im


def test_compare_subject_framing_passes_when_scale_matches() -> None:
    ref = _sprite_frame(top=45, bottom=145)
    f0 = _sprite_frame(top=44, bottom=147)
    res = framing_guard.compare_subject_framing(ref, f0)
    assert res["checked"] is True
    assert res["passed"] is True
    assert 0.98 <= res["scale_ratio"] <= 1.05
    assert res["ref_touches_bottom"] is False
    assert res["clip_touches_bottom"] is False
    assert res["reason"] is None


def test_compare_subject_framing_rejects_zoom_in_and_bottom_clip() -> None:
    ref = _sprite_frame(top=50, bottom=150)  # height 101/180 = 0.561, bottom margin 16%
    f0 = _sprite_frame(top=30, bottom=179)   # height 150/180 = 0.833 (~1.49x), touches bottom
    res = framing_guard.compare_subject_framing(ref, f0)
    assert res["checked"] is True
    assert res["passed"] is False
    assert res["scale_ratio"] > 1.4
    assert res["ref_touches_bottom"] is False
    assert res["clip_touches_bottom"] is True
    assert "zoomed in" in res["reason"] and "bottom edge" in res["reason"]


def test_compare_subject_framing_rejects_new_bottom_clip_even_under_15pct_drift() -> None:
    # Reference height 120 px (top=40..159), bottom margin 20/180 = 11.1% >= 3%.
    # Clip height 133 px (top=47..179) -> scale 1.108 (> 1.08, <= 1.15) but clips bottom.
    ref = _sprite_frame(top=40, bottom=159)
    f0 = _sprite_frame(top=47, bottom=179)
    res = framing_guard.compare_subject_framing(ref, f0)
    assert res["checked"] is True
    assert res["passed"] is False
    assert 1.08 < res["scale_ratio"] <= 1.15
    assert res["clip_touches_bottom"] is True
    assert "bottom edge" in res["reason"]


def test_compare_subject_framing_rejects_shrink_beyond_drift() -> None:
    ref = _sprite_frame(top=30, bottom=150)
    f0 = _sprite_frame(top=60, bottom=140)  # ~0.67x
    res = framing_guard.compare_subject_framing(ref, f0)
    assert res["checked"] is True
    assert res["passed"] is False
    assert res["scale_ratio"] < 0.85
    assert "shrunk" in res["reason"]


def test_compare_subject_framing_skips_non_chroma_reference() -> None:
    ref_ivory = _sprite_frame(bg=IVORY)
    res = framing_guard.compare_subject_framing(ref_ivory, _sprite_frame(bg=IVORY, top=10, bottom=179))
    assert res == {"checked": False, "passed": True, "reason": "non-chroma reference"}

    ref_rgba = Image.new("RGBA", (120, 120), (0, 0, 0, 0))
    ImageDraw.Draw(ref_rgba).rectangle((30, 30, 90, 90), fill=(*SUBJECT, 255))
    res_rgba = framing_guard.compare_subject_framing(ref_rgba, _sprite_frame(bg=GREEN))
    assert res_rgba == {"checked": False, "passed": True, "reason": "non-chroma reference"}


def test_run_frames_refuses_reframed_clip_before_keying_and_allows_with_flag(tmp_path: Path, monkeypatch) -> None:
    ref = tmp_path / "canvas.png"
    _sprite_frame(bg=MAGENTA, top=50, bottom=150).save(ref)
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"fake-mp4")

    raw_dir = tmp_path / "out_refused" / "raw"
    key_calls: list[int] = []

    def fake_probe(_clip: Path):
        return {"width": 320, "height": 180, "fps": 25.0, "nb_frames": 2, "duration": 0.08,
                "stream_index": 0, "codec": "h264", "pix_fmt": "yuv420p", "audio_streams": 0, "cover_streams": 0}

    def fake_extract(_clip: Path, dst_raw: Path, _stream: int | None):
        dst_raw.mkdir(parents=True, exist_ok=True)
        files = []
        for idx in range(2):
            p = dst_raw / f"frame-{idx:04d}.png"
            _sprite_frame(bg=MAGENTA, top=30, bottom=179).save(p)  # 1.49x zoomed
            files.append(p)
        return files

    def spy_key_frames(*args, **kwargs):
        key_calls.append(1)
        return {"frames": 2, "alpha_zero_pct_min": 80.0, "alpha_zero_pct_max": 80.0,
                "edge_contacts": [], "edge_policy": "border", "decontam": {"mode": "off", "applied_frames": 0}, "rows": []}

    monkeypatch.setattr(frames_mod, "probe", fake_probe)
    monkeypatch.setattr(frames_mod, "extract", fake_extract)
    monkeypatch.setattr(frames_mod, "key_frames", spy_key_frames)

    with pytest.raises(SystemExit, match="reframed.*--allow-reframe"):
        frames_mod.run_frames(
            clip, tmp_path / "out_refused", key="magenta", allow_edge_contact=False,
            report_path=None, spill="small", reference=ref,
        )
    assert key_calls == [], "framing guard must stop before expensive full-clip key_frames runs"
    rep = json.loads((tmp_path / "out_refused" / "frames.report.json").read_text())
    assert rep["status"] == "refused"
    assert rep["framing_guard"]["checked"] is True and rep["framing_guard"]["passed"] is False
    assert rep["framing_guard"]["scale_ratio"] > 1.4
    assert rep["suggested_fixes"] == ["camera-locked", "inside-frame"]

    # Passing check_framing=False (--allow-reframe) skips the gate and proceeds to key_frames.
    ok_rep = frames_mod.run_frames(
        clip, tmp_path / "out_allowed", key="magenta", allow_edge_contact=False,
        report_path=None, spill="small", reference=ref, check_framing=False,
    )
    assert ok_rep["status"] == "passed"
    assert len(key_calls) == 1


def test_run_frames_writes_suggested_fixes_on_framing_and_edge_contact_refusals(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    ref = tmp_path / "canvas.png"
    _sprite_frame(bg=MAGENTA, top=50, bottom=150).save(ref)
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"fake-mp4")

    def fake_probe(_clip: Path):
        return {"width": 320, "height": 180, "fps": 25.0, "nb_frames": 2, "duration": 0.08,
                "stream_index": 0, "codec": "h264", "pix_fmt": "yuv420p", "audio_streams": 0, "cover_streams": 0}

    def fake_extract_zoom(_clip: Path, dst_raw: Path, _stream: int | None):
        dst_raw.mkdir(parents=True, exist_ok=True)
        p = dst_raw / "frame-0000.png"
        _sprite_frame(bg=MAGENTA, top=30, bottom=179).save(p)
        return [p]

    monkeypatch.setattr(frames_mod, "probe", fake_probe)
    monkeypatch.setattr(frames_mod, "extract", fake_extract_zoom)

    with pytest.raises(SystemExit):
        frames_mod.run_frames(
            clip, tmp_path / "out_zoom", key="magenta", allow_edge_contact=False,
            report_path=None, spill="small", reference=ref,
        )
    stderr_zoom = capsys.readouterr().err
    assert "suggested --fix: camera-locked, inside-frame" in stderr_zoom
    rep_zoom = json.loads((tmp_path / "out_zoom" / "frames.report.json").read_text())
    assert rep_zoom["suggested_fixes"] == ["camera-locked", "inside-frame"]

    def fake_key_edge_refuse(*args, **kwargs):
        raise frames_mod.KeyingRefused(
            "video-frames: the subject touches the frame edge in 1 frame(s)",
            {"frames": 1, "edge_contacts": [{"frame": "frame-0000.png", "top": 12, "left": 0, "right": 0, "subject": 12, "residual": 0}]},
        )

    monkeypatch.setattr(frames_mod, "key_frames", fake_key_edge_refuse)
    with pytest.raises(frames_mod.KeyingRefused):
        frames_mod.run_frames(
            clip, tmp_path / "out_edge", key="magenta", allow_edge_contact=False,
            report_path=None, spill="small", reference=ref, check_framing=False,
        )
    stderr_edge = capsys.readouterr().err
    assert "suggested --fix: inside-frame" in stderr_edge
    rep_edge = json.loads((tmp_path / "out_edge" / "frames.report.json").read_text())
    assert rep_edge["suggested_fixes"] == ["inside-frame"]
