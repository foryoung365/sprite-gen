# SPDX-License-Identifier: Apache-2.0
"""Higgsfield transport and providers, offline: the CLI is a scripted fake and nothing is downloaded."""
import io
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sprite_gen import gen
from sprite_gen.audio import provider as sound
from sprite_gen.gen import higgsfield, higgsfield_provider, higgsfield_video
from sprite_gen.gen.base import GenRequest
from sprite_gen.workflow import access, catalog


def png_bytes(alpha_body=253, size=16):
    rgba = np.zeros((size, size, 4), np.uint8)
    rgba[4:12, 4:12] = (40, 80, 200, alpha_body)
    rgba[3, 4:12, 3] = 120  # a soft edge row
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    return buf.getvalue()


class FakeCli:
    """Answers `higgsfield <args> --json` from a script keyed by the subcommand, recording every call."""

    def __init__(self, *, credits=1.5, job_id="job-1", waits=None):
        self.calls = []
        self.credits = credits
        self.job_id = job_id
        self.waits = list(waits or [(0, {"id": job_id, "status": "completed", "result_url": "https://cdn/x.png"})])

    def __call__(self, command, **kwargs):
        args = command[1:-1]
        self.calls.append(args)
        if args[:2] == ["generate", "cost"]:
            return subprocess.CompletedProcess(command, 0, json.dumps({"credits": self.credits}), "")
        if args[:2] == ["generate", "create"]:
            return subprocess.CompletedProcess(command, 0, json.dumps([self.job_id]), "")
        if args[:2] == ["generate", "wait"]:
            code, answer = self.waits.pop(0)
            if code:
                return subprocess.CompletedProcess(command, code, "", answer)
            return subprocess.CompletedProcess(command, 0, json.dumps(answer), "")
        if args[:2] == ["account", "status"]:
            return subprocess.CompletedProcess(command, 0, json.dumps({"credits": 10, "email": "e",
                                                                       "subscription_plan_type": "creator"}), "")
        raise AssertionError(f"unexpected call {args}")

    def created(self):
        return [call for call in self.calls if call[:2] == ["generate", "create"]]


@pytest.fixture
def cli(monkeypatch):
    fake = FakeCli()
    monkeypatch.setattr(subprocess, "run", fake)
    monkeypatch.setattr(higgsfield, "download", lambda url, **_: png_bytes())
    monkeypatch.setattr(higgsfield.time, "sleep", lambda _s: None)
    return fake


def test_image_job_defaults_to_gpt_image_high_with_native_alpha(tmp_path):
    request = GenRequest(prompt="a knight", raw=tmp_path / "raw.png", refs=[tmp_path / "ref.png"],
                         aspect_ratio="2:3", native_alpha=True)
    job = higgsfield_provider.job_for(request)
    assert job.job_type == "gpt_image_2_5"
    assert job.params == {"prompt": "a knight", "quality": "high", "aspect_ratio": "2:3", "background": "transparent"}
    assert job.args()[-2:] == ["--image-references", str((tmp_path / "ref.png").resolve())]


@pytest.mark.parametrize("field, value, said", [("quality", "auto", None), ("resolution", "1.5k", "resolution"),
                                                ("quality", "ultra", "quality")])
def test_image_job_refuses_what_higgsfield_does_not_serve(tmp_path, field, value, said):
    request = GenRequest(prompt="p", raw=tmp_path / "raw.png", **{field: value})
    if said is None:
        assert "quality" in higgsfield_provider.job_for(request).params  # auto = the default model's own level
    else:
        with pytest.raises(SystemExit, match=said):
            higgsfield_provider.job_for(request)


def test_native_alpha_is_refused_on_a_model_without_background(tmp_path):
    request = GenRequest(prompt="p", raw=tmp_path / "raw.png", model="nano_banana_pro", native_alpha=True)
    with pytest.raises(SystemExit, match="alpha-mode chroma"):
        higgsfield_provider.job_for(request)


def test_alpha_knee_makes_the_body_opaque_and_keeps_the_edge():
    out, peak = higgsfield_provider.restore_alpha_peak(png_bytes(alpha_body=253))
    alpha = np.asarray(Image.open(io.BytesIO(out)))[..., 3]
    assert peak == 253
    assert alpha[8, 8] == 255
    assert alpha[3, 6] == round(120 * 255 / higgsfield_provider.ALPHA_KNEE)
    assert alpha[0, 0] == 0


def test_alpha_without_the_artifact_is_left_alone():
    data = png_bytes(alpha_body=255)
    assert higgsfield_provider.restore_alpha_peak(data) == (data, None)


def test_generate_is_the_default_and_prices_before_it_creates(tmp_path, cli, capsys):
    result = gen.generate_image(gen.resolve_default_provider()[0], "a knight", tmp_path / "out.png", transparent=True)
    assert result.provider == "higgsfield"
    assert [call[:2] for call in cli.calls] == [["generate", "cost"], ["generate", "create"], ["generate", "wait"]]
    assert "spends Higgsfield credits: 1.5 credits" in capsys.readouterr().err
    assert result.extra["job_id"] == "job-1" and result.extra["alpha_rescaled_from"] == 253
    assert result.alpha["strategy"] == "native"


def test_a_wait_with_no_answer_asks_again_and_never_recreates(tmp_path, monkeypatch):
    fake = FakeCli(waits=[(3, "Error: request failed (no response received)"),
                          (0, {"id": "job-1", "status": "completed", "result_url": "https://cdn/x.png"})])
    monkeypatch.setattr(subprocess, "run", fake)
    monkeypatch.setattr(higgsfield, "download", lambda url, **_: png_bytes())
    monkeypatch.setattr(higgsfield.time, "sleep", lambda _s: None)
    gen.generate_image("higgsfield", "p", tmp_path / "out.png")
    assert len(fake.created()) == 1
    assert sum(call[:2] == ["generate", "wait"] for call in fake.calls) == 2


def test_a_failed_job_names_its_id(tmp_path, monkeypatch):
    fake = FakeCli(waits=[(0, {"id": "job-1", "status": "nsfw"})])
    monkeypatch.setattr(subprocess, "run", fake)
    with pytest.raises(SystemExit, match="job job-1 ended nsfw"):
        gen.generate_image("higgsfield", "p", tmp_path / "out.png")
    assert len(fake.created()) == 1


def test_a_refused_parameter_spends_nothing(tmp_path, monkeypatch):
    def refuse(command, **kwargs):
        assert command[1:3] == ["generate", "cost"], "nothing may be created after a refused price"
        return subprocess.CompletedProcess(command, 4, "", "Error: Unknown params: quality")
    monkeypatch.setattr(subprocess, "run", refuse)
    with pytest.raises(SystemExit, match="refused before anything was spent"):
        gen.generate_image("higgsfield", "p", tmp_path / "out.png", model="nano_banana_pro", quality="low")


def _npm_shim(tmp_path, *, native: bool):
    """An npm install on Windows: `higgsfield.cmd` on PATH, with or without the package's own executable."""
    shim = tmp_path / "npm" / "higgsfield.cmd"
    shim.parent.mkdir()
    shim.write_text("@ECHO off\n")
    if native:
        exe = shim.parent / higgsfield.NATIVE_BESIDE_SHIM
        exe.parent.mkdir(parents=True)
        exe.write_bytes(b"MZ")
    return shim


def test_the_npm_shim_is_passed_over_for_the_package_executable(tmp_path, monkeypatch):
    shim = _npm_shim(tmp_path, native=True)
    monkeypatch.delenv(higgsfield.BINARY_ENV)
    monkeypatch.setattr(higgsfield, "provider_binary", lambda name: str(shim) if name == higgsfield.BINARY else name)
    assert higgsfield.binary() == str(shim.parent / higgsfield.NATIVE_BESIDE_SHIM)
    seen = []
    monkeypatch.setattr(subprocess, "run", lambda command, **_: seen.append(command) or
                        subprocess.CompletedProcess(command, 0, json.dumps({"credits": 1.5}), ""))
    job = higgsfield.Job("gpt_image_2_5", {"prompt": "a knight.\nThe view: front.", "quality": "high"})
    assert higgsfield.cost(job) == 1.5
    assert seen[0][0] == str(shim.parent / higgsfield.NATIVE_BESIDE_SHIM) and seen[0][-1] == "--json"


def test_a_shim_alone_refuses_a_prompt_of_several_lines_before_it_runs(tmp_path, monkeypatch):
    shim = _npm_shim(tmp_path, native=False)
    monkeypatch.delenv(higgsfield.BINARY_ENV)
    monkeypatch.setattr(higgsfield, "provider_binary", lambda name: str(shim) if name == higgsfield.BINARY else name)

    def never(command, **_):
        raise AssertionError("a shim must not be run with an argument it would cut short")
    monkeypatch.setattr(subprocess, "run", never)
    with pytest.raises(SystemExit, match="cuts an argument at its first line break"):
        higgsfield.cost(higgsfield.Job("gpt_image_2_5", {"prompt": "a knight.\nThe view: front."}))


def test_the_binary_env_still_names_the_cli(tmp_path, monkeypatch):
    exe = tmp_path / "hf.exe"
    monkeypatch.setenv(higgsfield.BINARY_ENV, str(exe))
    assert higgsfield.binary() == str(exe)


def test_sound_defaults_to_higgsfield_jobs(monkeypatch):
    monkeypatch.delenv(sound.PROVIDER_ENV, raising=False)
    assert sound.request("sfx", "whoosh", 2, False) == {
        "provider": "higgsfield", "job_type": "mirelo_text_to_audio",
        "params": {"prompt": "whoosh", "duration": 2.0}, "loop": False}
    assert sound.request("bgm", "calm", 30, True)["job_type"] == "sonilo_music"
    monkeypatch.setenv(sound.PROVIDER_ENV, "elevenlabs")
    assert sound.request("sfx", "whoosh", 2, False)["endpoint"] == "sound-generation"


def test_sound_on_higgsfield_returns_the_audio_and_its_receipt(cli, monkeypatch):
    monkeypatch.delenv(sound.PROVIDER_ENV, raising=False)
    monkeypatch.setattr(higgsfield, "download", lambda url, **_: b"ID3audio")
    data, meta = sound.generate(sound.request("sfx", "whoosh", 2, False))
    assert data == b"ID3audio"
    assert meta["provider"] == "higgsfield" and meta["model"] == "mirelo_text_to_audio"


class _Clip:
    def __init__(self, tmp_path, **kw):
        self.model = kw.get("model", "grok-imagine-video-1.5")
        self.last_frame = kw.get("last_frame")
        self.reference_images = kw.get("reference_images", [])
        self.prompt, self.duration, self.resolution = "walk", 3, "720p"
        self.aspect_ratio, self.generate_audio = None, False


@pytest.mark.parametrize("kw, model", [({}, "grok_video_v15"), ({"last_frame": "end.png"}, "wan3_0"),
                                       ({"reference_images": ["a.png"]}, "seedance_2_0"),
                                       ({"model": "minimax_h3"}, "minimax_h3")])
def test_video_model_follows_what_the_clip_pins(tmp_path, kw, model):
    assert higgsfield_video.model_for(_Clip(tmp_path, **kw)) == model


def test_pinned_video_job_takes_both_frames_and_auto_ratio(tmp_path):
    job = higgsfield_video.job_for(_Clip(tmp_path, last_frame="end.png"), Path("start.png"), Path("end.png"), [])
    assert job.params["aspect_ratio"] == "auto" and job.params["generate_audio"] is False
    assert job.media == {"start-image": [Path("start.png")], "end-image": [Path("end.png")]}


def test_the_model_env_films_every_clip_and_kling_is_told_its_mode(tmp_path, monkeypatch):
    monkeypatch.setenv(higgsfield_video.MODEL_ENV, "kling4_0")
    job = higgsfield_video.job_for(_Clip(tmp_path, last_frame="end.png"), Path("s.png"), Path("e.png"), [])
    assert job.job_type == "kling4_0"
    assert job.params["mode"] == "image-to-video" and job.params["aspect_ratio"] == "auto"
    assert "generate_audio" not in job.params


def test_access_reads_the_account_and_needs_no_confirmation(cli):
    state = access.probe_access("higgsfield")
    assert state["login"] == "ready" and state["billing"] == "credits" and state["known"]
    assert catalog.GUIDED_PROVIDERS[0] == "higgsfield"
