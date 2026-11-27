# SPDX-License-Identifier: Apache-2.0
"""`sprite-gen video` on Higgsfield (`--provider higgsfield`, the default here; `SPRITE_GEN_VIDEO_PROVIDER`).

Which Higgsfield job a clip becomes:

- a still alone: `grok_video_v15`, the same Grok Imagine Video 1.5 sprite-gen's clip prompts were measured on,
  which takes a start image and no end image;
- a pinned last frame: `magi_2`, which takes `start_image` and `end_image` at 1080p, 4 s minimum duration,
  an explicit aspect ratio matched to the input still (`21:9`, `16:9`, `4:3`, `1:1`, `3:4`, `9:16`), and
  `generate_audio`, sent false unless asked; a post-download first-frame framing guard checks for zoom/crop drift
  and records `extra.framing_guard` (use `--model wan3_0` or `SPRITE_GEN_HIGGSFIELD_VIDEO_MODEL=wan3_0` for Wan 3.0);
- reference images: `seedance_2_0`, which takes them beside a start and an end image (`wan3_0` and `magi_2` do not);
- `SPRITE_GEN_HIGGSFIELD_VIDEO_MODEL` names one job for every clip (how `video-set`, which passes no model, is
  filmed on another), e.g. `wan3_0` or `kling4_0`, which takes both frames at 720p or 1080p and is told its mode here;
- `--model <job_type>` names any other Higgsfield video job; a parameter it does not take is refused by
  `generate cost` before anything is spent. The Grok model names (`grok-imagine-video*`) are not Higgsfield jobs
  and pick the defaults above.

A job that does not finish in time is named by its id and not resubmitted: `sprite-gen video-resume
--request-id <id>` (or `higgsfield generate get <id>`) fetches it later. `video-extend` and `video-edit` run on
Grok only.
"""
from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from PIL import Image

from . import higgsfield, prompt_parts
from .base import PROVIDER_GROK, PROVIDER_HIGGSFIELD

PROVIDERS = (PROVIDER_HIGGSFIELD, PROVIDER_GROK)
PROVIDER_ENV = "SPRITE_GEN_VIDEO_PROVIDER"
MODEL_ENV = "SPRITE_GEN_HIGGSFIELD_VIDEO_MODEL"
STILL_MODEL = "grok_video_v15"
PINNED_MODEL = "magi_2"
REFERENCE_MODEL = "seedance_2_0"
MAGI2_MODELS = ("magi_2",)
MAGI2_RESOLUTION = "1080p"
MAGI2_MIN_DURATION = 4
MAGI2_ASPECT_RATIOS = ("21:9", "16:9", "4:3", "1:1", "3:4", "9:16")
# Jobs with an `auto` aspect ratio and a `generate_audio` switch (the others infer the ratio from the still).
AUTO_ASPECT_MODELS = ("seedance_2_0", "seedance_2_5", "wan3_0", "minimax_h3", "kling4_0")
# Jobs whose `mode` says what the inputs are.
MODE_MODELS = ("kling4_0",)
AUDIO_SWITCH_MODELS = ("magi_2", "seedance_2_0", "seedance_2_5", "wan3_0", "veo3_1", "veo3_1_lite")
TIMEOUT_SECONDS = 1800


def _parse_ratio(label: str) -> float:
    w, h = label.split(":")
    return float(w) / float(h)


def _nearest_aspect_ratio(
    image: Path | None,
    choices: tuple[str, ...] = MAGI2_ASPECT_RATIOS,
    default: str = "16:9",
) -> str:
    """Match `image`'s width/height ratio to the closest supported ratio label, falling back to `default`."""
    if image is None:
        return default
    try:
        with Image.open(image) as im:
            w, h = im.size
    except (OSError, ValueError):
        return default
    if w <= 0 or h <= 0:
        return default
    ratio = w / float(h)
    return min(choices, key=lambda label: abs(math.log(ratio / _parse_ratio(label))))


def video_provider(named: str | None = None) -> str:
    chosen = named or (os.environ.get(PROVIDER_ENV) or "").strip() or PROVIDERS[0]
    if chosen not in PROVIDERS:
        raise SystemExit(f"video: provider {chosen!r} is not one of {', '.join(PROVIDERS)} ({PROVIDER_ENV})")
    return chosen


def model_for(request) -> str:
    if request.model and not request.model.startswith("grok-imagine"):
        return request.model
    if (named := (os.environ.get(MODEL_ENV) or "").strip()):
        return named
    if request.reference_images:
        return REFERENCE_MODEL
    if request.last_frame is not None:
        return PINNED_MODEL
    return STILL_MODEL


def job_for(request, image: Path | None, last_frame: Path | None, references: list[Path]) -> higgsfield.Job:
    model = model_for(request)
    duration = int(request.duration)
    resolution = request.resolution
    if model in MAGI2_MODELS:
        resolution = MAGI2_RESOLUTION
        duration = max(duration, MAGI2_MIN_DURATION)
        aspect = request.aspect_ratio or _nearest_aspect_ratio(image)
    else:
        aspect = request.aspect_ratio or ("auto" if model in AUTO_ASPECT_MODELS else None)
    params: dict[str, Any] = {"prompt": request.prompt, "duration": duration,
                              "resolution": resolution}
    if aspect:
        params["aspect_ratio"] = aspect
    if model in AUDIO_SWITCH_MODELS:
        params["generate_audio"] = bool(request.generate_audio)
    if model in MODE_MODELS:
        params["mode"] = "reference-to-video" if references else "image-to-video"
    media: dict[str, list[Path]] = {}
    if image is not None:
        media["start-image"] = [image]
    if last_frame is not None:
        media["end-image"] = [last_frame]
    if references:
        media["image-references"] = list(references)
    return higgsfield.Job(model, params, media)


def _publish_mp4(data: bytes, out: Path) -> None:
    from .video import _is_mp4, _staged
    if not _is_mp4(data):
        raise SystemExit("video: the Higgsfield result is not an MP4; nothing published")
    out.parent.mkdir(parents=True, exist_ok=True)
    staged = _staged(out)
    staged.write_bytes(data)
    os.replace(staged, out)


def generate_video(request):
    """One clip on Higgsfield, as a VideoResult. Raises SystemExit on any failure."""
    from sprite_gen.video import facing as facing_mod

    from .higgsfield_provider import HiggsfieldProvider
    from .video import VideoResult, _resolved, _validate, side_view_prompt

    _validate(request)
    out = request.out.expanduser().resolve()
    image = _resolved(request.image)
    last_frame = _resolved(request.last_frame)
    references = [path.expanduser().resolve() for path in request.reference_images]
    facing_report = None
    if request.direction == "side":
        parts = side_view_prompt(request.prompt, request.facing)
        for line in prompt_parts.note_lines(parts):
            print(f"[video] {line}", file=sys.stderr)
        request = replace(request, prompt=parts.text)
        corrected = out.with_suffix(".facing.png")
        facing_report = facing_mod.prepare_still(image, corrected, facing=request.facing, fix=request.facing_fix,
                                                 vision=HiggsfieldProvider())
        image = corrected
    job = job_for(request, image, last_frame, references)
    started = time.monotonic()
    data, report = higgsfield.run_job(job, who="video", timeout=TIMEOUT_SECONDS,
                                      detail=f" ({job.params['duration']} s, {job.params['resolution']})")
    _publish_mp4(data, out)
    extra: dict[str, Any] = {"credits": report["credits"], "transport": report["transport"]}
    if job.job_type in MAGI2_MODELS and image is not None:
        try:
            from sprite_gen.video import framing_guard
            guard = framing_guard.check_first_frame_scale(out, image)
            extra["framing_guard"] = guard
            if guard.get("checked") and guard.get("passed") is False:
                print(
                    f"[video] warning: framing_guard flagged {out.name}: {guard.get('reason')} "
                    "(MP4 was kept; consider regenerating or passing --allow-reframe to video-frames)",
                    file=sys.stderr,
                )
        except Exception as exc:
            extra["framing_guard"] = {"checked": False, "passed": True, "reason": f"check failed: {exc}"}
            print(f"[video] warning: framing_guard could not inspect {out.name}: {exc}", file=sys.stderr)
    inputs: dict[str, Any] = {}
    if image is not None:
        inputs["image"] = str(image)
    if last_frame is not None:
        inputs["last_frame"] = str(last_frame)
    if references:
        inputs["reference_images"] = [str(path) for path in references]
    return VideoResult(
        out=out, bytes=len(data), model=job.job_type, request_id=report["job_id"],
        auth_source=report["auth_source"], elapsed_seconds=time.monotonic() - started, polls=0,
        duration_requested=request.duration, image=image, prompt=request.prompt,
        resolution=str(job.params.get("resolution") or request.resolution),
        aspect_ratio=job.params.get("aspect_ratio"), generate_audio=job.params.get("generate_audio"),
        mode=request.mode, inputs=inputs, facing=facing_report, provider=PROVIDER_HIGGSFIELD,
        extra=extra,
    )


def resume_video(request):
    """A Higgsfield job an earlier run left unfinished, by its id: waited for and downloaded, nothing bought."""
    from .video import VideoResult

    out = request.out.expanduser().resolve()
    started = time.monotonic()
    reply = higgsfield.wait(request.request_id, deadline=started + TIMEOUT_SECONDS)
    data = higgsfield.download(higgsfield.result_url(reply))
    _publish_mp4(data, out)
    params = reply.get("params") if isinstance(reply.get("params"), dict) else {}
    return VideoResult(
        out=out, bytes=len(data), model=reply.get("job_type"), request_id=request.request_id,
        auth_source="higgsfield-login", elapsed_seconds=time.monotonic() - started, polls=0,
        duration_requested=None, prompt=str(params.get("prompt") or ""), resolution=str(params.get("resolution") or ""),
        provider=PROVIDER_HIGGSFIELD, extra={"transport": "higgsfield-cli"},
    )
