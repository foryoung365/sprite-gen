# SPDX-License-Identifier: Apache-2.0
"""Image generation and editing on Higgsfield (`--provider higgsfield`), through the signed-in CLI.

The default model is GPT Image 2.5 (`gpt_image_2_5`), which returns a genuinely transparent PNG when asked
(`background=transparent`: 2026-10-30 probe, RGBA with 94 % of the pixels at alpha 0), so the provider's
transparency is `native`. `--model` names any other Higgsfield image job type (`higgsfield model list --image`,
e.g. `nano_banana_pro`); a parameter that model does not take is refused by `generate cost` before anything is
spent, and native alpha on a model with no `background` parameter is refused here (use `--alpha-mode chroma`).

Quality is GPT Image's own ladder, which is the shared vocabulary minus `auto`; omitted, the default model draws
at `high` (1.5 credits at 1k, 2026-10-30), other models at their own default. Resolution takes `1k` and `2k`;
`1.5k` is a Grok tier Higgsfield does not serve and is refused.

Higgsfield has no vision model: the facing check and the logo's text read run on the Claude subscription
(`facing_vision.claude_inspect`).
"""
from __future__ import annotations

import time
from pathlib import Path

from . import higgsfield
from .base import GEN_TIMEOUT_SECONDS, PROVIDER_HIGGSFIELD, TRANSPARENCY_NATIVE, GenRequest, ProviderRun, publish_png

DEFAULT_MODEL = "gpt_image_2_5"
DEFAULT_QUALITY = "high"
SUPPORTED_QUALITIES = ("low", "medium", "high", "xhigh", "max")
SUPPORTED_RESOLUTIONS = ("1k", "2k")
# Models whose `background` parameter can ask for alpha.
ALPHA_MODELS = ("gpt_image_2", "gpt_image_2_5")
# Alpha at or above this is opaque in a Higgsfield transparent PNG (`restore_alpha_peak`).
ALPHA_KNEE = 240
# A job waits at least this long: GPT Image at high quality takes longer than the shared 180 s default.
MIN_TIMEOUT_SECONDS = 600


def restore_alpha_peak(data: bytes) -> tuple[bytes, int | None]:
    """Higgsfield's transparent PNGs come back with no pixel quite opaque: a figure's body is noise between 246 and
    254 (2026-10-30, a 1024x1024 knight: 0 pixels at 255, 211,685 at 250-253, 6,700 at 246-249, 3,055 at 240-245,
    then a thin tail down to the outline). Downstream every such pixel reads as partial alpha. The channel is mapped
    through one knee: `ALPHA_KNEE` and above is opaque, below it is scaled by 255/`ALPHA_KNEE`, so an edge keeps a
    continuous gradient. Applied only when the visible pixels' 99th percentile sits in [knee, 255), the shape of
    this artifact. Returns the PNG and that percentile."""
    import io

    from PIL import Image

    with Image.open(io.BytesIO(data)) as source:
        if source.mode != "RGBA":
            return data, None
        image = source.convert("RGBA")
    alpha = image.getchannel("A")
    # The 99th percentile of the visible pixels, read off the channel's histogram (NumPy stays in one module).
    counts = alpha.histogram()
    visible = sum(counts[1:])
    if visible == 0:
        return data, None
    seen, peak = 0, 255
    for value in range(1, 256):
        seen += counts[value]
        if seen * 100 >= visible * 99:
            peak = value
            break
    if not ALPHA_KNEE <= peak < 255:
        return data, None
    image.putalpha(alpha.point([min(255, round(value * 255 / ALPHA_KNEE)) for value in range(256)]))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue(), peak


def job_for(request: GenRequest) -> higgsfield.Job:
    """The Higgsfield job a request asks for, checked before anything is priced."""
    if not request.prompt.strip():
        raise SystemExit("higgsfield-gen: empty prompt")
    model = request.model or DEFAULT_MODEL
    params: dict[str, object] = {"prompt": request.prompt}
    if request.quality is not None and request.quality != "auto":
        if request.quality not in SUPPORTED_QUALITIES:
            raise SystemExit(f"higgsfield-gen: unsupported quality {request.quality!r}; "
                             f"expected one of {', '.join(SUPPORTED_QUALITIES)}")
        params["quality"] = request.quality
    elif model == DEFAULT_MODEL:
        params["quality"] = DEFAULT_QUALITY
    if request.resolution is not None:
        if request.resolution not in SUPPORTED_RESOLUTIONS:
            raise SystemExit(f"higgsfield-gen: unsupported resolution {request.resolution!r}; "
                             f"expected one of {', '.join(SUPPORTED_RESOLUTIONS)}")
        params["resolution"] = request.resolution
    if request.aspect_ratio:
        params["aspect_ratio"] = request.aspect_ratio
    if request.native_alpha:
        if model not in ALPHA_MODELS:
            raise SystemExit(f"higgsfield-gen: {model} cannot return an alpha channel; use --alpha-mode chroma, "
                             f"or a model that can ({', '.join(ALPHA_MODELS)})")
        params["background"] = "transparent"
    media = {"image-references": [Path(ref) for ref in request.refs]} if request.refs else {}
    return higgsfield.Job(model, params, media)


class HiggsfieldProvider:
    name = PROVIDER_HIGGSFIELD
    transparency = TRANSPARENCY_NATIVE

    def inspect_facing(self, path: Path, workdir: Path) -> tuple[str, dict]:
        from .facing_vision import claude_inspect
        return claude_inspect(path)

    def inspect_image(self, path: Path, workdir: Path, prompt: str, *, label: str = "vision check") -> tuple[str, dict]:
        from .facing_vision import claude_inspect
        return claude_inspect(path, prompt=prompt, label=label)

    def generate(self, request: GenRequest, workdir: Path) -> ProviderRun:
        job = job_for(request)
        started = time.monotonic()
        data, report = higgsfield.run_job(job, who="gen", timeout=max(GEN_TIMEOUT_SECONDS, MIN_TIMEOUT_SECONDS),
                                          detail=f" ({len(request.refs)} reference image(s))" if request.refs else "")
        data, rescaled = restore_alpha_peak(data) if request.native_alpha else (data, None)
        publish_png(data, request.raw, label="higgsfield-gen")
        extra = {key: report[key] for key in ("job_id", "credits", "auth_source", "transport")}
        if rescaled is not None:
            extra["alpha_rescaled_from"] = rescaled
        extra.update({knob: job.params[knob] for knob in ("quality", "resolution", "aspect_ratio", "background")
                      if knob in job.params})
        return ProviderRun(provider=self.name, elapsed_seconds=time.monotonic() - started, model=job.job_type,
                           extra=extra)
