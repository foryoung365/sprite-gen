# SPDX-License-Identifier: Apache-2.0
"""Generate one named game sound and publish its immutable audio bundle."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

from sprite_gen.audio import cache, processing, provider, playback
from sprite_gen.audio.manifest import AUDIO_KINDS, BGM, KIND, SFX, event_name, load as load_sound
from sprite_gen.util.cli_kwargs import namespace_from_kwargs


def generate(*, kind, prompt, duration, event, out_dir, cache_dir=None, loop=None, target_lufs=None, crossfade=None):
    loop = kind == BGM if loop is None else loop
    request = provider.request(kind, prompt, duration, loop)
    event_name(event)
    target = (-18.0 if kind == SFX else -20.0) if target_lufs is None else float(target_lufs)
    if not math.isfinite(target) or not -36 <= target <= -9:
        raise ValueError('target_lufs must be between -36 and -9')
    crossfade = (min(.25, float(duration)/8) if crossfade is None else float(crossfade)) if loop else 0
    if loop and (not math.isfinite(crossfade) or not 2/processing.RATE <= crossfade < float(duration)/2):
        raise ValueError('crossfade must be positive and shorter than half the duration')
    out_dir = Path(out_dir).expanduser().resolve()
    recipe = {'request': cache.digest(request), 'event': event, 'loop': loop,
              'target_lufs': target, 'crossfade_seconds': crossfade, 'processing_version': processing.PROCESSING_VERSION}
    if out_dir.exists():
        manifest = json.loads((out_dir / 'sound.json').read_text()) if (out_dir / 'sound.json').is_file() else {}
        if manifest.get('recipe') != recipe:
            raise ValueError('output directory already exists with a different sound; use a new directory')
        load_sound(out_dir/'sound.json', 50_000_000)
        return {**manifest, 'cache_hit': True}
    processing.require_tools()
    cache_dir = Path(cache_dir).expanduser() if cache_dir else Path(os.environ.get('XDG_CACHE_HOME', Path.home()/'.cache'))/'sprite-gen'/'audio'
    # Resolve missing credentials before claiming a request, but allow fully offline cache use.
    def make():
        return provider.generate(request)
    if cache_dir.resolve() == out_dir or out_dir in cache_dir.resolve().parents:
        raise ValueError('cache directory must be outside the output bundle')
    data, receipt, hit = cache.obtain(cache_dir/'requests.sqlite3', request, make, prepare=provider.credential)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f'.{out_dir.name}-', dir=out_dir.parent))
    try:
        source = stage/'source.mp3'
        source.write_bytes(data)
        details = processing.export(source, stage, target_lufs=target, loop=loop, crossfade=crossfade)
        source.unlink()
        manifest = {'kind': KIND, 'version': playback.VERSION, 'event': event, 'audio_kind': kind,
                    'loop': loop, 'target_lufs': target, 'recipe': recipe, 'provider': receipt, **details}
        (stage/'sound.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
        stage.rename(out_dir)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return {**manifest, 'cache_hit': hit}


def add_arguments(parser):
    parser.add_argument('--kind', choices=AUDIO_KINDS, required=True)
    prompts = parser.add_mutually_exclusive_group(required=True)
    prompts.add_argument('--prompt')
    prompts.add_argument('--prompt-file', type=Path)
    parser.add_argument('--duration', type=float, required=True, help='provider duration in seconds; loops subtract crossfade')
    parser.add_argument('--event', required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--loop', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--target-lufs', type=float)
    parser.add_argument('--crossfade', type=float, help='loop overlap seconds (default <=0.25s)')


def run(**kwargs):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = namespace_from_kwargs(parser, kwargs)
    try:
        if (args.prompt is None) == (args.prompt_file is None):
            raise ValueError('provide exactly one of prompt or prompt_file')
        prompt = Path(args.prompt_file).read_text() if args.prompt_file is not None else args.prompt
        result = generate(kind=args.kind, prompt=prompt, duration=args.duration, event=args.event,
                          out_dir=args.out_dir, cache_dir=args.cache_dir, loop=args.loop,
                          target_lufs=args.target_lufs, crossfade=args.crossfade)
    except (ValueError, OSError) as exc:
        raise SystemExit(f'sound: {exc}') from None
    print(json.dumps({'ok': True, 'manifest': str(Path(args.out_dir)/'sound.json'),
                      'cache_hit': result['cache_hit'], 'duration_seconds': result['duration_seconds']}))
    return 0
