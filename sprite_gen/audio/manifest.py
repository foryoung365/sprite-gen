# SPDX-License-Identifier: Apache-2.0
"""Validated playback metadata and file digests for a sound bundle."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

from sprite_gen.audio import playback, processing

# A sound bundle's kind and its audio kinds, each written once: `sound` writes them into sound.json and offers
# the audio kinds as --kind, `load` reads them back, `provider` asks ElevenLabs for one.
KIND = 'sprite-gen-sound'
SFX, BGM = AUDIO_KINDS = ('sfx', 'bgm')


def event_name(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}', value):
        raise ValueError('event requires 1 to 128 letters, digits, dots, colons, dashes or underscores')
    return value


def _number(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f'invalid audio {name}')
    return value


def load(path, max_bytes):
    path = Path(path).resolve()
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or value.get('kind') != KIND or value.get('version') != playback.VERSION:
        raise ValueError('audio source must be a sprite-gen-sound version 2 manifest')
    if value.get('audio_kind') not in AUDIO_KINDS or type(value.get('loop')) is not bool:
        raise ValueError('invalid audio kind or loop flag')
    frames = playback.validate(value.get('playback'), value['loop'])
    duration = playback.duration(frames)
    if _number(value.get('duration_seconds'), 'duration', 0, 601) != duration:
        raise ValueError('audio duration must be derived from playback frames')
    result = {'event': event_name(value.get('event')), 'audio_kind': value['audio_kind'], 'loop': value['loop'],
              'duration_seconds': duration, 'target_lufs': _number(value.get('target_lufs'), 'target LUFS', -36, -9),
              'playback': frames, 'files': {}}
    files = value.get('files')
    if not isinstance(files, dict) or set(files) != {'ogg', 'm4a'}:
        raise ValueError('audio source requires ogg and m4a files')
    recipe = value.get('recipe')
    version = recipe.get('processing_version') if isinstance(recipe, dict) else None
    if type(version) is not int or version not in (3, processing.PROCESSING_VERSION):
        raise ValueError('audio source must come from sound processing version 3 or 4; make it again with sprite-gen '
                         'sound into a new --out-dir (its cached provider audio is reused)')
    if version == 3:
        # Version 3 had no peak-limited goal. Its files are measured again here, the way it measured
        # them, and must meet their target.
        processing.require_programs()
        goal = result['target_lufs']
    else:
        level = value.get('normalization')
        if not isinstance(level, dict) or not {'pcm_lufs', 'pcm_true_peak_dbtp'} <= set(level):
            raise ValueError('a sound processing version 4 source requires normalization.pcm_lufs and pcm_true_peak_dbtp')
        goal = processing.loudness_goal(result['target_lufs'], _number(level['pcm_lufs'], 'PCM LUFS', -70, 0),
                                        _number(level['pcm_true_peak_dbtp'], 'PCM true peak', -100, 0))
    payloads, sources, size = {}, {path}, 0
    for fmt in ('ogg', 'm4a'):
        file = files[fmt]
        name = file.get('path') if isinstance(file, dict) else None
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_-][A-Za-z0-9_.-]*\.'+fmt, name):
            raise ValueError('audio file must be a local filename of the declared format')
        source = (path.parent/name).resolve()
        if source.parent != path.parent:
            raise ValueError('audio source must stay within its bundle')
        size += source.stat().st_size
        if size > max_bytes:
            raise ValueError(f'audio exceeds {max_bytes} bytes')
        data = source.read_bytes()
        if hashlib.sha256(data).hexdigest() != file.get('sha256'):
            raise ValueError('audio file digest does not match its manifest')
        record = {'path': name, 'sha256': file['sha256'],
                  'duration_seconds': _number(file.get('duration_seconds'), 'file duration', frames['end_frame']/frames['sample_rate'], 602),
                  'lufs': _number(file.get('lufs'), 'LUFS', -70, 0),
                  'true_peak_dbfs': _number(file.get('true_peak_dbfs'), 'true peak', -100, -1)}
        if version == 3:
            measured = processing.measure_version_3(processing.file_frames(source, frames, value['loop'])[1], goal)
            record.update(lufs=measured['lufs'], true_peak_dbfs=measured['true_peak'])
        if not processing.meets(record['lufs'], record['true_peak_dbfs'], goal):
            raise ValueError('audio loudness does not match its target or its peak-limited PCM')
        result['files'][fmt] = record
        payloads[fmt] = data
        sources.add(source)
    return result, payloads, sources
