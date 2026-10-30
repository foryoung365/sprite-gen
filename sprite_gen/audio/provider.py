# SPDX-License-Identifier: Apache-2.0
"""Sound providers: Higgsfield (the default, through its signed-in CLI) or the ElevenLabs HTTP contract
(`SPRITE_GEN_SOUND_PROVIDER=elevenlabs`; retry decisions belong to gen.retry).

On Higgsfield an effect is a `mirelo_text_to_audio` job and music a `sonilo_music` job, each taking a prompt and a
duration (2026-10-30, `higgsfield model get`). Neither has a loop flag: a loop is made by `processing` from the
source, as it is for ElevenLabs music. The provider is part of the request document, so a cached sound is never
answered for the other provider's request.
"""
from __future__ import annotations

import json
import math
import os
import re
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from sprite_gen.gen import address, retry
from sprite_gen.gen.base import announce_api_billing, env_api_key
from sprite_gen.gen.xai import error_body, NO_ANSWER

AUTH_ENV = 'ELEVENLABS_API_KEY'
PROVIDER_ENV = 'SPRITE_GEN_SOUND_PROVIDER'
PROVIDERS = ('higgsfield', 'elevenlabs')
HIGGSFIELD_JOBS = {'sfx': 'mirelo_text_to_audio', 'bgm': 'sonilo_music'}
HIGGSFIELD_TIMEOUT_SECONDS = 900
FORMAT = 'mp3_44100_128'
SAFE_CODES = frozenset({'system_busy', 'too_many_concurrent_requests', 'quota_exceeded',
    'invalid_api_key', 'missing_permissions', 'bad_prompt', 'bad_composition_plan',
    'content_policy_violation', 'unauthorized', 'invalid_request'})


def sound_provider():
    named = (os.environ.get(PROVIDER_ENV) or '').strip() or PROVIDERS[0]
    if named not in PROVIDERS:
        raise ValueError(f'{PROVIDER_ENV}={named!r}: expected one of {", ".join(PROVIDERS)}')
    return named


def request(kind, prompt, duration, loop):
    if kind not in ('sfx', 'bgm'):
        raise ValueError('kind must be sfx or bgm')
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 4100:
        raise ValueError('prompt requires 1 to 4100 characters')
    duration = float(duration)
    low, high = (0.5, 30) if kind == 'sfx' else (3, 600)
    if not math.isfinite(duration) or not low <= duration <= high:
        raise ValueError(f'{kind} duration must be between {low} and {high} seconds')
    if type(loop) is not bool:
        raise ValueError('loop must be a boolean')
    if sound_provider() == 'higgsfield':
        return {'provider': 'higgsfield', 'job_type': HIGGSFIELD_JOBS[kind],
                'params': {'prompt': prompt, 'duration': duration}, 'loop': loop}
    body = ({'text': prompt, 'duration_seconds': duration, 'loop': loop,
             'model_id': 'eleven_text_to_sound_v2', 'prompt_influence': 0.3} if kind == 'sfx'
            else {'prompt': prompt, 'music_length_ms': round(duration * 1000),
                  'model_id': 'music_v1', 'force_instrumental': True})
    return {'endpoint': 'sound-generation' if kind == 'sfx' else 'music',
            'output_format': FORMAT, 'body': body}


def credential(document=None):
    if document is not None and document.get('provider') == 'higgsfield' or document is None and sound_provider() == 'higgsfield':
        from sprite_gen.gen import higgsfield
        try:
            signed_in = higgsfield.account()
        except SystemExit as exc:
            raise ValueError(str(exc)) from None
        if signed_in is None:
            raise ValueError(f'higgsfield is not signed in; {higgsfield.LOGIN_HINT}')
        return None
    # ElevenLabs has no subscription login to prefer: the key is the only route, billed per call.
    try:
        key = env_api_key(AUTH_ENV, who='sound')
    except SystemExit as exc:
        raise ValueError(str(exc)) from None
    if not key:
        raise ValueError(f'{AUTH_ENV} is required for a new sound request')
    return key


def _code(body):
    detail = body.get('detail') if isinstance(body, dict) else None
    code = detail.get('status') if isinstance(detail, dict) else None
    return code if isinstance(code, str) and code in SAFE_CODES else None


def _higgsfield(document):
    from sprite_gen.gen import higgsfield
    job = higgsfield.Job(document['job_type'], dict(document['params']))
    try:
        data, report = higgsfield.run_job(job, who='sound', timeout=HIGGSFIELD_TIMEOUT_SECONDS,
                                          detail=" (the audio is cached, so the same request is not bought twice)")
    except SystemExit as exc:
        raise ValueError(str(exc)) from None
    return data, {key: report[key] for key in ('provider', 'auth_source', 'model', 'job_id', 'credits')}


def generate(document):
    if document.get('provider') == 'higgsfield':
        return _higgsfield(document)
    key = credential()
    req = Request(address.url('elevenlabs', f"/{document['endpoint']}?output_format={document['output_format']}"),
                  data=json.dumps(document['body']).encode(),
                  headers={'xi-api-key': key, 'Content-Type': 'application/json'}, method='POST')

    def send():
        try:
            with urlopen(req, timeout=600) as response:
                return response.status, response.read(), response.headers
        except HTTPError as exc:
            try:
                body = error_body(exc)
            except NO_ANSWER:
                raise ValueError('ElevenLabs refusal body unavailable; request outcome unresolved; no resubmission') from None
            return exc.code, body, exc.headers
        except Exception:
            # Transport exceptions may include headers or body text. Never print them.
            raise ValueError('ElevenLabs response unavailable; request outcome unresolved; no resubmission') from None

    announce_api_billing('elevenlabs', AUTH_ENV, f" ({document['endpoint']}; the audio is cached, so the same "
                         "request is not bought twice).")
    status, body, headers, asks = retry.ask(send, provider='elevenlabs', who='ElevenLabs', label='sound', kind=_code)
    if not 200 <= status < 300:
        raise ValueError(f'ElevenLabs refused sound (HTTP {status}, {_code(body) or "unclassified"}); {retry.tail(asks)}')
    if not isinstance(body, bytes) or not body:
        raise ValueError('ElevenLabs returned no audio; request outcome unresolved')
    meta = {'provider': 'elevenlabs', 'auth_source': AUTH_ENV, 'model': document['body']['model_id'], 'asks': asks}
    # Only documented, constrained metadata; provider prose never enters a report or log.
    for header, field in (('request-id', 'request_id'), ('song-id', 'song_id')):
        value = headers.get(header, '')
        if value and key not in value and re.fullmatch(r'[A-Za-z0-9_-]{1,128}', value):
            meta[field] = value
    value = headers.get('character-cost', '')
    if str(value).isdigit() and len(str(value)) < 12:
        meta['character_cost'] = int(value)
    return body, meta
