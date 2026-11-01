# SPDX-License-Identifier: Apache-2.0
from concurrent.futures import ThreadPoolExecutor
import io
import http.client
import json
import re
import subprocess
import threading
import urllib.error
import wave

import numpy as np
import pytest

from sprite_gen.audio import cache, provider, processing, sound
from sprite_gen.audio.manifest import load as load_sound
from sprite_gen.gen import retry


@pytest.fixture(autouse=True)
def fake_credential(monkeypatch):
    monkeypatch.setenv('ELEVENLABS_API_KEY', 'synthetic-key')


def wav_bytes(seconds=3):
    t = np.arange(round(seconds * 48000)) / 48000
    x = .08 * np.sin(2 * np.pi * 321.3 * t)
    x[100:110] = .95  # sparse peak: gain alone would clip
    data = io.BytesIO()
    with wave.open(data, 'wb') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(48000)
        f.writeframes((x * 32767).astype('<i2').tobytes())
    return data.getvalue()


def stereo_wav(x, rate=48000):
    data = io.BytesIO()
    with wave.open(data, 'wb') as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes((np.clip(x, -1, 1) * 32767).astype('<i2').tobytes())
    return data.getvalue()


def hit(seconds, body_hz, body_tau, sweep, knock_hz, knock_tau, knock_gain, noise_tau, noise_gain, smooth, stereo, seed):
    """A synthetic hit: a pitch-dropping body, a knock and a noise burst decaying from one onset."""
    t = np.arange(round(seconds * 48000)) / 48000
    body = np.sin(2*np.pi*np.cumsum(body_hz*(1 + sweep*np.exp(-t/.03)))/48000) * np.exp(-t/body_tau)
    knock = knock_gain * np.sin(2*np.pi*knock_hz*t) * np.exp(-t/knock_tau)
    noise = np.random.default_rng(seed).standard_normal((len(t), 2))
    noise[:, 1] = (1-stereo)*noise[:, 0] + stereo*noise[:, 1]
    if smooth > 1:
        noise = np.stack([np.convolve(n, np.ones(smooth)/smooth, 'same') for n in noise.T], axis=1) * np.sqrt(smooth)
    x = .25 * ((body + knock)[:, None] + noise*(noise_gain*np.exp(-t/noise_tau))[:, None])
    x[:16] *= np.linspace(0, 1, 16)[:, None]
    return stereo_wav(x)


HITS = {
    # A thud whose AAC reconstruction overshoots the limited PCM peak by several dB.
    'thud': dict(seconds=.6, body_hz=77.19, body_tau=.1322, sweep=.3641, knock_hz=704.9, knock_tau=.03642,
                 knock_gain=.8461, noise_tau=.04146, noise_gain=.881, smooth=2, stereo=.7, seed=634996),
    # A few milliseconds of energy: at the -2 dBTP limiter ceiling it cannot reach -18 LUFS.
    'click': dict(seconds=.5, body_hz=900, body_tau=.003, sweep=0, knock_hz=2500, knock_tau=.002,
                  knock_gain=.5, noise_tau=.002, noise_gain=1, smooth=1, stereo=0, seed=7),
    # FFmpeg 5.1's loudnorm first pass measures its output for this hit at -inf LUFS (7.1's does not).
    'unanalysable': dict(seconds=.6, body_hz=55.03, body_tau=.09572, sweep=2.822, knock_hz=692.4, knock_tau=.01984,
                         knock_gain=.2113, noise_tau=.01464, noise_gain=.8967, smooth=1, stereo=.3, seed=212970),
}


def bundle(tmp_path, monkeypatch, name, seconds):
    monkeypatch.setattr(provider, 'generate', lambda req: (hit(**HITS[name]), {}))
    return sound.generate(kind='sfx', prompt='synthetic hit', duration=seconds, event='hit',
                          cache_dir=tmp_path/'cache', out_dir=tmp_path/'out')


@pytest.mark.slow
@pytest.mark.parametrize('name', HITS)
def test_short_hits_pass_the_one_loudness_rule(tmp_path, monkeypatch, name):
    result = bundle(tmp_path, monkeypatch, name, HITS[name]['seconds'])
    level = result['normalization']
    goal = processing.loudness_goal(-18, level['pcm_lufs'], level['pcm_true_peak_dbtp'])
    if name == 'click':
        assert level['pcm_true_peak_dbtp'] >= -2.2 and goal == level['pcm_lufs'] < -19
    else:
        assert goal == -18
    for file in result['files'].values():
        assert processing.meets(file['lufs'], file['true_peak_dbfs'], goal)
        assert file['true_peak_dbfs'] <= -1
    load_sound(tmp_path/'out'/'sound.json', 50_000_000)


def test_a_clip_loudnorm_cannot_analyse_is_normalized_by_the_limiter(tmp_path, monkeypatch):
    analyse = processing._loudnorm
    monkeypatch.setattr(processing, '_loudnorm', lambda *a: {**analyse(*a), 'target_offset': float('inf')})
    source = tmp_path/'source.wav'
    source.write_bytes(hit(**HITS['unanalysable']))
    _, level, _ = processing.normalize(processing.decode(source), -18, False)
    assert level['method'] == 'limiter' and abs(level['pcm_lufs'] + 18) <= .2


def test_a_bundle_is_refused_when_its_files_miss_the_rule(tmp_path, monkeypatch):
    bundle(tmp_path, monkeypatch, 'click', .5)
    path = tmp_path/'out'/'sound.json'
    original = json.loads(path.read_text())
    headroom = json.loads(path.read_text())
    headroom['normalization']['pcm_true_peak_dbtp'] = -8  # quiet with headroom: the goal is the target
    hot = json.loads(path.read_text())
    hot['files']['m4a']['true_peak_dbfs'] = -.5
    old = json.loads(path.read_text())
    del old['normalization']
    relabeled = json.loads(path.read_text())
    relabeled['recipe']['processing_version'] = 3  # version 3 had no peak-limited goal; its files measure against -18
    recipe2 = json.loads(path.read_text())
    recipe2['recipe']['processing_version'] = 2
    for manifest, refusal in ((headroom, 'does not match'), (hot, 'invalid audio true peak'), (old, 'version 4'),
                              (relabeled, 'does not match'), (recipe2, 'version 3 or 4')):
        path.write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match=refusal):
            load_sound(path, 50_000_000)
    path.write_text(json.dumps(original))
    load_sound(path, 50_000_000)


def test_the_rule_needs_a_peak_reason_and_never_relaxes_the_peak():
    assert processing.loudness_goal(-18, -24, -8) == -18  # quiet with headroom
    assert processing.loudness_goal(-18, -18.1, -2) == -18  # normalized to the target
    assert processing.loudness_goal(-18, -24, -2.1) == -24  # peak-limited
    assert processing.meets(-24.3, -1.9, -24) and not processing.meets(-25.2, -1.9, -24)
    assert not processing.meets(-24, -.5, -24) and not processing.meets(-18, -.9, -18)
    assert not processing.meets(float('-inf'), -50, -18)


def test_export_refuses_a_quiet_file_with_headroom_and_a_hot_peak(tmp_path, monkeypatch):
    # One file that fails meets() is enough: every failure reaches the same raise, and
    # the quiet-with-headroom and hot-peak conditions are pinned on meets() itself in
    # test_the_rule_needs_a_peak_reason_and_never_relaxes_the_peak.
    t = np.arange(48000) / 48000
    pcm = np.stack([np.sin(2*np.pi*440*t)]*2, axis=1) * .1365
    pcm[24000:24048] = .94  # about -18 LUFS with a -0.5 dBTP spike
    stats = processing.measure(pcm)
    record = {'pcm_lufs': stats['lufs'], 'pcm_true_peak_dbtp': stats['true_peak']}
    decode = processing.decode
    monkeypatch.setattr(processing, 'decode', lambda path: pcm.copy() if path.name == 'source' else decode(path))
    monkeypatch.setattr(processing, 'normalize', lambda samples, target, loop: (samples, record, stats))
    monkeypatch.setattr(processing, 'limit_gain', lambda samples, *a: samples)  # no correction can rescue it
    with pytest.raises(ValueError, match='ogg failed loudness/peak check') as refusal:
        processing.export(tmp_path/'source', tmp_path, target_lufs=-18, loop=False, crossfade=0)
    lufs, peak = map(float, re.search(r'\(([-\d.]+) LUFS, ([-\d.]+) dBTP', str(refusal.value)).groups())
    assert abs(lufs + 18) <= 1 and peak > -1


@pytest.mark.parametrize('seconds', [0, .3])
def test_silent_or_unmeasurable_audio_is_refused(tmp_path, seconds):
    source = tmp_path/'source.wav'
    t = np.arange(round(seconds*48000) or 48000) / 48000
    source.write_bytes(stereo_wav(np.stack([np.sin(2*np.pi*440*t)*(seconds > 0)*.5]*2, axis=1)))
    with pytest.raises(ValueError, match='silent or unmeasurable'):  # 0.3 s is shorter than one 400 ms block
        processing.export(source, tmp_path, target_lufs=-18, loop=False, crossfade=0)


def test_formats_are_measured_over_their_pcm_frames_and_blocks(tmp_path, monkeypatch):
    frames, at = 28800, (6000, 17000)
    x = np.zeros((frames, 2))
    k = np.arange(-64, 65)
    for frame in at:
        x[frame-64:frame+65] += (.5*np.sinc(k/2.5)*np.hanning(len(k)))[:, None]
    source = tmp_path/'source.wav'
    source.write_bytes(stereo_wav(x))
    measure, seen, pcm = processing.measure, [], {}
    def spy(samples, gate=None):
        seen.append((samples.copy(), gate))
        return measure(samples, gate)
    def settled(samples, target, loop):
        pcm.update(measure(samples))
        return samples, {'pcm_lufs': pcm['lufs'], 'pcm_true_peak_dbtp': pcm['true_peak']}, pcm
    monkeypatch.setattr(processing, 'measure', spy)
    monkeypatch.setattr(processing, 'normalize', settled)
    target = measure(processing.decode(source))['lufs']  # two clicks; their own loudness is the goal
    result = processing.export(source, tmp_path, target_lufs=target, loop=False, crossfade=0)
    assert {file['encode_passes'] for file in result['files'].values()} == {1} and len(seen) == 2
    for samples, gate in seen:
        assert gate is pcm['gate'] and len(samples) == frames
        peaks = [int(np.argmax(np.abs(samples[f-1000:f+1000, 0]))) + f - 1000 for f in at]
        assert all(abs(p - f) <= 2 for p, f in zip(peaks, at)), peaks


def test_a_block_at_the_relative_gate_cannot_move_a_file_by_whole_lu():
    t = np.arange(28800) / 48000
    level = np.where(t < .1, 1, np.where(t < .5, .107, .0333))  # the last block sits 0.2 dB under the gate
    pcm = np.stack([np.sin(2*np.pi*1000*t)*level*.3]*2, axis=1)
    codec = pcm * np.where(t < .5, 1, 2.5)[:, None]  # a change in the last 100 ms lifts it over the gate
    reference = processing.measure(pcm)
    assert abs(processing.measure(codec)['lufs'] - reference['lufs']) > 1
    assert abs(processing.measure(codec, reference['gate'])['lufs'] - reference['lufs']) < .05


def test_provider_contract_and_safe_retry(monkeypatch, capsys):
    requests = []
    class Answer:
        status = 200
        headers = {'character-cost': '120', 'request-id': 'req-synthetic'}
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self): return b'audio'
    def send(req, **kw):
        requests.append(req)
        if len(requests) == 1:
            raise urllib.error.HTTPError(req.full_url, 429, 'busy', {'retry-after':'0'},
                io.BytesIO(b'{"detail":{"status":"system_busy","message":"secret-key"}}'))
        return Answer()
    monkeypatch.setenv('ELEVENLABS_API_KEY', 'secret-key')
    monkeypatch.setattr(provider, 'urlopen', send)
    monkeypatch.setattr(retry, '_sleep', lambda _: None)
    req = provider.request('sfx', 'Impact', 3, True)
    audio, meta = provider.generate(req)
    assert audio == b'audio' and len(requests) == 2
    assert requests[0].full_url.endswith('/sound-generation?output_format=mp3_44100_128')
    assert json.loads(requests[0].data) == {'text':'Impact', 'duration_seconds':3.0,
        'loop':True, 'model_id':'eleven_text_to_sound_v2', 'prompt_influence':.3}
    assert meta['character_cost'] == 120 and meta['auth_source'] == 'ELEVENLABS_API_KEY'
    err = capsys.readouterr().err
    assert 'secret-key' not in err
    assert '[gen] elevenlabs: running on ELEVENLABS_API_KEY — this is a per-call API charge' in err
    monkeypatch.setenv('ELEVENLABS_API_KEY', ' ')
    with pytest.raises(ValueError, match='sound: ELEVENLABS_API_KEY is set but empty'):
        provider.generate(req)
    monkeypatch.delenv('ELEVENLABS_API_KEY')
    with pytest.raises(ValueError, match='ELEVENLABS_API_KEY is required for a new sound request'):
        provider.generate(req)
    bgm = provider.request('bgm', 'Adventure', 8, True)
    assert bgm['body'] == {'prompt':'Adventure', 'music_length_ms':8000,
                          'model_id':'music_v1', 'force_instrumental':True}


@pytest.mark.parametrize('status,body', [(500, {}), (429, {'detail':{'status':'quota_exceeded'}})])
def test_unproven_refusals_are_not_retried(status, body):
    assert retry.again_after('elevenlabs', 'POST', status, body, {}, 1) is None


@pytest.mark.parametrize('code', ['system_busy', 'too_many_concurrent_requests'])
def test_the_two_documented_busy_codes_are_retried_and_named(code):
    body = {'detail': {'status': code}}
    assert retry.again_after('elevenlabs', 'POST', 429, body, {}, 1) is not None
    assert provider._code(body) == code


def test_cache_serializes_paid_requests_and_survives_uncertain_results(tmp_path):
    db = tmp_path / 'cache.sqlite3'
    entered, release = threading.Event(), threading.Event()
    def make():
        entered.set()
        assert release.wait(5)
        return b'audio', {'request_id':'synthetic'}
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(cache.obtain, db, {'a':1}, make)
        assert entered.wait(5)
        with pytest.raises(ValueError, match='unresolved'):
            cache.obtain(db, {'a':1}, make)
        release.set()
        assert first.result()[0] == b'audio'
    assert cache.obtain(db, {'a':1}, lambda: pytest.fail('charged twice'))[2] is True
    def broken(): raise OSError('connection closed')
    with pytest.raises(OSError): cache.obtain(db, {'a':2}, broken)
    with pytest.raises(ValueError, match='unresolved'):
        cache.obtain(db, {'a':2}, lambda: pytest.fail('ambiguous POST resent'))


@pytest.mark.slow
def test_actual_exports_are_normalized_and_reused(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(provider, 'generate', lambda req: (called.append(req) or wav_bytes(), {}))
    kw = dict(kind='sfx', prompt='synthetic', duration=3, event='combat.hit',
              cache_dir=tmp_path/'cache', out_dir=tmp_path/'out')
    first = sound.generate(**kw)
    second = sound.generate(**{**kw, 'out_dir':tmp_path/'out2', 'event':'combat.other'})
    assert len(called) == 1 and second['cache_hit']
    assert first['event'] == 'combat.hit'
    for fmt, metadata in first['files'].items():
        path = tmp_path/'out'/metadata['path']
        assert path.suffix == '.' + fmt and path.stat().st_size > 100
        assert abs(metadata['lufs'] - first['target_lufs']) <= 1
        assert metadata['true_peak_dbfs'] <= -1
        assert abs(metadata['duration_seconds']-3) < .05


def test_processing_failure_reuses_paid_audio(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(provider, 'generate', lambda req: (called.append(req) or wav_bytes(), {}))
    original = processing.export
    monkeypatch.setattr(processing, 'export', lambda *a, **kw: (_ for _ in ()).throw(ValueError('processing failed')))
    kw = dict(kind='sfx', prompt='synthetic', duration=3, event='hit', cache_dir=tmp_path/'cache', out_dir=tmp_path/'out')
    with pytest.raises(ValueError, match='processing failed'): sound.generate(**kw)
    assert not kw['out_dir'].exists()
    monkeypatch.setattr(processing, 'export', original)
    assert sound.generate(**kw)['cache_hit'] and len(called) == 1


@pytest.mark.slow
@pytest.mark.parametrize('kind,duration,loop', [('sfx', .5, False), ('bgm', 3, True)])
def test_short_effect_and_loop_encoded_playback(tmp_path, monkeypatch, kind, duration, loop):
    monkeypatch.setattr(provider, 'generate', lambda req: (wav_bytes(duration), {}))
    result = sound.generate(kind=kind, prompt='synthetic', duration=duration, event='sample',
                            cache_dir=tmp_path/'cache', out_dir=tmp_path/'out')
    assert result['loop'] is loop
    meta = result['playback']
    assert meta == {'version': 2, 'sample_rate': 48000, 'start_frame': 0,
                    'end_frame': round(duration*48000), 'overlap_frames': 12000 if loop else 0}
    assert result['duration_seconds'] == (meta['end_frame']-meta['overlap_frames'])/48000
    assert 'loop_end_seconds' not in result and 'codec_taper_seconds' not in result
    for file in result['files'].values():
        assert file['duration_seconds'] >= duration
        assert len(processing.decode(tmp_path/'out'/file['path'])) >= meta['end_frame']


def test_preflight_does_not_claim_and_cached_source_works_without_key(tmp_path, monkeypatch):
    db = tmp_path/'cache.sqlite3'
    def no_key(): raise ValueError('missing key')
    with pytest.raises(ValueError, match='missing key'):
        cache.obtain(db, {'request':1}, lambda: pytest.fail('must preflight'), prepare=no_key)
    assert cache.obtain(db, {'request':1}, lambda: (b'audio', {}))[2] is False
    assert cache.obtain(db, {'request':1}, lambda: pytest.fail('cache must suffice'), prepare=no_key)[2] is True


def test_error_body_and_transport_cannot_disclose_key(monkeypatch, capsys):
    key = 'secret-api-key-should-never-appear'
    monkeypatch.setenv('ELEVENLABS_API_KEY', key)
    def refuse(req, **kwargs):
        raise urllib.error.HTTPError(req.full_url, 422, key, {}, io.BytesIO(json.dumps({'detail':{'message':key,'status':key}}).encode()))
    monkeypatch.setattr(provider, 'urlopen', refuse)
    with pytest.raises(ValueError) as err: provider.generate(provider.request('sfx','impact',1,False))
    assert key not in str(err.value) + capsys.readouterr().err
    def broken(*a, **kw): raise OSError(key)
    monkeypatch.setattr(provider, 'urlopen', broken)
    with pytest.raises(ValueError) as err: provider.generate(provider.request('sfx','impact',1,False))
    assert key not in str(err.value) + capsys.readouterr().err


def test_broken_refusal_body_is_not_retried_or_printed(monkeypatch, capsys):
    calls = []
    class Broken(io.BytesIO):
        def read(self, *args): raise http.client.IncompleteRead(b'secret-key', 100)
    def send(req, **kw):
        calls.append(req)
        raise urllib.error.HTTPError(req.full_url, 429, 'busy', {}, Broken())
    monkeypatch.setattr(provider, 'urlopen', send)
    with pytest.raises(ValueError, match='unresolved') as err:
        provider.generate(provider.request('sfx', 'impact', 1, False))
    assert len(calls) == 1 and 'secret-key' not in str(err.value) + capsys.readouterr().err


def test_cli_and_python_entry_publish_the_same_bundle(tmp_path, monkeypatch, capsys):
    from sprite_gen.cli import main
    monkeypatch.setattr(provider, 'generate', lambda req: (wav_bytes(1), {}))
    args = ['sound', '--kind', 'sfx', '--prompt', 'synthetic', '--duration', '1', '--event', 'hit',
            '--out-dir', str(tmp_path/'out'), '--cache-dir', str(tmp_path/'cache')]
    assert main(args) == 0
    monkeypatch.delenv('ELEVENLABS_API_KEY')
    assert sound.run(kind='sfx',prompt='synthetic',duration=1,event='hit',out_dir=str(tmp_path/'out'),
                     cache_dir=tmp_path/'cache') == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert lines[1]['cache_hit'] is True
    assert lines[0]['manifest'] == lines[1]['manifest']


def test_codec_reconstruction_keeps_a_broadband_transient_below_peak_ceiling(tmp_path, monkeypatch):
    t = np.arange(24000)/48000
    x = .8*np.sign(np.sin(2*np.pi*2200*t))*np.exp(-t*32)
    stream=io.BytesIO()
    with wave.open(stream,'wb') as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(48000)
        f.writeframes((x*32767).astype('<i2').tobytes())
    monkeypatch.setattr(provider,'generate',lambda _: (stream.getvalue(), {}))
    result=sound.generate(kind='sfx',prompt='synthetic transient',duration=.5,event='hit',
                          out_dir=tmp_path/'out',cache_dir=tmp_path/'cache')
    for file in result['files'].values():
        assert file['true_peak_dbfs'] <= -1
        assert abs(file['lufs']+18) <= 1


def test_export_keeps_source_frames_until_browser_preparation(tmp_path, monkeypatch):
    t = np.arange(48000*3)/48000
    x = np.stack([.1*np.sin(2*np.pi*121*t+.5), .4*np.cos(2*np.pi*73*t)], axis=1).astype('<f4')
    monkeypatch.setattr(processing, 'decode', lambda path: x.copy())
    seen = []
    def normalize(samples, target, loop):
        seen.append(samples.copy())
        return samples, {}, {'lufs': -20, 'true_peak': -2, 'gate': None}
    monkeypatch.setattr(processing, 'normalize', normalize)
    monkeypatch.setattr(processing, 'measure', lambda *a: {'lufs': -20, 'true_peak': -2, 'gate': None})
    processing.export(tmp_path/'source', tmp_path, target_lufs=-20, loop=True, crossfade=.25)
    np.testing.assert_array_equal(seen[0], x)


@pytest.mark.parametrize('overlap', [20, 21])
def test_export_reserves_overlap_for_minimum_playback_rate(tmp_path, monkeypatch, overlap):
    # At 8 kHz, the 20-frame overlap leaves only one blend frame after
    # interval quantization, loop rounding and a one-frame decoder shortfall.
    monkeypatch.setattr(processing, 'decode', lambda path: np.ones((48005, 2), dtype='<f4'))
    def reached_normalization(*args):
        raise RuntimeError('normalization reached')
    monkeypatch.setattr(processing, 'normalize', reached_normalization)
    if overlap == 20:
        with pytest.raises(ValueError, match='overlap too short.*8000 Hz'):
            processing.export(tmp_path/'source', tmp_path, target_lufs=-20, loop=True, crossfade=overlap/48000)
        assert not list(tmp_path.iterdir())
    else:
        with pytest.raises(RuntimeError, match='normalization reached'):
            processing.export(tmp_path/'source', tmp_path, target_lufs=-20, loop=True, crossfade=overlap/48000)


@pytest.mark.slow
@pytest.mark.parametrize('millisecond_timescale', [False, True])
def test_m4a_container_preserves_declared_frames(tmp_path, monkeypatch, millisecond_timescale):
    frames = 144003
    source = tmp_path/'source.wav'
    source.write_bytes(wav_bytes(frames/48000))
    if millisecond_timescale:
        encode = processing._ffmpeg
        def old_timescale(args, data=None):
            if str(args[-1]).endswith('.m4a'):
                args = [*args[:-1], '-movie_timescale', '1000', args[-1]]
            return encode(args, data)
        monkeypatch.setattr(processing, '_ffmpeg', old_timescale)
        with pytest.raises(ValueError, match='m4a container duration'):
            processing.export(source, tmp_path, target_lufs=-20, loop=True, crossfade=.25)
        return
    result = processing.export(source, tmp_path, target_lufs=-20, loop=True, crossfade=.25)
    probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'a:0',
                            '-show_entries', 'stream=duration_ts,time_base', '-of', 'json',
                            str(tmp_path/result['files']['m4a']['path'])],
                           capture_output=True, check=True)
    stream = json.loads(probe.stdout)['streams'][0]
    assert stream == {'time_base': '1/48000', 'duration_ts': frames}
    assert result['playback']['end_frame'] == frames


def test_missing_ffprobe_preflights_before_generation(tmp_path, monkeypatch):
    which = processing.shutil.which
    monkeypatch.setattr(processing.shutil, 'which', lambda name: None if name == 'ffprobe' else which(name))
    monkeypatch.setattr(provider, 'generate', lambda _: pytest.fail('generation before tool preflight'))
    with pytest.raises(ValueError, match='ffprobe is required'):
        sound.generate(kind='sfx', prompt='synthetic', duration=3, event='hit',
                       cache_dir=tmp_path/'cache', out_dir=tmp_path/'out')


@pytest.mark.slow
def test_old_processed_output_is_rejected_but_raw_cache_survives(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(provider, 'generate', lambda req: (calls.append(req) or wav_bytes(), {}))
    kw = dict(kind='bgm', prompt='synthetic', duration=3, event='music',
              cache_dir=tmp_path/'cache', out_dir=tmp_path/'out')
    result = sound.generate(**kw)
    assert result['version'] == 2
    assert result['recipe']['processing_version'] == 4
    # One old version is enough: any recipe that differs is refused by the same check.
    old = {**result, 'version': 3, 'recipe': {**result['recipe'], 'processing_version': 3}}
    (kw['out_dir']/'sound.json').write_text(json.dumps(old))
    with pytest.raises(ValueError, match='different sound'):
        sound.generate(**kw)
    monkeypatch.delenv('ELEVENLABS_API_KEY')
    fresh = sound.generate(**{**kw, 'out_dir': tmp_path/'new'})
    assert fresh['cache_hit'] and len(calls) == 1
