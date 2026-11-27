# Higgsfield provider

This fork's default for every generation that spends money: images, clips and sounds go to Higgsfield through
the signed-in `higgsfield` CLI and spend the plan's credits. Codex, Grok, OpenAI and ElevenLabs stay available,
each by name only.

| Verb | Higgsfield job | Other providers |
|---|---|---|
| `gen`, `gen-set`, `logo`, reroll and in-betweens in curation | `gpt_image_2_5` (any image job type with `--model`) | `--provider codex\|grok\|openai` |
| `video`, `video-set`, `video-resume` | `grok_video_v15` for a still alone; `magi_2` with `--last-frame` (1080p, >=4 s, auto aspect match, framing guard); `seedance_2_0` with `--reference` | `--provider grok`, or `SPRITE_GEN_VIDEO_PROVIDER=grok`; `--model wan3_0` or `SPRITE_GEN_HIGGSFIELD_VIDEO_MODEL=wan3_0` for Wan 3.0 |
| `sound` | `mirelo_text_to_audio` (sfx), `sonilo_music` (bgm) | `SPRITE_GEN_SOUND_PROVIDER=elevenlabs` |
| facing checks, a logo's text read | none: Higgsfield has no vision model, so these run on the Claude subscription (`claude` signed in) | the provider's own check |

`video-extend` and `video-edit` stay on Grok.

## Setup

```bash
higgsfield auth login
higgsfield account status
```

`sprite-gen workflow --kind sprite` shows the account, plan and credits under `access`. Higgsfield is listed first
among the image providers, and no confirmation question is asked for it, because the CLI reports the account itself.

### Windows

npm installs the CLI as a `higgsfield.cmd` shim, and cmd.exe cuts an argument at its first line break, so a prompt
of several lines (every prompt the engine assembles) lost the rest of the command, `--json` included, and came back
as `generate cost answered no JSON` (2026-10-30). sprite-gen now calls the executable the npm package carries beside
the shim (`node_modules/@higgsfield/cli/vendor/hf.exe`) instead. Where that is missing, a call with a prompt of
several lines is refused before it runs, naming `SPRITE_GEN_HIGGSFIELD_BIN`, which points sprite-gen at any CLI
binary and is always obeyed first.

## One job, three calls

1. `generate cost` prices the job. A parameter the model does not take is refused here (for example `--quality`
   on `nano_banana_pro`), so a mistake costs nothing.
2. The price is said on stderr: `[gen] higgsfield: gpt_image_2_5 spends Higgsfield credits: 1.5 credits`.
   `gen-set` and `video-set` forward that line from each item's log to the terminal.
3. `generate create` makes the job and `generate wait` waits for it. A wait that gets no answer is asked again for
   the same job id. A job is never created twice: one that fails or runs out of time is named by its id
   (`higgsfield generate get <id>`; for a clip, `sprite-gen video-resume --request-id <id>`).

The report keeps `job_id`, `credits`, `auth_source: higgsfield-login` and `transport: higgsfield-cli` under `extra`.

## Images

- The default model is GPT Image 2.5. `--transparent` asks it for a real alpha channel (`background=transparent`),
  so the strategy is `native`. With `--ref`, `auto` steps down to a chroma key, as it does for codex.
- Quality defaults to `high` on the default model. `--quality` takes `low`, `medium`, `high`, `xhigh` and `max`;
  `auto` leaves the default. `--resolution` takes `1k` and `2k`; `1.5k` is a Grok tier and is refused.
- Higgsfield's transparent PNGs have no fully opaque pixel: a figure's body comes back as alpha noise between 246
  and 254. Before the image is published, alpha at 240 or above is set to 255 and lower alpha is scaled by
  255/240, so edges keep a continuous gradient. The report records the 99th percentile it found as
  `extra.alpha_rescaled_from`.
- Another model is named with `--model`, e.g. `--model nano_banana_pro`. Such a model has no `background`
  parameter, so `--transparent` needs `--alpha-mode chroma`.

Prices measured 2026-10-30, at 1k: `low` 0.25, `medium` 0.5, `high` 1.5 credits; `high` at 2k 2.75.

## Clips

`video` takes `--provider higgsfield|grok`. `video-set --provider` sets it for every clip in the set; without it,
`SPRITE_GEN_VIDEO_PROVIDER` decides, and without that, Higgsfield.

- A still alone becomes `grok_video_v15`, the same Grok Imagine Video 1.5 model the clip prompts were tuned on.
- A clip pinned to a last frame (idles, attacks, and diagonal walks and runs in `video-set`) defaults to `magi_2`,
  normalized automatically to `1080p`, duration `>= 4` s, `generate_audio=false` (unless `--audio`), and an explicit
  `aspect_ratio` matched to the input still (`21:9`, `16:9`, `4:3`, `1:1`, `3:4`, `9:16`; fallback `16:9`).
  - **Magenta canvas routing (`video-set`)**: pinned `magi_2` items automatically rekey the canvas to `#FF00FF` magenta
    (`video-canvas --target-key magenta --bottom-pad 0.14`) and pass `key=magenta` to `video-frames`, avoiding green-screen
    grey-bar degradation and giving foot headroom against slight framing drift.
  - **First-frame framing guard (`video` & `video-frames`)**: after download (`extra.framing_guard`) and before expensive
    full-clip keying (`video-frames`), frame 0 is compared against the reference canvas; abnormal zoom/crop drift
    (e.g. >15% height change or new bottom clipping) is warned in `video` and refused early in `video-frames`
    (`--allow-reframe` skips the gate).
  - **Attack hold compaction (`video-loop` / `video-set`)**: `--action-retime auto` (default) trims dead mid-action holds
    and excess idle padding on `>= 3.2` s pinned/one-shot attack clips into a crisp ~0.8–1.1 s loop (`punch` always compacts;
    `off` keeps the clip as filmed).
  - To switch back to Wan 3.0 for pinned clips, pass `--model wan3_0` on `sprite-gen video` or set
    `SPRITE_GEN_HIGGSFIELD_VIDEO_MODEL=wan3_0` for `video-set` (`wan3_0` keeps `aspect_ratio=auto`, `generate_audio=false`,
    and 2–30 s duration at 480p/720p/1080p).
- A clip guided by reference images becomes `seedance_2_0`, which takes them beside a start and an end image.
- `SPRITE_GEN_HIGGSFIELD_VIDEO_MODEL=<job_type>` films every clip on one job, which is how `video-set` (it passes
  no model) is moved to another, e.g. `wan3_0` or `kling4_0`. Kling 4.0 takes both frames at 720p or 1080p and is sent
  `mode=image-to-video` (or `reference-to-video` with references). One measured idle (2026-10-30, a pixel-art
  knight, 3 s at 720p): `wan3_0` kept the still closely with a small breath and a blink, seam 0.70;
  `kling4_0` moved more (the shield sways, the head dips), seam 1.00, at about twice the credits.
- `--model <job_type>` names any other Higgsfield video job, e.g. `wan3_0`. The Grok model names are not
  Higgsfield jobs and pick the defaults above.

Prices measured 2026-10-30: `grok_video_v15` 3 s at 480p 7.5 credits, 6 s at 720p 27; `wan3_0` 3 s at 480p 3,
at 720p 5.25, 6 s at 720p 10.5; `seedance_2_0` 4 s at 480p 12, 6 s at 720p 27.

## Sounds

`sound` sends effects to `mirelo_text_to_audio` and music to `sonilo_music`, each with the prompt and duration.
Loops are made from the source by the same processing as before. The sound cache keys on the provider, so a sound
bought from one provider is never answered for the other. Prices measured 2026-10-30: a 2 s effect 0.5 credits,
10 s 2.5; 30 s of music 1.88, 120 s 7.5.

## Vision on the Claude subscription

The facing check (`gen --facing` with `--ref`, `video --direction side`, side views in `video-set`) and the
logo's text read send the image to `claude -p` on the signed-in Claude subscription. They never fall back to
`ANTHROPIC_API_KEY`: with no login, or with `SPRITE_GEN_LOGINS=off`, the check is refused and says so.
