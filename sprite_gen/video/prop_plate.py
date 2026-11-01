# SPDX-License-Identifier: Apache-2.0
"""A flat magenta plate painted into a clip's still for the subject to clear, and taken out of its keyed frames
again (`video-canvas --prop-plate`, `video-frames --prop-plate`).

A vault needs something to vault. Asked in words for a wall in the background colour, a video model draws a black
slab, and keying leaves the slab in the frames. So the wall is drawn into the still instead, as a plate the key
does not take and the erase does: flat magenta on the green key, standing on the ground in front of the subject.

One box is the plate. `place` sets it on the canvas, and `video-canvas` paints it and writes it into its report
(`prop_plate.box`, canvas px, right and bottom one past). `video-frames` reads it from that report (`read`) and
erases inside it on every keyed frame (`erase`), the box carried onto the clip's frames by the ratio of their size
to the canvas's where the video model returned another size.

The plate, of the subject's height on the canvas (`canvas.subject_box`): a wall as high as the thighs (HEIGHT), a
fifth of the height thick (WIDTH), its near side GAP ahead of the subject's front (the side it faces), its foot on
the subject's foot row.

The erase: inside the box and a rim of RIM px, every pixel led by red and blue over green (by more than
MAGENTA_LEAD, red and blue within MAGENTA_BALANCE of each other: the plate, its rim blended into the key too) goes,
as does every pixel within STILL_TOLERANCE on each channel of what that pixel shows over the whole clip (the median:
the plate's keyed rim, a ground line drawn under it), since the subject crosses the box for part of the clip only.
Alpha is one of those channels, so a pixel keyed out over most of the clip is not read as black: the subject's dark
outline crossing the rim stays, as does the rest of the subject crossing in front of the plate.

A plate the video model draws a pixel off in a frame leaves its edge there, the codec's blend of magenta and green
(grey, so neither rule takes it), a line a pixel wide down each side.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from sprite_gen._deps import np
from sprite_gen.frames.cutout import KEY_TARGETS, MAGENTA

RGB = KEY_TARGETS[MAGENTA]
HEIGHT = 0.49  # of the subject's height
WIDTH = 0.20
GAP = 0.84
RIM = 3  # px added round the box on each side, on the clip's frames
MAGENTA_LEAD = 35
MAGENTA_BALANCE = 90
STILL_TOLERANCE = 40
CANVAS_REPORT = "sprite-gen-video-canvas-report"


def place(subject: tuple[int, int, int, int], *, facing: str, size: tuple[int, int]) -> tuple[int, int, int, int]:
    """The plate's box (left, top, right, bottom; right and bottom one past) on a canvas of `size`, in front of the
    `subject` box facing `facing`. Refused where it would leave the canvas."""
    left, top, right, bottom = subject
    height = bottom - top
    width, gap = round(WIDTH * height), round(GAP * height)
    if facing == "right":
        x0 = right + gap
        x1 = x0 + width
    else:
        x1 = left - gap
        x0 = x1 - width
    if x0 < 0 or x1 > size[0]:
        raise SystemExit(
            f"video-canvas: --prop-plate: the plate ({x0}..{x1}) falls past the canvas's {facing} edge ({size[0]} px wide); "
            "frame the subject with room in front (--shape wide, --lead)"
        )
    return x0, bottom - round(HEIGHT * height), x1, bottom


def paint(canvas: Image.Image, subject: tuple[int, int, int, int], *, facing: str) -> dict[str, Any]:
    """Paint the plate in front of `subject` on `canvas` (in place); the report's `prop_plate`."""
    box = place(subject, facing=facing, size=canvas.size)
    canvas.paste(RGB, box)
    return {"box": list(box), "rgb": list(RGB), "subject": list(subject),
            "of_subject_height": {"height": HEIGHT, "width": WIDTH, "gap": GAP}}


def read(report: Path) -> dict[str, Any]:
    """The plate a `video-canvas --report` holds: its `box` and the `canvas` size it is on, and the report's path."""
    path = Path(report).expanduser().resolve()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise SystemExit(f"video-frames: --prop-plate: cannot read the canvas report {path}: {e}") from e
    if not isinstance(data, dict) or data.get("kind") != CANVAS_REPORT:
        raise SystemExit(f"video-frames: --prop-plate: {path} is not a video-canvas report ({CANVAS_REPORT})")
    plate = data.get("prop_plate")
    if not isinstance(plate, dict):
        raise SystemExit(f"video-frames: --prop-plate: {path} holds no plate; the canvas was made without --prop-plate")
    return {"report": str(path), "box": [int(v) for v in plate["box"]], "canvas": [int(v) for v in data["canvas"]]}


def frame_box(box: list[int], canvas: list[int], frame: tuple[int, int]) -> tuple[int, int, int, int]:
    """The canvas `box` on a frame of size `frame`: scaled by frame / canvas each way, rounded outward, widened by
    RIM px, kept on the frame."""
    (x0, y0, x1, y1), (cw, ch), (fw, fh) = box, canvas, frame
    x0, y0 = x0 * fw // cw, y0 * fh // ch
    x1, y1 = -(-x1 * fw // cw), -(-y1 * fh // ch)
    return max(0, x0 - RIM), max(0, y0 - RIM), min(fw, x1 + RIM), min(fh, y1 + RIM)


def erase(files: list[Path], plate: dict[str, Any]) -> dict[str, Any]:
    """Erase the plate from every keyed frame in `files` (in place); the frames report's `prop_plate`."""
    with Image.open(files[0]) as first:
        x0, y0, x1, y1 = frame_box(plate["box"], plate["canvas"], first.size)
    crops = []
    for f in files:
        with Image.open(f) as im:
            crops.append(np.asarray(im.convert("RGBA"))[y0:y1, x0:x1])
    still = np.median(np.stack(crops), axis=0).astype(np.int32)
    changed = erased = 0
    for f in files:
        with Image.open(f) as im:
            a = np.array(im.convert("RGBA"))
        reg = a[y0:y1, x0:x1]
        rgba = reg.astype(np.int32)
        r, g, b = rgba[..., 0], rgba[..., 1], rgba[..., 2]
        magenta = (np.minimum(r, b) - g > MAGENTA_LEAD) & (np.abs(r - b) < MAGENTA_BALANCE)
        gone = magenta | (np.abs(rgba - still).max(axis=2) < STILL_TOLERANCE)
        if not reg[gone].any():
            continue
        erased += int((reg[gone][:, 3] > 0).sum())
        reg[gone] = 0
        Image.fromarray(a, "RGBA").save(f)
        changed += 1
    return {"canvas_report": plate["report"], "erase_box": [x0, y0, x1, y1], "rim": RIM, "frames": len(files),
            "frames_changed": changed, "erased_px": erased}
