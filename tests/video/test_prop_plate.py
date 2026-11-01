# SPDX-License-Identifier: Apache-2.0
"""A magenta plate painted into the still (`video-canvas --prop-plate`) and erased from the keyed frames by the same
box (`video-frames --prop-plate`), so a clip filmed over it keeps no plate and the cut `video-loop` makes has none.

The pictures are synthetic: an outlined figure on the green key, filmed by a "model" that returns the canvas at twice
its size, keeps the plate where the canvas has it (a pixel off in two frames), draws a ground line under it and runs
the figure across in front of it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from sprite_gen._deps import np
from sprite_gen.video import canvas as canvas_mod
from sprite_gen.video import frames as frames_mod
from sprite_gen.video import loop as loop_mod
from sprite_gen.video import prop_plate

GREEN = (0, 255, 0)
OUTLINE = (28, 18, 16)
SKIN = (238, 206, 178)
SHIRT = (245, 245, 240)
NAVY = (40, 52, 110)
HAIR = (170, 52, 36)
GROUND = (60, 60, 60)
SS = 4  # supersamples per pixel side: the figure's edge is antialiased as a drawn one is
STILL = (160, 120)
AT = (10, 20)  # the figure's place on the still
SCALE = 2  # the clip's frames are the canvas at twice its size
FRAMES = 16
STEP = 22  # frame px the figure runs each frame: in front of the plate in frames 8..12
OFF = {3: 1, 11: 1}  # frames the model draws the plate a pixel right in
OUTLINE_MAX = 60  # an outline pixel: no channel above it


def figure(scale: int) -> Image.Image:
    """A 40 x 96 runner at `scale`, RGBA: a dark outline round a head, a shirt, shorts, legs and shoes."""
    k = SS * scale
    im = Image.new("RGBA", (40 * k, 96 * k))
    d = ImageDraw.Draw(im)
    line = 2 * k
    d.ellipse((6 * k, 0, 30 * k, 22 * k), fill=HAIR, outline=OUTLINE, width=line)
    d.ellipse((12 * k, 4 * k, 34 * k, 26 * k), fill=SKIN, outline=OUTLINE, width=line)
    for x in (12, 23):
        d.rectangle(((x) * k, 62 * k, (x + 8) * k, 90 * k), fill=SKIN, outline=OUTLINE, width=line)
        d.rectangle(((x - 2) * k, 88 * k, (x + 11) * k, 96 * k - 1), fill=SHIRT, outline=OUTLINE, width=line)
    d.rectangle((10 * k, 26 * k, 34 * k, 54 * k), fill=SHIRT, outline=OUTLINE, width=line)
    d.rectangle((10 * k, 52 * k, 34 * k, 66 * k), fill=NAVY, outline=OUTLINE, width=line)
    return im.resize((40 * scale, 96 * scale), Image.BOX)


def body_mask(scale: int) -> np.ndarray:
    """Where the figure at `scale` covers at least half a pixel."""
    return np.asarray(figure(scale))[..., 3] >= 128


def still(path: Path) -> Path:
    im = Image.new("RGB", STILL, GREEN)
    f = figure(1)
    im.paste(f, AT, f)
    im.save(path)
    return path


def make_canvas(tmp: Path, *, facing: str = "right", plate: bool = True, report: bool = True, **kwargs) -> dict:
    return canvas_mod.run_canvas(still(tmp / "still.png"), tmp / "canvas.png", state="vault", shape="wide", facing=facing,
                                 headroom=None, lead=None, report_path=tmp / "canvas.json" if report else None,
                                 prop_plate=plate, **kwargs)


def film(tmp: Path, report: dict) -> tuple[Path, list[tuple[int, int]]]:
    """The clip a model makes of the canvas: the canvas at SCALE, the plate kept (a pixel to the right in frames 3
    and 11), a ground line under it, the figure running right across in front of it. Returns the clip and the
    figure's place in each frame."""
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg required")
    w, h = report["canvas"]
    x0, y0, x1, y1 = (v * SCALE for v in report["prop_plate"]["box"])
    ox, oy = report["offset"]
    made = tmp / "made"
    made.mkdir()
    runner = figure(SCALE)
    places = []
    for k in range(FRAMES):
        im = Image.new("RGB", (w * SCALE, h * SCALE), GREEN)
        d = ImageDraw.Draw(im)
        off = OFF.get(k, 0)
        d.rectangle((x0 + off, y0, x1 - 1 + off, y1 - 1), fill=prop_plate.RGB)
        d.rectangle((x0 - 2, y1, x1 + 1, y1 + 1), fill=GROUND)
        at = ((ox + AT[0]) * SCALE + STEP * k, (oy + AT[1]) * SCALE)
        im.paste(runner, at, runner)
        places.append(at)
        im.save(made / f"frame-{k:04d}.png")
    subprocess.run(["ffmpeg", "-v", "error", "-framerate", "24", "-i", str(made / "frame-%04d.png"), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "12", str(tmp / "clip.mp4")], check=True)
    return tmp / "clip.mp4", places


def key(tmp: Path, clip: Path, out: str, plate: Path | None) -> dict:
    return frames_mod.run_frames(clip, tmp / out, key="green", allow_edge_contact=False, report_path=None, spill="auto",
                                 reference=tmp / "canvas.png", prop_plate=plate)


def keyed(tmp: Path, out: str) -> list[np.ndarray]:
    return [np.asarray(Image.open(f).convert("RGBA")) for f in sorted((tmp / out / "keyed").glob("*.png"))]


def placed(mask: np.ndarray, at: tuple[int, int], size: tuple[int, int]) -> np.ndarray:
    """`mask` put at `at` on a frame of `size`."""
    out = np.zeros((size[1], size[0]), dtype=bool)
    x, y = at
    out[y:y + mask.shape[0], x:x + mask.shape[1]] = mask
    return out


def grown(mask: np.ndarray, px: int) -> np.ndarray:
    out = mask.copy()
    for _ in range(px):
        out[1:] |= out[:-1].copy(); out[:-1] |= out[1:].copy()
        out[:, 1:] |= out[:, :-1].copy(); out[:, :-1] |= out[:, 1:].copy()
    return out


def shrunk(mask: np.ndarray, px: int) -> np.ndarray:
    return ~grown(~mask, px)


@pytest.fixture(scope="module")
def filmed(tmp_path_factory):
    """One clip keyed twice: with the plate erased by the canvas report's box, and without."""
    tmp = tmp_path_factory.mktemp("plate")
    report = make_canvas(tmp)
    clip, places = film(tmp, report)
    with_plate = key(tmp, clip, "erased", tmp / "canvas.json")
    without = key(tmp, clip, "kept", None)
    return tmp, report, places, with_plate, without


# --- video-canvas ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("facing", ["right", "left"])
def test_the_canvas_paints_the_plate_in_front_of_the_subject_on_its_foot_row(tmp_path: Path, facing: str) -> None:
    report = make_canvas(tmp_path, facing=facing)
    bare, _ = canvas_mod.pad_canvas(Image.open(tmp_path / "still.png"), canvas_mod.profile_for("vault", "wide"), facing=facing)
    left, top, right, bottom = canvas_mod.subject_box(bare, GREEN, 0)
    height = bottom - top
    width, gap = round(prop_plate.WIDTH * height), round(prop_plate.GAP * height)
    x0 = right + gap if facing == "right" else left - gap - width
    box = [x0, bottom - round(prop_plate.HEIGHT * height), x0 + width, bottom]
    assert report["prop_plate"]["box"] == box
    assert report["prop_plate"]["subject"] == [left, top, right, bottom]
    assert json.loads((tmp_path / "canvas.json").read_text())["prop_plate"]["box"] == box
    painted, plain = np.asarray(Image.open(tmp_path / "canvas.png")), np.asarray(bare)
    inside = np.zeros(plain.shape[:2], dtype=bool)
    inside[box[1]:box[3], box[0]:box[2]] = True
    assert (painted[inside] == prop_plate.RGB).all()
    assert (painted[~inside] == plain[~inside]).all(), "the plate is the only change"


def test_without_the_option_the_canvas_is_the_padded_still_and_its_report_has_no_plate(tmp_path: Path) -> None:
    report = make_canvas(tmp_path, plate=False)
    bare, _ = canvas_mod.pad_canvas(Image.open(tmp_path / "still.png"), canvas_mod.profile_for("vault", "wide"))
    assert "prop_plate" not in report and "prop_plate" not in json.loads((tmp_path / "canvas.json").read_text())
    assert (np.asarray(Image.open(tmp_path / "canvas.png")) == np.asarray(bare)).all()


def test_a_plate_the_canvas_has_no_room_for_is_refused(tmp_path: Path) -> None:
    """The still's figure stands at its left: facing left on its own square, the plate would stand off the canvas."""
    with pytest.raises(SystemExit, match="falls past the canvas's left edge"):
        canvas_mod.run_canvas(still(tmp_path / "still.png"), tmp_path / "c.png", state="vault", shape="square", facing="left",
                              headroom=None, lead=None, report_path=tmp_path / "c.json", prop_plate=True)
    assert not (tmp_path / "c.png").exists()


def test_the_plate_needs_a_green_key_and_a_report(tmp_path: Path) -> None:
    im = Image.new("RGB", STILL, (255, 0, 255))
    f = figure(1)
    im.paste(f, AT, f)
    im.save(tmp_path / "magenta.png")
    with pytest.raises(SystemExit, match="green key"):
        canvas_mod.run_canvas(tmp_path / "magenta.png", tmp_path / "c.png", state="vault", shape="wide", facing="right",
                              headroom=None, lead=None, report_path=tmp_path / "c.json", prop_plate=True)
    with pytest.raises(SystemExit, match="--report"):
        make_canvas(tmp_path, report=False)


# --- video-frames ------------------------------------------------------------------------------------------------


def magenta(a: np.ndarray) -> np.ndarray:
    """Where `a` (RGBA) is led by red and blue over green as the erase reads the plate."""
    r, g, b = (a[..., i].astype(np.int32) for i in range(3))
    return (a[..., 3] > 0) & (np.minimum(r, b) - g > prop_plate.MAGENTA_LEAD) & (np.abs(r - b) < prop_plate.MAGENTA_BALANCE)


def test_the_plate_is_gone_from_every_keyed_frame(filmed) -> None:
    """Gone where the model kept the plate where the canvas has it. A frame it is drawn a pixel off in keeps the
    codec's grey blend of its sides, a line a pixel wide down each (the erase's limit, in its docstring)."""
    tmp, report, places, with_plate, without = filmed
    frames = keyed(tmp, "erased")
    size = (frames[0].shape[1], frames[0].shape[0])
    x0, y0, x1, y1 = prop_plate.frame_box(report["prop_plate"]["box"], report["canvas"], size)
    assert with_plate["prop_plate"]["erase_box"] == [x0, y0, x1, y1]
    assert with_plate["prop_plate"]["canvas_report"] == str((tmp / "canvas.json").resolve())
    px0, py0, px1, py1 = (v * SCALE for v in report["prop_plate"]["box"])
    for k, (a, at) in enumerate(zip(frames, places)):
        left = (a[y0:y1, x0:x1, 3] > 0) & ~grown(placed(body_mask(SCALE), at, size), 3)[y0:y1, x0:x1]
        ys, xs = np.nonzero(left)
        assert not (left & magenta(a[y0:y1, x0:x1])).any(), f"frame {k}: magenta left in the box"
        if k not in OFF:
            assert len(xs) == 0, f"frame {k}: {len(xs)} plate pixels left in the box"
            continue
        sides = (px0 + OFF[k], px1 - 1 + OFF[k])  # the first and last column of the plate as drawn in frame k
        assert all(min(abs(x0 + x - side) for side in sides) <= 1 for x in xs), f"frame {k}: left off the plate's sides"
        assert len(xs) <= 2 * 2 * (py1 - py0 + 2), f"frame {k}: {len(xs)} px left, more than a line down each side"
    kept = [int((a[y0:y1, x0:x1, 3] > 0).sum()) for a in keyed(tmp, "kept")]
    assert min(kept) > (x1 - x0 - 2 * prop_plate.RIM) * (y1 - y0 - 2 * prop_plate.RIM) // 2, "keyed without the erase, the plate is there"


def test_the_figure_crossing_in_front_of_the_plate_keeps_its_pixels(filmed) -> None:
    """The figure crosses the plate and its rim, its dark outline over the rim's pixels that are keyed out over most of
    the clip. None of it goes but the few edge pixels the codec tinted with the plate's magenta."""
    tmp, report, places, with_plate, without = filmed
    erased, kept = keyed(tmp, "erased"), keyed(tmp, "kept")
    size = (erased[0].shape[1], erased[0].shape[0])
    x0, y0, x1, y1 = with_plate["prop_plate"]["erase_box"]
    px0, py0, px1, py1 = (v * SCALE for v in report["prop_plate"]["box"])
    rim = np.zeros(size[::-1], dtype=bool)
    rim[y0:y1, x0:x1] = True
    rim[py0:py1, px0:px1] = False
    lost, tinted, over_rim = [], [], []
    for a, b, at in zip(erased, kept, places):
        core = shrunk(placed(body_mask(SCALE), at, size), 2)
        gone = core & (a != b).any(axis=2)
        lost.append(int(gone.sum()))
        tinted.append(int((gone & magenta(b)).sum()))
        over_rim.append(int((core & rim & (b[..., :3].astype(np.int32).max(axis=2) < OUTLINE_MAX)).sum()))
    assert sum(1 for n in over_rim if n) >= 4, f"the outline crosses the rim in too few frames: {over_rim}"
    assert lost == tinted, f"figure pixels erased that the codec did not tint magenta: lost {lost}, tinted {tinted}"
    assert max(lost) <= 2, f"figure pixels erased in the box, frame by frame: {lost}"


def test_outside_the_box_the_frames_are_the_frames_keyed_without_it(filmed) -> None:
    tmp, report, places, with_plate, without = filmed
    x0, y0, x1, y1 = with_plate["prop_plate"]["erase_box"]
    for a, b in zip(keyed(tmp, "erased"), keyed(tmp, "kept")):
        a, b = a.copy(), b.copy()
        a[y0:y1, x0:x1] = b[y0:y1, x0:x1] = 0
        assert (a == b).all()
    assert "prop_plate" not in without


def test_the_erase_reads_the_box_the_canvas_report_holds(filmed, tmp_path: Path) -> None:
    """One box: moved in the canvas report, the erase moves with it."""
    tmp, report, places, with_plate, without = filmed
    moved = json.loads((tmp / "canvas.json").read_text())
    moved["prop_plate"]["box"] = [v + 5 for v in moved["prop_plate"]["box"]]
    (tmp_path / "moved.json").write_text(json.dumps(moved))
    files = sorted((tmp / "erased" / "keyed").glob("*.png"))
    for f in files:
        shutil.copy(f, tmp_path / f.name)
    done = prop_plate.erase(sorted(tmp_path.glob("frame-*.png")), prop_plate.read(tmp_path / "moved.json"))
    size = Image.open(files[0]).size
    assert done["erase_box"] == list(prop_plate.frame_box(moved["prop_plate"]["box"], moved["canvas"], size))
    assert done["erase_box"] != with_plate["prop_plate"]["erase_box"]


def test_a_canvas_report_without_a_plate_is_refused_before_anything_is_keyed(filmed, tmp_path: Path) -> None:
    tmp = filmed[0]
    bare = json.loads((tmp / "canvas.json").read_text())
    del bare["prop_plate"]
    (tmp_path / "bare.json").write_text(json.dumps(bare))
    (tmp_path / "other.json").write_text(json.dumps({"kind": "sprite-gen-video-frames-report"}))
    for report, words in ((tmp_path / "bare.json", "holds no plate"), (tmp_path / "other.json", "not a video-canvas report")):
        with pytest.raises(SystemExit, match=words):
            frames_mod.run_frames(tmp / "clip.mp4", tmp_path / "out", key="green", allow_edge_contact=False, report_path=None,
                                  prop_plate=report)
        assert not (tmp_path / "out" / "raw").exists()


def test_frame_box_at_the_canvas_size_is_the_box_and_its_rim() -> None:
    assert prop_plate.frame_box([700, 729, 777, 919], [1920, 1080], (1920, 1080)) == (697, 726, 780, 922)
    assert prop_plate.frame_box([10, 10, 21, 31], [100, 50], (300, 150)) == (27, 27, 66, 96)
    assert prop_plate.frame_box([0, 40, 5, 50], [100, 50], (100, 50)) == (0, 37, 8, 50)


# --- video-loop: the cut is made of the erased frames -------------------------------------------------------------


def test_the_cut_of_the_erased_frames_has_no_plate(filmed) -> None:
    tmp = filmed[0]
    magenta = []
    for out in ("erased", "kept"):
        loop_mod.run_loop(tmp / out / "keyed", tmp / f"loop-{out}", fps=24, state="vault", min_len=None, max_len=None,
                          n_out=None, seam_max=0.001, name="vault", report_path=None, cycle_mode="once", start=0,
                          length=FRAMES, fill_held="off")
        strip = np.asarray(Image.open(tmp / f"loop-{out}" / "vault.strip.png").convert("RGBA")).astype(np.int32)
        r, g, b = strip[..., 0], strip[..., 1], strip[..., 2]
        magenta.append(int(((strip[..., 3] > 0) & (np.minimum(r, b) - g > prop_plate.MAGENTA_LEAD)
                            & (np.abs(r - b) < prop_plate.MAGENTA_BALANCE)).sum()))
    assert magenta[0] == 0 and magenta[1] > 0, magenta
