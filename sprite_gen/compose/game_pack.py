# SPDX-License-Identifier: Apache-2.0
"""Resize finished assets and publish an immutable game atlas pack.

Request version 2 adds painted scene art (docs/game-pack.md "Version 2"): single images with
anchor and footprint, repeating tiles as their own files, and a UI kit of nine-slice elements
and icons. A version 1 request is packed exactly as before.
"""
from __future__ import annotations

import argparse
from io import BytesIO
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile

from PIL import Image

from sprite_gen.audio.manifest import load as load_sound
from sprite_gen.audio.playback import player as audio_player
from sprite_gen.frames.ui_kit import (REPORT_KIND as UI_KIT_KIND, SIDES, STRETCHES, content_rect, element_path,
                                     measure_insets)
from sprite_gen._deps import np
from sprite_gen.spec.assets import finite, load_asset, load_asset_states
from sprite_gen.spec.runio import inside_folder, same_folder
from sprite_gen.util.resample import resize_cell
from sprite_gen.util.console import utf8_streams

MAX_PAGE_SIZE = 4096
MAX_BYTES = 50_000_000
SCENE_SECTIONS = ("images", "tiles", "ui")
# A "base" footprint counts pixels at least this opaque (docs/game-pack.md "Version 2 request").
FOOTPRINT_ALPHA = 128


def _integer(value, name, *, maximum=None):
    number = finite(value, name, positive=True)
    if number != int(number) or maximum is not None and number > maximum:
        raise ValueError(f"{name} must be an integer between 1 and {maximum or 'its limit'}")
    return int(number)


def _name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError("asset id and state must use letters, digits, underscore, dot or dash")
    return value


def _policy(sequence, entry):
    count = len(sequence.frames)
    metadata = sequence.metadata
    result = {}
    for key, default in (("loop", True), ("hold_last", False), ("end_frame", count - 1),
                         ("blend", "normal"), ("events", []), ("direction", None)):
        result[key] = entry.get(key, metadata.get(key, default))
    for key in ("loop", "hold_last"):
        if type(result[key]) is not bool:
            raise ValueError(f"{key} must be a bool")
    if result["blend"] not in ("normal", "add"):
        raise ValueError("blend must be normal or add")
    if type(result["end_frame"]) is not int or not 0 <= result["end_frame"] < count:
        raise ValueError("end_frame must index a playback frame (zero-based)")
    if result["direction"] is not None and not isinstance(result["direction"], str):
        raise ValueError("direction must be a string or null")
    if not isinstance(result["events"], list):
        raise ValueError("events must be a list")
    for event in result["events"]:
        if (not isinstance(event, dict) or type(event.get("frame")) is not int
                or not 0 <= event["frame"] < count or not isinstance(event.get("name"), str) or not event["name"]):
            raise ValueError("each event needs a zero-based playback frame and a name")
    result["durations_ms"] = [duration * 1000 for duration in sequence.durations]
    return result


def _display(sequence, entry):
    display = entry.get("display")
    if not isinstance(display, dict) or len(display) != 1 or not set(display) <= {"width", "height", "body_height"}:
        raise ValueError("display requires exactly one of width, height or body_height")
    dimension, value = next(iter(display.items()))
    target = _integer(value, f"display {dimension}")
    width, height = sequence.size
    if dimension == "body_height":
        reference = finite(entry.get("source_body_height", sequence.metadata.get("body_h")),
                           "source_body_height (required when the asset has no body_h)", positive=True)
    else:
        reference = width if dimension == "width" else height
    scale = target / reference
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    anchor = [sequence.anchor[0] * size[0] / width, sequence.anchor[1] * size[1] / height]
    sampling = entry.get("resample", "nearest" if sequence.metadata.get("frame_variant") == "pixel" else "lanczos")
    if sampling not in ("nearest", "lanczos"):
        raise ValueError("resample must be nearest or lanczos")
    return size, anchor, sampling


def _version(request):
    if not isinstance(request, dict) or type(request.get("version")) is not int or request["version"] not in (1, 2):
        raise ValueError("game pack request version must be 1 or 2")
    if request["version"] == 1 and any(key in request for key in SCENE_SECTIONS):
        raise ValueError("images, tiles and ui need request version 2")
    return request["version"]


def _cell(image, size, sampling):
    return image.resize(size, Image.Resampling.NEAREST) if sampling == "nearest" else resize_cell(image, size)


def _prepare(request, base, page_size):
    entries = request.get("assets", [])
    if not isinstance(entries, list):
        raise ValueError("assets must be a list")
    ids, animations, frames, sources = set(), {}, [], set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("each asset must be an object")
        asset_id = _name(entry.get("id"))
        if asset_id in ids:
            raise ValueError(f"duplicate asset id: {asset_id}")
        ids.add(asset_id)
        if not isinstance(entry.get("source"), str) or not entry["source"]:
            raise ValueError("each asset requires a source path")
        sequences = load_asset_states(base / entry["source"], state=entry.get("state"), fps=entry.get("fps"), anchor=entry.get("anchor"))
        if len(sequences) > 1 and "direction" in entry:
            raise ValueError("direction override requires one animation; select a source state")
        for state, sequence in sequences.items():
            key = asset_id if state is None else f"{asset_id}/{_name(state)}"
            size, anchor, sampling = _display(sequence, entry)
            # One extruded pixel on each edge keeps filtered textures from bleeding.
            if max(size) + 2 > page_size:
                raise ValueError(f"{key}: frame {size} plus padding exceeds page size {page_size}")
            animation = _policy(sequence, entry)
            animation.update(anchor=anchor, size={"w": size[0], "h": size[1]}, frames=[])
            for index, image in enumerate(sequence.frames):
                name = f"{key}/{index:06d}"
                animation["frames"].append(name)
                frames.append((name, _cell(image, size, sampling), anchor))
            animations[key] = animation
            sources.update(sequence.source_files)
    return dict(sorted(animations.items())), sorted(frames, key=lambda f: f[0]), sources


def _entries(owner, key, allowed):
    """A version 2 list section: objects with a unique id, a source path and only the documented keys."""
    entries = owner.get(key, [])
    if not isinstance(entries, list):
        raise ValueError(f"{key} must be a list")
    ids = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"each {key} entry must be an object")
        unknown = sorted(set(entry) - allowed)
        if unknown:
            raise ValueError(f"unknown {key} key(s): {', '.join(unknown)}")
        ident = _name(entry.get("id"))
        if ident in ids:
            raise ValueError(f"duplicate {key} id: {ident}")
        ids.add(ident)
        if not isinstance(entry.get("source"), str) or not entry["source"]:
            raise ValueError(f"each {key} entry requires a source path")
    return entries


def _still(path, key, anchor=None):
    sequence = load_asset(path, anchor=anchor)
    if len(sequence.frames) != 1:
        raise ValueError(f"{key}: needs exactly one frame, the source has {len(sequence.frames)}")
    return sequence


def _fixed_display(entry, key):
    display = entry.get("display")
    if not isinstance(display, dict) or len(display) != 1 or not set(display) <= {"width", "height"}:
        raise ValueError(f"{key}: display requires exactly one of width or height")


def _fits(key, size, page_size):
    if max(size) + 2 > page_size:
        raise ValueError(f"{key}: frame {size} plus padding exceeds page size {page_size}")


def _footprint(entry, image, size, key):
    value = entry.get("footprint")
    if value is None:
        return None
    width, height = image.size
    if value == "base":
        opaque = np.asarray(image.getchannel("A")) >= FOOTPRINT_ALPHA
        rows = np.flatnonzero(opaque.any(axis=1))
        if not rows.size:
            raise ValueError(f"{key}: no pixel reaches alpha {FOOTPRINT_ALPHA}, so the footprint has no base")
        bottom = int(rows[-1]) + 1
        top = bottom - max(1, math.ceil((bottom - int(rows[0])) / 5))
        band = opaque[top:bottom]
        cols, band_rows = np.flatnonzero(band.any(axis=0)), np.flatnonzero(band.any(axis=1))
        x0, x1 = int(cols[0]), int(cols[-1]) + 1
        y0, y1 = top + int(band_rows[0]), top + int(band_rows[-1]) + 1
    elif isinstance(value, dict) and set(value) == {"x", "y", "w", "h"} and all(type(v) is int for v in value.values()):
        x0, y0 = value["x"], value["y"]
        x1, y1 = x0 + value["w"], y0 + value["h"]
        if value["w"] < 1 or value["h"] < 1 or x0 < 0 or y0 < 0 or x1 > width or y1 > height:
            raise ValueError(f"{key}: footprint must be a nonempty rectangle inside the {width}x{height} source")
    else:
        raise ValueError(f'{key}: footprint must be "base" or integer {{x, y, w, h}} in source pixels')
    # outward to whole stored pixels: left and top down, right and bottom up, inside the frame
    fx0, fy0 = x0 * size[0] // width, y0 * size[1] // height
    fx1, fy1 = min(size[0], -(-x1 * size[0] // width)), min(size[1], -(-y1 * size[1] // height))
    return {"x": fx0, "y": fy0, "w": max(1, fx1 - fx0), "h": max(1, fy1 - fy0)}


def _scaled_up(value, scale):
    return math.ceil(round(value * scale, 9))


def _sides(record, field, key):
    sides = record.get(field)
    if not isinstance(sides, dict) or set(sides) != set(SIDES) or any(type(sides[s]) is not int or sides[s] < 0 for s in SIDES):
        raise ValueError(f"{key}: {field} needs non-negative integer left, right, top and bottom")
    return sides


def _ends_kept(cell, insets, width):
    """``cell`` as a three-slice drawn ``width`` wide: its two end slices as they are, its middle columns
    repeated or dropped (nearest), never blended."""
    if cell.width == width:
        return cell
    left, right = insets["left"], insets["right"]
    stored = Image.new("RGBA", (width, cell.height), (0, 0, 0, 0))
    stored.paste(cell.crop((0, 0, left, cell.height)), (0, 0))
    middle = cell.crop((left, 0, cell.width - right, cell.height))
    stored.paste(middle.resize((width - left - right, cell.height), Image.Resampling.NEAREST), (left, 0))
    stored.paste(cell.crop((cell.width - right, 0, cell.width, cell.height)), (width - right, 0))
    return stored


def _fill(name, record, image, frame, page_size):
    """A fill stored at its frame's content rectangle (docs/game-pack.md "Version 2 manifest"): resized by the
    one factor that makes it as tall as the rectangle, then drawn as wide as it with its ends kept."""
    key = f"ui/{_name(name)}"
    rect = content_rect(frame)
    factor = rect["h"] / image.height
    cell = resize_cell(image, (max(1, round(image.width * factor)), rect["h"]))
    insets = {side: _scaled_up(record["insets"][side], factor) for side in SIDES}
    measured, _notes = measure_insets(cell, "x")
    insets = {side: max(insets[side], measured[side]) for side in SIDES}
    if insets["left"] + insets["right"] >= min(cell.width, rect["w"]):
        raise ValueError(f"{key}: scaled to the height of {record['inside']['element']}'s content area, its fixed "
                         "ends are wider than that area")
    padding = {side: _scaled_up(record["padding"][side], factor) for side in SIDES}
    if padding["left"] + padding["right"] >= rect["w"] or padding["top"] + padding["bottom"] >= rect["h"]:
        raise ValueError(f"{key}: in its frame's content area the padding leaves no content area")
    size = (rect["w"], rect["h"])
    _fits(key, size, page_size)
    entry = {"frame": key, "size": {"w": size[0], "h": size[1]}, "stretch": "x", "insets": insets,
             "padding": padding, "inside": {"element": record["inside"]["element"], "rect": rect}}
    return entry, (key, _ends_kept(cell, insets, size[0]), (size[0] / 2, size[1] / 2))


def _ui_kit(path, scale, page_size):
    """Nine-slice entries and atlas frames from a `ui-kit` report (docs/ui-kit.md "Report")."""
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or report.get("kind") != UI_KIT_KIND or report.get("version") != 1:
        raise ValueError(f"ui.kit must be a version 1 ui-kit report ({UI_KIT_KIND}): {path.name}")
    elements = report.get("elements")
    if not isinstance(elements, dict) or not elements:
        raise ValueError("ui-kit report lists no elements")
    entries, frames, sources, fills = {}, [], {path}, []
    for name, record in elements.items():
        key = f"ui/{_name(name)}"
        if not isinstance(record, dict) or record.get("stretch") not in STRETCHES:
            raise ValueError(f"{key}: stretch must be one of {', '.join(STRETCHES)}")
        stretch = record["stretch"]
        insets, padding = _sides(record, "insets", key), _sides(record, "padding", key)
        fixed = {"x": ("top", "bottom"), "none": SIDES}.get(stretch, ())
        if any(insets[side] for side in fixed):
            raise ValueError(f"{key}: stretch {stretch} needs {', '.join(fixed)} insets of 0")
        if not isinstance(record.get("file"), str) or not record["file"]:
            raise ValueError(f"{key}: the report names no file")
        sequence = _still(element_path(path, record["file"]), key)
        image = sequence.frames[0]
        if record.get("size") != {"w": image.width, "h": image.height}:
            raise ValueError(f"{key}: the report's size does not match {record['file']}")
        sources.update(sequence.source_files)
        if "inside" in record:
            # a fill takes its size from its frame's stored content area, so it is packed after every frame
            inside = record["inside"]
            if not isinstance(inside, dict) or not isinstance(inside.get("element"), str):
                raise ValueError(f"{key}: inside must name the element the fill goes in")
            if stretch != "x":
                raise ValueError(f"{key}: a fill stretches in width only (stretch x), not {stretch}")
            fills.append((name, record, image))
            continue
        size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        _fits(key, size, page_size)
        cell = image if size == image.size else resize_cell(image, size)
        insets = {side: _scaled_up(insets[side], scale) for side in SIDES}
        if cell is not image:
            # Resampling spreads an ornament's edge a little past its scaled place; the stored pixels are
            # measured the way `ui-kit` measured the source, and the larger inset of the two is kept.
            measured, _notes = measure_insets(cell, stretch)
            insets = {side: max(insets[side], measured[side]) for side in SIDES}
        padding = {side: _scaled_up(padding[side], scale) for side in SIDES}
        if (stretch != "none" and insets["left"] + insets["right"] >= size[0]) or (
                stretch == "xy" and insets["top"] + insets["bottom"] >= size[1]):
            raise ValueError(f"{key}: at ui scale {scale} the insets leave no stretchable middle")
        if padding["left"] + padding["right"] >= size[0] or padding["top"] + padding["bottom"] >= size[1]:
            raise ValueError(f"{key}: at ui scale {scale} the padding leaves no content area")
        frames.append((key, cell, (size[0] / 2, size[1] / 2)))
        entries[name] = {"frame": key, "size": {"w": size[0], "h": size[1]}, "stretch": stretch,
                         "insets": insets, "padding": padding}
    frame_names = set(entries)
    for name, record, image in fills:
        target = record["inside"]["element"]
        if target not in frame_names:
            raise ValueError(f"ui/{_name(name)}: inside names {target}, which is not a frame of this report")
        entries[name], frame = _fill(name, record, image, entries[target], page_size)
        frames.append(frame)
    return entries, frames, sources


def _encode_image(image, fmt):
    buffer = BytesIO()
    if fmt == "PNG":
        image.save(buffer, format="PNG", compress_level=9)
    else:
        image.save(buffer, format="WEBP", lossless=True, exact=True, method=6)
    return buffer.getvalue()


def _prepare_scene(request, base, page_size):
    """The version 2 sections: manifest entries, atlas frames, tile files and the sources they read."""
    frames, files, sources = [], {}, set()
    images = {}
    for entry in _entries(request, "images", {"id", "source", "display", "resample", "anchor", "footprint"}):
        key = f"image/{entry['id']}"
        sequence = _still(base / entry["source"], key, anchor=entry.get("anchor"))
        _fixed_display(entry, key)
        size, anchor, sampling = _display(sequence, entry)
        _fits(key, size, page_size)
        frames.append((key, _cell(sequence.frames[0], size, sampling), anchor))
        images[entry["id"]] = {"frame": key, "size": {"w": size[0], "h": size[1]}, "anchor": anchor,
                               "footprint": _footprint(entry, sequence.frames[0], size, key)}
        sources.update(sequence.source_files)
    tiles = {}
    for entry in _entries(request, "tiles", {"id", "source", "display"}):
        key = f"tile/{entry['id']}"
        sequence = _still(base / entry["source"], key)
        _fixed_display(entry, key)
        image = sequence.frames[0]
        dimension, value = next(iter(entry["display"].items()))
        target = _integer(value, f"display {dimension}")
        scale = target / (image.width if dimension == "width" else image.height)
        png, webp = f"tile-{entry['id']}.png", f"tile-{entry['id']}.webp"
        files[png], files[webp] = _encode_image(image, "PNG"), _encode_image(image, "WEBP")
        tiles[entry["id"]] = {"image": png, "webp": webp, "size": {"w": image.width, "h": image.height},
                              "display_size": {"w": max(1, round(image.width * scale)),
                                               "h": max(1, round(image.height * scale))}}
        sources.update(sequence.source_files)
    nine_slice, icons = {}, {}
    ui = request.get("ui")
    if ui is not None:
        if not isinstance(ui, dict):
            raise ValueError("ui must be an object")
        unknown = sorted(set(ui) - {"kit", "scale", "icons"})
        if unknown:
            raise ValueError(f"unknown ui key(s): {', '.join(unknown)}")
        scale = finite(ui.get("scale", 1), "ui scale", positive=True)
        if "kit" in ui:
            if not isinstance(ui["kit"], str) or not ui["kit"]:
                raise ValueError("ui.kit must be the path of a ui-kit report")
            nine_slice, kit_frames, kit_sources = _ui_kit((base / ui["kit"]).resolve(), scale, page_size)
            frames += kit_frames
            sources |= kit_sources
        for entry in _entries(ui, "icons", {"id", "source", "display", "resample"}):
            key = f"icon/{entry['id']}"
            sequence = _still(base / entry["source"], key)
            _fixed_display(entry, key)
            size, _anchor, sampling = _display(sequence, entry)
            _fits(key, size, page_size)
            frames.append((key, _cell(sequence.frames[0], size, sampling), (size[0] / 2, size[1] / 2)))
            icons[entry["id"]] = {"frame": key, "size": {"w": size[0], "h": size[1]}}
            sources.update(sequence.source_files)
    sections = {"images": dict(sorted(images.items())), "tiles": dict(sorted(tiles.items())),
                "ui": {"nine_slice": dict(sorted(nine_slice.items())), "icons": dict(sorted(icons.items()))}}
    return sections, frames, files, sources


def _layout(frames, limit):
    """Deterministic shelf placement; playback order is independent of texture position."""
    pages, placed = [], []
    x = y = row_height = width = height = 0
    for name, image, anchor in frames:
        w, h = image.width + 2, image.height + 2
        if x + w > limit:
            x, y, row_height = 0, y + row_height, 0
        if y + h > limit:
            pages.append((width, height, placed))
            placed = []
            x = y = row_height = width = height = 0
        placed.append((name, image, anchor, x + 1, y + 1))
        x += w
        row_height = max(row_height, h)
        width, height = max(width, x), max(height, y + h)
    pages.append((width, height, placed))
    return pages


def _paste_extruded(sheet, image, x, y):
    w, h = image.size
    sheet.paste(image, (x, y))
    sheet.paste(image.crop((0, 0, w, 1)), (x, y - 1))
    sheet.paste(image.crop((0, h - 1, w, h)), (x, y + h))
    sheet.paste(image.crop((0, 0, 1, h)), (x - 1, y))
    sheet.paste(image.crop((w - 1, 0, w, h)), (x + w, y))
    for dx, sx in ((-1, 0), (w, w - 1)):
        for dy, sy in ((-1, 0), (h, h - 1)):
            sheet.putpixel((x + dx, y + dy), image.getpixel((sx, sy)))


def _encode(pages, animations, max_bytes, sections=None, files=None):
    """Pages, then any version 2 files, then the manifest; ``sections`` makes it a version 2 manifest."""
    payloads, textures = {}, []
    total = 0
    for index, (width, height, placed) in enumerate(pages):
        sheet = Image.new("RGBA", (width, height))
        records = {}
        for name, image, anchor, x, y in placed:
            _paste_extruded(sheet, image, x, y)
            size = {"w": image.width, "h": image.height}
            records[name] = {"frame": {"x": x, "y": y, **size}, "rotated": False, "trimmed": False,
                             "spriteSourceSize": {"x": 0, "y": 0, **size}, "sourceSize": size,
                             "pivot": {"x": anchor[0]/image.width, "y": anchor[1]/image.height}}
        png_name, webp_name = f"atlas-{index:03d}.png", f"atlas-{index:03d}.webp"
        for name, fmt, options in ((png_name, "PNG", {"compress_level": 9}),
                                   (webp_name, "WEBP", {"lossless": True, "exact": True, "method": 6})):
            buffer = BytesIO()
            sheet.save(buffer, format=fmt, **options)
            payloads[name] = buffer.getvalue()
            total += len(payloads[name])
            if total > max_bytes:
                raise ValueError(f"game pack exceeds {max_bytes} bytes (50 MB maximum)")
        textures.append({"image": png_name, "webp": webp_name, "frames": records,
                         "meta": {"image": png_name, "format": "RGBA8888", "size": {"w": width, "h": height}, "scale": "1"}})
    for name, data in (files or {}).items():
        payloads[name] = data
        total += len(data)
        if total > max_bytes:
            raise ValueError(f"game pack exceeds {max_bytes} bytes (50 MB maximum)")
    result = {"kind": "sprite-gen-game-pack", "version": 1, "textures": textures, "animations": animations}
    if sections is not None:
        result["version"] = 2
        result.update(sections)
    payloads["game-pack.json"] = (json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode()
    if total + len(payloads["game-pack.json"]) > max_bytes:
        raise ValueError(f"game pack exceeds {max_bytes} bytes (50 MB maximum, including its manifest)")
    return result, payloads


def _publish(out_dir, payloads):
    """One atomic directory rename; concurrent publishers cannot interleave packs."""
    def identical():
        return (out_dir.is_dir() and {p.name for p in out_dir.iterdir()} == set(payloads)
                and all((out_dir / name).is_file() and (out_dir / name).read_bytes() == data
                        for name, data in payloads.items()))

    if out_dir.exists():
        if identical():
            return
        raise ValueError("output directory already exists with different content; use a new directory")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}-", dir=out_dir.parent))
    try:
        for name, data in payloads.items():
            with (stage / name).open("wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        try:
            stage.rename(out_dir)
        except OSError:
            if not identical():
                raise
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def pack(request: Path, out_dir: Path, *, page_size=MAX_PAGE_SIZE, max_bytes=MAX_BYTES):
    page_size = _integer(page_size, "page size", maximum=MAX_PAGE_SIZE)
    max_bytes = _integer(max_bytes, "max bytes", maximum=MAX_BYTES)
    request, out_dir = Path(request).expanduser().resolve(), Path(out_dir).expanduser().resolve()
    document = json.loads(request.read_text(encoding="utf-8"))
    version = _version(document)
    animations, frames, sources = _prepare(document, request.parent, page_size)
    # `files` is the audio loop's name below; the version 2 tile files must not share it.
    sections = tile_files = None
    if version == 2:
        sections, scene_frames, tile_files, scene_sources = _prepare_scene(document, request.parent, page_size)
        frames = sorted(frames + scene_frames, key=lambda f: f[0])
        sources |= scene_sources
        names = [name for name, _image, _anchor in frames]
        if len(set(names)) != len(names):
            raise ValueError("two atlas frames share a name; rename an asset or scene entry")
    audio_entries = document.get("audio", [])
    if not isinstance(audio_entries, list):
        raise ValueError("audio must be a list")
    audio, audio_payloads = {}, {}
    for entry in audio_entries:
        if not isinstance(entry, dict):
            raise ValueError("each audio entry must be an object")
        asset_id = _name(entry.get("id"))
        if asset_id in audio:
            raise ValueError(f"duplicate audio id: {asset_id}")
        if not isinstance(entry.get("source"), str) or not entry["source"]:
            raise ValueError("each audio entry requires a source path")
        remaining = max_bytes - sum(map(len, audio_payloads.values()))
        record, files, paths = load_sound(request.parent / entry["source"], remaining)
        for fmt, data in files.items():
            name = f"audio-{asset_id}.{fmt}"
            record["files"][fmt]["path"] = name
            audio_payloads[name] = data
        audio[asset_id] = record
        sources.update(paths)
    if not frames and not tile_files and not audio:
        raise ValueError("game pack request declares nothing to pack: give it an asset, a scene entry or a sound")
    if any(same_folder(path, out_dir) or inside_folder(path, out_dir) for path in sources | {request}):
        raise ValueError("output directory must not contain any source file")
    result, payloads = _encode(_layout(frames, page_size) if frames else [], animations, max_bytes, sections, tile_files)
    if audio:
        helper, module = audio_player()
        audio_payloads[helper["path"]] = module
        for record in audio.values():
            record["player"] = helper
        result["audio"] = dict(sorted(audio.items()))
        payloads.update(audio_payloads)
        payloads["game-pack.json"] = (json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode()
        if sum(map(len, payloads.values())) > max_bytes:
            raise ValueError(f"game pack exceeds {max_bytes} bytes (including audio)")
    _publish(out_dir, payloads)
    return result


def add_arguments(parser):
    parser.epilog = "events[].frame and end_frame are zero-based indices in output playback order."
    parser.add_argument("--request", type=Path, required=True, help="JSON result list with display sizes")
    parser.add_argument("--out-dir", type=Path, required=True, help="new directory for the immutable pack")
    parser.add_argument("--page-size", type=int, default=MAX_PAGE_SIZE)
    parser.add_argument("--max-bytes", type=int, default=MAX_BYTES)


def run(**kwargs):
    try:
        result = pack(kwargs["request"], kwargs["out_dir"], page_size=kwargs.get("page_size", MAX_PAGE_SIZE),
                      max_bytes=kwargs.get("max_bytes", MAX_BYTES))
    except (ValueError, OSError) as exc:
        raise SystemExit(f"game-pack: {exc}") from exc
    print(json.dumps({"ok": True, "manifest": str(Path(kwargs["out_dir"]) / "game-pack.json"),
                      "pages": len(result["textures"]), "animations": len(result["animations"])}))
    return 0


def main():
    utf8_streams()
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    return run(**vars(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
