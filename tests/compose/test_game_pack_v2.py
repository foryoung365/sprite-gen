# SPDX-License-Identifier: Apache-2.0
"""Pack contract version 2 (docs/game-pack.md "Version 2"): images, tiles and ui sections.

The manifest is checked field by field against the contract, with exact key sets, so a
consumer never meets a field the doc does not define. A version 1 request keeps its bytes:
the manifest text and every page's pixels are frozen below from the packer before version 2.
"""
import hashlib
import json
import math

import numpy as np
import pytest
from PIL import Image, ImageDraw

from sprite_gen.compose import game_pack
from sprite_gen.frames import ui_kit
from sprite_gen.frames.ui_kit import measure_insets
from sprite_gen.util.resample import resize_cell


def write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def prop(path, size=(40, 60)):
    """A tall prop on a transparent canvas: a pole over a wide base slab, so the base differs from the bbox."""
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((16, 4, 23, 49), fill=(90, 120, 200, 255))
    draw.rectangle((6, 50, 33, 57), fill=(120, 110, 90, 255))
    image.save(path)
    return image


def tile(path, size=(24, 16)):
    image = Image.new("RGBA", size)
    image.putdata([((x * 37) % 256, (y * 53) % 256, (x * y) % 256, 255) for y in range(size[1]) for x in range(size[0])])
    image.save(path)
    return image


def element(path, size, color):
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, size[0] - 1, size[1] - 1), outline=color, width=3)
    draw.rectangle((3, 3, size[0] - 4, size[1] - 4), fill=(20, 30, 50, 200))
    image.save(path)
    return image


UI_ELEMENTS = {
    "panel": ((40, 30), "xy", {"left": 7, "right": 5, "top": 6, "bottom": 3}, {"left": 4, "right": 4, "top": 4, "bottom": 5}),
    "hp_bar_frame": ((48, 12), "x", {"left": 9, "right": 5, "top": 0, "bottom": 0}, {"left": 10, "right": 6, "top": 3, "bottom": 3}),
    "timer_badge": ((16, 16), "none", {"left": 0, "right": 0, "top": 0, "bottom": 0}, {"left": 5, "right": 5, "top": 5, "bottom": 5}),
}


def ui_kit_report(directory):
    directory.mkdir()
    elements, images = {}, {}
    for index, (name, (size, stretch, insets, padding)) in enumerate(UI_ELEMENTS.items()):
        images[name] = element(directory / f"{name}.png", size, (200, 40 * index, 90, 255))
        elements[name] = {"file": f"{name}.png", "box": {"x": 0, "y": 0, "w": 64, "h": 64},
                          "rect": {"x": 1, "y": 1, "w": size[0], "h": size[1]},
                          "size": {"w": size[0], "h": size[1]}, "stretch": stretch,
                          "insets": insets, "padding": padding, "parts": 1}
    write_json(directory / "ui-kit.json", {
        "kind": "sprite-gen-ui-kit", "version": 1, "sheet": "/unused/sheet.png", "sheet_size": [256, 256],
        "status": "pass", "review_reasons": [], "alpha_cleanup": None, "dropped_parts": 0, "unassigned": [],
        "elements": elements,
    })
    return images


def scene_request(tmp_path, **override):
    prop(tmp_path / "pillar.png")
    prop(tmp_path / "stone.png", size=(30, 20))
    tile(tmp_path / "ground.png")
    Image.new("RGBA", (10, 10), (250, 200, 40, 255)).save(tmp_path / "sword.png")
    ui_kit_report(tmp_path / "ui")
    document = {
        "version": 2,
        "assets": [{"id": "coin", "source": "sword.png", "display": {"height": 5}}],
        "images": [
            {"id": "pillar", "source": "pillar.png", "display": {"height": 30}, "footprint": "base"},
            {"id": "stone", "source": "stone.png", "display": {"width": 15}, "anchor": [10, 18],
             "footprint": {"x": 3, "y": 11, "w": 9, "h": 5}},
            {"id": "plain", "source": "stone.png", "display": {"width": 30}},
        ],
        "tiles": [{"id": "ground", "source": "ground.png", "display": {"width": 12}}],
        "ui": {"kit": "ui/ui-kit.json", "scale": 0.5,
               "icons": [{"id": "sword", "source": "sword.png", "display": {"height": 20}, "resample": "nearest"}]},
    }
    document.update(override)
    return write_json(tmp_path / "game.json", document)


def atlas_frames(result):
    return {name: (page, record) for page in result["textures"] for name, record in page["frames"].items()}


def crop(out, result, name):
    page, record = atlas_frames(result)[name]
    rect = record["frame"]
    with Image.open(out / page["image"]) as image:
        return image.crop((rect["x"], rect["y"], rect["x"] + rect["w"], rect["y"] + rect["h"])), record


def all_ints(value):
    if isinstance(value, dict):
        return all(all_ints(v) for v in value.values())
    if isinstance(value, list):
        return all(all_ints(v) for v in value)
    return type(value) is int


def test_the_sides_of_a_nine_slice_are_the_ui_kit_s():
    assert game_pack.SIDES is ui_kit.SIDES


def test_version_2_manifest_has_exactly_the_documented_sections(tmp_path):
    out = tmp_path / "out"
    result = game_pack.pack(scene_request(tmp_path), out)
    assert json.loads((out / "game-pack.json").read_text()) == result
    assert result["version"] == 2
    assert set(result) == {"kind", "version", "textures", "animations", "images", "tiles", "ui"}
    assert set(result["ui"]) == {"nine_slice", "icons"}
    assert list(result["images"]) == sorted(result["images"]) == ["pillar", "plain", "stone"]
    assert "coin" in result["animations"]


def test_images_are_atlas_frames_with_anchor_and_footprint_in_stored_pixels(tmp_path):
    out = tmp_path / "out"
    result = game_pack.pack(scene_request(tmp_path), out)
    pillar, stone, plain = (result["images"][key] for key in ("pillar", "stone", "plain"))
    for entry in (pillar, stone, plain):
        assert set(entry) == {"frame", "size", "anchor", "footprint"}
    # height 30 from 60: scale 0.5, a 20 x 30 frame; default anchor is the bottom centre
    assert pillar["frame"] == "image/pillar"
    assert pillar["size"] == {"w": 20, "h": 30}
    assert pillar["anchor"] == [10, 30]
    # base: opaque bbox rows 4..57 (54 rows), lowest fifth = ceil(54 / 5) = 11 rows (47..57); opaque there:
    # the pole x 16..23 for rows 47..49 and the slab x 6..33 rows 50..57 -> x 6..33, y 47..57 in source pixels,
    # scaled outward: x 3..17 (floor 3, ceil 17), y 23..29 (floor 23.5 -> 23, ceil 29) -> {3, 23, 14, 6}
    assert pillar["footprint"] == {"x": 3, "y": 23, "w": 14, "h": 6}
    # width 15 from 30: scale 0.5; anchor [10, 18] -> [5, 9]; footprint x 3..12, y 11..16 -> 1..6, 5..8
    assert stone["size"] == {"w": 15, "h": 10}
    assert stone["anchor"] == [5, 9]
    assert stone["footprint"] == {"x": 1, "y": 5, "w": 5, "h": 3}
    assert plain["footprint"] is None
    image, record = crop(out, result, "image/pillar")
    assert image.size == (20, 30)
    with Image.open(tmp_path / "pillar.png") as source:
        assert image.tobytes() == resize_cell(source.convert("RGBA"), (20, 30)).tobytes()
    assert record["pivot"] == {"x": 0.5, "y": 1.0}
    assert record["trimmed"] is False and record["rotated"] is False
    _, record = crop(out, result, "image/stone")
    assert record["pivot"] == {"x": 5 / 15, "y": 9 / 10}
    assert all_ints([[entry["size"], entry["footprint"] or {}] for entry in result["images"].values()])


def test_tiles_are_separate_files_with_stored_and_display_sizes(tmp_path):
    out = tmp_path / "out"
    result = game_pack.pack(scene_request(tmp_path), out)
    ground = result["tiles"]["ground"]
    assert ground == {"image": "tile-ground.png", "webp": "tile-ground.webp",
                      "size": {"w": 24, "h": 16}, "display_size": {"w": 12, "h": 8}}
    assert "tile-ground" not in json.dumps(result["textures"])
    with Image.open(tmp_path / "ground.png") as source, Image.open(out / ground["image"]) as png, \
            Image.open(out / ground["webp"]) as webp:
        assert png.format == "PNG" and webp.format == "WEBP"
        pixels = source.convert("RGBA").tobytes()
        assert png.convert("RGBA").tobytes() == pixels
        assert webp.convert("RGBA").tobytes() == pixels


def test_ui_nine_slice_insets_and_padding_are_in_stored_frame_pixels(tmp_path):
    out = tmp_path / "out"
    result = game_pack.pack(scene_request(tmp_path), out)
    nine = result["ui"]["nine_slice"]
    assert list(nine) == sorted(UI_ELEMENTS)
    for name, (size, stretch, insets, padding) in UI_ELEMENTS.items():
        entry = nine[name]
        assert set(entry) == {"frame", "size", "stretch", "insets", "padding"}
        assert entry["frame"] == f"ui/{name}"
        stored = (round(size[0] * 0.5), round(size[1] * 0.5))
        assert entry["size"] == {"w": stored[0], "h": stored[1]}
        assert entry["stretch"] == stretch
        image, record = crop(out, result, entry["frame"])
        with Image.open(tmp_path / "ui" / f"{name}.png") as source:
            assert image.tobytes() == resize_cell(source.convert("RGBA"), stored).tobytes()
        # scaled and rounded up, or what the ui-kit measure finds on the stored pixels if that is larger:
        # an ornament and its resampled fringe stay inside the fixed slices; content only shrinks
        measured, _ = measure_insets(image, stretch)
        assert entry["insets"] == {side: max(math.ceil(value * 0.5), measured[side]) for side, value in insets.items()}
        assert all(entry["insets"][side] >= math.ceil(value * 0.5) for side, value in insets.items())
        assert entry["padding"] == {side: math.ceil(value * 0.5) for side, value in padding.items()}
        assert set(entry["insets"]) == set(entry["padding"]) == {"left", "right", "top", "bottom"}
        assert record["pivot"] == {"x": 0.5, "y": 0.5}
    assert nine["hp_bar_frame"]["insets"]["top"] == nine["hp_bar_frame"]["insets"]["bottom"] == 0
    assert set(nine["timer_badge"]["insets"].values()) == {0}
    assert all_ints({name: {k: v for k, v in entry.items() if k not in ("frame", "stretch")} for name, entry in nine.items()})


def test_ui_icons_are_fixed_atlas_frames(tmp_path):
    out = tmp_path / "out"
    result = game_pack.pack(scene_request(tmp_path), out)
    assert result["ui"]["icons"] == {"sword": {"frame": "icon/sword", "size": {"w": 20, "h": 20}}}
    image, record = crop(out, result, "icon/sword")
    assert image.getpixel((7, 7)) == (250, 200, 40, 255)
    assert record["pivot"] == {"x": 0.5, "y": 0.5}


def test_version_2_sections_default_to_empty_and_a_tile_only_pack_has_no_atlas(tmp_path):
    tile(tmp_path / "ground.png")
    request = write_json(tmp_path / "game.json", {
        "version": 2, "tiles": [{"id": "ground", "source": "ground.png", "display": {"height": 32}}]})
    out = tmp_path / "out"
    result = game_pack.pack(request, out)
    assert result["textures"] == [] and result["animations"] == {} and result["images"] == {}
    assert result["ui"] == {"nine_slice": {}, "icons": {}}
    assert result["tiles"]["ground"]["display_size"] == {"w": 48, "h": 32}
    assert sorted(p.name for p in out.iterdir()) == ["game-pack.json", "tile-ground.png", "tile-ground.webp"]


def test_version_2_output_is_deterministic(tmp_path):
    request = scene_request(tmp_path)
    game_pack.pack(request, tmp_path / "a")
    game_pack.pack(request, tmp_path / "b")
    read = lambda d: {p.name: p.read_bytes() for p in d.iterdir()}
    assert read(tmp_path / "a") == read(tmp_path / "b")
    assert str(tmp_path) not in (tmp_path / "a" / "game-pack.json").read_text()


@pytest.mark.parametrize("change,message", [
    ({"version": 3}, "version must be 1 or 2"),
    ({"images": [{"id": "pillar", "source": "pillar.png", "display": {"body_height": 30}}]}, "width or height"),
    ({"images": [{"id": "a", "source": "pillar.png", "display": {"height": 9}},
                 {"id": "a", "source": "stone.png", "display": {"height": 9}}]}, "duplicate"),
    ({"images": [{"id": "pillar", "source": "pillar.png", "display": {"height": 9}, "footprint": "top"}]}, "footprint"),
    ({"images": [{"id": "pillar", "source": "pillar.png", "display": {"height": 9},
                  "footprint": {"x": 0, "y": 0, "w": 0, "h": 4}}]}, "footprint"),
    ({"tiles": [{"id": "ground", "source": "ground.png", "display": {"width": 12, "height": 8}}]}, "width or height"),
    ({"ui": {"kit": "ui/ui-kit.json", "scale": 0}}, "scale"),
    ({"ui": {"kit": "ui/ui-kit.json", "scale": 0.5, "extra": 1}}, "unknown ui key"),
    ({"ui": {"kit": "ui/ui-kit.json", "scale": 0.05}}, "inset"),
])
def test_bad_version_2_requests_are_refused(tmp_path, change, message):
    request = scene_request(tmp_path, **change)
    with pytest.raises(ValueError, match=message):
        game_pack.pack(request, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_an_image_needs_exactly_one_frame(tmp_path):
    strip = Image.new("RGBA", (8, 4), (255, 0, 0, 255))
    strip.save(tmp_path / "two.png")
    write_json(tmp_path / "two.json", {"frames": 2, "w": 4, "h": 4, "delay_ms": 100})
    request = write_json(tmp_path / "game.json", {
        "version": 2, "images": [{"id": "two", "source": "two.json", "display": {"height": 4}}]})
    with pytest.raises(ValueError, match="one frame"):
        game_pack.pack(request, tmp_path / "out")


def test_a_ui_kit_report_must_be_a_ui_kit_report(tmp_path):
    request = scene_request(tmp_path)
    report = json.loads((tmp_path / "ui" / "ui-kit.json").read_text())
    report["elements"]["panel"]["stretch"] = "y"
    write_json(tmp_path / "ui" / "ui-kit.json", report)
    with pytest.raises(ValueError, match="stretch"):
        game_pack.pack(request, tmp_path / "out")
    report["elements"]["panel"]["stretch"] = "xy"
    report["kind"] = "sprite-gen-game-pack"
    write_json(tmp_path / "ui" / "ui-kit.json", report)
    with pytest.raises(ValueError, match="ui-kit"):
        game_pack.pack(request, tmp_path / "out")


def fill_strip(path, size, cap):
    """A bar fill: a green strip with a lighter end cap at each end."""
    image = Image.new("RGBA", size, (40, 180, 90, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, cap - 1, size[1] - 1), fill=(200, 250, 210, 255))
    draw.rectangle((size[0] - cap, 0, size[0] - 1, size[1] - 1), fill=(200, 250, 210, 255))
    image.save(path)
    return image


def fill_kit(tmp_path, *, scale=1, inside="hp_bar_frame", stretch="x", cap=3, size=(40, 8)):
    """A report with a bar frame (content area 32 x 6 at scale 1) and a fill drawn inside it."""
    ui_kit_report(tmp_path / "ui")
    report = json.loads((tmp_path / "ui" / "ui-kit.json").read_text())
    fill_strip(tmp_path / "ui" / "hp_bar_fill.png", size, cap)
    record = {"file": "hp_bar_fill.png", "box": {"x": 0, "y": 0, "w": 64, "h": 16},
              "rect": {"x": 1, "y": 1, "w": size[0], "h": size[1]}, "size": {"w": size[0], "h": size[1]},
              "stretch": stretch, "insets": {"left": cap + 2, "right": cap + 2, "top": 0, "bottom": 0},
              "padding": {"left": 0, "right": 0, "top": 0, "bottom": 0}, "parts": 1}
    if inside is not None:
        record["inside"] = {"element": inside, "rect": {"x": 10, "y": 3, "w": 32, "h": 6}, "scale": 0.75}
    report["elements"]["hp_bar_fill"] = record
    write_json(tmp_path / "ui" / "ui-kit.json", report)
    return write_json(tmp_path / "game.json", {"version": 2, "ui": {"kit": "ui/ui-kit.json", "scale": scale}})


@pytest.mark.parametrize("scale", [1, 0.5, 2])
def test_a_fill_is_stored_at_its_frames_content_area(tmp_path, scale):
    out = tmp_path / "out"
    result = game_pack.pack(fill_kit(tmp_path, scale=scale), out)
    nine = result["ui"]["nine_slice"]
    frame, fill = nine["hp_bar_frame"], nine["hp_bar_fill"]
    # the frame is packed as before; the fill's content rectangle is the frame's, in the frame's stored pixels
    assert set(frame) == {"frame", "size", "stretch", "insets", "padding"}
    pad = frame["padding"]
    rect = {"x": pad["left"], "y": pad["top"], "w": frame["size"]["w"] - pad["left"] - pad["right"],
            "h": frame["size"]["h"] - pad["top"] - pad["bottom"]}
    assert set(fill) == {"frame", "size", "stretch", "insets", "padding", "inside"}
    assert fill["frame"] == "ui/hp_bar_fill" and fill["stretch"] == "x"
    assert fill["inside"] == {"element": "hp_bar_frame", "rect": rect}
    assert fill["size"] == {"w": rect["w"], "h": rect["h"]}
    assert fill["padding"] == {"left": 0, "right": 0, "top": 0, "bottom": 0}
    # its ends are the fill's own, resized by the one factor that makes it as tall as the content area
    factor = rect["h"] / 8
    resized = resize_cell(Image.open(tmp_path / "ui" / "hp_bar_fill.png").convert("RGBA"),
                          (round(40 * factor), rect["h"]))
    measured, _ = measure_insets(resized, "x")
    insets = {side: max(math.ceil(value * factor), measured[side])
              for side, value in {"left": 5, "right": 5, "top": 0, "bottom": 0}.items()}
    assert fill["insets"] == insets
    image, record = crop(out, result, "ui/hp_bar_fill")
    assert image.size == (rect["w"], rect["h"]) and record["pivot"] == {"x": 0.5, "y": 0.5}
    left, right = insets["left"], insets["right"]
    stored, source = np.asarray(image), np.asarray(resized)
    assert np.array_equal(stored[:, :left], source[:, :left])
    assert np.array_equal(stored[:, rect["w"] - right:], source[:, resized.width - right:])
    # the middle is the fill's own middle columns, repeated or dropped, never blended
    middle = {tuple(map(tuple, source[:, x])) for x in range(left, resized.width - right)}
    assert all(tuple(map(tuple, stored[:, x])) in middle for x in range(left, rect["w"] - right))


@pytest.mark.parametrize("change,message", [
    ({"inside": "mana_bar_frame"}, "hp_bar_fill: inside names mana_bar_frame, which is not a frame of this report"),
    ({"inside": "hp_bar_fill"}, "hp_bar_fill: inside names hp_bar_fill, which is not a frame of this report"),
    ({"stretch": "xy"}, "hp_bar_fill: a fill stretches in width only"),
    # 4 px tall: scaled 1.5x to the 6 px content area, its 11 px ends are 17 px each, 34 of the 32 px it has
    ({"cap": 9, "size": (40, 4)}, "hp_bar_fill: scaled to the height of hp_bar_frame's content area, its fixed ends"),
])
def test_bad_fills_are_refused(tmp_path, change, message):
    request = fill_kit(tmp_path, **change)
    with pytest.raises(ValueError, match=message):
        game_pack.pack(request, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_a_fill_must_name_its_frame(tmp_path):
    request = fill_kit(tmp_path)
    report = json.loads((tmp_path / "ui" / "ui-kit.json").read_text())
    report["elements"]["hp_bar_fill"]["inside"] = {"rect": {"x": 10, "y": 3, "w": 32, "h": 6}}
    write_json(tmp_path / "ui" / "ui-kit.json", report)
    with pytest.raises(ValueError, match="hp_bar_fill: inside must name the element"):
        game_pack.pack(request, tmp_path / "out")


def test_version_1_refuses_the_version_2_sections(tmp_path):
    tile(tmp_path / "ground.png")
    for section, value in (("images", []), ("tiles", []), ("ui", {})):
        request = write_json(tmp_path / "game.json", {
            "version": 1, "assets": [{"id": "g", "source": "ground.png", "display": {"height": 4}}], section: value})
        with pytest.raises(ValueError, match="version 2"):
            game_pack.pack(request, tmp_path / "out")


def test_output_directory_must_not_hold_a_ui_kit_element(tmp_path):
    request = scene_request(tmp_path)
    with pytest.raises(ValueError, match="source"):
        game_pack.pack(request, tmp_path / "ui")


# A version 1 request, packed by the packer before version 2 existed (sprite-gen 2.44.0): the manifest's
# exact bytes and each page's decoded pixels. Encoded PNG/WebP bytes depend on the zlib/libwebp build, so
# the pages are compared as pixels; everything the packer itself decides is frozen.
V1_MANIFEST_SHA256 = "f0c5d43dcb92d7717de82c7e284a1070dcda3493d80ad8e1e720d053141113de"
V1_PAGE_PIXELS_SHA256 = ["36a1d358264c2b7c155fe07dac88b9b7d0f6c3d403eea86739cb59bfd6e811fa"]


def v1_request(tmp_path):
    strip = Image.new("RGBA", (36, 20))
    for index, color in enumerate(((255, 0, 0, 255), (0, 128, 0, 200), (0, 0, 255, 90))):
        strip.paste(Image.new("RGBA", (12, 20), color), (index * 12, 0))
    strip.save(tmp_path / "walk.strip.png")
    write_json(tmp_path / "walk.strip.json", {"frames": 3, "w": 12, "h": 20, "delay_ms": 80, "body_h": 16,
                                              "anchor": [4, 18], "events": [{"frame": 1, "name": "impact"}]})
    prop(tmp_path / "pillar.png")
    return write_json(tmp_path / "game.json", {"version": 1, "assets": [
        {"id": "walk", "source": "walk.strip.json", "display": {"body_height": 8}},
        {"id": "pillar", "source": "pillar.png", "display": {"height": 30}},
        {"id": "pixel", "source": "pillar.png", "display": {"height": 12}, "resample": "nearest"},
    ]})


def test_version_1_request_keeps_its_manifest_bytes_and_page_pixels(tmp_path):
    out = tmp_path / "out"
    game_pack.pack(v1_request(tmp_path), out)
    manifest = (out / "game-pack.json").read_bytes()
    assert hashlib.sha256(manifest).hexdigest() == V1_MANIFEST_SHA256
    result = json.loads(manifest)
    assert result["version"] == 1 and set(result) == {"kind", "version", "textures", "animations"}
    pages = []
    for page in result["textures"]:
        with Image.open(out / page["image"]) as image:
            pages.append(hashlib.sha256(image.convert("RGBA").tobytes()).hexdigest())
    assert pages == V1_PAGE_PIXELS_SHA256
    assert sorted(p.name for p in out.iterdir()) == ["atlas-000.png", "atlas-000.webp", "game-pack.json"]
