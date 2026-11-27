# SPDX-License-Identifier: Apache-2.0
"""Verified failure-gate fix clause library for video sprite retakes.

Maps numeric quality-gate refusals (framing zoom/crop, edge contact, keying spill, colour drift,
facing turns, stride drift) to concise, battle-tested prompt clauses that can be appended to a
clip prompt via ``--fix`` without rewriting the whole prompt.

Enforces the gait safety rule: ``walk``, ``run`` and ``jump`` must never receive ``feet-planted``
(which freezes stride or takeoff); any request or gate recommendation for ``feet-planted`` on those
states is automatically rewritten to ``feet-ground-line`` and recorded in the result.

Adapted from 0x0funky/agent-sprite-forge (MIT, commit 25203d03900b46b444bdd72a75768c33d1895ad6).
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

# States where pinning both feet planted freezes the stride or takeoff.
NON_PLANTED_STATES = frozenset({"walk", "run", "jump"})

# 22 atomic prompt fix clauses verified across image-to-video sprite takes.
CLAUSES: dict[str, str] = {
    "never-turn": (
        "The character keeps facing {facing} the whole time: never turns around, "
        "never shows {pos} back and never faces the camera."
    ),
    "camera-locked": (
        "Locked camera and constant character size: no zoom, no push-in, no pull-out, no pan; "
        "the character is exactly the same size in every frame, before and after the action."
    ),
    "constant-size": (
        "The character keeps exactly the same size and proportions as the still in every frame; "
        "arms, legs and the weapon never grow longer."
    ),
    "feet-planted": (
        "Both feet stay planted on the same ground line and the body stays centered in the frame; "
        "no stepping away, no sliding, no drifting."
    ),
    "feet-ground-line": (
        "Every step lands on the same ground line and the body stays centered in the frame; "
        "the character never drifts forward, backward, up or down."
    ),
    "inside-frame": (
        "The whole body, the weapon and every moving part stay fully inside the frame with a clear "
        "margin on every side; nothing is cut off by the frame edge."
    ),
    "flat-background": (
        "The background stays solid flat pure {key} for the whole shot: no floor, no ground, "
        "no shadow, no gradient, no glow, no haze, no light."
    ),
    "no-extras": (
        "Nothing else appears in the frame: no dust, no dirt, no smoke, no puffs, no pebbles, "
        "no particles, no orbs, no sparks, no flashes, no projectiles, no arrows, no motion trails, "
        "no effects of any kind."
    ),
    "identity-lock": (
        "CHARACTER LOCK for the whole clip: the face, hair, colours, costume, proportions and "
        "every design detail stay exactly as in the still; nothing new grows or appears on the "
        "character (no horns, no crests, no extra straps or weapons)."
    ),
    "face-visible": (
        "The whole face is visible in every frame: no hand, weapon, sleeve, hair or effect ever "
        "passes in front of the face."
    ),
    "keep-colours": (
        "The colours stay exactly as in the still in every frame: no tint, no colour shift, "
        "no darkening and no glow on the character."
    ),
    "same-start-end": (
        "The clip starts AND ends in exactly the same pose as the still and holds that pose still "
        "at the end."
    ),
    "starts-from-still": (
        "The first frame is exactly the supplied image: the same pose, position and size; "
        "the motion starts from there."
    ),
    "single-action": (
        "Only one {state}: no repeats, no second strike, no spinning, no jumping, no leaving the frame."
    ),
    "weapon-shape": (
        "The weapon stays one single weapon of the same length and shape the whole time: "
        "never bending, never doubling, no second blade, no ghost blades."
    ),
    "no-text": "No text, letters, glyphs, writing or logos anywhere.",
    "no-flares": "No cross flares, no starbursts, no four-point sparkles, no lens flares.",
    "visible-motion": (
        "The motion is clearly visible at game size: a full, readable {state} with real movement "
        "of the body, not a still image."
    ),
    "in-place": (
        "The character stays in the same spot and never travels across the frame; "
        "the game moves the sprite."
    ),
    "fluid-motion": (
        "One continuous, fluid motion at natural speed: no pauses, no frozen poses, no hovering, "
        "no slow motion."
    ),
    "steady-cycle": (
        "A steady, evenly timed cycle: every step looks the same, the legs clearly alternate and "
        "the pose at the end of each cycle matches its start."
    ),
    "no-blur": "No motion-blur smears, no afterimages, no colour fringing on the weapon.",
}

# Mapping from quality-gate names to recommended fix clause IDs.
GATE_FIXES: dict[str, tuple[str, ...]] = {
    "framing_guard": ("camera-locked", "inside-frame"),
    "zoom": ("camera-locked",),
    "edge_contact": ("inside-frame",),
    "edge": ("inside-frame",),
    "keying": ("flat-background", "inside-frame"),
    "residual_background": ("flat-background",),
    "background": ("flat-background",),
    "turn": ("never-turn",),
    "colour": ("keep-colours",),
    "color": ("keep-colours",),
    "area": ("constant-size", "camera-locked"),
    "feet": ("feet-planted", "in-place"),
    "identity": ("identity-lock", "face-visible"),
    "extra": ("no-extras",),
    "motion": ("visible-motion", "single-action"),
    "end-pose": ("same-start-end",),
    "timing": ("fluid-motion",),
    "loop": ("steady-cycle", "in-place"),
    "registration": ("starts-from-still", "camera-locked"),
}


def _render_placeholders(
    template: str,
    *,
    state: str,
    facing: str,
    key: str,
    pos: str,
) -> str:
    """Safely substitute known placeholders without failing on unrelated braces in free text."""
    action = state or "action"
    rendered = template
    for token, value in (
        ("{facing}", facing or "right"),
        ("{key}", key or "magenta"),
        ("{state}", action),
        ("{action}", action),
        ("{pos}", pos or "their"),
    ):
        rendered = rendered.replace(token, value)
    return rendered


def _rewrite_for_state(clause_id: str, state_norm: str) -> tuple[str, dict[str, str] | None]:
    if clause_id == "feet-planted" and state_norm in NON_PLANTED_STATES:
        return (
            "feet-ground-line",
            {
                "from": "feet-planted",
                "to": "feet-ground-line",
                "reason": (
                    f"state {state_norm!r} moves or lifts its feet; "
                    "feet-planted freezes stride/takeoff, rewritten to feet-ground-line"
                ),
            },
        )
    return clause_id, None


def suggest_fixes_for_gates(gates: Iterable[str], *, state: str = "") -> list[str]:
    """Return deduplicated, state-safe clause IDs recommended for the given failed gate names."""
    state_norm = (state or "").strip().lower()
    out: list[str] = []
    seen: set[str] = set()
    for gate in gates:
        gate_key = (gate or "").strip().lower()
        for raw_id in GATE_FIXES.get(gate_key, ()):
            clause_id, _ = _rewrite_for_state(raw_id, state_norm)
            if clause_id not in seen:
                seen.add(clause_id)
                out.append(clause_id)
    return out


def resolve_fix_clauses(
    fixes: Sequence[str] | None,
    *,
    state: str = "",
    facing: str = "right",
    key: str = "magenta",
    pos: str = "their",
) -> dict[str, Any]:
    """Resolve clause IDs and free-text fix entries into deduplicated rendered prompt sentences.

    Applies placeholder substitution (``{facing}``, ``{key}``, ``{state}``, ``{action}``, ``{pos}``)
    and enforces the gait safety rule rewriting ``feet-planted`` to ``feet-ground-line`` for
    ``walk``, ``run`` and ``jump``.
    """
    state_norm = (state or "").strip().lower()
    facing_norm = (facing or "right").strip()
    key_norm = (key or "magenta").strip().lower()

    ids: list[str] = []
    clauses: list[str] = []
    rewritten: list[dict[str, str]] = []
    entries: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_texts: set[str] = set()

    for raw in fixes or ():
        item = " ".join(str(raw).split())
        if not item:
            continue
        lower = item.lower()
        if lower in CLAUSES:
            clause_id, rewrite = _rewrite_for_state(lower, state_norm)
            if rewrite is not None:
                rewritten.append(rewrite)
            if clause_id in seen_ids:
                continue
            rendered = _render_placeholders(
                CLAUSES[clause_id], state=state_norm, facing=facing_norm, key=key_norm, pos=pos
            )
            norm_text = rendered.lower().rstrip(".!? ")
            if norm_text in seen_texts:
                continue
            seen_ids.add(clause_id)
            seen_texts.add(norm_text)
            ids.append(clause_id)
            clauses.append(rendered)
            entries.append({"input": item, "id": clause_id, "text": rendered, "custom": False})
        else:
            rendered = _render_placeholders(item, state=state_norm, facing=facing_norm, key=key_norm, pos=pos)
            norm_text = rendered.lower().rstrip(".!? ")
            if norm_text in seen_texts:
                continue
            seen_texts.add(norm_text)
            clauses.append(rendered)
            entries.append({"input": item, "id": None, "text": rendered, "custom": True})

    return {
        "ids": ids,
        "clauses": clauses,
        "text": " ".join(clauses),
        "rewritten": rewritten,
        "entries": entries,
    }
