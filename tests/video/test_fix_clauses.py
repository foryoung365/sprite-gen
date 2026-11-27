# SPDX-License-Identifier: Apache-2.0
"""Offline unit tests for failure-gate fix clauses (`fix_clauses`, `video-prompt`, `batch`)."""

from __future__ import annotations

import pytest

from sprite_gen.video import batch as batch_mod
from sprite_gen.video import clip_prompt
from sprite_gen.video import fix_clauses as fc

EXPECTED_CLAUSE_IDS = {
    "never-turn",
    "camera-locked",
    "constant-size",
    "feet-planted",
    "feet-ground-line",
    "inside-frame",
    "flat-background",
    "no-extras",
    "identity-lock",
    "face-visible",
    "keep-colours",
    "same-start-end",
    "starts-from-still",
    "single-action",
    "weapon-shape",
    "no-text",
    "no-flares",
    "visible-motion",
    "in-place",
    "fluid-motion",
    "steady-cycle",
    "no-blur",
}


def test_clause_library_has_all_22_verified_clauses() -> None:
    assert set(fc.CLAUSES) == EXPECTED_CLAUSE_IDS
    assert len(fc.CLAUSES) == 22


def test_resolve_fix_clauses_substitutes_placeholders_and_deduplicates() -> None:
    res = fc.resolve_fix_clauses(
        [
            "never-turn",
            "flat-background",
            "single-action",
            "never-turn",
            "Keep the spear horizontal during {state} facing {facing} on {key}.",
            "Keep the spear horizontal during {state} facing {facing} on {key}.",
        ],
        state="attack",
        facing="left",
        key="magenta",
    )
    assert res["ids"] == ["never-turn", "flat-background", "single-action"]
    assert len(res["clauses"]) == 4
    assert "facing left" in res["clauses"][0]
    assert "pure magenta" in res["clauses"][1]
    assert "Only one attack" in res["clauses"][2]
    assert "during attack facing left on magenta" in res["clauses"][3]
    assert res["rewritten"] == []


@pytest.mark.parametrize("state", ["walk", "run", "jump"])
def test_gait_and_jump_rewrite_feet_planted_to_feet_ground_line(state: str) -> None:
    res = fc.resolve_fix_clauses(["feet-planted", "camera-locked", "feet-ground-line"], state=state)
    assert res["ids"] == ["feet-ground-line", "camera-locked"]
    assert len(res["clauses"]) == 2
    assert len(res["rewritten"]) == 1
    assert res["rewritten"][0]["from"] == "feet-planted"
    assert res["rewritten"][0]["to"] == "feet-ground-line"
    assert state in res["rewritten"][0]["reason"]
    assert "Both feet stay planted" not in res["text"]
    assert "Every step lands on the same ground line" in res["text"]


@pytest.mark.parametrize("state", ["idle", "attack", "hurt"])
def test_standing_states_keep_feet_planted(state: str) -> None:
    res = fc.resolve_fix_clauses(["feet-planted"], state=state)
    assert res["ids"] == ["feet-planted"]
    assert res["rewritten"] == []
    assert "Both feet stay planted" in res["text"]


def test_suggest_fixes_for_gates_maps_and_applies_gait_safety() -> None:
    assert fc.suggest_fixes_for_gates(["framing_guard"]) == ["camera-locked", "inside-frame"]
    assert fc.suggest_fixes_for_gates(["edge_contact"]) == ["inside-frame"]
    assert fc.suggest_fixes_for_gates(["keying"]) == ["flat-background", "inside-frame"]
    assert fc.suggest_fixes_for_gates(["zoom", "turn", "colour"]) == ["camera-locked", "never-turn", "keep-colours"]
    assert fc.suggest_fixes_for_gates(["feet"], state="attack") == ["feet-planted", "in-place"]
    assert fc.suggest_fixes_for_gates(["feet"], state="walk") == ["feet-ground-line", "in-place"]
    assert fc.suggest_fixes_for_gates(["feet"], state="run") == ["feet-ground-line", "in-place"]
    assert fc.suggest_fixes_for_gates(["feet"], state="jump") == ["feet-ground-line", "in-place"]


def test_video_prompt_fix_appends_without_altering_default_prompt() -> None:
    base_record = clip_prompt.plan_prompt(direction="side", state="walk", facing="left", key="magenta")
    assert "fix" not in base_record
    assert base_record["prompt"] == batch_mod.build_prompt("side", "walk", None, facing="left")

    fixed_record = clip_prompt.plan_prompt(
        direction="side",
        state="walk",
        facing="left",
        key="magenta",
        fix=["camera-locked", "feet-planted"],
    )
    assert fixed_record["prompt"].startswith(base_record["prompt"])
    assert "Locked camera and constant character size" in fixed_record["prompt"]
    assert "Every step lands on the same ground line" in fixed_record["prompt"]
    assert "Both feet stay planted" not in fixed_record["prompt"]
    assert fixed_record["fix"]["ids"] == ["camera-locked", "feet-ground-line"]
    assert fixed_record["fix"]["rewritten"][0]["to"] == "feet-ground-line"
