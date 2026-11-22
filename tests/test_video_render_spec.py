"""Stage 5 tests: ScenePlan -> render specification, assets, generic rules."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from agents.scene_planner.scene_models import Mascot, Scene, ScenePlan
from agents.video_generation.assets import AssetKind, AssetResolver
from agents.video_generation.render_spec import (
    ANIMATION_TYPES,
    ELEMENT_TYPES,
    RenderSpecError,
)
from agents.video_generation.translator import (
    camera_animations,
    collect_scene_asset_ids,
    translate_scene_plan,
)

CANONICAL_MASCOT = "assets/channels/finance_mascot/mascot.png"
REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_PLAN_PATH = REPO_ROOT / "data" / "e2e-20261005-stage4-live-01" / "scene_plan.json"


def make_scene(
    scene_id: str,
    order: int = 0,
    duration: int = 5,
    *,
    camera: str = "medium shot",
    required_assets: list[str] | None = None,
    mascot_present: bool = True,
    narration: str = "Narration words for the scene.",
) -> Scene:
    return Scene(
        scene_id=scene_id,
        order=order,
        duration_seconds=duration,
        narration=narration,
        visual_description="generic visual description",
        camera=camera,
        composition="centered",
        environment="neutral",
        generation_prompt="generic prompt",
        required_assets=list(required_assets or [CANONICAL_MASCOT]),
        transition_hint="cut",
        mascot=Mascot(
            present=mascot_present,
            action="generic action",
            asset=CANONICAL_MASCOT,
        ),
    )


def make_plan(scenes: list[Scene], **overrides) -> ScenePlan:
    """Build a ScenePlan directly so each scene's own `order` field is kept."""
    payload = {
        "topic_id": "t_generic",
        "title": "Generic title",
        "scenes": scenes,
        "run_id": "run_test",
        "channel_id": "finance_mascot",
        "mascot_asset_path": CANONICAL_MASCOT,
    }
    payload.update(overrides)
    return ScenePlan(**payload)


# 1 - valid ScenePlan -> render specification ---------------------------------


def test_valid_scene_plan_translates_to_render_spec():
    plan = make_plan(
        [
            make_scene("sc01", order=0, duration=6),
            make_scene("sc02", order=1, duration=4),
        ]
    )
    spec = translate_scene_plan(plan, width=1080, height=1920, fps=30)
    spec.validate()

    assert spec.scene_ids == ["sc01", "sc02"]
    assert spec.width == 1080 and spec.height == 1920 and spec.fps == 30
    first = spec.get_scene("sc01")
    types = {element.type for element in first.elements}
    assert "image" in types
    assert "mascot" in types
    assert spec.metadata["translator_version"]


def test_translated_spec_uses_only_generic_element_and_animation_types():
    plan = make_plan(
        [
            make_scene("sc01", order=0, duration=5),
            make_scene("sc02", order=1, duration=5, camera="push in"),
        ]
    )
    spec = translate_scene_plan(plan, width=320, height=568, fps=10)
    for scene in spec.scenes:
        for element in scene.elements:
            assert element.type in ELEMENT_TYPES
            for animation in element.animations:
                assert animation.type in ANIMATION_TYPES


# 2 - deterministic render-spec generation ------------------------------------


def test_render_spec_generation_is_deterministic():
    plan = make_plan(
        [
            make_scene("sc01", order=0, duration=6),
            make_scene("sc02", order=1, duration=4, camera="push in"),
        ]
    )
    first = translate_scene_plan(plan, width=1080, height=1920, fps=30).to_json()
    second = translate_scene_plan(plan, width=1080, height=1920, fps=30).to_json()
    assert first == second
    assert json.loads(first) == json.loads(second)


# 3 - scene ordering ----------------------------------------------------------


def test_scene_ordering_follows_order_field_not_input_position():
    scenes = [
        make_scene("sc03", order=2, duration=5),
        make_scene("sc01", order=0, duration=5),
        make_scene("sc02", order=1, duration=5),
    ]
    spec = translate_scene_plan(make_plan(scenes), width=160, height=288, fps=10)
    assert spec.scene_ids == ["sc01", "sc02", "sc03"]
    assert [s.order for s in spec.ordered_scenes] == [0, 1, 2]


# 4 - asset id / path resolution ---------------------------------------------


def test_scene_asset_id_resolves_into_run_assets_directory(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "assets").mkdir(parents=True)
    (run_dir / "assets" / "sc01_main.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (run_dir / "assets" / "sc01_chart.jpg").write_bytes(b"\xff\xd8\xff")

    resolver = AssetResolver(run_dir, project_root=tmp_path)

    by_extension = resolver.resolve("sc01_main")
    assert by_extension.kind is AssetKind.SCENE
    assert by_extension.status == "resolved"
    assert by_extension.path == run_dir / "assets" / "sc01_main.png"

    probed = resolver.resolve("sc01_chart")
    assert probed.status == "resolved"
    assert probed.path.suffix == ".jpg"

    missing = resolver.resolve("sc09_missing")
    assert missing.status == "missing"
    assert missing.path == run_dir / "assets" / "sc09_missing.png"


def test_path_style_asset_ref_resolves_against_project_root(tmp_path):
    (tmp_path / "assets" / "gen").mkdir(parents=True)
    (tmp_path / "assets" / "gen" / "frame.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    resolver = AssetResolver(run_dir, project_root=tmp_path)
    resolved = resolver.resolve("assets/gen/frame.png")
    assert resolved.kind is AssetKind.PATH
    assert resolved.status == "resolved"
    assert resolved.path == tmp_path / "assets" / "gen" / "frame.png"


def test_translate_scene_plan_exposes_bare_scene_asset_ids():
    plan = make_plan(
        [
            make_scene(
                "sc05",
                order=0,
                duration=5,
                required_assets=[CANONICAL_MASCOT, "sc05_background", "sc05_chart"],
            )
        ]
    )
    spec = translate_scene_plan(plan, width=160, height=288, fps=10)
    assert collect_scene_asset_ids(spec) == ["sc05_background", "sc05_chart"]


# 5 - canonical mascot path handling -----------------------------------------


def test_mascot_element_keeps_canonical_channel_asset_path():
    plan = make_plan([make_scene("sc01", order=0, duration=5)])
    spec = translate_scene_plan(plan, width=160, height=288, fps=10)
    mascot = next(e for e in spec.get_scene("sc01").elements if e.type == "mascot")
    assert mascot.asset == CANONICAL_MASCOT
    assert mascot.metadata["speaks"] is False
    assert mascot.metadata["action"] == "generic action"


def test_resolver_classifies_canonical_mascot_as_read_only_channel_asset(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    resolver = AssetResolver(run_dir, project_root=tmp_path)

    resolved = resolver.resolve(CANONICAL_MASCOT, scene_id="sc01")
    assert resolved.kind is AssetKind.CHANNEL
    assert resolved.path == tmp_path / CANONICAL_MASCOT
    # resolution is read-only: nothing was created
    assert not resolved.path.exists()
    assert resolved.status == "missing"
    assert not (run_dir / "assets").exists()

    # relative to the real repository root the reference still points at the
    # canonical location, whether or not the file exists yet
    repo_resolver = AssetResolver(run_dir)
    repo_resolved = repo_resolver.resolve(CANONICAL_MASCOT)
    assert repo_resolved.kind is AssetKind.CHANNEL
    assert repo_resolved.path.as_posix().endswith(CANONICAL_MASCOT)


# 6 - missing mascot does not corrupt the render-spec contract ----------------


def test_missing_mascot_does_not_corrupt_render_spec(tmp_path):
    plan = make_plan([make_scene("sc01", order=0, duration=5)])
    spec = translate_scene_plan(plan, width=160, height=288, fps=10)
    spec.validate()  # must not raise even though the mascot file does not exist

    mascot = next(e for e in spec.get_scene("sc01").elements if e.type == "mascot")
    assert mascot.asset == CANONICAL_MASCOT
    assert mascot.type == "mascot"
    assert spec.to_json()  # serializes cleanly


def test_mascot_absent_scene_emits_no_mascot_element():
    plan = make_plan([make_scene("sc01", order=0, duration=5, mascot_present=False)])
    spec = translate_scene_plan(plan, width=160, height=288, fps=10)
    assert all(e.type != "mascot" for e in spec.get_scene("sc01").elements)


# camera vocabulary -> generic motion ----------------------------------------


def test_camera_vocabulary_maps_to_generic_motion_not_scene_logic():
    push = camera_animations("push in", 8.0)
    assert [a.type for a in push] == ["zoom"]
    assert push[0].to_value > push[0].from_value

    tracking = camera_animations("medium tracking shot", 8.0)
    assert [a.type for a in tracking] == ["slide"]

    unknown = camera_animations("some brand new camera word", 8.0)
    assert [a.type for a in unknown] == ["zoom"]  # generic default
    assert unknown[0].duration == 8.0


def test_text_elements_are_data_driven():
    plan_dict = make_plan(
        [make_scene("sc01", order=0, duration=5)]
    ).to_dict()
    plan_dict["scenes"][0]["on_screen_text"] = "A GENERIC HEADLINE"
    spec = translate_scene_plan(plan_dict, width=160, height=288, fps=10)
    text_elements = [e for e in spec.get_scene("sc01").elements if e.type == "text"]
    assert len(text_elements) == 1
    assert text_elements[0].text == "A GENERIC HEADLINE"

    override = translate_scene_plan(
        make_plan([make_scene("sc02", order=0, duration=5)]),
        width=160,
        height=288,
        fps=10,
        text_overrides={"sc02": "OVERRIDE"},
    )
    assert any(
        e.type == "text" and e.text == "OVERRIDE"
        for e in override.get_scene("sc02").elements
    )


def test_scene_without_text_emits_no_text_element():
    plan = make_plan([make_scene("sc01", order=0, duration=5)])
    spec = translate_scene_plan(plan, width=160, height=288, fps=10)
    assert all(e.type != "text" for e in spec.get_scene("sc01").elements)


# 14 - normalized coordinates -------------------------------------------------


def test_translated_coordinates_are_normalized():
    plan = make_plan(
        [
            make_scene("sc01", order=0, duration=5, camera="pan right"),
            make_scene("sc02", order=1, duration=5, camera="push in"),
        ]
    )
    spec = translate_scene_plan(plan, width=1080, height=1920, fps=30)
    for scene in spec.scenes:
        assert 0.0 <= scene.background[0] <= 255
        for element in scene.elements:
            assert 0.0 <= element.transform.x <= 1.0
            assert 0.0 <= element.transform.y <= 1.0
            assert element.transform.scale > 0.0
            assert 0.0 <= element.transform.opacity <= 1.0
            assert element.transform.rotation == 0.0
            for animation in element.animations:
                if animation.type == "zoom":
                    assert float(animation.from_value) > 0
                    assert float(animation.to_value) > 0
                if animation.type == "fade":
                    assert 0.0 <= float(animation.from_value) <= 1.0
                    assert 0.0 <= float(animation.to_value) <= 1.0


# 15 - scene duration handling ------------------------------------------------


def test_scene_duration_is_carried_from_plan():
    plan = make_plan(
        [
            make_scene("sc01", order=0, duration=8),
            make_scene("sc02", order=1, duration=7),
            make_scene("sc03", order=2, duration=12),
        ]
    )
    spec = translate_scene_plan(plan, width=160, height=288, fps=10)
    assert [s.duration_seconds for s in spec.ordered_scenes] == [8.0, 7.0, 12.0]
    assert spec.total_duration_seconds == 27.0

    plan_dict = plan.to_dict()
    plan_dict["scenes"][1]["duration_seconds"] = 0
    with pytest.raises(Exception):
        translate_scene_plan(plan_dict, width=160, height=288, fps=10)


def test_invalid_render_spec_is_rejected():
    plan = make_plan([make_scene("sc01", order=0, duration=5)])
    spec = translate_scene_plan(plan, width=160, height=288, fps=10)
    broken = spec.to_dict()
    broken["scenes"][0]["elements"][0]["type"] = "talking_head"
    with pytest.raises(RenderSpecError):
        type(spec).from_dict(broken).validate()


# real Stage 4 artifact ------------------------------------------------------


@pytest.mark.skipif(not REAL_PLAN_PATH.exists(), reason="real Stage 4 artifact not present")
def test_real_scene_plan_translates_to_seven_scenes():
    from agents.video_generation.stage import load_scene_plan

    plan = load_scene_plan(REAL_PLAN_PATH)
    spec = translate_scene_plan(plan, width=1080, height=1920, fps=30)
    assert spec.scene_ids == ["sc01", "sc02", "sc03", "sc04", "sc05", "sc06", "sc07"]
    assert spec.total_duration_seconds == 68.0
    for scene in spec.ordered_scenes:
        mascot = [e for e in scene.elements if e.type == "mascot"]
        assert len(mascot) == 1
        assert mascot[0].asset == CANONICAL_MASCOT
        assert mascot[0].metadata["action"]
        assert scene.metadata["narration"]
        assert scene.metadata["visual_description"]
        assert scene.metadata["transition_hint"]


# 19 + 23 - no scene-specific or pipeline-concept logic ----------------------

STAGE5_SOURCES = sorted(
    str(path) for path in Path(__file__).resolve().parent.parent.joinpath(
        "agents", "video_generation"
    ).glob("*.py")
)

FORBIDDEN_LOGIC = [
    (r"\bsc0[1-9]\b", "hardcoded scene id"),
    (r"if\s+scene_id\s*==", "scene_id branch"),
    (r"scene_id\s*in\s*\{", "scene_id lookup table"),
    (r"topic_id\s*==", "topic branch"),
    (r"visual_description\s*==", "visual description branch"),
    (r"if\s+topic\s*==", "topic branch"),
    (r"\b(paycheck|wallet|coins?|equity|startup|inflation)\b", "scene content word"),
]

FORBIDDEN_CONCEPTS = [
    (r"evidence", "evidence concept"),
    (r"corrobor", "corroboration concept"),
    (r"fact[_ ]?check", "fact-checking concept"),
    (r"fact_checker", "fact checker concept"),
    (r"citation", "citation concept"),
    (r"\bcitations?\b", "citation concept"),
    (r"verdict", "verdict concept"),
    (r"pass/fail", "pass/fail concept"),
    (r"\bqa\b", "QA concept"),
    (r"qa_", "QA concept"),
    (r"_qa\b", "QA concept"),
    (r"approval", "approval concept"),
    (r"quality[_ ]gate", "quality gate concept"),
    (r"source_section", "source section concept"),
    (r"\bsources?\b", "sources concept"),
    (r"retriev", "retrieval concept"),
    (r"web[_ ]?search", "web search concept"),
]


def _read_stage5_sources() -> dict[str, str]:
    return {Path(p).name: Path(p).read_text(encoding="utf-8") for p in STAGE5_SOURCES}


def test_no_scene_specific_hardcoded_logic():
    for name, text in _read_stage5_sources().items():
        for pattern, label in FORBIDDEN_LOGIC:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            assert match is None, f"{name}: {label} matched {match.group(0)!r}"


def test_no_fact_check_evidence_or_qa_concepts_in_stage5():
    for name, text in _read_stage5_sources().items():
        for pattern, label in FORBIDDEN_CONCEPTS:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            assert match is None, f"{name}: {label} matched {match.group(0)!r}"
