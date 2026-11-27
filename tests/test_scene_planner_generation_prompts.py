"""Stage 4 tests: compact SDXL-compatible generation_prompt rules.

Covers the two visual concepts every scene must produce (planning-level
`visual_description` + compact `generation_prompt`), the channel image-style
configuration, the deterministic validator (fail clearly, never truncate),
and the pre-existing Stage 4 behaviours (narration reconstruction, mascot
identity, structural validation).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from agents.scene_planner.provider import MockScenePlannerProvider
from agents.scene_planner.scene_models import (
    CHANNEL_IMAGE_STYLE,
    MASCOT_ASSET_PATH,
    MASCOT_SPEAKS,
    Mascot,
    Scene,
    ScenePlan,
    ScenePlanningError,
)
from agents.scene_planner.stage import build_scene_plan_context, run_scene_planning
from agents.scene_planner.validator import (
    FORBIDDEN_PROMPT_CONCEPTS,
    GENERATION_PROMPT_HARD_CEILING_TOKENS,
    GENERATION_PROMPT_TARGET_TOKENS,
    NARRATION_WINDOW_WORDS,
    check_generation_prompt,
    contains_narration,
    estimate_clip_tokens,
    repeats_channel_style,
    validate_scene_plan_generation,
)
from agents.script.script_models import Script, ScriptSection
from config.channel import CHANNEL

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_PLAN_PATH = REPO_ROOT / "data" / "e2e-20261005-stage4-live-01" / "scene_plan.json"

STYLE_FRAGMENTS = (
    "hand-drawn ink illustration",
    "dark ink outlines",
    "warm wood and stone",
    "muted brown and beige tones",
    "subtle blue accents",
)

SECTION_SPECS = [
    ("s1", "hook", "The startup offer looked generous on paper."),
    ("s2", "setup", "But the equity portion was worth nothing yet."),
    ("s3", "explanation", "Vesting schedules quietly freeze your income."),
    ("s4", "example", "A four year vest means one full year of waiting."),
    ("s5", "reveal", "The salary was real but the wealth was not."),
    ("s6", "payoff", "Count the full package before you say yes."),
    ("s7", "closing", "Always read the equity fine print carefully."),
]


def build_script() -> Script:
    sections = [
        ScriptSection(section_id, role, narration, estimated_seconds=8)
        for section_id, role, narration in SECTION_SPECS
    ]
    narration = " ".join(text for _, _, text in SECTION_SPECS)
    return Script(
        topic_id="t_startup_pay_cut",
        title="Autopsy: The Startup Salary That Disappeared",
        narration=narration,
        sections=sections,
        pillar="financial_autopsies",
        channel_id="finance_mascot",
    )


@pytest.fixture(scope="module")
def script() -> Script:
    return build_script()


@pytest.fixture(scope="module")
def plan(script) -> ScenePlan:
    return run_scene_planning(
        script, MockScenePlannerProvider(), run_id="genprompt-test"
    )


# 1 - generation_prompt exists for every scene --------------------------------


def test_every_scene_has_a_generation_prompt(plan):
    assert plan.scenes, "plan has no scenes"
    for scene in plan.scenes:
        assert isinstance(scene.generation_prompt, str)
        assert scene.generation_prompt.strip(), (
            f"{scene.scene_id} has an empty generation_prompt"
        )


def test_every_scene_keeps_both_visual_concepts(plan):
    for scene in plan.scenes:
        assert scene.visual_description.strip(), (
            f"{scene.scene_id} lost its planning-level visual_description"
        )
        assert scene.generation_prompt.strip(), (
            f"{scene.scene_id} lost its generation_prompt"
        )
        assert scene.visual_description != scene.generation_prompt


# 2 - generation_prompt is compact -------------------------------------------


def test_generation_prompts_are_compact(plan):
    for scene in plan.scenes:
        estimated = estimate_clip_tokens(scene.generation_prompt)
        assert estimated <= GENERATION_PROMPT_TARGET_TOKENS, (
            f"{scene.scene_id} generation_prompt is ~{estimated} tokens, "
            f"above the {GENERATION_PROMPT_TARGET_TOKENS} target"
        )
        assert estimated >= 25, (
            f"{scene.scene_id} generation_prompt is only ~{estimated} tokens, "
            "below the ~25-token target floor"
        )
        assert estimated <= GENERATION_PROMPT_HARD_CEILING_TOKENS


def test_token_budget_is_ordered_under_clip_limit():
    assert GENERATION_PROMPT_TARGET_TOKENS <= GENERATION_PROMPT_HARD_CEILING_TOKENS
    # 77 includes CLIP's 2 special tokens; the ceiling must leave that room.
    assert GENERATION_PROMPT_HARD_CEILING_TOKENS + 2 < 77


def test_estimate_clip_tokens_is_deterministic_and_conservative():
    assert estimate_clip_tokens("") == 0
    assert estimate_clip_tokens("   ") == 0
    short = estimate_clip_tokens("coins on a desk")
    long = estimate_clip_tokens(
        "scattered golden coins rest upon the worn wooden detective desk"
    )
    assert short > 0
    assert long > short
    # Never under-counts the word count (conservative bias toward failing early).
    words = "one two three four five six seven eight nine ten".split()
    assert estimate_clip_tokens(" ".join(words)) >= len(words)


# 3 - forbidden visual-generation concepts ------------------------------------


FORBIDDEN_SAMPLE_PROMPTS = {
    "photorealistic": (
        "Cinematic shot of a glowing job offer letter on a dark modern desk, "
        "professional lighting, photorealistic"
    ),
    "3d render": (
        "A running track with a glowing finish line banner made of currency "
        "symbols, modern 3D render"
    ),
    "animation": (
        "Minimalist graphic animation of a financial pie chart shrinking and "
        "dividing into tiny slices"
    ),
    "video": (
        "A golden parachute transforming into a paper envelope floating in a "
        "stormy sky, vertical video scene"
    ),
}


def test_generation_prompts_avoid_forbidden_concepts(plan):
    for scene in plan.scenes:
        prompt = scene.generation_prompt
        lowered = prompt.lower()
        for forbidden in ("photorealistic", "3d render", "animation", "video"):
            assert forbidden not in lowered, (
                f"{scene.scene_id} generation_prompt contains {forbidden!r}"
            )
        for concept, pattern in FORBIDDEN_PROMPT_CONCEPTS.items():
            if concept == "mascot":
                continue
            assert not re.search(pattern, prompt, flags=re.IGNORECASE), (
                f"{scene.scene_id} generation_prompt matches forbidden "
                f"concept {concept!r}"
            )


@pytest.mark.parametrize("concept", sorted(FORBIDDEN_SAMPLE_PROMPTS))
def test_validator_rejects_old_style_prompts(concept):
    issues = check_generation_prompt(
        FORBIDDEN_SAMPLE_PROMPTS[concept],
        scene_id="sc01",
        narration="Some narration that must not leak into prompts.",
    )
    assert issues, f"validator accepted the old {concept!r} prompt"
    assert {issue.code for issue in issues} == {"gp04"}
    assert all(issue.severity == "error" for issue in issues)


# 4 - no mascot / fox ---------------------------------------------------------


def test_generation_prompts_do_not_mention_mascot_or_fox(plan):
    for scene in plan.scenes:
        prompt = scene.generation_prompt.lower()
        assert "mascot" not in prompt
        assert not re.search(r"\bfox\b|\bfoxes\b", prompt)


@pytest.mark.parametrize(
    "prompt",
    [
        "Finance_mascot fox inspecting coins on a wooden detective desk",
        "A fox mascot beside stacked coins, simple composition",
    ],
)
def test_validator_rejects_mascot_mentions(prompt):
    codes = [
        issue.code
        for issue in check_generation_prompt(prompt, scene_id="sc02")
        if issue.severity == "error"
    ]
    assert "gp05" in codes


# 5 - no narration in generation_prompt --------------------------------------


def test_generation_prompts_do_not_contain_narration(plan):
    for scene in plan.scenes:
        assert not contains_narration(scene.generation_prompt, scene.narration), (
            f"{scene.scene_id} generation_prompt embeds its narration"
        )
        assert scene.narration.lower() not in scene.generation_prompt.lower()


def test_validator_rejects_narration_leakage():
    narration = "The startup offer looked generous on paper but equity was frozen."
    leaked = (
        "The startup offer looked generous on paper but equity was frozen, "
        "coins on a desk."
    )
    codes = [
        issue.code
        for issue in check_generation_prompt(leaked, scene_id="sc01", narration=narration)
    ]
    assert "gp06" in codes
    # A run as long as the detection window always fires.
    window = " ".join(narration.split()[:NARRATION_WINDOW_WORDS])
    assert contains_narration(f"Scene with {window} inside.", narration)
    # Short run of common words alone is not narration.
    assert not contains_narration("Coins on a desk", narration)


# 6 - channel image style -----------------------------------------------------


def test_channel_image_style_is_configured():
    assert CHANNEL.image_style
    assert CHANNEL_IMAGE_STYLE == CHANNEL.image_style
    for fragment in STYLE_FRAGMENTS:
        assert fragment in CHANNEL.image_style
    assert CHANNEL.to_dict()["image_style"] == CHANNEL.image_style


def test_channel_image_style_is_carried_on_the_plan(plan):
    assert plan.channel_image_style == CHANNEL_IMAGE_STYLE
    assert ScenePlan(topic_id="t1").channel_image_style == CHANNEL_IMAGE_STYLE
    reloaded = ScenePlan.from_dict(plan.to_dict())
    assert reloaded.channel_image_style == plan.channel_image_style


def test_channel_image_style_is_not_repeated_in_scene_prompts(plan):
    for scene in plan.scenes:
        assert not repeats_channel_style(
            scene.generation_prompt, plan.channel_image_style
        ), f"{scene.scene_id} repeats the channel style block"
        for fragment in STYLE_FRAGMENTS:
            assert fragment not in scene.generation_prompt.lower()


def test_validator_rejects_style_block_repetition():
    prompt = "Coins on a wooden desk, dark ink outlines, muted brown and beige tones."
    codes = [
        issue.code
        for issue in check_generation_prompt(
            prompt, scene_id="sc03", channel_style=CHANNEL_IMAGE_STYLE
        )
    ]
    assert "gp07" in codes


# 7 - mock ScenePlanner generates valid compact prompts -----------------------


def test_mock_scene_planner_generates_valid_compact_prompts(script):
    provider = MockScenePlannerProvider()
    generated = provider.generate(build_scene_plan_context(script))
    validation = validate_scene_plan_generation(generated)
    assert validation.passed, validation.to_dict()
    assert validation.issues == [], validation.to_dict()
    for scene in generated.scenes:
        estimated = estimate_clip_tokens(scene.generation_prompt)
        assert 25 <= estimated <= GENERATION_PROMPT_TARGET_TOKENS


def test_run_scene_planning_accepts_mock_output(script):
    result = run_scene_planning(script, MockScenePlannerProvider(), run_id="mock-ok")
    assert result.validate() is result
    assert validate_scene_plan_generation(result).passed


# 8 - narration reconstruction -----------------------------------------------


def test_narration_reconstruction_is_preserved(script, plan):
    rebuilt = " ".join(scene.narration for scene in plan.scenes)
    assert " ".join(rebuilt.split()) == " ".join(script.narration.split())
    assert [scene.source_section_id for scene in plan.scenes] == [
        spec[0] for spec in SECTION_SPECS
    ]


# 9 - mascot behaviour --------------------------------------------------------


def test_mascot_identity_and_non_speaking_rule(plan, script):
    assert MASCOT_SPEAKS is False
    assert script.mascot_speaks is False
    assert plan.mascot_asset_path == MASCOT_ASSET_PATH
    assert Mascot().asset == MASCOT_ASSET_PATH
    for scene in plan.scenes:
        assert scene.mascot.present is True
        assert scene.mascot.asset == MASCOT_ASSET_PATH
        assert scene.mascot.action, f"{scene.scene_id} mascot has no action"
        assert MASCOT_ASSET_PATH in scene.required_assets


# 10 - existing Stage 4 validation remains intact -----------------------------


def test_structural_validation_still_behaves_as_before():
    valid = ScenePlan(
        topic_id="t1",
        scenes=[Scene(scene_id="sc01", narration="Words.", duration_seconds=5)],
    )
    assert valid.validate() is valid

    with pytest.raises(ScenePlanningError):
        ScenePlan(topic_id="").validate()
    with pytest.raises(ScenePlanningError):
        ScenePlan(topic_id="t1").validate()
    with pytest.raises(ScenePlanningError):
        ScenePlan(
            topic_id="t1",
            scenes=[Scene(scene_id="sc01", narration="")],
        ).validate()
    with pytest.raises(ScenePlanningError):
        ScenePlan(
            topic_id="t1",
            scenes=[Scene(scene_id="sc01", narration="words", duration_seconds=0)],
        ).validate()


def test_structural_layer_tolerates_legacy_empty_generation_prompt():
    """Stage 5 loads historical artifacts through ScenePlan.validate(); the
    structural layer must not newly reject them. The SDXL prompt gate is the
    separate Stage 4 layer (run_scene_planning)."""
    legacy = ScenePlan(
        topic_id="t1",
        scenes=[
            Scene(scene_id="sc01", narration="Legacy narration.", generation_prompt="")
        ],
    )
    assert legacy.validate() is legacy
    validation = validate_scene_plan_generation(legacy)
    assert not validation.passed
    assert [issue.code for issue in validation.errors] == ["gp01"]


# validation: fail clearly, never truncate ------------------------------------


def _stub_plan(generation_prompt: str) -> ScenePlan:
    return ScenePlan(
        topic_id="t_stub",
        title="Stub",
        scenes=[
            Scene(
                scene_id="sc01",
                order=0,
                duration_seconds=8,
                narration="Stub narration for the scene.",
                visual_description="planning level description",
                generation_prompt=generation_prompt,
                mascot=Mascot(present=True, action="standing"),
            )
        ],
    )


class _StubProvider:
    provider_name = "stub"

    def __init__(self, plan: ScenePlan) -> None:
        self.plan = plan

    def generate(self, context: dict) -> ScenePlan:
        return self.plan


def test_empty_generation_prompt_fails_clearly():
    issues = check_generation_prompt("", scene_id="sc07")
    assert [issue.code for issue in issues] == ["gp01"]
    assert issues[0].severity == "error"


def test_over_ceiling_prompt_fails_without_truncation(script):
    long_prompt = (
        "scattered golden coins rest upon the worn wooden detective desk "
        "beside a folded document and a brass magnifying glass, warm light. "
    ) * 3
    long_prompt = long_prompt.strip()
    assert estimate_clip_tokens(long_prompt) > GENERATION_PROMPT_HARD_CEILING_TOKENS

    bad_plan = _stub_plan(long_prompt)
    with pytest.raises(ScenePlanningError) as excinfo:
        run_scene_planning(script, _StubProvider(bad_plan), run_id="too_long")

    message = str(excinfo.value)
    assert "gp02" in message
    assert "never truncated" in message
    # Nothing was silently truncated: the source prompt is untouched.
    assert bad_plan.scenes[0].generation_prompt == long_prompt


def test_prompt_between_target_and_ceiling_warns_but_passes():
    clause = "coins and documents rest on the wooden desk beside the brass scale"
    prompt = ", ".join([clause] * 3)
    estimated = estimate_clip_tokens(prompt)
    assert GENERATION_PROMPT_TARGET_TOKENS < estimated <= GENERATION_PROMPT_HARD_CEILING_TOKENS

    issues = check_generation_prompt(prompt, scene_id="sc04")
    assert [issue.code for issue in issues] == ["gp03"]
    assert issues[0].severity == "warning"

    plan = _stub_plan(prompt)
    assert validate_scene_plan_generation(plan).passed
    assert validate_scene_plan_generation(plan).errors == []


def test_full_pipeline_reports_every_error_scene(script):
    bad_plan = ScenePlan(
        topic_id="t_bad",
        title="Bad",
        scenes=[
            Scene(
                scene_id="sc01",
                order=0,
                duration_seconds=8,
                narration="Valid narration line here.",
                generation_prompt="",
            ),
            Scene(
                scene_id="sc02",
                order=1,
                duration_seconds=8,
                narration="Another valid narration line.",
                generation_prompt="Photorealistic fox mascot in a 3D render video",
            ),
        ],
    )
    with pytest.raises(ScenePlanningError) as excinfo:
        run_scene_planning(script, _StubProvider(bad_plan), run_id="mixed-bad")
    message = str(excinfo.value)
    assert "gp01" in message
    assert "gp04" in message
    assert "gp05" in message
    assert "sc01" in message and "sc02" in message


# historical migration note ----------------------------------------------------


@pytest.mark.skipif(
    not REAL_PLAN_PATH.exists(), reason="historical Stage 4 artifact not present"
)
def test_historical_live_plan_prompts_no_longer_pass():
    """The pre-SDXL live plan (photorealistic / 3D render prompts) is read-only
    evidence that the old prompt style exists; the new gate rejects it."""
    import json

    data = json.loads(REAL_PLAN_PATH.read_text(encoding="utf-8"))
    historical = ScenePlan.from_dict(data)
    validation = validate_scene_plan_generation(historical)
    assert not validation.passed
    assert any(issue.code == "gp04" for issue in validation.errors)


def test_scene_prompt_v3_steers_subjects_away_from_text_money_and_people():
    from providers.gemini.scene_planner_provider import DEFAULT_PROMPT_PATH, PROMPT_VERSION

    assert PROMPT_VERSION == "scene_planner/short_scene-v3" and DEFAULT_PROMPT_PATH.is_file()
    text = DEFAULT_PROMPT_PATH.read_text(encoding="utf-8")
    assert "ignores the" in text and "solid, inanimate physical object" in text
    for banned in ("statements", "price tags", "banknotes", "screens", "silhouettes", "shadows"):
        assert banned in text
    assert "ledger on a plain desk" not in text  # the old example invited fake text
