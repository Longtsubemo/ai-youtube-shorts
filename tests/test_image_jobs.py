"""Tests for the Stage 4 -> image-job conversion (agents/image_jobs)."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from agents.image_jobs import (
    DEFAULT_NEGATIVE_PROMPT,
    FINAL_PROMPT_HARD_CEILING_TOKENS,
    FINAL_PROMPT_TARGET_TOKENS,
    ImageJob,
    ImageJobBatch,
    ImageJobError,
    MascotJobInfo,
    build_image_job_batch,
    check_final_prompt,
    check_negative_prompt,
    derive_seed,
)
from agents.image_jobs.__main__ import main as image_jobs_main
from agents.scene_planner.scene_models import (
    CHANNEL_IMAGE_STYLE,
    MASCOT_ASSET_PATH,
    Mascot,
    Scene,
    ScenePlan,
)
from agents.scene_planner.validator import estimate_clip_tokens

REAL_PLAN = Path("data/e2e-20261008-stage4-live-02/scene_plan.json")


def _stub_plan(*, prompts=None, run_id="run-test-01") -> ScenePlan:
    prompts = prompts or [
        "Magnifying glass over scattered coins on a wooden desk. Simple composition.",
        "Balance scale with coins on both pans. Minimalist composition.",
    ]
    scenes = []
    for i, prompt in enumerate(prompts):
        scenes.append(
            Scene(
                scene_id=f"sc{i + 1:02d}",
                order=i,
                duration_seconds=8 + i,
                narration=f"Scene {i + 1} narration words that must stay out of prompts.",
                generation_prompt=prompt,
                required_assets=[MASCOT_ASSET_PATH],
                mascot=Mascot(
                    present=True, action="investigating", asset=MASCOT_ASSET_PATH
                ),
            )
        )
    return ScenePlan(
        topic_id="t_test",
        title="Test plan",
        scenes=scenes,
        run_id=run_id,
        channel_image_style=CHANNEL_IMAGE_STYLE,
    )


# --- seeds -------------------------------------------------------------------


def test_seed_is_deterministic():
    assert derive_seed("run-a", "sc01") == derive_seed("run-a", "sc01")
    assert derive_seed("run-a", "sc01") != derive_seed("run-a", "sc02")
    assert derive_seed("run-a", "sc01") != derive_seed("run-b", "sc01")


def test_seed_matches_documented_derivation():
    digest = hashlib.sha256(b"run-test-01:sc01").digest()
    expected = int.from_bytes(digest[:4], "big")
    assert derive_seed("run-test-01", "sc01") == expected
    assert 0 <= expected < 2**32


def test_all_scene_seeds_distinct_within_batch():
    plan = _stub_plan(prompts=["Coins on a desk. Simple composition."] * 4)
    batch = build_image_job_batch(plan, run_id="run-test-01")
    seeds = [job.seed for job in batch.jobs]
    assert len(set(seeds)) == len(seeds)


# --- field preservation and composition --------------------------------------


def test_required_fields_are_preserved():
    plan = _stub_plan()
    batch = build_image_job_batch(plan, run_id="run-test-01")
    assert len(batch.jobs) == 2
    for scene, job in zip(plan.scenes, batch.jobs):
        assert job.scene_id == scene.scene_id
        assert job.order == scene.order
        assert job.duration_seconds == scene.duration_seconds
        assert job.scene_generation_prompt == scene.generation_prompt
        assert job.channel_image_style == CHANNEL_IMAGE_STYLE
        assert job.mascot.present is True
        assert job.mascot.action == "investigating"
        assert job.mascot.asset == MASCOT_ASSET_PATH
        assert job.mascot.in_positive_prompt is False
        assert job.seed == derive_seed("run-test-01", scene.scene_id)


def test_final_prompt_follows_composition_rule():
    plan = _stub_plan()
    batch = build_image_job_batch(plan, run_id="run-test-01")
    for job in batch.jobs:
        assert (
            job.positive_prompt
            == f"{CHANNEL_IMAGE_STYLE}, {job.scene_generation_prompt}"
        )


def test_mascot_and_narration_never_in_final_prompt():
    plan = _stub_plan()
    batch = build_image_job_batch(plan, run_id="run-test-01")
    for scene, job in zip(plan.scenes, batch.jobs):
        lowered = job.positive_prompt.lower()
        assert "mascot" not in lowered
        assert "fox" not in lowered
        assert MASCOT_ASSET_PATH not in job.positive_prompt
        scene_words = " ".join(scene.narration.lower().split()[:6])
        assert scene_words not in lowered


def test_output_asset_mapping_matches_stage5_convention():
    plan = _stub_plan()
    batch = build_image_job_batch(plan, run_id="run-test-01")
    for job in batch.jobs:
        assert job.asset_id == f"{job.scene_id}_main"
        assert job.output_path == f"data/run-test-01/assets/{job.scene_id}_main.png"
        assert job.job_id == f"run-test-01-{job.scene_id}"


# --- final-prompt validation -------------------------------------------------


def test_final_prompt_budget_constants_are_clipsafe():
    assert FINAL_PROMPT_HARD_CEILING_TOKENS <= 75
    assert FINAL_PROMPT_TARGET_TOKENS < FINAL_PROMPT_HARD_CEILING_TOKENS


def test_channel_style_alone_is_within_ceiling():
    assert estimate_clip_tokens(CHANNEL_IMAGE_STYLE) <= FINAL_PROMPT_HARD_CEILING_TOKENS


def test_channel_style_has_no_forbidden_concepts():
    issues = check_final_prompt(
        f"{CHANNEL_IMAGE_STYLE}, coins on a desk. Simple composition.",
        channel_style=CHANNEL_IMAGE_STYLE,
    )
    assert [i for i in issues if i.severity == "error"] == []


def test_over_ceiling_final_prompt_fails_without_truncation():
    long_scene_prompt = ("scattered coins rest upon the wooden desk " * 8).strip()
    plan = _stub_plan(prompts=[long_scene_prompt])
    combined = f"{CHANNEL_IMAGE_STYLE}, {long_scene_prompt}"
    assert estimate_clip_tokens(combined) > FINAL_PROMPT_HARD_CEILING_TOKENS
    with pytest.raises(ImageJobError, match="ij02"):
        build_image_job_batch(plan, run_id="run-test-01")
    # the source prompt was never truncated anywhere
    assert plan.scenes[0].generation_prompt == long_scene_prompt


def test_empty_generation_prompt_fails():
    plan = _stub_plan(prompts=["   "])
    with pytest.raises(ImageJobError, match="ij01"):
        build_image_job_batch(plan, run_id="run-test-01")


def test_narration_leakage_fails_in_final_prompt():
    narration = "The startup offer looked generous on paper but equity was frozen hard."
    scene = Scene(
        scene_id="sc01",
        order=0,
        duration_seconds=8,
        narration=narration,
        generation_prompt=f"{narration} Coins on a desk.",
    )
    plan = ScenePlan(
        topic_id="t_test",
        scenes=[scene],
        run_id="run-test-01",
        channel_image_style=CHANNEL_IMAGE_STYLE,
    )
    with pytest.raises(ImageJobError, match="ij04"):
        build_image_job_batch(plan, run_id="run-test-01")


def test_mascot_mention_in_final_prompt_fails():
    plan = _stub_plan(prompts=["A fox mascot detective at a desk. Simple composition."])
    with pytest.raises(ImageJobError, match="ij05"):
        build_image_job_batch(plan, run_id="run-test-01")


def test_above_target_but_under_ceiling_warns_without_failing():
    scene_prompt = ("coins and documents rest on the wooden desk beside the scale " * 2).strip()
    plan = _stub_plan(prompts=[scene_prompt])
    batch = build_image_job_batch(plan, run_id="run-test-01")
    final = batch.jobs[0].positive_prompt
    est = estimate_clip_tokens(final)
    assert FINAL_PROMPT_TARGET_TOKENS < est <= FINAL_PROMPT_HARD_CEILING_TOKENS
    assert any(w["code"] == "ij03" for w in batch.warnings)


# --- negative prompt ---------------------------------------------------------


def test_default_negative_prompt_is_valid_and_stable():
    issues = check_negative_prompt(DEFAULT_NEGATIVE_PROMPT)
    assert issues == []
    assert DEFAULT_NEGATIVE_PROMPT == (
        "photorealistic, 3d render, animation, text, words, logo, watermark, "
        "people, faces, fox, mascot, blurry, deformed, extra limbs"
    )


def test_every_job_carries_the_negative_prompt():
    plan = _stub_plan()
    batch = build_image_job_batch(plan, run_id="run-test-01")
    for job in batch.jobs:
        assert job.negative_prompt == DEFAULT_NEGATIVE_PROMPT
    assert batch.negative_prompt == DEFAULT_NEGATIVE_PROMPT


def test_empty_negative_prompt_fails():
    plan = _stub_plan()
    with pytest.raises(ImageJobError, match="ij07"):
        build_image_job_batch(plan, run_id="run-test-01", negative_prompt="")


def test_over_ceiling_negative_prompt_fails():
    long_negative = "blurry deformed extra limbs " * 20
    assert estimate_clip_tokens(long_negative) > FINAL_PROMPT_HARD_CEILING_TOKENS
    plan = _stub_plan()
    with pytest.raises(ImageJobError, match="ij08"):
        build_image_job_batch(plan, run_id="run-test-01", negative_prompt=long_negative)


# --- batch-level behaviour ---------------------------------------------------


def test_run_id_is_required():
    plan = _stub_plan(run_id="")
    with pytest.raises(ImageJobError, match="run_id"):
        build_image_job_batch(plan, run_id="")


def test_plan_without_scenes_fails():
    plan = ScenePlan(topic_id="t_test", scenes=[], run_id="run-test-01")
    with pytest.raises(ImageJobError, match="no scenes"):
        build_image_job_batch(plan, run_id="run-test-01")


def test_roundtrip_serialization():
    plan = _stub_plan()
    batch = build_image_job_batch(plan, run_id="run-test-01")
    restored = ImageJobBatch.from_dict(batch.to_dict())
    assert restored.to_dict() == batch.to_dict()
    assert isinstance(restored.jobs[0], ImageJob)
    assert isinstance(restored.jobs[0].mascot, MascotJobInfo)


def test_build_is_deterministic_across_calls():
    plan = _stub_plan()
    first = build_image_job_batch(plan, run_id="run-test-01").to_dict()
    second = build_image_job_batch(plan, run_id="run-test-01").to_dict()
    # created_at-free contract: identical output for identical input
    assert first == second


# --- real live artifact ------------------------------------------------------


@pytest.mark.skipif(not REAL_PLAN.is_file(), reason="live Stage 4 artifact missing")
def test_real_live_plan_converts_cleanly():
    from agents.image_jobs import load_scene_plan

    plan = load_scene_plan(REAL_PLAN)
    batch = build_image_job_batch(plan, run_id="e2e-20261008-stage4-live-02")
    assert len(batch.jobs) == 7
    # The channel style block alone is ~38 estimated tokens, so every combined
    # prompt is expected to sit above the 60-token soft target but at or below
    # the 75-token hard ceiling (warnings only, no errors).
    assert batch.warnings != []
    assert {w["code"] for w in batch.warnings} == {"ij03"}
    assert batch.channel_image_style == CHANNEL_IMAGE_STYLE
    assert batch.schema == "image-jobs/1.0"
    for job in batch.jobs:
        est = estimate_clip_tokens(job.positive_prompt)
        assert est <= FINAL_PROMPT_HARD_CEILING_TOKENS
        assert job.mascot.in_positive_prompt is False
        assert job.mascot.asset == MASCOT_ASSET_PATH
        assert job.negative_prompt == DEFAULT_NEGATIVE_PROMPT
    # determinism against a second build
    again = build_image_job_batch(plan, run_id="e2e-20261008-stage4-live-02")
    assert [j.seed for j in again.jobs] == [j.seed for j in batch.jobs]


# --- CLI ---------------------------------------------------------------------


def test_cli_end_to_end(tmp_path, monkeypatch, capsys):
    plan = _stub_plan()
    run_id = "cli-test-01"
    run_dir = tmp_path / "data" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "scene_plan.json").write_text(
        json.dumps(plan.to_dict()), encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["image_jobs", "--run-id", run_id]
    )
    assert image_jobs_main() == 0
    out = run_dir / "image_jobs.json"
    assert out.is_file()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["schema"] == "image-jobs/1.0"
    assert len(data["jobs"]) == 2
    captured = capsys.readouterr().out
    assert f"run_id={run_id} jobs=2" in captured


def test_cli_missing_plan_returns_2(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["image_jobs", "--run-id", "does-not-exist"]
    )
    assert image_jobs_main() == 2
