"""Shared builders for the Stage-9 (policy/safety QA) test suite.

Everything here is MOCK, built deterministically from the committed
`data/mock/*` octopus fixtures (the same rule as the Stage-3..8 builders):
no fake claims, no real-world evidence, no external services.

The committed fixtures evaluate CLEAN under the policy gate (all 14 rule checks
pass and Stage-8 <-> policy remain consistent). Each scenario below introduces
exactly one defect on top of that healthy baseline with `dataclasses.replace`,
so a failing 15-check run isolates the rule under test.
"""

from __future__ import annotations

import sys
from pathlib import Path

from agents.qa.policy.models import PolicyInputs, PolicyQAConfig
from agents.qa.policy.stage import DEFAULT_QA_RESULT_PATH, load_committed_fixtures

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

__all__ = [
    "REPO_ROOT",
    "committed_inputs",
    "build_inputs",
    "build_no_qa_inputs",
    "build_review_qa_inputs",
    "build_stale_qa_inputs",
    "build_harmful_inputs",
    "build_dangerous_inputs",
    "build_overstatement_inputs",
    "build_misinfo_framing_inputs",
    "build_fake_footage_inputs",
    "build_synthetic_no_disclosure_inputs",
    "build_unknown_rights_inputs",
    "build_watermarked_inputs",
    "build_licensed_external_inputs",
    "build_privacy_pii_inputs",
    "build_privacy_leak_inputs",
    "build_spam_metadata_inputs",
    "build_near_duplicate_inputs",
    "build_adsensitive_inputs",
    "build_sensational_title_inputs",
    "build_missing_provenance_inputs",
    "build_missing_plan_inputs",
    "build_qa_result_dict",
    "invalid_version_config",
]


def committed_inputs() -> PolicyInputs:
    """The committed data/mock fixtures (clean PASS baseline).

    Loaded from absolute paths so it works regardless of cwd (e.g. under
    ``monkeypatch.chdir`` in purity tests).
    """
    from agents.qa.policy.stage import (
        build_policy_inputs,
        load_video_qa_result,
        load_video_generation_result,
    )
    from agents.qa.script.stage import load_fact_check_result, load_script
    from agents.video_generation.stage import load_production_plan

    mock = REPO_ROOT / "data" / "mock"
    video_qa = None
    qa_path = mock / "qa_result.json"
    if qa_path.is_file():
        video_qa = load_video_qa_result(qa_path)
    return build_policy_inputs(
        script=load_script(mock / "script.json"),
        fact_check=load_fact_check_result(mock / "fact_check_result.json"),
        plan=load_production_plan(mock / "production.json"),
        video_result=load_video_generation_result(mock / "video_generation_result.json"),
        video_qa=video_qa,
    )


def build_inputs(**changes) -> PolicyInputs:
    """Committed inputs with selected artifacts swapped out."""
    return dataclasses_replace(committed_inputs(), **changes)


def dataclasses_replace(obj, **changes):
    import dataclasses

    return dataclasses.replace(obj, **changes)


# --------------------------------------------------------------------------- #
# Plan / script helpers
# --------------------------------------------------------------------------- #


def _scene(plan, scene_id: str):
    return next(scene for scene in plan.scenes if scene.scene_id == scene_id)


def _set_scene(plan, scene_id: str, **changes):
    import dataclasses

    rows = list(plan.scenes)
    for index, scene in enumerate(rows):
        if scene.scene_id == scene_id:
            rows[index] = dataclasses.replace(scene, **changes)
    return dataclasses.replace(plan, scenes=tuple(rows))


def _set_asset(plan, scene_id: str, index: int = 0, **changes):
    import dataclasses

    rows = list(plan.scenes)
    for position, scene in enumerate(rows):
        if scene.scene_id == scene_id:
            assets = list(scene.asset_requirements)
            assets[index] = dataclasses.replace(assets[index], **changes)
            rows[position] = dataclasses.replace(scene, asset_requirements=tuple(assets))
    return dataclasses.replace(plan, scenes=tuple(rows))


def _set_line(script, line_id: str, **changes):
    import dataclasses

    lines = list(script.lines)
    for index, line in enumerate(lines):
        if line.line_id == line_id:
            lines[index] = dataclasses.replace(line, **changes)
    return dataclasses.replace(script, lines=tuple(lines))


# --------------------------------------------------------------------------- #
# Scenarios (each introduces exactly the rule's defect over the clean baseline)
# --------------------------------------------------------------------------- #


def build_no_qa_inputs() -> PolicyInputs:
    """Archived nothing fell through: the pipeline forgot the Stage-8 gate."""
    return build_inputs(video_qa=None)


def build_review_qa_inputs() -> PolicyInputs:
    """Stage-8 returned REVIEW; a policy PASS must not silently republish it."""
    qa = committed_inputs().video_qa
    import dataclasses

    return build_inputs(video_qa=dataclasses.replace(qa, overall_verdict="REVIEW"))


def build_stale_qa_inputs() -> PolicyInputs:
    """Stage-8 result came from a different production plan than current."""
    qa = committed_inputs().video_qa
    import dataclasses

    return build_inputs(video_qa=dataclasses.replace(qa, target_ref="pp_archive_0001"))


def build_harmful_inputs() -> PolicyInputs:
    plan = committed_inputs().plan
    plan = _set_scene(
        plan,
        "scn_01",
        narration="Step-by-step guide: how to build a bomb out of household items.",
    )
    return build_inputs(plan=plan)


def build_dangerous_inputs() -> PolicyInputs:
    plan = committed_inputs().plan
    plan = _set_scene(
        plan,
        "scn_02",
        narration="Extreme challenge: try this at home without any protection.",
    )
    return build_inputs(plan=plan)


def build_overstatement_inputs() -> PolicyInputs:
    script = committed_inputs().script
    script = _set_line(
        script,
        "ln_02",
        text=(
            "Scientists PROVED octopuses have three hearts, so this is "
            "absolutely certain."
        ),
    )
    return build_inputs(script=script)


def build_misinfo_framing_inputs() -> PolicyInputs:
    script = committed_inputs().script
    script = _set_line(
        script,
        "ln_03",
        text="This video will definitely prove that this cure is guaranteed.",
    )
    return build_inputs(script=script)


def build_fake_footage_inputs() -> PolicyInputs:
    plan = committed_inputs().plan
    plan = _set_scene(
        plan,
        "scn_01",
        narration="This documentary uses actual footage shot on location at the reef.",
    )
    return build_inputs(plan=plan)


def build_synthetic_no_disclosure_inputs() -> PolicyInputs:
    plan = committed_inputs().plan
    plan = _set_scene(
        plan,
        "scn_01",
        generation_prompt=(
            "photoreal rendering of the president delivering a speech, "
            "public figure close-up"
        ),
        visual_description="realistic depiction of a real person on a podium",
    )
    return build_inputs(plan=plan)


def build_unknown_rights_inputs() -> PolicyInputs:
    plan = committed_inputs().plan
    plan = _set_asset(
        plan,
        "scn_07",
        index=0,
        asset_type="stock_asset",
        subject="archive coral reef footage",
    )
    return build_inputs(plan=plan)


def build_watermarked_inputs() -> PolicyInputs:
    plan = committed_inputs().plan
    plan = _set_asset(
        plan,
        "scn_07",
        index=0,
        asset_type="stock_asset",
        subject="watermarked proprietary footage from a news archive",
    )
    return build_inputs(plan=plan)


def build_licensed_external_inputs() -> PolicyInputs:
    plan = committed_inputs().plan
    plan = _set_asset(
        plan,
        "scn_07",
        index=0,
        asset_type="stock_asset",
        subject="royalty-free licensed reef footage, CC0 public domain",
    )
    return build_inputs(plan=plan)


def build_privacy_pii_inputs() -> PolicyInputs:
    script = committed_inputs().script
    script = _set_line(
        script,
        "ln_03",
        text="Contact the team at sarah.smith@example.com for the full report.",
    )
    return build_inputs(script=script)


def build_privacy_leak_inputs() -> PolicyInputs:
    script = committed_inputs().script
    script = _set_line(
        script,
        "ln_03",
        text="We leaked her email address sarah.smith@example.com and phone number.",
    )
    return build_inputs(script=script)


def build_spam_metadata_inputs() -> PolicyInputs:
    script = committed_inputs().script
    script = dataclasses_replace(
        script,
        title="Exclusive FREE secret: shocking and guaranteed, #1!",
        hook="Act now, limited time offer, click here without waiting!!!",
    )
    return build_inputs(script=script)


def build_near_duplicate_inputs() -> PolicyInputs:
    script = committed_inputs().script
    script = _set_line(
        script,
        "ln_03",
        text="Octopuses have three hearts and blue blood pumped by branchial hearts.",
    )
    script = _set_line(
        script,
        "ln_04",
        text="Octopuses have three hearts and blue blood pumped by branchial hearts.",
    )
    return build_inputs(script=script)


def build_adsensitive_inputs() -> PolicyInputs:
    script = committed_inputs().script
    script = _set_line(
        script,
        "ln_05",
        text="Our premium wine cellar is the best way to celebrate any milestone.",
    )
    return build_inputs(script=script)


def build_sensational_title_inputs() -> PolicyInputs:
    script = committed_inputs().script
    script = dataclasses_replace(
        script,
        title="The exclusive interview nobody wanted: leaked footage exposed.",
    )
    return build_inputs(script=script)


def build_missing_provenance_inputs() -> PolicyInputs:
    plan = committed_inputs().plan
    import dataclasses

    return build_inputs(plan=dataclasses.replace(plan, provenance=None))


def build_missing_plan_inputs() -> PolicyInputs:
    return build_inputs(plan=None)


def build_qa_result_dict(**changes) -> dict:
    """Serializable Stage-8 VideoQAResult for CLI scenario tests (default clean)."""
    qa = committed_inputs().video_qa
    import dataclasses

    if changes:
        qa = dataclasses.replace(qa, **changes)
    return qa.to_dict()


def invalid_version_config() -> PolicyQAConfig:
    return PolicyQAConfig(policy_version="policy_rules_bogus")


def qa_result_path() -> Path:
    return DEFAULT_QA_RESULT_PATH