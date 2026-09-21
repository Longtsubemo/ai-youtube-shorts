"""Stage-8 video-QA orchestration: checks -> verdict -> artifact.

This module owns the overall verdict (PASS / REVIEW / REJECT), the issue id
assignment, and the assembly of the VideoQAResult artifact. It is fully
deterministic and network-free.

Verdict rules (docs/DATA_CONTRACTS.md §11 / PROJECT_PROGRESS Stage 8, mirroring
the §8.1 gate model):

    REJECT  any severity-ERROR issue that is a HARD gate
    REVIEW  any severity-ERROR issue that is not a hard gate
    PASS    otherwise (warnings/info only, or clean)

Hard gates cover the fatal cases: missing/pending/failed scenes, wrong order,
broken timing, malformed artifacts, phantom scene references, provider
coupling, a result that failed its own Stage-7 generation validation, an
inconsistent overall status, and any mock/truthfulness violation (mock
artifacts presented as real, or real media claimed when nothing was produced).
"""

from __future__ import annotations

import dataclasses

from agents.qa.video.checks import GATE_HARD, run_checks
from agents.qa.video.models import (
    PRODUCED_BY,
    VideoQAConfig,
    VideoQAResult,
    qa_result_id_for,
)
from agents.seo.models import Provenance, utc_now

VERDICT_REJECT = "REJECT"
VERDICT_REVIEW = "REVIEW"
VERDICT_PASS = "PASS"


def aggregate_verdict(issues) -> str:
    """Overall video-QA verdict from a severity-ERROR stream (hard gate wins)."""
    errors = [issue for issue in issues if issue.severity == "ERROR"]
    if any(issue.gate == GATE_HARD for issue in errors):
        return VERDICT_REJECT
    if errors:
        return VERDICT_REVIEW
    return VERDICT_PASS


def evaluate_video(
    result,
    plan,
    config: VideoQAConfig | None = None,
    run_id: str = "",
    created_at: str = "",
) -> VideoQAResult:
    """Run every Stage-8 check against a Stage-7 VideoGenerationResult and its
    ProductionPlan, producing a VideoQAResult.

    Deliberately does NOT call `result.validate()`: Stage 8 is a gate, not a
    re-validation of Stage 7. Structurally broken results (missing scenes,
    phantom ids, malformed artifacts, provider coupling) must produce QA issues
    rather than raise, so every check reads attributes defensively.
    """
    config = (config or VideoQAConfig()).validate()
    run_id = run_id or result.run_id or plan.run_id
    created_at = created_at or result.created_at or plan.created_at or utc_now()

    issues, check_records = run_checks(result, plan, config)

    # Stable, content-independent issue ids (deterministic ordering).
    issues = [
        dataclasses.replace(issue, issue_id=f"vqa_{number:02d}")
        for number, issue in enumerate(issues)
    ]

    verdict = aggregate_verdict(issues)

    artifact = VideoQAResult(
        qa_result_id=qa_result_id_for(run_id, plan.production_plan_id),
        target_ref=plan.production_plan_id,
        topic_id=plan.topic_id,
        topic_title=plan.topic_title,
        script_id=plan.script_id,
        video_generation_result_id=result.video_generation_result_id,
        overall_verdict=verdict,
        checked_at=created_at,
        checks=tuple(check_records),
        issues=tuple(issue for issue in issues if issue.severity == "ERROR"),
        warnings=tuple(issue for issue in issues if issue.severity == "WARNING"),
        infos=tuple(issue for issue in issues if issue.severity == "INFO"),
        provenance=Provenance(
            generated_by="video_qa_agent_v1",
            source="DeterministicVideoQA",
            created_at=created_at,
            provider=None,
            model="VideoQAValidator",
            prompt_version=config.schema_version,
        ),
        created_at=created_at,
        run_id=run_id,
        produced_by=dict(PRODUCED_BY),
    )
    return artifact.validate()