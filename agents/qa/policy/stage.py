"""Stage-9 entrypoint: artifacts -> PolicyQAResult (policy/safety gate).

QA-side only (docs/ARCHITECTURE.md): the stage consumes the Stage-3 Script,
the Stage-5 FactCheckResult, the Stage-6 ProductionPlan, the Stage-7
VideoGenerationResult, and the Stage-8 VideoQAResult, runs the deterministic
15-check policy/safety gate (rules.py/checks.py/validator.py), and emits a
`PolicyQAResult` under `data/<run_id>/production/policy_qa_result.json`.

It never calls a provider, never inspects media bytes, never consults an LLM,
and does NOT upload or schedule anything -- Stage 10 (human approval) and
Stage 11 (YouTube upload) are out of scope for this stage and are not
implemented here.

Pairing guard: the Stage-7 result must target the SAME ProductionPlan being
assessed; a mismatched pairing raises before anything is evaluated (mirroring
Stage 8). A stale/mismatched Stage-8 VideoQAResult is instead GRADED by check
P15 (soft ERROR -> REVIEW), because gating against it is a policy concern the
gate can surface rather than abort on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agents.qa.fact_checker.models import FactCheckResult
from agents.qa.policy.classifier import PolicyRiskClassifier
from agents.qa.policy.models import (
    PolicyInputs,
    PolicyQAConfig,
    PolicyQAError,
    PolicyQAResult,
)
from agents.qa.policy.validator import evaluate_policy
from agents.qa.script.stage import load_fact_check_result, load_script
from agents.qa.video.models import VideoQAResult
from agents.scene_planner.models import ProductionPlan
from agents.script.models import Script
from agents.video_generation.models import VideoGenerationResult
from agents.video_generation.stage import load_production_plan

DEFAULT_SCRIPT_PATH = Path("data") / "mock" / "script.json"
DEFAULT_FACT_CHECK_PATH = Path("data") / "mock" / "fact_check_result.json"
DEFAULT_PRODUCTION_PLAN_PATH = Path("data") / "mock" / "production.json"
DEFAULT_VIDEO_RESULT_PATH = Path("data") / "mock" / "video_generation_result.json"
DEFAULT_QA_RESULT_PATH = Path("data") / "mock" / "qa_result.json"
DEFAULT_POLICY_RESULT_NAME = "policy_qa_result.json"


def load_video_generation_result(path: str | Path) -> VideoGenerationResult:
    return VideoGenerationResult.from_json(Path(path).read_text(encoding="utf-8"))


def load_video_qa_result(path: str | Path) -> VideoQAResult:
    return VideoQAResult.from_json(Path(path).read_text(encoding="utf-8"))


def load_policy_qa_result(path: str | Path) -> PolicyQAResult:
    return PolicyQAResult.from_json(Path(path).read_text(encoding="utf-8"))


def build_policy_inputs(
    script: Script | None = None,
    fact_check: FactCheckResult | None = None,
    plan: ProductionPlan | None = None,
    video_result: VideoGenerationResult | None = None,
    video_qa: VideoQAResult | None = None,
) -> PolicyInputs:
    """Assemble the typed input bundle the gate evaluates over."""
    return PolicyInputs(
        script=script,
        fact_check=fact_check,
        plan=plan,
        video_result=video_result,
        video_qa=video_qa,
    )


def run_policy_qa(
    inputs: PolicyInputs,
    config: PolicyQAConfig | None = None,
    run_id: str = "",
    created_at: str = "",
    classifier: PolicyRiskClassifier | None = None,
    resumed_from_policy_result_id: str = "",
) -> PolicyQAResult:
    """Run the Stage-9 policy/safety gate over `inputs`.

    Guards the plan/video-result pairing (raises on mismatch, mirroring Stage 8),
    then delegates verdict aggregation to `evaluate_policy`.
    """
    plan = inputs.plan
    video_result = inputs.video_result
    if plan is not None and video_result is not None:
        if video_result.production_plan_id != plan.production_plan_id:
            raise PolicyQAError(
                f"video result {video_result.video_generation_result_id} targets "
                f"production plan {video_result.production_plan_id}, not "
                f"{plan.production_plan_id}"
            )
    return evaluate_policy(
        inputs,
        config=config,
        run_id=run_id,
        created_at=created_at,
        classifier=classifier,
        resumed_from_policy_result_id=resumed_from_policy_result_id,
    )


def load_committed_fixtures() -> PolicyInputs:
    """Load the committed data/mock artifacts (missing QA result -> None)."""
    script = load_script(DEFAULT_SCRIPT_PATH)
    fact_check = load_fact_check_result(DEFAULT_FACT_CHECK_PATH)
    plan = load_production_plan(DEFAULT_PRODUCTION_PLAN_PATH)
    video_result = load_video_generation_result(DEFAULT_VIDEO_RESULT_PATH)
    video_qa = None
    if DEFAULT_QA_RESULT_PATH.is_file():
        video_qa = load_video_qa_result(DEFAULT_QA_RESULT_PATH)
    return build_policy_inputs(
        script=script,
        fact_check=fact_check,
        plan=plan,
        video_result=video_result,
        video_qa=video_qa,
    )


def policy_inputs_from_paths(
    script_path: str | Path,
    fact_check_path: str | Path,
    plan_path: str | Path,
    video_result_path: str | Path,
    video_qa_path: str | Path | None = None,
    **_: Any,
) -> PolicyInputs:
    """Load an input bundle from explicit artifact JSON paths (for the CLI)."""
    video_qa = None
    if video_qa_path is not None:
        video_qa = load_video_qa_result(video_qa_path)
    return build_policy_inputs(
        script=load_script(script_path),
        fact_check=load_fact_check_result(fact_check_path),
        plan=load_production_plan(plan_path),
        video_result=load_video_generation_result(video_result_path),
        video_qa=video_qa,
    )