"""Deterministic validation for Stage-6 ProductionPlan artifacts.

The validator implements the documented gate rules (docs/DATA_CONTRACTS.md §10
and PROJECT_PROGRESS.md Stage 6). It is defense in depth: even a future LLM
scene planner cannot make a scene silently become a source of scientific
knowledge, because every factual visual must trace to an approved claim.

Checks implemented here (rule ids r01..r21):

  r01  plan non-empty
  r02  unique scene ids
  r03  scene ordering valid (contiguous 1..N)
  r04  scene durations positive
  r05  total duration internally consistent
  r06  total duration within configured tolerance of the script duration
  r07  scenes map >= 1 script line unless explicitly a transition
  r08  every script line covered unless explicitly allowed
  r09  every referenced script line exists
  r10  every claim_ref exists in the fact-check result
  r11  every factual claim_ref points to an approved claim
  r12  no rejected claim is referenced
  r13  no review claim is referenced
  r14  factual visual scenes contain claim_refs (covering their factual lines)
  r15  factual on-screen text has claim traceability
  r16  generation prompts exist where required
  r17  scene visual types are valid
  r18  no provider-specific generation parameters in the contract
       (the plan-level provenance record is excluded: it is audit metadata)
  r19  provenance exists
  r20  JSON round-trip works (skip when structural rules already failed)
  r21  determinism is a planner-level guarantee, verified in tests (re-run the
       planner twice and compare); a pure validator cannot re-run itself.

The validator returns a ScenePlanningValidation (issues + passed) instead of
raising, so it can grade arbitrary (even structurally invalid) plans; the
artifact-level model invariants are enforced separately in ProductionPlan.validate().
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Sequence

from agents.qa.fact_checker.models import FactCheckResult
from agents.scene_planner.models import (
    FACTUAL_TEXT_KINDS,
    GENERATIVE_VISUAL_TYPES,
    VISUAL_TYPES,
    ScenePlanningConfig,
    _utc_now,
)
from agents.script.models import FACTUAL_PURPOSES, Script

SEVERITY_ERROR = "ERROR"
SEVERITY_WARNING = "WARNING"


@dataclass(frozen=True)
class ValidationIssue:
    """One scene-planning validation finding."""

    rule_id: str
    severity: str
    message: str
    scene_id: str = ""

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "severity": self.severity,
            "message": self.message,
            "scene_id": self.scene_id,
        }


@dataclass
class ScenePlanningValidation:
    """Aggregate result of validating a ProductionPlan."""

    issues: tuple[ValidationIssue, ...] = ()
    checked_at: str = ""

    def __post_init__(self):
        if not self.checked_at:
            self.checked_at = _utc_now()

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == SEVERITY_ERROR)

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == SEVERITY_WARNING)

    @property
    def passed(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "checked_at": self.checked_at,
            "issues": [issue.to_dict() for issue in self.issues],
        }


def _provider_primitives(value):
    """Yield (kind, key, text) leaf primitives for provider-hint scanning.

    `key` is the dict key path for keys; `text` is string VALUES that are
    scanned for denylisted provider vocabulary.
    """
    if isinstance(value, dict):
        for key, child in value.items():
            if isinstance(child, (dict, list)):
                yield from _provider_primitives(child)
            else:
                yield ("key", str(key), None)
                if isinstance(child, str):
                    yield ("value", str(key), child)
    elif isinstance(value, list):
        for child in value:
            if isinstance(child, (dict, list)):
                yield from _provider_primitives(child)
            else:
                if isinstance(child, str):
                    yield ("value", "", child)


def validate_production_plan(
    plan,
    script: Script,
    result: FactCheckResult,
    config: ScenePlanningConfig | None = None,
) -> ScenePlanningValidation:
    """Validate a ProductionPlan against its Script + FactCheckResult.

    `result` supplies the approved/rejected/review claim sets (the whitelist);
    a claim id may only appear in a scene if it is in `result.approved_claims`.
    """
    config = config or ScenePlanningConfig().validate()
    issues: list[ValidationIssue] = []

    def error(rule_id: str, message: str, scene_id: str = ""):
        issues.append(
            ValidationIssue(rule_id=rule_id, severity=SEVERITY_ERROR, message=message, scene_id=scene_id)
        )

    def warn(rule_id: str, message: str, scene_id: str = ""):
        issues.append(
            ValidationIssue(rule_id=rule_id, severity=SEVERITY_WARNING, message=message, scene_id=scene_id)
        )

    script_by_id = {line.line_id: line for line in script.lines}
    script_line_ids = [line.line_id for line in script.lines]
    approved = set(result.approved_claims)
    rejected = set(result.rejected_claims)
    review = set(result.review_items)
    known_claim_ids = {vc.claim_id for vc in result.claim_results}

    # r01 -- plan non-empty
    if not plan.scenes:
        error("r01", "production plan contains no scenes")
        return ScenePlanningValidation(issues=tuple(issues))

    # r02 -- unique scene ids
    seen: set[str] = set()
    for scene in plan.scenes:
        if scene.scene_id in seen:
            error("r02", f"duplicate scene_id '{scene.scene_id}'", scene.scene_id)
        seen.add(scene.scene_id)

    # r03 -- scene ordering valid (contiguous 1..N)
    if [scene.order for scene in plan.scenes] != list(range(1, len(plan.scenes) + 1)):
        error("r03", "scene orders must be the contiguous sequence 1..N")

    # r04 -- scene durations positive
    for scene in plan.scenes:
        if scene.duration_seconds <= 0:
            error("r04", f"scene {scene.scene_id} has non-positive duration", scene.scene_id)

    # r05 -- total duration internally consistent
    derived = round(sum(scene.duration_seconds for scene in plan.scenes), 2)
    if abs(derived - plan.total_duration_seconds) > 0.01:
        error(
            "r05",
            f"total_duration_seconds {plan.total_duration_seconds} != derived sum {derived}",
        )

    # r06 -- total duration within tolerance of the script duration
    tolerance = config.duration_tolerance_seconds
    if not (
        script.estimated_duration_seconds - tolerance
        <= plan.total_duration_seconds
        <= script.estimated_duration_seconds + tolerance
    ):
        error(
            "r06",
            f"total_duration_seconds {plan.total_duration_seconds} is outside "
            f"script duration {script.estimated_duration_seconds} +/- {tolerance}",
        )

    covered_script_lines: set[str] = set()
    for scene in plan.scenes:
        # r07 -- scenes map >= 1 script line unless explicitly a transition
        if not scene.script_line_ids and scene.visual_type != "transition":
            error(
                "r07",
                f"scene {scene.scene_id} maps no script lines and is not a transition",
                scene.scene_id,
            )
        for line_id in scene.script_line_ids:
            # r09 -- every referenced script line exists
            if line_id not in script_by_id:
                error(
                    "r09",
                    f"scene {scene.scene_id} references unknown script line '{line_id}'",
                    scene.scene_id,
                )
            else:
                covered_script_lines.add(line_id)

        scene_claims = set(scene.claim_refs)
        # r10/r11/r12/r13 -- claim reference categories
        for claim_id in scene.claim_refs:
            if claim_id in rejected:
                error(
                    "r12",
                    f"scene {scene.scene_id} references rejected claim '{claim_id}'",
                    scene.scene_id,
                )
            elif claim_id in review:
                error(
                    "r13",
                    f"scene {scene.scene_id} references review (unapproved) claim '{claim_id}'",
                    scene.scene_id,
                )
            elif claim_id not in approved:
                error(
                    "r10" if claim_id not in known_claim_ids else "r11",
                    f"scene {scene.scene_id} references claim '{claim_id}' which is not approved",
                    scene.scene_id,
                )

        # r14 -- factual visual scenes contain claim_refs covering their factual lines
        factual_lines = [
            script_by_id[line_id]
            for line_id in scene.script_line_ids
            if line_id in script_by_id and script_by_id[line_id].purpose in FACTUAL_PURPOSES
        ]
        if factual_lines and scene.visual_type != "transition":
            missing = sorted(
                {claim_id for line in factual_lines for claim_id in line.claim_refs} - scene_claims
            )
            if not scene_claims:
                error(
                    "r14",
                    f"factual scene {scene.scene_id} contains no claim_refs",
                    scene.scene_id,
                )
            elif missing:
                error(
                    "r14",
                    f"factual scene {scene.scene_id} is missing claim refs for its factual "
                    f"lines: {', '.join(missing)}",
                    scene.scene_id,
                )

        # r15 -- factual on-screen text has traceability
        for item in scene.on_screen_text:
            if item.kind in FACTUAL_TEXT_KINDS and config.caption_traceability:
                if not item.claim_refs:
                    error(
                        "r15",
                        f"factual on-screen text in scene {scene.scene_id} has no claim_refs",
                        scene.scene_id,
                    )
            for claim_id in item.claim_refs:
                if claim_id in rejected:
                    error("r12", f"on-screen text references rejected claim '{claim_id}'", scene.scene_id)
                elif claim_id in review:
                    error("r13", f"on-screen text references review claim '{claim_id}'", scene.scene_id)
                elif claim_id not in approved:
                    error(
                        "r10" if claim_id not in known_claim_ids else "r11",
                        f"on-screen text references claim '{claim_id}' which is not approved",
                        scene.scene_id,
                    )

        # r16 -- generation prompts exist where required
        if scene.visual_type in config.generation_prompt_required_types:
            if not scene.generation_prompt.strip():
                error(
                    "r16",
                    f"scene {scene.scene_id} of generative type '{scene.visual_type}' "
                    "has no generation_prompt",
                    scene.scene_id,
                )

        # r17 -- scene visual types are valid
        if scene.visual_type not in VISUAL_TYPES:
            error(
                "r17",
                f"scene {scene.scene_id} has invalid visual_type '{scene.visual_type}'",
                scene.scene_id,
            )

    # r08 -- every script line covered unless explicitly allowed
    uncovered_script_lines = [line_id for line_id in script_line_ids if line_id not in covered_script_lines]
    if uncovered_script_lines:
        if config.require_full_coverage:
            error(
                "r08",
                "uncovered script line(s): " + ", ".join(uncovered_script_lines),
            )
        else:
            warn(
                "r08",
                "uncovered script line(s): " + ", ".join(uncovered_script_lines),
            )

    # r18 -- no provider-specific generation parameters in the contract
    # (the plan-level `provenance` record is audit metadata, not render
    # parameters, so it is excluded from this scan)
    payload = dict(plan.to_dict()["data"])
    payload.pop("provenance", None)
    provider_hits: list[str] = []
    for kind, key, text in _provider_primitives(payload):
        if kind == "key":
            lowered = key.lower()
            if any(
                marker in lowered
                for marker in ("provider", "api_", "generator", "params", "model_", "scheduler", "sampler")
            ):
                provider_hits.append(f"key '{key}'")
        elif text:
            lowered = text.lower()
            for term in config.provider_hint_denylist:
                if term.lower() in lowered:
                    provider_hits.append(f"{key or 'value'} contains '{term}'")
                    break
    if provider_hits:
        error("r18", "provider-specific wording leaked into the plan: " + "; ".join(provider_hits))

    # r19 -- provenance exists
    if plan.provenance is None:
        error("r19", "production plan provenance is missing")

    # r20 -- JSON round-trip works (only meaningful for structurally sound plans)
    if not any(issue.rule_id in ("r01", "r02", "r03", "r04", "r05") for issue in issues):
        try:
            payload = plan.to_dict()
            cloned = plan.__class__.from_dict(json.loads(plan.to_json()))
            if cloned.to_dict() != payload:
                error("r20", "JSON round-trip changed the plan contents")
        except Exception as exc:  # noqa: BLE001 - a round-trip failure is a validation result
            error("r20", f"JSON round-trip failed: {exc}")

    return ScenePlanningValidation(issues=tuple(issues))