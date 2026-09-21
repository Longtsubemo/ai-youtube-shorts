"""Deterministic validation for Stage-7 VideoGenerationResult artifacts.

The validator is defensive: malformed results produce validation issues instead
of silent acceptance, but it never crashes on a corrupt artifact (the
artifact-level model invariants are enforced separately in
`VideoGenerationResult.validate()`). A result that fails the gate is reported
back as issues so the stage runner / CLI can exit cleanly without writing an
artifact.

Checks implemented here (rule ids g01..g13):

  g01  result contains at least one scene record
  g02  scene ids are unique
  g03  every scene status is a known value (completed / failed / pending)
  g04  every result scene exists in the ProductionPlan (no phantom scenes)
  g05  every ProductionPlan scene is represented in the result (no missing scenes)
  g06  per-scene order matches the plan's order
  g07  request timing is preserved (request.duration == plan scene duration)
  g08  completed scenes carry a valid artifact whose duration mirrors the plan
  g09  failed scenes carry a message + retryable flag; retryable failures
       respect the retry limit (a retryable flag at max_attempts is
       inconsistent and must be recorded as non-retryable)
  g10  overall_status is consistent with the per-scene statuses
  g11  provenance exists
  g12  provider isolation: the result contract must not leak API providers,
       API keys, GPU/CUDA requirements, cloud-vendor details, or
       provider-specific runtime assumptions (the top-level provenance record
       is audit metadata and is excluded - it legitimately names the local
       generator abstraction)
  g13  JSON round-trip works (skipped for structurally broken results)

Like the Stage-6 validator, this returns a validation object (issues + passed)
instead of raising, so it can grade arbitrary (even structurally invalid)
results.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable

from agents.scene_planner.models import ProductionPlan
from agents.video_generation.models import (
    RESULT_STATUSES,
    SCENE_STATUS_COMPLETED,
    SCENE_STATUS_FAILED,
    SCENE_STATUSES,
    VideoGenerationConfig,
    _utc_now,
    derive_overall_status,
)

SEVERITY_ERROR = "ERROR"
SEVERITY_WARNING = "WARNING"


@dataclass(frozen=True)
class ValidationIssue:
    """One video-generation validation finding."""

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
class VideoGenerationValidation:
    """Aggregate result of validating a VideoGenerationResult."""

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


# --------------------------------------------------------------------------- #
# Provider-isolation scanning helpers (explicit, unit-testable)
# --------------------------------------------------------------------------- #


def scan_value_terms(text: str, denylist: Iterable[str]) -> tuple[str, ...]:
    """Denylisted terms present in a single text value (lowercased match)."""
    lowered = str(text or "").lower()
    return tuple(term for term in denylist if term in lowered)


def scan_key_terms(key: str, denylist: Iterable[str]) -> tuple[str, ...]:
    """Denylisted terms present in a single dict key (lowercased match)."""
    lowered = str(key or "").lower()
    return tuple(term for term in denylist if term in lowered)


def _leaf_primitives(value: Any):
    """Yield ("key", key, None) at non-container dict keys and
    ("value", context_key_or_"", text) for string leaves.

    Keys are scanned for denylisted vocabulary; string values are scanned too,
    so provider-specific wording cannot hide inside the contract.
    """
    if isinstance(value, dict):
        for key, child in value.items():
            if isinstance(child, (dict, list)):
                yield from _leaf_primitives(child)
            else:
                yield ("key", str(key), None)
                if isinstance(child, str):
                    yield ("value", str(key), child)
    elif isinstance(value, list):
        for child in value:
            if isinstance(child, (dict, list)):
                yield from _leaf_primitives(child)
            else:
                if isinstance(child, str):
                    yield ("value", "", child)


def provider_isolation_hits(payload: dict, denylist: Iterable[str]) -> tuple[str, ...]:
    """Scan a contract payload for provider/API/GPU/cloud coupling.

    Returns human-readable hit descriptions. Empty result means the payload is
    provider-independent. The top-level `provenance` record of an artifact is
    excluded by the caller (it is audit metadata and legitimately names the
    generator abstraction that produced the artifact).
    """
    hits: list[str] = []
    for kind, key, text in _leaf_primitives(payload):
        if kind == "key":
            for term in scan_key_terms(key, denylist):
                hits.append(f"key '{key}' contains '{term}'")
        elif text:
            for term in scan_value_terms(text, denylist):
                hits.append(f"{key or 'value'} contains '{term}'")
    return tuple(hits)


# --------------------------------------------------------------------------- #
# Validator entrypoint
# --------------------------------------------------------------------------- #


def validate_video_generation_result(
    result,
    plan: ProductionPlan,
    config: VideoGenerationConfig | None = None,
) -> VideoGenerationValidation:
    """Validate a Stage-7 VideoGenerationResult against its ProductionPlan.

    `plan` supplies the authoritative scene set, orders, and durations (timing
    preservation). `result` is any object with the VideoGenerationResult
    surface; structural corruption yields issues, never exceptions.
    """
    config = config or VideoGenerationConfig().validate()
    issues: list[ValidationIssue] = []

    def error(rule_id: str, message: str, scene_id: str = ""):
        issues.append(
            ValidationIssue(
                rule_id=rule_id, severity=SEVERITY_ERROR, message=message, scene_id=scene_id
            )
        )

    plan_by_id = {scene.scene_id: scene for scene in plan.scenes}
    result_by_id = {row.scene_id: row for row in result.scenes}

    # g01 -- at least one scene record
    if not result.scenes:
        error("g01", "video generation result contains no scene records")
        return VideoGenerationValidation(issues=tuple(issues))

    # g02 -- unique scene ids
    seen: set[str] = set()
    for row in result.scenes:
        if row.scene_id in seen:
            error("g02", f"duplicate scene_id '{row.scene_id}' in result", row.scene_id)
        seen.add(row.scene_id)

    for row in result.scenes:
        # g03 -- known status value
        if row.status not in SCENE_STATUSES:
            allowed = ", ".join(SCENE_STATUSES)
            error(
                "g03",
                f"scene {row.scene_id} has unknown status "
                f"'{row.status}' (allowed: {allowed})",
                row.scene_id,
            )

        # g04 -- every result scene exists in the plan
        if row.scene_id not in plan_by_id:
            error(
                "g04",
                f"result row references unknown plan scene '{row.scene_id}'",
                row.scene_id,
            )
            continue

        plan_scene = plan_by_id[row.scene_id]

        # g06 -- scene order matches the plan
        if row.order != plan_scene.order:
            error(
                "g06",
                f"scene {row.scene_id} order {row.order} does not match "
                f"plan order {plan_scene.order}",
                row.scene_id,
            )

        # g07 -- request timing preserved
        if (
            abs(row.request.duration_seconds - plan_scene.duration_seconds)
            > config.duration_tolerance_seconds
        ):
            error(
                "g07",
                f"scene {row.scene_id} request duration {row.request.duration_seconds} "
                f"does not match planned scene duration {plan_scene.duration_seconds}",
                row.scene_id,
            )

        # g08 -- completed scenes carry a valid, duration-preserving artifact
        if row.status == SCENE_STATUS_COMPLETED:
            if row.artifact is None:
                error(
                    "g08",
                    f"completed scene {row.scene_id} has no generated artifact",
                    row.scene_id,
                )
            else:
                if not row.artifact.artifact_id.strip():
                    error(
                        "g08",
                        f"completed scene {row.scene_id} artifact has no artifact_id",
                        row.scene_id,
                    )
                if not isinstance(row.artifact.mock, bool):
                    error(
                        "g08",
                        f"completed scene {row.scene_id} artifact mock flag "
                        "must be a bool",
                        row.scene_id,
                    )
                if (
                    abs(row.artifact.duration_seconds - plan_scene.duration_seconds)
                    > config.duration_tolerance_seconds
                ):
                    error(
                        "g08",
                        f"completed scene {row.scene_id} artifact duration "
                        f"{row.artifact.duration_seconds} does not match planned "
                        f"scene duration {plan_scene.duration_seconds}",
                        row.scene_id,
                    )

        # g09 -- failed scenes record a message + retryable flag within limits
        if row.status == SCENE_STATUS_FAILED:
            if not row.failure.strip():
                error(
                    "g09",
                    f"failed scene {row.scene_id} has no failure message",
                    row.scene_id,
                )
            if row.failure_retryable is None:
                error(
                    "g09",
                    f"failed scene {row.scene_id} has no failure_retryable flag",
                    row.scene_id,
                )
            if row.attempts < 1:
                error(
                    "g09",
                    f"failed scene {row.scene_id} must have attempts >= 1",
                    row.scene_id,
                )
            if row.failure_retryable is True and row.attempts >= config.max_attempts:
                error(
                    "g09",
                    f"scene {row.scene_id} is marked retryable at attempts "
                    f"{row.attempts} but the retry limit is "
                    f"{config.max_attempts}; the failure must be recorded as "
                    "non-retryable",
                    row.scene_id,
                )

    # g05 -- every plan scene is represented in the result
    for scene in plan.scenes:
        if scene.scene_id not in result_by_id:
            error(
                "g05",
                f"plan scene '{scene.scene_id}' is missing from the result",
                scene.scene_id,
            )

    # g10 -- overall status consistent with per-scene statuses
    derived = derive_overall_status(result.scenes)
    if result.overall_status != derived:
        error(
            "g10",
            f"overall_status '{result.overall_status}' does not match "
            f"per-scene statuses (derived '{derived}')",
        )

    # g11 -- provenance exists
    if result.provenance is None:
        error("g11", "video generation result provenance is missing")

    # g12 -- provider isolation (the result-level provenance record is audit
    # metadata and is excluded, mirroring the Stage-6 r18 handling)
    payload = dict(result.to_dict()["data"])
    payload.pop("provenance", None)
    for hit in provider_isolation_hits(payload, config.provider_isolation_denylist):
        error("g12", f"provider-specific coupling leaked into the result: {hit}")

    # g13 -- JSON round-trip works (only meaningful for structurally sound results)
    structural = {"g01", "g02", "g03", "g04", "g05", "g06"}
    if not any(issue.rule_id in structural for issue in issues):
        try:
            payload_before = result.to_dict()
            cloned = result.__class__.from_dict(json.loads(result.to_json()))
            if cloned.to_dict() != payload_before:
                error("g13", "JSON round-trip changed the result contents")
        except Exception as exc:  # noqa: BLE001 - a round-trip failure is a validation result
            error("g13", f"JSON round-trip failed: {exc}")

    return VideoGenerationValidation(issues=tuple(issues))