"""Modular, deterministic Stage-8 video-QA checks (15 rules).

Every check is a pure function over (result, plan, config) -> list[VideoQAIssue]
where `result` is a Stage-7 VideoGenerationResult and `plan` is the Stage-6
ProductionPlan it was generated from. Nothing here talks to the network, to an
LLM, to a GPU, or to a video provider: all judgements are deterministic over
the result records and the plan metadata (docs/VIDEO_PIPELINE.md §6).

Because no assembled media bytes exist in the mock-first pipeline, checks are
either metadata/contract-level (provable from the Stage-7 result + plan) or
truthfulness checks (nothing is ever presented as a real uploaded video). A
future real provider must revisit the two mock/truthfulness checks when real
artifacts exist - that is documented in the registry notes.

Checks are registered in `CHECKS` so future rules can be added independently:
write a function with the same signature and append it to the registry.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from agents.video_generation.models import RESULT_STATUSES
from agents.video_generation.validator import (
    provider_isolation_hits,
    validate_video_generation_result,
)
from agents.qa.video.models import (
    VideoQAConfig,
    VideoQAIssue,
    VideoQACheckRecord,
)

GATE_HARD = "hard"
GATE_SOFT = "soft"

# Terms that would assert a real, uploaded/published video exists. The current
# pipeline produces NO media bytes and NO uploads, so any occurrence inside the
# Stage-7 result records is a false claim (truthfulness violation).
_REAL_MEDIA_CLAIMS = (
    "uploaded",
    "publishing",
    "published",
    "youtube",
    "youtu.be",
    "live now",
    "video url",
    "video_id",
)

SCENE_STATUS_COMPLETED = "completed"
SCENE_STATUS_FAILED = "failed"
SCENE_STATUS_PENDING = "pending"


def _rows(result) -> tuple:
    return tuple(getattr(result, "scenes", ()) or ())


def _plan_scenes(plan) -> tuple:
    return tuple(getattr(plan, "scenes", ()) or ())


def _artifact(row):
    return getattr(row, "artifact", None)


def _issue(
    category: str,
    severity: str,
    message: str,
    *,
    scene_ids=(),
    artifact_ids=(),
    gate: str = GATE_SOFT,
    check: str = "",
) -> VideoQAIssue:
    return VideoQAIssue(
        category=category,
        severity=severity,
        message=message,
        affected_scene_ids=tuple(scene_ids),
        affected_artifact_ids=tuple(artifact_ids),
        gate=gate,
        check=check,
    ).validate()


# --------------------------------------------------------------------------- #
# 1. Duration window
# --------------------------------------------------------------------------- #


def check_duration_window(result, plan, config: VideoQAConfig) -> list[VideoQAIssue]:
    """Total rendered duration (from the per-scene record timings) sits in
    the 30-60s Shorts window. Metadata/contract level: the timing is the
    record's declared duration, not a measurement of a real file."""
    rows = _rows(result)
    durations = [
        getattr(getattr(row, "request", None), "duration_seconds", 0.0) for row in rows
    ]
    if not durations:
        return []  # structure_integrity reports an empty result
    total = sum(d for d in durations if isinstance(d, (int, float)) and d > 0)
    if total < config.hard_min_duration_seconds:
        return [
            _issue(
                "duration_window",
                "ERROR",
                f"recorded total duration {total:.1f}s is below the usable "
                f"minimum of {config.hard_min_duration_seconds:.0f}s",
                gate=GATE_HARD,
                check="duration_window",
            )
        ]
    if total > config.hard_max_duration_seconds:
        return [
            _issue(
                "duration_window",
                "ERROR",
                f"recorded total duration {total:.1f}s clearly exceeds the "
                f"usable Short-form maximum of {config.hard_max_duration_seconds:.0f}s",
                gate=GATE_HARD,
                check="duration_window",
            )
        ]
    if total < config.min_duration_seconds:
        return [
            _issue(
                "duration_window",
                "ERROR",
                f"recorded total duration {total:.1f}s is below the preferred "
                f"{config.min_duration_seconds:.0f}s Shorts window",
                gate=GATE_SOFT,
                check="duration_window",
            )
        ]
    if total > config.max_duration_seconds:
        return [
            _issue(
                "duration_window",
                "ERROR",
                f"recorded total duration {total:.1f}s exceeds the preferred "
                f"{config.max_duration_seconds:.0f}s Shorts window",
                gate=GATE_SOFT,
                check="duration_window",
            )
        ]
    return []


# --------------------------------------------------------------------------- #
# 2. Aspect-ratio metadata
# --------------------------------------------------------------------------- #


def check_aspect_ratio_metadata(result, plan, config: VideoQAConfig) -> list[VideoQAIssue]:
    """The deliverable must be a vertical 9:16 Short. Metadata/contract level:
    this verifies the ProductionPlan's declared global style; no real frame is
    inspected yet."""
    issues: list[VideoQAIssue] = []
    style = getattr(plan, "global_style", None)
    if style is None:
        issues.append(
            _issue(
                "aspect_ratio_metadata",
                "ERROR",
                "ProductionPlan carries no global_style metadata; the target "
                "format cannot be verified",
                gate=GATE_HARD,
                check="aspect_ratio_metadata",
            )
        )
        return issues
    aspect = getattr(style, "aspect_ratio", "") or ""
    if aspect != config.expected_aspect_ratio:
        issues.append(
            _issue(
                "aspect_ratio_metadata",
                "ERROR",
                f"declared aspect_ratio {aspect!r} is not the required vertical "
                f"Short format {config.expected_aspect_ratio!r}",
                gate=GATE_HARD,
                check="aspect_ratio_metadata",
            )
        )
    resolution = getattr(style, "target_resolution", "") or ""
    if resolution and resolution != config.expected_resolution:
        issues.append(
            _issue(
                "aspect_ratio_metadata",
                "ERROR",
                f"declared target_resolution {resolution!r} differs from the "
                f"expected {config.expected_resolution!r}",
                gate=GATE_SOFT,
                check="aspect_ratio_metadata",
            )
        )
    fps = getattr(style, "fps", 0) or 0
    if fps and fps != config.expected_fps:
        issues.append(
            _issue(
                "aspect_ratio_metadata",
                "ERROR",
                f"declared fps {fps} differs from the expected {config.expected_fps}",
                gate=GATE_SOFT,
                check="aspect_ratio_metadata",
            )
        )
    return issues


# --------------------------------------------------------------------------- #
# 3. Scene coverage (all plan scenes have results; none missing)
# --------------------------------------------------------------------------- #


def check_scene_coverage(result, plan, config: VideoQAConfig) -> list[VideoQAIssue]:
    """Every ProductionPlan scene must be represented in the result; a missing
    scene means the video is incomplete."""
    issues: list[VideoQAIssue] = []
    result_ids = {row.scene_id for row in _rows(result)}
    for scene in _plan_scenes(plan):
        if scene.scene_id not in result_ids:
            issues.append(
                _issue(
                    "scene_coverage",
                    "ERROR",
                    f"plan scene {scene.scene_id} is missing from the "
                    "VideoGenerationResult",
                    scene_ids=[scene.scene_id],
                    gate=GATE_HARD,
                    check="scene_coverage",
                )
            )
    return issues


# --------------------------------------------------------------------------- #
# 4. No pending scenes
# --------------------------------------------------------------------------- #


def check_scene_pending(result, plan, config: VideoQAConfig) -> list[VideoQAIssue]:
    """A finished video must have zero pending scenes: pending means the
    generation never completed."""
    issues: list[VideoQAIssue] = []
    for row in _rows(result):
        if getattr(row, "status", "") == SCENE_STATUS_PENDING:
            issues.append(
                _issue(
                    "scene_pending",
                    "ERROR",
                    f"scene {row.scene_id} is still pending; the video did not "
                    "complete generation",
                    scene_ids=[row.scene_id],
                    gate=GATE_HARD,
                    check="scene_pending",
                )
            )
    return issues


# --------------------------------------------------------------------------- #
# 5. No failed scenes
# --------------------------------------------------------------------------- #


def check_scene_failures(result, plan, config: VideoQAConfig) -> list[VideoQAIssue]:
    """A finished video must have zero failed scenes: failure means that scene
    never produced an artifact."""
    issues: list[VideoQAIssue] = []
    for row in _rows(result):
        if getattr(row, "status", "") == SCENE_STATUS_FAILED:
            message = getattr(row, "failure", "") or "no failure message"
            issues.append(
                _issue(
                    "scene_failures",
                    "ERROR",
                    f"scene {row.scene_id} failed generation: {message}",
                    scene_ids=[row.scene_id],
                    gate=GATE_HARD,
                    check="scene_failures",
                )
            )
    return issues


# --------------------------------------------------------------------------- #
# 6. Scene ordering
# --------------------------------------------------------------------------- #


def check_scene_ordering(result, plan, config: VideoQAConfig) -> list[VideoQAIssue]:
    """Result scene orders must match the ProductionPlan orders, so the final
    video follows the intended story sequence."""
    issues: list[VideoQAIssue] = []
    plan_order = {scene.scene_id: scene.order for scene in _plan_scenes(plan)}
    for row in _rows(result):
        expected = plan_order.get(row.scene_id)
        if expected is not None and getattr(row, "order", None) != expected:
            issues.append(
                _issue(
                    "scene_ordering",
                    "ERROR",
                    f"scene {row.scene_id} order {row.order} does not match the "
                    f"ProductionPlan order {expected}",
                    scene_ids=[row.scene_id],
                    gate=GATE_HARD,
                    check="scene_ordering",
                )
            )
    return issues


# --------------------------------------------------------------------------- #
# 7. Timing fidelity
# --------------------------------------------------------------------------- #


def check_timing_fidelity(result, plan, config: VideoQAConfig) -> list[VideoQAIssue]:
    """Declared scene timing (request and artifact) must match the ProductionPlan
    timing within tolerance; drifting timing breaks narration sync."""
    issues: list[VideoQAIssue] = []
    plan_duration = {scene.scene_id: scene.duration_seconds for scene in _plan_scenes(plan)}
    for row in _rows(result):
        expected = plan_duration.get(row.scene_id)
        if expected is None:
            continue
        # Skipping == false when the attribute is genuinely missing; this check's
        # scope is timing only (structure/reference checks own other classes of
        # breakage).
        request = getattr(row, "request", None)
        request_duration = getattr(request, "duration_seconds", None)
        if request_duration is not None and abs(request_duration - expected) > config.duration_tolerance_seconds:
            issues.append(
                _issue(
                    "timing_fidelity",
                    "ERROR",
                    f"scene {row.scene_id} request duration {request_duration} "
                    f"does not match the planned {expected}",
                    scene_ids=[row.scene_id],
                    gate=GATE_HARD,
                    check="timing_fidelity",
                )
            )
        artifact = _artifact(row)
        artifact_duration = getattr(artifact, "duration_seconds", None)
        if artifact_duration is not None and abs(artifact_duration - expected) > config.duration_tolerance_seconds:
            issues.append(
                _issue(
                    "timing_fidelity",
                    "ERROR",
                    f"scene {row.scene_id} artifact duration {artifact_duration} "
                    f"does not match the planned {expected}",
                    scene_ids=[row.scene_id],
                    artifact_ids=[getattr(artifact, "artifact_id", "")],
                    gate=GATE_HARD,
                    check="timing_fidelity",
                )
            )
    return issues


# --------------------------------------------------------------------------- #
# 8. Artifact assignment
# --------------------------------------------------------------------------- #


def check_artifact_assignment(result, plan, config: VideoQAConfig) -> list[VideoQAIssue]:
    """Completed scenes carry exactly one well-formed artifact for their scene;
    artifact ids are unique. Non-completed scenes must not carry artifacts."""
    issues: list[VideoQAIssue] = []
    seen_ids: set[str] = set()
    for row in _rows(result):
        status = getattr(row, "status", "")
        artifact = _artifact(row)
        if status == SCENE_STATUS_COMPLETED:
            if artifact is None:
                issues.append(
                    _issue(
                        "artifact_assignment",
                        "ERROR",
                        f"completed scene {row.scene_id} has no generated artifact",
                        scene_ids=[row.scene_id],
                        gate=GATE_HARD,
                        check="artifact_assignment",
                    )
                )
                continue
            artifact_id = getattr(artifact, "artifact_id", "") or ""
            if not artifact_id.strip():
                issues.append(
                    _issue(
                        "artifact_assignment",
                        "ERROR",
                        f"completed scene {row.scene_id} artifact has no artifact_id",
                        scene_ids=[row.scene_id],
                        gate=GATE_HARD,
                        check="artifact_assignment",
                    )
                )
            elif artifact_id in seen_ids:
                issues.append(
                    _issue(
                        "artifact_assignment",
                        "ERROR",
                        f"artifact_id {artifact_id!r} is reused across scenes",
                        scene_ids=[row.scene_id],
                        artifact_ids=[artifact_id],
                        gate=GATE_HARD,
                        check="artifact_assignment",
                    )
                )
            else:
                seen_ids.add(artifact_id)
            if not isinstance(getattr(artifact, "mock", None), bool):
                issues.append(
                    _issue(
                        "artifact_assignment",
                        "ERROR",
                        f"scene {row.scene_id} artifact mock flag must be a bool",
                        scene_ids=[row.scene_id],
                        artifact_ids=[artifact_id or ""],
                        gate=GATE_HARD,
                        check="artifact_assignment",
                    )
                )
            request = getattr(row, "request", None)
            request_duration = getattr(request, "duration_seconds", None)
            artifact_duration = getattr(artifact, "duration_seconds", None)
            if (
                request_duration is not None
                and artifact_duration is not None
                and abs(artifact_duration - request_duration) > config.duration_tolerance_seconds
            ):
                issues.append(
                    _issue(
                        "artifact_assignment",
                        "ERROR",
                        f"scene {row.scene_id} artifact duration {artifact_duration} "
                        f"does not match its request duration {request_duration}",
                        scene_ids=[row.scene_id],
                        artifact_ids=[artifact_id or ""],
                        gate=GATE_HARD,
                        check="artifact_assignment",
                    )
                )
        elif artifact is not None:
            issues.append(
                _issue(
                    "artifact_assignment",
                    "ERROR",
                    f"scene {row.scene_id} is not completed but carries an artifact",
                    scene_ids=[row.scene_id],
                    artifact_ids=[getattr(artifact, "artifact_id", "") or ""],
                    gate=GATE_HARD,
                    check="artifact_assignment",
                )
            )
    return issues


# --------------------------------------------------------------------------- #
# 9. Structure integrity
# --------------------------------------------------------------------------- #


def check_structure_integrity(result, plan, config: VideoQAConfig) -> list[VideoQAIssue]:
    """The result record itself must be structurally usable: non-empty, unique
    scene ids, contiguous orders, known statuses, and well-formed rows."""
    issues: list[VideoQAIssue] = []
    rows = _rows(result)
    if not rows:
        issues.append(
            _issue(
                "structure_integrity",
                "ERROR",
                "VideoGenerationResult contains no scene records",
                gate=GATE_HARD,
                check="structure_integrity",
            )
        )
        return issues
    scene_ids = [row.scene_id for row in rows]
    if len(set(scene_ids)) != len(scene_ids):
        issues.append(
            _issue(
                "structure_integrity",
                "ERROR",
                "duplicate scene_id in the VideoGenerationResult",
                gate=GATE_HARD,
                check="structure_integrity",
            )
        )
    orders = [getattr(row, "order", None) for row in rows]
    if orders != list(range(1, len(rows) + 1)):
        issues.append(
            _issue(
                "structure_integrity",
                "ERROR",
                "scene orders must be the contiguous sequence 1..N",
                gate=GATE_HARD,
                check="structure_integrity",
            )
        )
    for row in rows:
        if getattr(row, "request", None) is None:
            issues.append(
                _issue(
                    "structure_integrity",
                    "ERROR",
                    f"scene {row.scene_id} carries no generation request",
                    scene_ids=[row.scene_id],
                    gate=GATE_HARD,
                    check="structure_integrity",
                )
            )
        status = getattr(row, "status", "")
        if status not in (SCENE_STATUS_COMPLETED, SCENE_STATUS_FAILED, SCENE_STATUS_PENDING):
            issues.append(
                _issue(
                    "structure_integrity",
                    "ERROR",
                    f"scene {row.scene_id} has unknown status {status!r}",
                    scene_ids=[row.scene_id],
                    gate=GATE_HARD,
                    check="structure_integrity",
                )
            )
    return issues


# --------------------------------------------------------------------------- #
# 10. Reference integrity
# --------------------------------------------------------------------------- #


def check_reference_integrity(result, plan, config: VideoQAConfig) -> list[VideoQAIssue]:
    """No unexpected, phantom, blank, or malformed scene references: every
    result row must point at a real ProductionPlan scene."""
    issues: list[VideoQAIssue] = []
    plan_ids = {scene.scene_id for scene in _plan_scenes(plan)}
    for row in _rows(result):
        scene_id = getattr(row, "scene_id", None)
        if scene_id is None or (isinstance(scene_id, str) and not scene_id.strip()):
            issues.append(
                _issue(
                    "reference_integrity",
                    "ERROR",
                    "result row carries a blank scene_id",
                    gate=GATE_HARD,
                    check="reference_integrity",
                )
            )
        elif scene_id not in plan_ids:
            issues.append(
                _issue(
                    "reference_integrity",
                    "ERROR",
                    f"result row references unknown plan scene {scene_id!r}",
                    scene_ids=[scene_id],
                    gate=GATE_HARD,
                    check="reference_integrity",
                )
            )
    return issues


# --------------------------------------------------------------------------- #
# 11. Provider isolation
# --------------------------------------------------------------------------- #


def check_provider_isolation(result, plan, config: VideoQAConfig) -> list[VideoQAIssue]:
    """The result contract must not leak API providers, model/vendor names, API
    keys, GPU/CUDA requirements, or cloud-vendor details. The top-level
    provenance record is audit metadata and is excluded (it legitimately names
    the local generator abstraction)."""
    issues: list[VideoQAIssue] = []
    try:
        payload = dict(result.to_dict()["data"])
    except Exception as exc:  # noqa: BLE001 - corrupt results still get graded
        return [
            _issue(
                "provider_isolation",
                "ERROR",
                f"result could not be serialized for isolation scanning: {exc}",
                gate=GATE_HARD,
                check="provider_isolation",
            )
        ]
    payload.pop("provenance", None)
    for hit in provider_isolation_hits(payload, config.provider_isolation_denylist):
        issues.append(
            _issue(
                "provider_isolation",
                "ERROR",
                f"provider-specific coupling leaked into the result: {hit}",
                gate=GATE_HARD,
                check="provider_isolation",
            )
        )
    return issues


# --------------------------------------------------------------------------- #
# 12. Generation validation
# --------------------------------------------------------------------------- #


def check_generation_validation(result, plan, config: VideoQAConfig) -> list[VideoQAIssue]:
    """The VideoGenerationResult must pass its own Stage-7 validation against
    the ProductionPlan (g01..g13). QA consumes that gate instead of re-inventing
    it; a result the generator's own validator rejects cannot be passed."""
    issues: list[VideoQAIssue] = []
    from agents.video_generation.models import VideoGenerationConfig

    try:
        validation = validate_video_generation_result(
            result,
            plan,
            config=VideoGenerationConfig(
                duration_tolerance_seconds=config.duration_tolerance_seconds
            ),
        )
    except Exception as exc:  # noqa: BLE001 - grading must never crash
        return [
            _issue(
                "generation_validation",
                "ERROR",
                f"Stage-7 generation validation could not run: {exc}",
                gate=GATE_HARD,
                check="generation_validation",
            )
        ]
    for issue in validation.errors:
        message = f"Stage-7 validation [{issue.rule_id}]: {issue.message}"
        if issue.scene_id:
            message += f" (scene {issue.scene_id})"
        issues.append(
            _issue(
                "generation_validation",
                "ERROR",
                message,
                scene_ids=[issue.scene_id] if issue.scene_id else (),
                gate=GATE_HARD,
                check="generation_validation",
            )
        )
    return issues


# --------------------------------------------------------------------------- #
# 13. Overall status consistency
# --------------------------------------------------------------------------- #


def check_overall_status_consistency(result, plan, config: VideoQAConfig) -> list[VideoQAIssue]:
    """The whole-result status must be a known value AND must match the
    per-scene statuses; an impossible/inconsistent status cannot be trusted."""
    issues: list[VideoQAIssue] = []
    from agents.video_generation.models import derive_overall_status

    overall = getattr(result, "overall_status", "")
    if overall not in RESULT_STATUSES:
        issues.append(
            _issue(
                "overall_status_consistency",
                "ERROR",
                f"overall_status {overall!r} is not a known result status",
                gate=GATE_HARD,
                check="overall_status_consistency",
            )
        )
        return issues
    try:
        derived = derive_overall_status(_rows(result))
    except Exception as exc:  # noqa: BLE001 - grading must not crash
        return [
            _issue(
                "overall_status_consistency",
                "ERROR",
                f"overall status could not be derived: {exc}",
                gate=GATE_HARD,
                check="overall_status_consistency",
            )
        ]
    if overall != derived:
        issues.append(
            _issue(
                "overall_status_consistency",
                "ERROR",
                f"overall_status {overall!r} does not match the per-scene "
                f"statuses (derived {derived!r})",
                gate=GATE_HARD,
                check="overall_status_consistency",
            )
        )
    return issues


# --------------------------------------------------------------------------- #
# 14. Mock recognition
# --------------------------------------------------------------------------- #


def check_mock_recognition(result, plan, config: VideoQAConfig) -> list[VideoQAIssue]:
    """Every artifact must be EXPLICITLY flagged as a mock stand-in. The current
    pipeline has no real video provider, so an artifact that is not marked mock
    (or carries a URI like a real file) is misleading. NOTE: this check must be
    revisited when a real provider starts producing genuine assets."""
    issues: list[VideoQAIssue] = []
    for row in _rows(result):
        if getattr(row, "status", "") != SCENE_STATUS_COMPLETED:
            continue
        artifact = _artifact(row)
        if artifact is None:
            continue  # artifact_assignment reports the missing artifact
        artifact_id = getattr(artifact, "artifact_id", "") or ""
        mock = getattr(artifact, "mock", None)
        uri = getattr(artifact, "uri", "") or ""
        if mock is not True:
            issues.append(
                _issue(
                    "mock_recognition",
                    "ERROR",
                    f"artifact {artifact_id} for scene {row.scene_id} is not "
                    "explicitly flagged as a mock stand-in (mock=False/absent); "
                    "the current pipeline has no real video provider so this "
                    "record cannot be a genuine asset",
                    scene_ids=[row.scene_id],
                    artifact_ids=[artifact_id],
                    gate=GATE_HARD,
                    check="mock_recognition",
                )
            )
        elif uri:
            issues.append(
                _issue(
                    "mock_recognition",
                    "ERROR",
                    f"mock artifact {artifact_id} for scene {row.scene_id} "
                    "carries a URI as if a real file existed",
                    scene_ids=[row.scene_id],
                    artifact_ids=[artifact_id],
                    gate=GATE_SOFT,
                    check="mock_recognition",
                )
            )
    return issues


# --------------------------------------------------------------------------- #
# 15. Media honesty
# --------------------------------------------------------------------------- #


def check_media_honesty(result, plan, config: VideoQAConfig) -> list[VideoQAIssue]:
    """No Stage-7 record may present the deliverable as a real, uploaded, or
    published video. Nothing in the current pipeline uploads or publishes;
    such a claim would be a fabricated assertion."""
    issues: list[VideoQAIssue] = []
    for row in _rows(result):
        artifact = _artifact(row)
        if artifact is None:
            continue
        artifact_id = getattr(artifact, "artifact_id", "") or ""
        for field_name in ("note", "uri", "provider", "subject"):
            value = str(getattr(artifact, field_name, "") or "")
            for marker in _REAL_MEDIA_CLAIMS:
                if marker in value.lower():
                    issues.append(
                        _issue(
                            "media_honesty",
                            "ERROR",
                            f"artifact {artifact_id} ({field_name}) claims real "
                            f"media with {marker!r}; the pipeline has not "
                            "uploaded or published anything",
                            scene_ids=[row.scene_id],
                            artifact_ids=[artifact_id],
                            gate=GATE_HARD,
                            check="media_honesty",
                        )
                    )
                    break
    return issues


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class VideoQACheck:
    """One registered rule. `hard_gate` is informational (issues carry their
    own per-issue gate); it documents which checks default to fatal. `note`
    documents the check for reviewers, including metadata/contract-level
    checks that cannot inspect media that does not exist yet."""

    name: str
    category: str
    hard_gate: bool
    note: str
    fn: Callable[[object, object, VideoQAConfig], list[VideoQAIssue]]


CHECKS: tuple[VideoQACheck, ...] = (
    VideoQACheck(
        "duration_window",
        "duration_window",
        True,
        "Recorded total duration is within the 30-60s Shorts window. "
        "Metadata/contract-level: timing comes from the Stage-7 records, not a "
        "real file measurement.",
        check_duration_window,
    ),
    VideoQACheck(
        "aspect_ratio_metadata",
        "aspect_ratio_metadata",
        True,
        "Deliverable is a vertical 9:16 Short. Metadata/contract-level: "
        "verified from the ProductionPlan global style; no real frame exists "
        "to inspect yet.",
        check_aspect_ratio_metadata,
    ),
    VideoQACheck(
        "scene_coverage",
        "scene_coverage",
        True,
        "Every ProductionPlan scene is represented in the result (none missing).",
        check_scene_coverage,
    ),
    VideoQACheck(
        "scene_pending",
        "scene_pending",
        True,
        "No scene is left pending.",
        check_scene_pending,
    ),
    VideoQACheck(
        "scene_failures",
        "scene_failures",
        True,
        "No scene failed generation.",
        check_scene_failures,
    ),
    VideoQACheck(
        "scene_ordering",
        "scene_ordering",
        True,
        "Result scene order matches the ProductionPlan order.",
        check_scene_ordering,
    ),
    VideoQACheck(
        "timing_fidelity",
        "timing_fidelity",
        True,
        "Declared scene timings match the ProductionPlan within tolerance.",
        check_timing_fidelity,
    ),
    VideoQACheck(
        "artifact_assignment",
        "artifact_assignment",
        True,
        "Completed scenes carry one unique, well-formed artifact; "
        "non-completed scenes carry none.",
        check_artifact_assignment,
    ),
    VideoQACheck(
        "structure_integrity",
        "structure_integrity",
        True,
        "The result record is structurally usable (non-empty, unique scene "
        "ids, contiguous orders, known statuses).",
        check_structure_integrity,
    ),
    VideoQACheck(
        "reference_integrity",
        "reference_integrity",
        True,
        "No unexpected, blank, or phantom scene references.",
        check_reference_integrity,
    ),
    VideoQACheck(
        "provider_isolation",
        "provider_isolation",
        True,
        "No provider, model-vendor, credential, or accelerator-vendor details "
        "leaked into the result contract.",
        check_provider_isolation,
    ),
    VideoQACheck(
        "generation_validation",
        "generation_validation",
        True,
        "The VideoGenerationResult passed its own Stage-7 generation "
        "validation against the ProductionPlan.",
        check_generation_validation,
    ),
    VideoQACheck(
        "overall_status_consistency",
        "overall_status_consistency",
        True,
        "Overall status is a known value and matches the per-scene statuses.",
        check_overall_status_consistency,
    ),
    VideoQACheck(
        "mock_recognition",
        "mock_recognition",
        True,
        "Every artifact is explicitly flagged as a mock stand-in. Revisit "
        "when a real provider produces genuine assets.",
        check_mock_recognition,
    ),
    VideoQACheck(
        "media_honesty",
        "media_honesty",
        True,
        "Nothing is presented as a real, uploaded, or published video.",
        check_media_honesty,
    ),
)


def run_checks(
    result,
    plan,
    config: VideoQAConfig,
) -> tuple[list[VideoQAIssue], list[VideoQACheckRecord]]:
    """Run every registered check in order; deterministic issue order.

    The overall issue id numbering is assigned later by the validator so ids
    stay stable (vqa_00, vqa_01, ...) regardless of content.
    """
    issues: list[VideoQAIssue] = []
    records: list[VideoQACheckRecord] = []
    for check in CHECKS:
        found = check.fn(result, plan, config) or []
        issues.extend(found)
        errors = [issue for issue in found if issue.severity == "ERROR"]
        records.append(
            VideoQACheckRecord(
                check=check.name,
                category=check.category,
                hard_gate=check.hard_gate,
                errors=len(errors),
                hard_errors=sum(1 for issue in errors if issue.gate == GATE_HARD),
                warnings=sum(1 for issue in found if issue.severity == "WARNING"),
                infos=sum(1 for issue in found if issue.severity == "INFO"),
                note=check.note,
            )
        )
    return issues, records