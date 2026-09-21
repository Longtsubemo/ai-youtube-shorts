"""Shared builders/stubs for the Stage-8 (video QA) test suite.

Everything here is synthetic and MOCK - same rule as the Stage-3..7 builders.
The helpers construct a Stage-7 VideoGenerationResult + Stage-6 ProductionPlan
pair and produce exact healthy / missing / failed / pending / ordering /
timing / phantom / provider-leak / mock-truthfulness scenarios offline so the
QA gate can be tested deterministically.

Scenario results are built from a HEALTY run (which carries provenance and a
matching plan) and mutated with `dataclasses.replace`, so each scenario
introduces exactly the defect under test. Malformed cases that cannot survive a
JSON round-trip (e.g. a blank artifact_id) are still buildable as objects for
check-level tests; the CLI covers only scenarios that round-trip.
"""

from __future__ import annotations

import dataclasses

from agents.video_generation.models import (
    VideoGenerationResult,
    derive_overall_status,
    video_generation_result_id_for,
)
from agents.video_generation.mock_generator import MockVideoGenerator
from video_generation_test_utils import FIXED_AT, make_healthy_plan

__all__ = [
    "FIXED_AT",
    "healthy_pair",
    "result_failed_scene",
    "result_pending_scene",
    "result_missing_scene",
    "result_swapped_ordering",
    "result_phantom_scene",
    "result_timing_mismatch",
    "result_malformed_artifact",
    "result_provider_marker",
    "result_non_mock_artifact",
    "result_mock_with_uri",
    "result_media_claim",
    "result_bad_overall_status",
]


def healthy_pair(run_id: str = "video-qa-test", created_at: str = FIXED_AT):
    """Deterministically produce a HEALTHY 7-scene mock generation outcome."""
    plan = make_healthy_plan(run_id=run_id, created_at=created_at)
    result = MockVideoGenerator().generate(plan, run_id=run_id, created_at=created_at)
    return plan, result


def _rebuild(plan, result, rows, overall_status: str = ""):
    """Reconstruct the result object (no validate) around the given rows."""
    status = overall_status or derive_overall_status(rows)
    return VideoGenerationResult(
        video_generation_result_id=result.video_generation_result_id,
        production_plan_id=result.production_plan_id,
        topic_id=result.topic_id,
        topic_title=result.topic_title,
        script_id=result.script_id,
        script_qa_result_id=result.script_qa_result_id,
        scenes=tuple(rows),
        overall_status=status,
        resumed_from_result_id=result.resumed_from_result_id,
        provenance=result.provenance,
        created_at=result.created_at,
        run_id=result.run_id,
    )


def _replace_row(rows, index: int, **changes):
    rows = list(rows)
    rows[index] = dataclasses.replace(rows[index], **changes)
    return rows


def result_failed_scene(plan, result, index: int = 1):
    rows = _replace_row(
        result.scenes,
        index,
        status="failed",
        artifact=None,
        failure="simulated permanent provider failure",
        failure_retryable=False,
    )
    return _rebuild(plan, result, rows)


def result_pending_scene(plan, result, index: int = 1):
    rows = _replace_row(
        result.scenes,
        index,
        status="pending",
        artifact=None,
        attempts=0,
    )
    return _rebuild(plan, result, rows)


def result_missing_scene(plan, result, index: int = 0):
    """Drop one scene and renumber the rest (so the result still round-trips)."""
    rows = list(result.scenes)
    del rows[index]
    rows = [
        dataclasses.replace(row, order=position)
        for position, row in enumerate(rows, start=1)
    ]
    return _rebuild(plan, result, rows)


def result_swapped_ordering(plan, result):
    """Swap the first two rows and renumber (contiguous, but order-mismatched
    vs the ProductionPlan, which still plans scn_01 then scn_02)."""
    rows = list(result.scenes)
    rows[0], rows[1] = rows[1], rows[0]
    rows = [
        dataclasses.replace(row, order=position)
        for position, row in enumerate(rows, start=1)
    ]
    return _rebuild(plan, result, rows)


def result_phantom_scene(plan, result):
    rows = list(result.scenes)
    source = rows[0]
    ghost = dataclasses.replace(
        source,
        scene_id="scn_99",
        order=len(rows) + 1,
        artifact=dataclasses.replace(source.artifact, artifact_id="mockva_00000000"),
    )
    return _rebuild(plan, result, [*rows, ghost])


def result_timing_mismatch(plan, result, index: int = 1, delta: float = 5.0):
    rows = list(result.scenes)
    row = rows[index]
    rows[index] = dataclasses.replace(
        row,
        request=dataclasses.replace(row.request, duration_seconds=row.request.duration_seconds + delta),
        artifact=dataclasses.replace(row.artifact, duration_seconds=row.artifact.duration_seconds + delta),
    )
    return _rebuild(plan, result, rows)


def result_malformed_artifact(plan, result, index: int = 1):
    rows = list(result.scenes)
    rows[index] = dataclasses.replace(
        rows[index], artifact=dataclasses.replace(rows[index].artifact, artifact_id="")
    )
    return _rebuild(plan, result, rows)


def result_provider_marker(plan, result, index: int = 1):
    rows = list(result.scenes)
    rows[index] = dataclasses.replace(
        rows[index],
        artifact=dataclasses.replace(
            rows[index].artifact, note="rendered with a runway-style API endpoint"
        ),
    )
    return _rebuild(plan, result, rows)


def result_non_mock_artifact(plan, result, index: int = 1):
    rows = list(result.scenes)
    rows[index] = dataclasses.replace(
        rows[index],
        artifact=dataclasses.replace(
            rows[index].artifact, mock=False, uri="video.mp4", note="real rendered asset"
        ),
    )
    return _rebuild(plan, result, rows)


def result_mock_with_uri(plan, result, index: int = 1):
    rows = list(result.scenes)
    rows[index] = dataclasses.replace(
        rows[index],
        artifact=dataclasses.replace(rows[index].artifact, uri="file:///tmp/short.mp4"),
    )
    return _rebuild(plan, result, rows)


def result_media_claim(plan, result, index: int = 1):
    rows = list(result.scenes)
    rows[index] = dataclasses.replace(
        rows[index],
        artifact=dataclasses.replace(
            rows[index].artifact, note="clip was uploaded to youtube already"
        ),
    )
    return _rebuild(plan, result, rows)


def result_bad_overall_status(plan, result):
    return _rebuild(plan, result, result.scenes, overall_status="banana")


def result_scaled_duration(plan, result, target_total: float = 70.0):
    """Scale every scene duration so the recorded total hits `target_total`.
    Round-trips cleanly (request and artifact durations stay consistent)."""
    rows = list(result.scenes)
    current = sum(row.request.duration_seconds for row in rows)
    factor = target_total / current
    rows = [
        dataclasses.replace(
            row,
            request=dataclasses.replace(
                row.request, duration_seconds=round(row.request.duration_seconds * factor, 3)
            ),
            artifact=dataclasses.replace(
                row.artifact, duration_seconds=round(row.artifact.duration_seconds * factor, 3)
            ),
        )
        for row in rows
    ]
    return _rebuild(plan, result, rows)