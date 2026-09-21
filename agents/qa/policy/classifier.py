"""Bounded inputs for a future policy-risk classifier (LLM/provider boundary).

Stage 9 is deterministic and runs with ``classifier=None``; this module exists
so a future provider-based policy overview can plug in WITHOUT changing the
deterministic gate. To keep that true:

* ``PolicyRiskContext`` exposes only a fixed, bounded summary of the upstream
  artifacts -- arbitrary artifact access is impossible through it.
* ``PolicyClassifierSuggestion`` is advisory. ``evaluate_policy`` may only use
  it to make the verdict MORE conservative; a suggestion can never turn a
  deterministic PASS/REVIEW/REJECT into a different PASS/REVIEW, and a
  deterministic hard REJECT stays a REJECT regardless of any suggestion.

No network, no model, no keys/libraries are required. The default pipeline
does not invoke a classifier at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from agents.qa.policy.models import POLICY_VERDICTS, PolicyInputs, PolicyQAConfig, PolicyQAError

#: Order used when combining a deterministic verdict with a classifier
#: suggestion: the more conservative value always wins.
POLICY_VERDICT_RANK = {"PASS": 0, "REVIEW": 1, "REJECT": 2}

#: Cap on the free-text summary exposed to a classifier (bounded inputs).
CONTEXT_TEXT_LIMIT = 8000


@dataclass(frozen=True)
class PolicyClassifierSuggestion:
    """Advisory verdict from a provider classifier (never authoritative)."""

    suggested_verdict: str
    model_name: str
    notes: str = ""

    def validate(self) -> "PolicyClassifierSuggestion":
        if self.suggested_verdict not in POLICY_VERDICTS:
            raise PolicyQAError(
                f"suggested_verdict must be one of {POLICY_VERDICTS}, "
                f"got {self.suggested_verdict!r}"
            )
        return self


@dataclass(frozen=True)
class PolicyRiskContext:
    """The bounded, fixed-by-contract view a classifier is allowed to see."""

    policy_version: str
    title: str
    hook: str
    line_count: int
    scene_count: int
    total_estimated_duration_seconds: float
    has_fact_check: bool
    fact_check_verdict: str
    has_video_result: bool
    video_provider: str
    video_qa_verdict: str
    generated_scene_ids: tuple[str, ...] = ()
    external_asset_ids: tuple[str, ...] = ()
    presentation_summary: str = ""

    def to_dict(self) -> dict:
        return {
            "policy_version": self.policy_version,
            "title": self.title,
            "hook": self.hook,
            "line_count": self.line_count,
            "scene_count": self.scene_count,
            "total_estimated_duration_seconds": self.total_estimated_duration_seconds,
            "has_fact_check": self.has_fact_check,
            "fact_check_verdict": self.fact_check_verdict,
            "has_video_result": self.has_video_result,
            "video_provider": self.video_provider,
            "video_qa_verdict": self.video_qa_verdict,
            "generated_scene_ids": list(self.generated_scene_ids),
            "external_asset_ids": list(self.external_asset_ids),
            "presentation_summary": self.presentation_summary,
        }


@runtime_checkable
class PolicyRiskClassifier(Protocol):
    """A provider that turns a bounded context into an advisory verdict."""

    def classify(self, context: PolicyRiskContext) -> PolicyClassifierSuggestion:
        """Return an advisory suggestion (never overrides the gate)."""


def build_policy_context(inputs: PolicyInputs) -> PolicyRiskContext:
    """Build the bounded context from the gate's inputs (no external calls)."""
    script = inputs.script
    plan = inputs.plan
    fact_check = inputs.fact_check
    video_result = inputs.video_result

    lines = tuple(getattr(script, "lines", ()) or ()) if script is not None else ()
    scenes = tuple(getattr(plan, "scenes", ()) or ()) if plan is not None else ()
    summary = " ".join(item.text for item in inputs.presentation_items())
    summary = summary[:CONTEXT_TEXT_LIMIT]

    generated_scene_ids = tuple(
        getattr(scene, "scene_id", "")
        for scene in scenes
        if (getattr(scene, "visual_type", "") or "") in ("generated_video", "generated_image", "motion_graphic")
    )
    external_asset_ids = tuple(
        f"{getattr(scene, 'scene_id', '')}:{getattr(asset, 'asset_type', '')}"
        for scene in scenes
        for asset in getattr(scene, "asset_requirements", ()) or ()
        if (getattr(asset, "asset_type", "") or "") not in (
            "generated_video", "generated_image", "motion_graphic", "text_only"
        )
    )

    total_duration = 0.0
    if plan is not None:
        total_duration = float(getattr(plan, "estimated_duration_seconds", 0.0) or 0.0)
        if not total_duration:
            total_duration = sum(
                float(getattr(scene, "duration_seconds", 0.0) or 0.0) for scene in scenes
            )

    config = PolicyQAConfig()
    return PolicyRiskContext(
        policy_version=config.policy_version,
        title=getattr(script, "title", "") if script is not None else "",
        hook=getattr(script, "hook", "") if script is not None else "",
        line_count=len(lines),
        scene_count=len(scenes),
        total_estimated_duration_seconds=total_duration,
        has_fact_check=fact_check is not None,
        fact_check_verdict=getattr(fact_check, "package_verdict", "") if fact_check is not None else "",
        has_video_result=video_result is not None,
        video_provider=getattr(video_result, "provider", "") if video_result is not None else "",
        video_qa_verdict=getattr(inputs.video_qa, "overall_verdict", "") if inputs.video_qa is not None else "",
        generated_scene_ids=generated_scene_ids,
        external_asset_ids=external_asset_ids,
        presentation_summary=summary,
    )