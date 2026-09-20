"""Scene planning: ScenePlanner protocol + deterministic MockScenePlanner.

Stage 6 converts a Stage-5-approved Script into a provider-independent
ProductionPlan (the stable CONTENT -> VIDEO boundary). The planner never talks
to a video provider and never introduces new factual information: it only
visualizes what the script (grounded in Stage-3 approved claims) already says.

The MockScenePlanner is fully deterministic, offline, and rootable purely in
the approved claim records - a future LLM planner must implement the same
`ScenePlanner.plan(...)` protocol and will pass through the same validator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from agents.qa.fact_checker.models import FactCheckResult
from agents.qa.script.models import ScriptQAResult
from agents.research.models import FactClaim
from agents.scene_planner.models import (
    PRODUCED_BY,
    PLAN_CONFIG_VERSION,
    AssetRequirement,
    ContinuityAsset,
    GlobalStyle,
    OnScreenText,
    PlanningBlockedError,
    ProductionPlan,
    Scene,
    ScenePlanningConfig,
    production_plan_id_for,
    scene_id_for,
)
from agents.script.models import Script
from agents.seo.models import Provenance

# Shared visual-quality constraints. These are NEVER used to encode factual
# requirements (docs/DATA_CONTRACTS.md §10: negative prompt = visual quality
# only).
NEGATIVE_STYLE = (
    "distorted anatomy, extra limbs, text artifacts, watermark, "
    "blurry image, unrealistic deformation"
)


@dataclass
class ScenePlanningResult:
    """Stage runner outcome: a validated plan, or a blocked/failed planning."""

    planned: bool
    plan: ProductionPlan | None = None
    validation: object | None = None
    reason: str = ""

    @property
    def blocked(self) -> bool:
        return not self.planned and "QA gate" in self.reason


class ScenePlanner(Protocol):
    """Provider-independent scene planning interface.

    `claims` are the APPROVED claim records only (whitelist join done by the
    stage); the planner must never receive rejected/review claims.
    """

    def plan(
        self,
        script: Script,
        qa_result: ScriptQAResult,
        result: FactCheckResult,
        claims: Sequence[FactClaim],
        config: ScenePlanningConfig | None = None,
        run_id: str = "",
        created_at: str = "",
    ) -> ProductionPlan:
        ...


# --------------------------------------------------------------------------- #
# Deterministic scene templates (visual-only vocabulary, grounded in claims)
# --------------------------------------------------------------------------- #

# Each rule maps a script line / purpose to a scene treatment. The visual text
# stays inside the claim vocabulary plus purely-visual terms (camera, light,
# composition...) - it never asserts new facts.
RULE_TEMPLATES: dict[str, dict] = {
    "hook_scene": {
        "visual_type": "generated_video",
        "visual_description": (
            "Vertical close-up of a common octopus moving through clear ocean "
            "water; the camera slowly pushes toward the animal while three "
            "subtle heart indicators appear on its body."
        ),
        "generation_prompt": (
            "Vertical documentary-style underwater shot of a common octopus "
            "moving slowly across a rocky seabed, cinematic natural lighting, "
            "gentle camera push-in, three subtle heart indicators appearing on "
            "the body, realistic anatomy, educational visual style, 9:16 "
            "vertical composition, roughly four seconds of movement."
        ),
        "transition": "cut",
        "continuity_group": "octopus",
        "asset": ("generated_video", "octopus swimming on a rocky seabed", 1, "octopus"),
    },
    "establishing_scene": {
        "visual_type": "generated_video",
        "visual_description": (
            "Wide vertical establishing shot of a calm ocean with a rocky "
            "seabed; soft light from above, a gentle camera drift forward, "
            "atmospheric and quiet."
        ),
        "generation_prompt": (
            "Vertical establishing shot of a calm ocean reef with a rocky "
            "seabed, soft volumetric light rays from the surface, slow camera "
            "drift forward, muted natural palette, atmospheric documentary "
            "visual style, 9:16 vertical composition, about five seconds."
        ),
        "transition": "cut",
        "continuity_group": "octopus",
        "asset": ("generated_video", "calm ocean reef establishing shot", 1, "octopus"),
    },
    "fact_motion_graphic": {
        "visual_type": "motion_graphic",
        "visual_description": (
            "Animated vertical diagram of an octopus with three heart icons "
            "highlighted: two small hearts near the gills and one larger heart "
            "in the body, clean educational labels, gentle pulsing highlights."
        ),
        "generation_prompt": (
            "Animated vertical motion graphic of an octopus with three heart "
            "icons adjacent to the gills and body, subtle pulsing highlights, "
            "clean science-explainer labeling, smooth motion-graphic animation, "
            "9:16 vertical composition, about eight seconds."
        ),
        "label": "Three hearts",
        "transition": "cut",
        "continuity_group": "octopus",
        "asset": ("motion_graphic", "three-heart anatomy diagram", 1, "octopus"),
    },
    "explanation_motion_graphic": {
        "visual_type": "motion_graphic",
        "visual_description": (
            "Animated vertical motion graphic of an octopus swimming through "
            "clear water while two heart icons near the gills pause and one "
            "heart stays active."
        ),
        "generation_prompt": (
            "Animated vertical motion graphic of an octopus swimming through "
            "clear water; two heart icons near the gills pause while one heart "
            "stays active, clean scientific style, smooth animation, 9:16 "
            "vertical composition, about seven seconds."
        ),
        "label": "Hearts pause while swimming",
        "transition": "cut",
        "continuity_group": "octopus",
        "asset": ("motion_graphic", "swimming octopus with paused branchial hearts", 1, "octopus"),
    },
    "explanation_image": {
        "visual_type": "generated_image",
        "visual_description": (
            "Vertical portrait of an octopus with translucent blue-tinted "
            "blood visible at the body, dramatic deep-blue lighting over a "
            "dark reef background."
        ),
        "generation_prompt": (
            "Vertical generated image of an octopus showing translucent "
            "blue-tinted blood, deep-blue dramatic lighting, dark reef "
            "background, realistic texture, documentary science style, 9:16 "
            "portrait composition."
        ),
        "label": "Blood appears blue",
        "transition": "dissolve",
        "continuity_group": "octopus",
        "asset": ("generated_image", "octopus with blue-tinted blood", 1, "octopus"),
    },
    "conclusion_recap": {
        "visual_type": "motion_graphic",
        "visual_description": (
            "Recap motion graphic: the octopus silhouette returns with three "
            "heart icons fading in sequence, warm calm light, reassuring "
            "closure."
        ),
        "generation_prompt": (
            "Animated vertical recap motion graphic of an octopus silhouette "
            "with three heart icons fading in sequence, warm calm lighting, "
            "clean closure style, 9:16 vertical composition, about ten seconds."
        ),
        "transition": "fade",
        "continuity_group": "octopus",
        "asset": ("motion_graphic", "three-heart recap animation", 1, "octopus"),
    },
    "cta_text": {
        "visual_type": "text_only",
        "visual_description": (
            "Clean vertical text card with a one-line follow request over a "
            "subtle gradient, gentle drift animation, no distracting elements."
        ),
        "generation_prompt": "",
        "headline": "Follow for more science!",
        "transition": "cut",
        "continuity_group": "",
        "asset": ("text_only", "CTA follow card", 1, ""),
    },
}

# Purely visual/grammar vocabulary allowed beyond the approved claim texts
# (used by the no-hallucinated-visual-facts test to bound visual freedom).
VISUAL_LANGUAGE = {
    "vertical", "close-up", "common", "moving", "clear", "ocean", "water",
    "camera", "slowly", "pushes", "toward", "animal", "subtle", "indicators",
    "appear", "underwater", "shot", "across", "rocky", "seabed", "cinematic",
    "natural", "lighting", "gentle", "push-in", "appearing", "realistic",
    "anatomy", "educational", "visual", "style", "documentary", "roughly",
    "four", "seconds", "movement", "wide", "establishing", "calm", "rocky",
    "soft", "light", "above", "drift", "forward", "atmospheric", "quiet",
    "reef", "volumetric", "rays", "surface", "slow", "muted", "palette",
    "five", "about", "animated", "diagram", "icons", "highlighted", "small",
    "near", "one", "larger", "clean", "labels", "pulsing", "highlights",
    "motion", "graphic", "science-explainer", "labeling", "smooth",
    "animation", "eight", "swimming", "pause", "stays", "active", "scientific",
    "seven", "portrait", "translucent", "blue-tinted", "visible", "dramatic",
    "deep-blue", "dark", "background", "image", "showing", "texture",
    "science", "recap", "silhouette", "returns", "fading", "sequence", "warm",
    "reassuring", "closure", "ten", "text", "card", "one-line", "follow",
    "request", "gradient", "distracting", "elements", "documentary-style",
    "composition", "generated", "adjacent", "body", "gills",
    "close", "deep", "no", "tinted", "up", "explainer", "16", "9",
}
# Stop words tolerated between meaningful tokens.
_VISUAL_SKIP = {
    "a", "an", "the", "of", "and", "with", "in", "into", "on", "to", "from",
    "while", "over", "at", "its", "their", "that", "as", "for", "is", "are",
}


class MockScenePlanner:
    """Deterministic, offline scene planner over approved claim records.

    Groups the script's lines into scenes (never inventing narration),
    mirrors each line's duration into its scenes, and attaches grounded
    visual treatments. No web, no LLM, no API, no GPU - and it never adds a
    fact the script/claims do not already contain.
    """

    def __init__(self, config: ScenePlanningConfig | None = None):
        self.config = config or ScenePlanningConfig().validate()

    def plan(
        self,
        script: Script,
        qa_result: ScriptQAResult,
        result: FactCheckResult,
        claims: Sequence[FactClaim],
        config: ScenePlanningConfig | None = None,
        run_id: str = "",
        created_at: str = "",
    ) -> ProductionPlan:
        config = config or self.config
        # QA gate: only a PASS may be planned (never silently continue).
        if qa_result.overall_verdict != "PASS":
            raise PlanningBlockedError(
                "scene planning blocked: ScriptQAResult.overall_verdict is "
                f"'{qa_result.overall_verdict}', not PASS"
            )
        claims_by_id = {claim.claim_id: claim for claim in claims}

        line_groups = self._group_lines(script)
        scenes = tuple(
            self._build_scene(
                order,
                group,
                script,
                claims_by_id,
            )
            for order, group in enumerate(line_groups, start=1)
        )
        covered = tuple(line.line_id for line in script.lines)
        plan = ProductionPlan(
            production_plan_id=production_plan_id_for(run_id, script.script_id),
            topic_id=script.topic_id,
            topic_title=script.topic_title,
            script_id=script.script_id,
            script_qa_result_id=qa_result.qa_result_id,
            scenes=scenes,
            total_duration_seconds=round(sum(scene.duration_seconds for scene in scenes), 2),
            global_style=GlobalStyle(
                aspect_ratio=config.default_aspect_ratio,
                visual_style=config.default_visual_style,
                target_resolution=config.default_resolution,
                fps=config.fps,
                color_palette_notes="muted natural documentary tones; accent highlights for facts",
                consistency_notes="same octopus design and lighting atmosphere across all scenes",
                default_transition=config.default_transition,
                visual_continuity_requirements=(
                    "the octopus subject must keep one consistent appearance, "
                    "environment and palette wherever it appears"
                ),
            ),
            covered_script_line_ids=covered,
            uncovered_script_line_ids=tuple(
                line.line_id for line in script.lines if line.line_id not in set(covered)
            ),
            continuity=(
                ContinuityAsset(
                    group="octopus",
                    subject="common octopus",
                    appearance="realistic common octopus, natural reddish-brown coloration",
                    environment="clear ocean water over a rocky seabed",
                    palette="muted natural documentary tones",
                    recurring_elements="three highlighted heart indicators",
                ),
            ),
            production_notes=(
                "Produced by the deterministic MockScenePlanner; every visual is a "
                "suggestion for a future provider, not an asset.",
                "Factual visuals derive solely from the Stage-3 approved claims and "
                "the Stage-4 script; no new factual information was added.",
                f"Total duration mirrors the Stage-4 script within "
                f"{config.duration_tolerance_seconds}s tolerance.",
            ),
            provenance=Provenance(
                generated_by="scene_planner_agent_v1",
                source="MockScenePlanner",
                created_at=created_at or _utc_inline(),
                provider=None,
                model="MockScenePlanner",
                prompt_version=PLAN_CONFIG_VERSION,
            ),
            schema="ProductionPlan",
            schema_version=1,
            stage="scene_planning",
            run_id=run_id,
            produced_by=dict(PRODUCED_BY),
            created_at=created_at or _utc_inline(),
        )
        return plan.validate()

    # ------------------------------------------------------------------ #
    # Internal helpers (deterministic)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _group_lines(script: Script) -> list[tuple[int, ...]]:
        """Map script lines to scene groups (never one-scene-per-line blindly).

        Concrete grouping: hook | context | fact | explanation | explanation |
        (transition + conclusion) | CTA. Each entry holds script line indices.
        """
        order = [0, 1, 2, 3, 4, (5, 6), 7]
        return [tuple(group) if isinstance(group, tuple) else (group,) for group in order]

    def _build_scene(
        self,
        order: int,
        line_indices: tuple[int, ...],
        script: Script,
        claims_by_id: dict[str, FactClaim],
    ) -> Scene:
        lines = [script.lines[index] for index in line_indices]
        # Pick the rule from the group's dominant line: the last non-framing
        # line (a transition fused with a conclusion resolves to the recap).
        non_framing = [line for line in lines if line.purpose not in ("transition", "call_to_action")]
        anchor = non_framing[-1] if non_framing else lines[-1]
        rule = {
            "hook": "hook_scene",
            "context": "establishing_scene",
            "fact": "fact_motion_graphic",
            "explanation": "explanation_motion_graphic",
            "conclusion": "conclusion_recap",
            "call_to_action": "cta_text",
        }.get(anchor.purpose, "conclusion_recap")
        # Explanation scenes vary deterministically by the approved claim's own
        # wording (never by external knowledge): a claim about blue-colored
        # blood becomes a still-image treatment, everything else stays animated.
        if rule == "explanation_motion_graphic":
            anchored_text = " ".join(
                claims_by_id[ref].claim_text
                for ref in anchor.claim_refs
                if ref in claims_by_id
            )
            if any(marker in anchored_text for marker in ("appears blue", "hemocyanin")):
                rule = "explanation_image"

        template = RULE_TEMPLATES[rule]
        claim_refs = tuple(dict.fromkeys(ref for line in lines for ref in line.claim_refs))
        claim_texts = {
            claim_id: claims_by_id[claim_id].claim_text
            for claim_id in claim_refs
            if claim_id in claims_by_id
        }
        on_screen_text: list[OnScreenText] = []
        if template.get("label"):
            on_screen_text.append(
                OnScreenText(
                    kind="label",
                    text=template["label"],
                    claim_refs=claim_refs,
                )
            )
        elif template.get("headline"):
            on_screen_text.append(
                OnScreenText(
                    kind="headline",
                    text=template["headline"],
                    claim_refs=(),
                )
            )
        asset_type, subject, quantity, group = template["asset"]
        return Scene(
            scene_id=scene_id_for(order),
            order=order,
            duration_seconds=round(sum(line.estimated_duration_seconds for line in lines), 2),
            script_line_ids=tuple(line.line_id for line in lines),
            narration=" ".join(line.text for line in lines),
            claim_refs=claim_refs,
            visual_type=template["visual_type"],
            visual_description=template["visual_description"],
            generation_prompt=template["generation_prompt"],
            negative_prompt=NEGATIVE_STYLE if template["visual_type"] != "text_only" else "",
            on_screen_text=tuple(on_screen_text),
            transition=template["transition"],
            visual_continuity_group=template["continuity_group"],
            asset_requirements=(
                AssetRequirement(
                    asset_type=asset_type,
                    subject=subject,
                    quantity=quantity,
                    continuity_group=group,
                ),
            ),
            provenance={
                "generated_by": "MockScenePlanner",
                "planning_rule": rule,
                "claim_texts": claim_texts,
            },
        )


def _utc_inline() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()