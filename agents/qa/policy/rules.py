"""Stage-9 Policy/Safety QA: the deterministic pre-publication policy ruleset.

This module is a plain-Python "small policy registry" (mirroring the genre/QA
`CHECKS` registries used elsewhere in this repo).  It defines:

* the semantic version id for the ruleset (`POLICY_VERSION`),
* the set of known ruleset versions (`POLICY_VERSIONS`),
* a `PolicyRule` record describing each check in `POLICY_RULES`.

The registry is deliberately data-only: the deterministic marker vocabularies
and matching logic live in `checks.py`, and verdict aggregation lives in
`validator.py`.  Keeping the registry here lets Stage-9 emit, trace, and
version every issue produced by every check without importing matching code.

Scope and honesty: this ruleset is a *pre-publication risk assessment*.  It is
NOT a YouTube (or any platform) compliance guarantee: platform policies change,
are applied contextually by human reviewers, and cannot be reduced to phrase
matching.  Borderline cases therefore resolve to REVIEW rather than PASS.
"""

from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# Ruleset versioning
# --------------------------------------------------------------------------- #

#: The ruleset version the Stage-9 engine evaluates under by default.  The same
#: value is stamped on every produced PolicyQAResult (data.policy_version and
#: provenance.prompt_version) so a result is always auditable against the exact
#: rules that produced it.
POLICY_VERSION = "policy_rules_v1"

#: Every ruleset version the engine knows how to evaluate with.  A config that
#: names a version outside this tuple is caught by check P14 and resolves to a
#: hard REJECT ("cannot assess under an unregistered ruleset").
POLICY_VERSIONS = (POLICY_VERSION,)

#: Short note embedded in every audit trail (``RiskSummary.reviewer_note``).
POLICY_RULESET_NOTE = (
    "Deterministic pre-publication risk assessment. Not a YouTube or platform "
    "compliance guarantee; unresolved classifications resolve to REVIEW and "
    "human approval happens in a later stage."
)


# --------------------------------------------------------------------------- #
# Policy rule records
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PolicyRule:
    """Metadata describing one deterministic policy check (P01 .. P15)."""

    check_id: str
    category: str
    name: str
    description: str
    evidence_sources: tuple[str, ...] = ()
    #: What the check returns when it cannot confirm a safe disposition.  True
    #: means the check's "found" path is a hard-gate ERROR (cannot be waived by
    #: a classifier suggestion); False means the borderline path may resolve to
    #: REVIEW.  The deterministic verdict is always authoritative regardless.
    hard_default: bool = False


POLICY_RULES: tuple[PolicyRule, ...] = (
    PolicyRule(
        check_id="P01",
        category="harmful_content",
        name="Harmful content",
        description=(
            "Flags content that promotes serious harm or is generally "
            "unacceptable on a public platform (violence, self-harm "
            "instruction, hard prohibited phrases)."
        ),
        evidence_sources=("script_title", "script_hook", "script_line", "scene_narration", "scene_visual"),
        hard_default=True,
    ),
    PolicyRule(
        check_id="P02",
        category="dangerous_instructions",
        name="Dangerous / instructional harm",
        description=(
            "Flags actionable instructions that can injure the viewer "
            "('try this at home', unguarded stunts) and borderline-advisory "
            "danger language."
        ),
        evidence_sources=("script_line", "scene_narration", "scene_visual"),
        hard_default=True,
    ),
    PolicyRule(
        check_id="P03",
        category="misleading_presentation",
        name="Misleading / deceptive presentation",
        description=(
            "Checks that factual narration does not overstate what the "
            "approved claims actually establish (overstatement markers vs "
            "claim strength reported by the fact-checker)."
        ),
        evidence_sources=("script_line", "claim_result"),
        hard_default=False,
    ),
    PolicyRule(
        check_id="P04",
        category="misinformation_framing",
        name="Misinformation / policy-risk framing",
        description=(
            "Flags definitive-sounding framing about policy-sensitive topics "
            "(medical, financial, elections) that is not hedged as "
            "informational."
        ),
        evidence_sources=("script_title", "script_hook", "script_line", "scene_narration"),
        hard_default=False,
    ),
    PolicyRule(
        check_id="P05",
        category="manipulated_fabricated_media",
        name="Manipulated / fabricated media",
        description=(
            "Flags presentation claiming real footage/photos/audio when the "
            "video-generation record proves the artifact is mock/placeholder."
        ),
        evidence_sources=("scene_narration", "scene_visual", "video_artifact"),
        hard_default=True,
    ),
    PolicyRule(
        check_id="P06",
        category="synthetic_media_disclosure",
        name="Synthetic-media disclosure",
        description=(
            "Flags AI-generated visuals that depict a realistic identifiable "
            "person/event without an accompanying AI-disclosure term."
        ),
        evidence_sources=("scene_visual", "on_screen_text", "script_line"),
        hard_default=False,
    ),
    PolicyRule(
        check_id="P07",
        category="asset_rights_provenance",
        name="Copyright / asset-rights provenance",
        description=(
            "Evaluates rights evidence for external (stock/existing) assets: "
            "license noted passes, unknown rights REVIEW, prohibited/watermark "
            "markers REJECT.  Internally generated assets pass."
        ),
        evidence_sources=("scene_asset_requirement", "scene_visual", "production_notes"),
        hard_default=False,
    ),
    PolicyRule(
        check_id="P08",
        category="privacy_personal_information",
        name="Privacy / personal information",
        description=(
            "Flags private-identifying information (email, phone, address) or "
            "leak/doxx framing in presented copy."
        ),
        evidence_sources=("script_title", "script_hook", "script_line", "scene_narration", "on_screen_text"),
        hard_default=False,
    ),
    PolicyRule(
        check_id="P09",
        category="spam_deceptive_metadata",
        name="Spam / deceptive metadata",
        description=(
            "Flags spammy or keyword-stuffed title/hook patterns that look "
            "like engagement bait."
        ),
        evidence_sources=("script_title", "script_hook"),
        hard_default=False,
    ),
    PolicyRule(
        check_id="P10",
        category="repetitive_reused_content",
        name="Repetitive / reused content",
        description=(
            "Flags near-duplicate factual narration (intentional bookends -- "
            "hook, conclusion, call-to-action, transitions -- are excluded) and "
            "repost markers."
        ),
        evidence_sources=("script_line", "scene_narration"),
        hard_default=False,
    ),
    PolicyRule(
        check_id="P11",
        category="advertiser_suitability",
        name="Advertiser suitability",
        description=(
            "Flags advertiser-sensitive topics (alcohol, gambling, quick-fix "
            "products) that lack a dependency disclaimer; disclaimer resolves "
            "to an informational note."
        ),
        evidence_sources=("script_title", "script_hook", "script_line", "scene_narration"),
        hard_default=False,
    ),
    PolicyRule(
        check_id="P12",
        category="title_content_consistency",
        name="Title / description / content consistency",
        description=(
            "Flags sensational title/hook subjects (secret, leak, interview) "
            "that never appear in the actual content."
        ),
        evidence_sources=("script_title", "script_hook", "script_line"),
        hard_default=False,
    ),
    PolicyRule(
        check_id="P13",
        category="provenance_completeness",
        name="Provenance completeness",
        description=(
            "Requires each consumed artifact (script, fact-check, plan, video "
            "result, video QA) to carry provenance; missing provenance or a "
            "missing artifact resolves to REVIEW."
        ),
        evidence_sources=("artifact_provenance"),
        hard_default=False,
    ),
    PolicyRule(
        check_id="P14",
        category="policy_version_validity",
        name="Policy-version validity",
        description=(
            "A config naming an unregistered policy version cannot be assessed -- "
            "emits a hard-gate warning to guarantee a failure rather than a "
            "silent PASS."
        ),
        evidence_sources=("config"),
        hard_default=True,
    ),
    PolicyRule(
        check_id="P15",
        category="prior_gate_consistency",
        name="Prior-gate consistency",
        description=(
            "A Comprehensive-QA (Stage-8) REVIEW/REJECT or a missing/stale QA "
            "result must not silently become a policy PASS."
        ),
        evidence_sources=("video_qa_result"),
        hard_default=False,
    ),
)

#: Stable traversal order for running checks.
POLICY_CHECK_ORDER: tuple[str, ...] = tuple(rule.check_id for rule in POLICY_RULES)

#: Convenience lookup: check_id -> PolicyRule.
POLICY_RULES_BY_ID: dict[str, PolicyRule] = {rule.check_id: rule for rule in POLICY_RULES}

#: Stale attribute kept for line-oriented tooling that scans for check ids.
POLICY_CATEGORIES: tuple[str, ...] = tuple(rule.category for rule in POLICY_RULES)