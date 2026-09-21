"""Stage-9 deterministic policy/safety checks P01..P15.

Design and honesty (mirrors docs/POLICY_AND_SAFETY.md):

* Every check is deterministic and provable from the upstream artifacts
  (Script, FactCheckResult, ProductionPlan, VideoGenerationResult,
  VideoQAResult) through the `PolicyInputs` bundle -- no external
  verification, no policy API, no LLM, no network.
* Matching uses conservative marker vocabularies. Hard-gate findings are
  unambiguous, documented phrases (e.g. a line claiming real footage while
  the artifact record proves it is a mock placeholder). Borderline or
  uncertain states are encoded as SOFT errors so they resolve to REVIEW and
  can NEVER become a PASS.
* The registry metadata lives in rules.py; this module only supplies the
  matching logic. Each check returns a tuple of `PolicyIssue` objects whose
  affected ids keep the audit chain line -> claim -> scene -> asset/artifact.
"""

from __future__ import annotations

import re
from typing import Callable

from agents.qa.policy.models import (
    PolicyInputs,
    PolicyIssue,
    PolicyQAConfig,
    PolicyQAError,
)
from agents.qa.policy.rules import POLICY_RULES_BY_ID, POLICY_VERSION

# --------------------------------------------------------------------------- #
# Vocabulary (see docs/POLICY_AND_SAFETY.md §2 for rationale)
# --------------------------------------------------------------------------- #

# Purposes that intentionally bookend the video (Stage-5 decision); excluded
# from near-duplicate detection because the hook/conclusion/CTA/transition are
# deliberate echoes, not reused segments.
BOOKEND_PURPOSES = frozenset({"hook", "conclusion", "call_to_action", "transition"})

FACTUAL_PURPOSES = frozenset({"fact", "explanation", "conclusion"})

HARMFUL_HARD_MARKERS = (
    "how to build a bomb",
    "how to make a bomb",
    "make a bomb",
    "build a bomb",
    "homemade explosive",
    "make an explosive",
    "how to poison",
    "instructions to torture",
    "how to self-harm",
    "how to harm yourself",
    "plans to harm",
    "weapon of mass destruction",
)

HARMFUL_SOFT_MARKERS = (
    "weapon",
    "violence",
    "gore",
    "extremist",
    "terror",
    "shooting",
    "brutal",
    "explicit",
)

# Dangerous-instruction phrases; negated disclaimers ("don't try this at home")
# are explicitly guarded out so standard advisories do not self-flag.
DANGEROUS_HARD_MARKERS = (
    ("try this at home", ("don't try this at home", "do not try this at home", "not try this at home")),
    ("without any safety gear", ()),
    ("without any protection", ()),
    ("drive with your eyes closed", ()),
    ("walk on the freeway", ()),
    ("eat this wild mushroom", ()),
    ("swallow this pill", ()),
    ("cut the wire", ()),
)

DANGEROUS_SOFT_MARKERS = (
    "dangerous",
    "extreme stunt",
    "risky",
    "challenge me to",
    "do not imitate",
)

OVERSTATEMENT_MARKERS = (
    "proved",
    "proof",
    "guaranteed",
    "definitely",
    "certainly",
    "100%",
    "miracle",
    "secret",
    "shocking",
    "will change",
    "doctors hate",
    "scientists stunned",
    "mind-blowing",
    "they don't want you to know",
)

RISK_TOPIC_MARKERS = (
    "medical advice",
    "diabetes",
    "vaccine",
    "cure",
    "cancer",
    "financial advice",
    "invest",
    "stock market",
    "crypto",
    "bitcoin",
    "election",
)

DEFINITIVE_MARKERS = ("will ", "guaranteed", "definitely", "100%", "must", "always", "proved")

HEDGE_MARKERS = ("may ", "might ", "could ", "possibly", "probably", "likely", "seems", "appears", "consider")

REAL_MEDIA_MARKERS = (
    "actual footage",
    "real footage",
    "real video",
    "real photo",
    "real audio",
    "actual audio",
    "authentic footage",
    "actual event",
    "real event",
    "genuine footage",
    "shot on location",
    "documentary footage",
)

MANIPULATED_MEDIA_MARKERS = (
    "deepfake",
    "manipulated footage",
    "altered footage",
    "fake video",
    "forged",
)

GENERATED_VISUAL_TYPES = ("generated_video", "generated_image", "motion_graphic")

# P06 deliberately targets REAL identifiable PEOPLE/EVENTS, not generic
# "realistic" illustrations. "realistic anatomy" or "portrait of an octopus"
# must NOT trigger a synthetic-media disclosure concern; "the president",
# "a real person", "a famous celebrity" DO.
REALISTIC_SUBJECT_MARKERS = (
    "a real person",
    "real person",
    "real people",
    "famous",
    "celebrity",
    "president",
    "public figure",
    "interview with",
    "depiction of a real",
    "recreation of a real",
    "real event",
    "real interview",
    "portrait of the",
    "a human",
    "appears to show a human",
)

DISCLOSURE_MARKERS = (
    "ai-generated",
    "ai generated",
    "synthetic",
    "artificial intelligence",
    "generated imagery",
    "simulation",
)

PROHIBITED_ASSET_MARKERS = ("watermark", "copyrighted", "proprietary", "trademark")

LICENSE_OK_MARKERS = (
    "license",
    "licensed",
    "royalty-free",
    "royalty free",
    "public domain",
    "creative commons",
    "cc0",
)

PII_MARKERS = (
    "email address",
    "phone number",
    "home address",
    "social security",
    "passport number",
    "date of birth",
    "bank account",
    "credit card number",
)

LEAK_MARKERS = ("leaked", "leaking", "doxx", "exposed private", "revealed a")

PII_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

SPAM_MARKERS = (
    "free",
    "exclusive",
    "limited time",
    "act now",
    "click here",
    "guaranteed",
    "!!!",
    "???",
    "shocking",
    "secret",
    "will change",
    "number 1",
    "please subscribe",
)

REPOST_MARKERS = ("repost", "re-upload", "reused clip", "posted before", "already uploaded")

ADSENSITIVE_MARKERS = (
    "alcohol",
    "beer",
    "wine",
    "liquor",
    "tobacco",
    "cigarette",
    "vape",
    "cannabis",
    "marijuana",
    "gambling",
    "casino",
    "betting",
    "lottery",
    "prescription",
    "weight loss",
    "diet pill",
    "supplement",
    "bitcoin",
    "crypto",
)

DEPENDENCY_DISCLAIMER_MARKERS = (
    "not financial advice",
    "not medical advice",
    "consult",
    "disclaimer",
    "does not replace",
)

SENSATIONAL_SUBJECT_MARKERS = (
    "secret",
    "exclusive",
    "leak",
    "interview",
    "footage",
    "president",
    "celebrity",
    "celebrit",
    "war",
    "attack",
    "crash",
    "recording",
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _markers_found(text: str, markers: tuple[str, ...]) -> tuple[str, ...]:
    """All markers present (case-insensitive substring), in given order."""
    lowered = (text or "").lower()
    return tuple(marker for marker in markers if marker in lowered)


def _guarded_markers_found(text: str, spec: tuple[tuple[str, tuple[str, ...]], ...]):
    """Return markers whose text match is not negated by a preceding disclaimer."""
    lowered = (text or "").lower()
    found = []
    for marker, negators in spec:
        if marker in lowered and not any(negator in lowered for negator in negators):
            found.append(marker)
    return found


def _scenes(inputs: PolicyInputs) -> tuple:
    plan = inputs.plan
    if plan is None:
        return ()
    return tuple(getattr(plan, "scenes", ()) or ())


def _artifact_row_for_scene(inputs: PolicyInputs, scene_id: str):
    result = inputs.video_result
    if result is None or not scene_id:
        return None
    for row in getattr(result, "scenes", ()) or ():
        if getattr(row, "scene_id", "") == scene_id:
            return row
    return None


def _artifact_mock(inputs: PolicyInputs, scene_id: str) -> bool | None:
    row = _artifact_row_for_scene(inputs, scene_id)
    if row is None:
        return None
    artifact = getattr(row, "artifact", None)
    if artifact is None:
        return None
    return bool(getattr(artifact, "mock", True))


def _strong_words(fact_check, claim_id: str) -> tuple[str, ...] | None:
    """Words the fact-checker found STRONG in the claim, parsed from checks.

    Returns None when the claim or the strength record is missing (unverifiable),
    an empty tuple when the claim has NO strong words, or the strong words.
    """
    if fact_check is None or not claim_id:
        return None
    verdict = fact_check.claim_verdict(claim_id)
    if verdict is None:
        return None
    for entry in verdict.checks_performed or ():
        if entry.startswith("strong_in_claim="):
            value = entry[len("strong_in_claim="):]
            if value == "none":
                return ()
            return tuple(word for word in value.split(",") if word)
    return None


def _all_text(inputs: PolicyInputs) -> tuple[str, ...]:
    return tuple(item.text for item in inputs.presentation_items())


def _evidence_snippet(text: str, marker: str, width: int = 40) -> str:
    lowered = text.lower()
    index = lowered.find(marker)
    if index < 0:
        return text[:width * 2]
    start = max(0, index - width)
    return f"...{text[start:index + width]}..."


def _rule(check_id: str):
    rule = POLICY_RULES_BY_ID.get(check_id)
    if rule is None:
        raise PolicyQAError(f"no registered policy rule for {check_id}")
    return rule


def _issue(
    check_id: str,
    severity: str,
    gate: str,
    message: str,
    rationale: str,
    scene_ids=(),
    line_ids=(),
    claim_ids=(),
    asset_ids=(),
    evidence: str = "",
    resolution_hint: str = "",
) -> PolicyIssue:
    rule = _rule(check_id)
    return PolicyIssue(
        check_id=check_id,
        category=rule.category,
        severity=severity,
        gate=gate,
        message=message,
        rationale=rationale,
        affected_scene_ids=tuple(scene_ids),
        affected_line_ids=tuple(line_ids),
        affected_claim_ids=tuple(claim_ids),
        affected_asset_ids=tuple(asset_ids),
        policy_rule=f"{check_id}@{POLICY_VERSION}",
        evidence=evidence,
        resolution_hint=resolution_hint,
    )


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #


def check_harmful_content(inputs: PolicyInputs, config: PolicyQAConfig):
    """P01: content that promotes serious harm that a public platform rejects."""
    issues: list[PolicyIssue] = []
    for item in inputs.presentation_items():
        if item.source in ("production_notes", "video.artifact.subject"):
            continue
        hard = _markers_found(item.text, HARMFUL_HARD_MARKERS)
        if hard:
            issues.append(
                _issue(
                    "P01", "ERROR", "hard",
                    "Presented copy contains a harmful-content phrase.",
                    "Unambiguous prohibited-theme phrasing cannot be blessed for a "
                    "public platform and cannot be waived by an automated suggestion.",
                    scene_ids=(item.scene_id,) if item.scene_id else (),
                    line_ids=(item.line_id,) if item.line_id else (),
                    asset_ids=(item.asset_id,) if item.asset_id else (),
                    evidence=_evidence_snippet(item.text, hard[0]),
                    resolution_hint="Rewrite the offending segment to remove the "
                    "prohibited phrase and describe the topic at a summary level.",
                )
            )
            continue
        soft = _markers_found(item.text, HARMFUL_SOFT_MARKERS)
        if soft:
            if item.source in ("script.title", "script.hook"):
                continue
            issues.append(
                _issue(
                    "P01", "ERROR", "soft",
                    "Borderline harmful-content language appears in presented copy.",
                    "Borderline policy classifications must not resolve to PASS "
                    "(uncertainty -> REVIEW); the audience is public.",
                    scene_ids=(item.scene_id,) if item.scene_id else (),
                    line_ids=(item.line_id,) if item.line_id else (),
                    asset_ids=(item.asset_id,) if item.asset_id else (),
                    evidence=_evidence_snippet(item.text, soft[0]),
                    resolution_hint="Review the segment manually; remove or soften "
                    "the borderline language before PASS.",
                )
            )
    return tuple(issues)


def check_dangerous_instructions(inputs: PolicyInputs, config: PolicyQAConfig):
    """P02: actionable instructions that can injure the viewer."""
    issues: list[PolicyIssue] = []
    # Hard actionable phrases -> REJECT (with disclaimer guard).
    for item in inputs.presentation_items():
        if item.source in ("production_notes", "video.artifact.subject", "video.artifact.note"):
            continue
        hard = _guarded_markers_found(item.text, DANGEROUS_HARD_MARKERS)
        if hard:
            issues.append(
                _issue(
                    "P02", "ERROR", "hard",
                    "Presented copy gives actionable, unsafe instructions.",
                    "Step-by-step unsafe behavior in public content is a hard "
                    "policy/liability finding; advisories phrased 'don't do this' "
                    "are guarded out.",
                    scene_ids=(item.scene_id,) if item.scene_id else (),
                    line_ids=(item.line_id,) if item.line_id else (),
                    asset_ids=(item.asset_id,) if item.asset_id else (),
                    evidence=_evidence_snippet(item.text, hard[0]),
                    resolution_hint="Remove the instructional segment and add a "
                    "clear non-imitation disclaimer before PASS.",
                )
            )
            continue
        soft = _markers_found(item.text, DANGEROUS_SOFT_MARKERS)
        if soft and item.source in (
            "script.line",
            "plan.scene.narration",
            "plan.scene.visual_description",
        ):
            issues.append(
                _issue(
                    "P02", "ERROR", "soft",
                    "Borderline danger/stunt language appears in presented copy.",
                    "Stunt risk is borderline: resolved by a human reviewer rather "
                    "than an automated PASS.",
                    scene_ids=(item.scene_id,) if item.scene_id else (),
                    line_ids=(item.line_id,) if item.line_id else (),
                    asset_ids=(item.asset_id,) if item.asset_id else (),
                    evidence=_evidence_snippet(item.text, soft[0]),
                    resolution_hint="Confirm a disclaimer (e.g. 'do not try this at "
                    "home') is present; otherwise soften the language.",
                )
            )
    return tuple(issues)


def check_misleading_presentation(inputs: PolicyInputs, config: PolicyQAConfig):
    """P03: factual narration must not overstate what the claims establish."""
    issues: list[PolicyIssue] = []
    fact_check = inputs.fact_check
    for item in inputs.presentation_items():
        if item.source not in ("script.line", "script.title", "script.hook"):
            continue
        markers = _markers_found(item.text, OVERSTATEMENT_MARKERS)
        if not markers:
            continue
        if item.source == "script.line" and item.purpose in FACTUAL_PURPOSES:
            unresolvable: list[str] = []
            weak: list[str] = []
            substantiated: list[str] = []
            for claim_id in item.claim_refs or ():
                strength = _strong_words(fact_check, claim_id)
                if strength is None:
                    unresolvable.append(claim_id)
                elif not strength:
                    weak.append(claim_id)
                else:
                    substantiated.append(claim_id)
            if not (item.claim_refs or ()):
                unresolvable.append("(no claim_refs recorded)")
            if unresolvable:
                issues.append(
                    _issue(
                        "P03", "ERROR", "soft",
                        "Factual line uses strong presentation language but its "
                        "claim strength could not be verified.",
                        "Uncertain claim ground-truth cannot resolve to PASS.",
                        line_ids=(item.line_id,) if item.line_id else (),
                        claim_ids=tuple(unresolvable),
                        evidence=f"{_evidence_snippet(item.text, markers[0])}; "
                        f"unverifiable claims: {unresolvable}",
                        resolution_hint="Re-check the claim or soften the narration.",
                    )
                )
            elif weak:
                issues.append(
                    _issue(
                        "P03", "ERROR", "soft",
                        "Factual line overstates a claim the fact-checker found "
                        "to contain no strong support.",
                        "Strong presentation language on a weak claim is "
                        "misleading; the fact-checker reported strong_in_claim=none.",
                        line_ids=(item.line_id,) if item.line_id else (),
                        claim_ids=tuple(weak),
                        evidence=f"{_evidence_snippet(item.text, markers[0])}; "
                        f"strong_in_claim=none on {weak}",
                        resolution_hint="Align the narration with the weak claim "
                        "(remove the overstatement) before PASS.",
                    )
                )
        elif item.source in ("script.title", "script.hook"):
            issues.append(
                _issue(
                    "P03", "ERROR", "soft",
                    "Title/hook uses overstatement that overpromises the content.",
                    "Deceptive framing in the metadata is a policy concern; "
                    "overclaiming hooks read as misleading.",
                    line_ids=(item.line_id,) if item.line_id else (),
                    evidence=_evidence_snippet(item.text, markers[0]),
                    resolution_hint="Reword the hook to promise exactly what the "
                    "episode delivers.",
                )
            )
    return tuple(issues)


def check_misinformation_framing(inputs: PolicyInputs, config: PolicyQAConfig):
    """P04: definitive framing about policy-sensitive topics needs hedging."""
    issues: list[PolicyIssue] = []
    for item in inputs.presentation_items():
        if item.source in ("production_notes", "video.artifact.note", "video.artifact.subject"):
            continue
        topics = _markers_found(item.text, RISK_TOPIC_MARKERS)
        definitive = _markers_found(item.text, DEFINITIVE_MARKERS)
        if not topics or not definitive:
            continue
        if any(hedge in item.text.lower() for hedge in HEDGE_MARKERS):
            continue
        issues.append(
            _issue(
                "P04", "ERROR", "soft",
                "Definitive framing about a policy-sensitive topic without hedging.",
                "Definitive claims about medical/financial/political topics carry "
                "policy risk when packaged as advice; hedged informational "
                "framing is acceptable.",
                scene_ids=(item.scene_id,) if item.scene_id else (),
                line_ids=(item.line_id,) if item.line_id else (),
                evidence=f"topics={topics} definitive={definitive}",
                resolution_hint="Add hedged informational framing (e.g. 'may', "
                "'consult a professional') before PASS.",
            )
        )
    return tuple(issues)


def check_manipulated_fabricated_media(inputs: PolicyInputs, config: PolicyQAConfig):
    """P05: presentation must not claim real media the record proves is mock."""
    issues: list[PolicyIssue] = []
    for item in inputs.presentation_items():
        if item.source not in (
            "plan.scene.narration",
            "plan.scene.visual_description",
            "plan.scene.on_screen_text",
            "script.title",
            "script.hook",
        ):
            continue
        real = _markers_found(item.text, REAL_MEDIA_MARKERS)
        if real and item.scene_id:
            row = _artifact_row_for_scene(inputs, item.scene_id)
            asset_ids = ()
            if row is not None:
                artifact = getattr(row, "artifact", None)
                if artifact is not None and getattr(artifact, "artifact_id", ""):
                    asset_ids = (artifact.artifact_id,)
            mock = _artifact_mock(inputs, item.scene_id)
            if mock is True:
                issues.append(
                    _issue(
                        "P05", "ERROR", "hard",
                        "Content claims real footage/audio while the generation "
                        "record proves the artifact is a mock placeholder.",
                        "Self-claimed authenticity is a fabrication finding when "
                        "the artifact record contradicts it (deterministic, "
                        "cannot be overridden).",
                        scene_ids=(item.scene_id,) if item.scene_id else (),
                        line_ids=(item.line_id,) if item.line_id else (),
                        asset_ids=asset_ids,
                        evidence=f"{_evidence_snippet(item.text, real[0])}; artifact mock=True",
                        resolution_hint="Produce real captured media or drop the "
                        "authenticity claim before upload.",
                    )
                )
            elif mock is None:
                issues.append(
                    _issue(
                        "P05", "ERROR", "soft",
                        "Content claims real media but the scene has no recorded "
                        "artifact to verify against.",
                        "An unverifiable authenticity claim cannot PASS.",
                        scene_ids=(item.scene_id,) if item.scene_id else (),
                        line_ids=(item.line_id,) if item.line_id else (),
                        asset_ids=asset_ids,
                        evidence=f"{_evidence_snippet(item.text, real[0])}; no artifact row",
                        resolution_hint="Check whether the scene is covered by the "
                        "video-generation result before PASS.",
                    )
                )
        manipulated = _markers_found(item.text, MANIPULATED_MEDIA_MARKERS)
        if manipulated and item.scene_id:
            issues.append(
                _issue(
                    "P05", "ERROR", "soft",
                    "Content references manipulated/fake media without disclosure.",
                    "Manipulated-media references must be disclosed; presence "
                    "without a disclosure term is a borderline policy finding.",
                    scene_ids=(item.scene_id,) if item.scene_id else (),
                    line_ids=(item.line_id,) if item.line_id else (),
                    evidence=_evidence_snippet(item.text, manipulated[0]),
                    resolution_hint="Add a synthetic/manipulated content disclosure "
                    "or remove the reference.",
                )
            )
    return tuple(issues)


def check_synthetic_media_disclosure(inputs: PolicyInputs, config: PolicyQAConfig):
    """P06: AI-generated visuals of real people/events need a disclosure term."""
    issues: list[PolicyIssue] = []
    all_text = " ".join(_all_text(inputs)).lower()
    disclosed = any(marker in all_text for marker in DISCLOSURE_MARKERS)
    for scene in _scenes(inputs):
        visual_type = getattr(scene, "visual_type", "") or ""
        scene_id = getattr(scene, "scene_id", "")
        if visual_type not in GENERATED_VISUAL_TYPES:
            continue
        subjects = " ".join(
            getattr(asset, "subject", "") or ""
            for asset in getattr(scene, "asset_requirements", ()) or ()
        )
        description = getattr(scene, "visual_description", "") or ""
        prompt = getattr(scene, "generation_prompt", "") or ""
        on_screen = " ".join(
            getattr(chunk, "text", "") or "" for chunk in getattr(scene, "on_screen_text", ()) or ()
        )
        blob = f"{subjects} {description} {prompt} {on_screen}"
        realistic = _markers_found(blob, REALISTIC_SUBJECT_MARKERS)
        if not realistic:
            continue
        if disclosed:
            continue
        issues.append(
            _issue(
                "P06", "ERROR", "soft",
                "AI-generated visual depicts a realistic identifiable subject "
                "without an AI-disclosure term.",
                "Synthetic media depicting real people/events must be disclosed; "
                "its absence is a borderline (uncertain) classification.",
                scene_ids=(scene_id,) if scene_id else (),
                evidence=f"scene visual={visual_type}; realistic markers={realistic}",
                resolution_hint="Add an explicit AI-generated/synthetic disclosure "
                "(or use non-photoreal imagery) before PASS.",
            )
        )
    return tuple(issues)


def check_asset_rights_provenance(inputs: PolicyInputs, config: PolicyQAConfig):
    """P07: external assets need recorded rights; generated assets pass."""
    issues: list[PolicyIssue] = []
    production_notes = ""
    if inputs.plan is not None:
        production_notes = getattr(inputs.plan, "production_notes", "") or ""
    for scene in _scenes(inputs):
        scene_id = getattr(scene, "scene_id", "")
        provenance_text = ""
        provenance = getattr(scene, "provenance", None)
        if isinstance(provenance, dict):
            provenance_text = " ".join(str(v) for v in provenance.values())
        elif provenance is not None:
            provenance_text = str(provenance)
        description = getattr(scene, "visual_description", "") or ""
        prompt = getattr(scene, "generation_prompt", "") or ""
        evidence_blob = f"{description} {prompt} {provenance_text} {production_notes}"
        for asset in getattr(scene, "asset_requirements", ()) or ():
            asset_type = getattr(asset, "asset_type", "") or ""
            asset_id = asset.asset_id if hasattr(asset, "asset_id") else f"{scene_id}:{asset_type}"
            subject = getattr(asset, "subject", "") or ""
            if asset_type in GENERATED_VISUAL_TYPES or asset_type == "text_only":
                continue
            blob = f"{subject} {evidence_blob}"
            prohibited = _markers_found(blob, PROHIBITED_ASSET_MARKERS)
            if prohibited:
                issues.append(
                    _issue(
                        "P07", "ERROR", "hard",
                        "External asset carries a prohibited rights marker.",
                        "Explicit watermark/copyrighted markers cannot be waived "
                        "by an automated suggestion.",
                        scene_ids=(scene_id,) if scene_id else (),
                        asset_ids=(asset_id,) if asset_id else (),
                        evidence=f"asset={asset_type} markers={prohibited}",
                        resolution_hint="Secure proper licensing/rights for the "
                        "asset or replace it with a licensed/generated asset.",
                    )
                )
                continue
            licensed = _markers_found(blob, LICENSE_OK_MARKERS)
            if not licensed:
                issues.append(
                    _issue(
                        "P07", "ERROR", "soft",
                        "External asset has no recorded rights/license evidence.",
                        "Unknown asset rights cannot resolve to PASS; a reviewer "
                        "must confirm licensing.",
                        scene_ids=(scene_id,) if scene_id else (),
                        asset_ids=(asset_id,) if asset_id else (),
                        evidence=f"asset={asset_type} subject={subject[:60]}",
                        resolution_hint="Record the license/royalty-free source for "
                        "the asset before PASS.",
                    )
                )
    return tuple(issues)


def check_privacy(inputs: PolicyInputs, config: PolicyQAConfig):
    """P08: private-identifying information in presented copy."""
    issues: list[PolicyIssue] = []
    for item in inputs.presentation_items():
        if item.source in ("production_notes", "video.artifact.note", "video.artifact.subject", "scene_visual", "generation_prompt"):
            continue
        text = item.text.lower()
        pii = _markers_found(item.text, PII_MARKERS)
        if PII_EMAIL_RE.search(item.text):
            pii += ("email-address-pattern",)
        if not pii:
            continue
        leak = _markers_found(item.text, LEAK_MARKERS)
        if leak:
            issues.append(
                _issue(
                    "P08", "ERROR", "hard",
                    "Content exposes private-identifying information.",
                    "Leaking/doxxing phrasing around PII is a hard privacy "
                    "finding that automated review cannot waive.",
                    scene_ids=(item.scene_id,) if item.scene_id else (),
                    line_ids=(item.line_id,) if item.line_id else (),
                    evidence=f"pii={pii} leak={leak}",
                    resolution_hint="Remove the identifying details entirely.",
                )
            )
        else:
            issues.append(
                _issue(
                    "P08", "ERROR", "soft",
                    "Private-identifying information appears in presented copy.",
                    "Personally-identifying details in public content are a "
                    "privacy review item.",
                    scene_ids=(item.scene_id,) if item.scene_id else (),
                    line_ids=(item.line_id,) if item.line_id else (),
                    evidence=f"pii={pii}",
                    resolution_hint="Remove or anonymize the identifying details.",
                )
            )
    return tuple(issues)


def check_spam_metadata(inputs: PolicyInputs, config: PolicyQAConfig):
    """P09: spammy / engagement-bait title and hook."""
    issues: list[PolicyIssue] = []
    hook = ""
    title = ""
    script = inputs.script
    if script is not None:
        title = getattr(script, "title", "") or ""
        hook = getattr(script, "hook", "") or ""
    blob = f"{title} {hook}".lower().rstrip()
    markers = tuple(
        marker for marker in SPAM_MARKERS if marker in blob
    )
    if len(markers) >= 3:
        issues.append(
            _issue(
                "P09", "ERROR", "soft",
                "Title/hook reads as spammy or keyword-stuffed metadata.",
                "Engagement-bait metadata degrades trust and looks like spam; a "
                "reviewer should confirm the title is accurate.",
                evidence=f"markers={markers}",
                resolution_hint="Rewrite the title/hook to be informative rather "
                "than bait-like before PASS.",
            )
        )
    elif len(markers) == 2:
        issues.append(
            _issue(
                "P09", "ERROR", "soft",
                "Title/hook shows borderline engagement-bait patterns.",
                "Two spam markers is borderline; uncertainty resolves to REVIEW.",
                evidence=f"markers={markers}",
                resolution_hint="Reword the bait-y phrases or confirm they are "
                "accurate descriptors.",
            )
        )
    return tuple(issues)


def check_repetitive(inputs: PolicyInputs, config: PolicyQAConfig):
    """P10: near-duplicate factual narration (bookends excluded) + repost marks."""
    issues: list[PolicyIssue] = []
    lines = []
    script = inputs.script
    if script is not None:
        for line in getattr(script, "lines", ()) or ():
            purpose = getattr(line, "purpose", "") or ""
            if purpose in BOOKEND_PURPOSES or not purpose:
                continue
            text = getattr(line, "text", "") or ""
            if text:
                lines.append((getattr(line, "line_id", ""), text))
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            line_id_a, text_a = lines[i]
            line_id_b, text_b = lines[j]
            tokens_a = {t for t in text_a.lower().split() if len(t) > 1}
            tokens_b = {t for t in text_b.lower().split() if len(t) > 1}
            if not tokens_a or not tokens_b:
                continue
            overlap = len(tokens_a & tokens_b) / min(len(tokens_a), len(tokens_b))
            if overlap >= config.repetition_token_overlap_threshold:
                issues.append(
                    _issue(
                        "P10", "ERROR", "soft",
                        "Two factual lines are near-duplicates of each other.",
                        "Near-duplicate factual narration looks like reused "
                        "content; accidental echoes in the body are REVIEWed "
                        "(deliberate hook/conclusion bookends are excluded).",
                        line_ids=(line_id_a, line_id_b),
                        evidence=f"token overlap {overlap:.2f}",
                        resolution_hint="Differentiate the two lines or remove the "
                        "redundant one before PASS.",
                    )
                )
    for item in inputs.presentation_items():
        if item.source in ("production_notes", "script.title", "script.hook"):
            continue
        repost = _markers_found(item.text, REPOST_MARKERS)
        if repost:
            issues.append(
                _issue(
                    "P10", "ERROR", "soft",
                    "Content explicitly marks itself as reposted/reused.",
                    "Re-uploaded content is a policy-review item.",
                    scene_ids=(item.scene_id,) if item.scene_id else (),
                    line_ids=(item.line_id,) if item.line_id else (),
                    evidence=f"markers={repost}",
                    resolution_hint="Confirm original ownership or remove the "
                    "reused segment before PASS.",
                )
            )
    return tuple(issues)


def check_advertiser_suitability(inputs: PolicyInputs, config: PolicyQAConfig):
    """P11: advertiser-sensitive topics need a dependency disclaimer."""
    issues: list[PolicyIssue] = []
    all_blob = " ".join(_all_text(inputs)).lower()
    disclaimed = any(marker in all_blob for marker in DEPENDENCY_DISCLAIMER_MARKERS)
    for item in inputs.presentation_items():
        sensitive = _markers_found(item.text, ADSENSITIVE_MARKERS)
        if not sensitive:
            continue
        if disclaimed:
            continue
        issues.append(
            _issue(
                "P11", "ERROR", "soft",
                "Advertiser-sensitive topic appears without a dependency disclaimer.",
                "Spontaneous promotion of sensitive categories is a monetization/"
                "suitability review item.",
                scene_ids=(item.scene_id,) if item.scene_id else (),
                line_ids=(item.line_id,) if item.line_id else (),
                evidence=f"markers={sensitive}",
                resolution_hint="Add e.g. 'not financial advice' / 'consult a "
                "professional' or remove the topic before PASS.",
            )
        )
    return tuple(issues)


def check_title_content_consistency(inputs: PolicyInputs, config: PolicyQAConfig):
    """P12: a sensational title must be backed by matching content."""
    issues: list[PolicyIssue] = []
    script = inputs.script
    if script is None:
        return ()
    title = getattr(script, "title", "") or ""
    hook = getattr(script, "hook", "") or ""
    bag = " ".join(
        [item.text for item in inputs.presentation_items() if item.source.startswith(("script.line", "plan.scene"))]
    ).lower()
    for text, kind in ((title, "title"), (hook, "hook")):
        markers = _markers_found(text, SENSATIONAL_SUBJECT_MARKERS)
        if not markers:
            continue
        missing = [m for m in markers if m not in bag]
        if missing:
            issues.append(
                _issue(
                    "P12", "ERROR", "soft",
                    f"{kind.capitalize()} promises a subject that never appears in the content.",
                    "A sensational subject in the metadata with no matching "
                    "content is a misleading-metadata finding.",
                    evidence=f"promised={missing}",
                    resolution_hint="Either deliver the promised subject in the "
                    "content or reword the title/hook.",
                )
            )
    return tuple(issues)


def check_provenance_completeness(inputs: PolicyInputs, config: PolicyQAConfig):
    """P13: upstream content artifacts must carry provenance."""
    issues: list[PolicyIssue] = []

    def provenance_ok(artifact) -> bool:
        if artifact is None:
            return False
        provenance = getattr(artifact, "provenance", None)
        if provenance is None:
            return False
        if isinstance(provenance, dict):
            return bool(provenance.get("generated_by"))
        return bool(getattr(provenance, "generated_by", None))

    for key, artifact in (
        ("script", inputs.script),
        ("fact-check result", inputs.fact_check),
        ("production plan", inputs.plan),
        ("video-generation result", inputs.video_result),
    ):
        if artifact is None:
            issues.append(
                _issue(
                    "P13", "ERROR", "soft",
                    f"{key} artifact is missing from the policy gate inputs.",
                    "A missing upstream content artifact means the gate cannot "
                    "trace every decision; that is a REVIEW-level gap.",
                    resolution_hint=f"Provide the {key} artifact before re-running "
                    "the policy gate.",
                )
            )
            continue
        if not provenance_ok(artifact):
            issues.append(
                _issue(
                    "P13", "ERROR", "soft",
                    f"{key} artifact carries no usable provenance.",
                    "Provenance is required for audit; its absence resolves to "
                    "REVIEW rather than PASS.",
                    resolution_hint=f"Record provenance on the {key} artifact.",
                )
            )
    return tuple(issues)


def check_policy_version_validity(inputs: PolicyInputs, config: PolicyQAConfig):
    """P14: an unregistered policy version cannot be assessed."""
    from agents.qa.policy.rules import POLICY_VERSIONS

    if config.policy_version not in POLICY_VERSIONS:
        return (
            _issue(
                "P14", "ERROR", "hard",
                f"Policy version '{config.policy_version}' is not a registered ruleset.",
                "An unregistered ruleset cannot be evaluated; the gate fails "
                "closed so an invalid config can never produce a silent PASS.",
                evidence=f"registered={list(POLICY_VERSIONS)}",
                resolution_hint="Re-run with a registered policy_version "
                f"(e.g. '{POLICY_VERSIONS[0]}').",
            ),
        )
    return ()


def check_prior_gate_consistency(inputs: PolicyInputs, config: PolicyQAConfig):
    """P15: Stage-8 REVIEW/REJECT (or missing/stale QA) cannot become a PASS."""
    issues: list[PolicyIssue] = []
    video_qa = inputs.video_qa
    if video_qa is None:
        issues.append(
            _issue(
                "P15", "ERROR", "soft",
                "No Stage-8 video-QA result was supplied to the policy gate.",
                "Publishing without a prior QA gate cannot be blessed; a missing "
                "gate is uncertainty and resolves to REVIEW.",
                resolution_hint="Run Stage-8 (agents.qa.video) first and pass the "
                "result to the policy gate.",
            )
        )
        return tuple(issues)
    verdict = getattr(video_qa, "overall_verdict", "") or ""
    if verdict != "PASS":
        issues.append(
            _issue(
                "P15", "ERROR", "soft",
                "Stage-8 QA did not PASS this content.",
                "A prior-gate REVIEW/REJECT must not silently become a policy "
                "PASS; policy runs after quality but does not override it.",
                evidence=f"qa_result_id={getattr(video_qa, 'qa_result_id', '')} "
                f"verdict={verdict}",
                resolution_hint="Resolve the Stage-8 findings and re-run Stage-8 "
                "to a PASS before the policy gate.",
            )
        )
    plan = inputs.plan
    target_ref = getattr(video_qa, "target_ref", "") or ""
    if plan is not None and target_ref and target_ref != getattr(plan, "production_plan_id", ""):
        issues.append(
            _issue(
                "P15", "ERROR", "soft",
                "The Stage-8 QA result references a different production plan.",
                "Gating content against a stale QA result is an uncertainty that "
                "must resolve to REVIEW.",
                evidence=f"qa target_ref={target_ref}; plan={getattr(plan, 'production_plan_id', '')}",
                resolution_hint="Re-run Stage-8 against this production plan and "
                "re-run the policy gate.",
            )
        )
    return tuple(issues)


# Registry: check_id -> implementation. Iterates in POLICY_CHECK_ORDER.
CHECK_FUNCTIONS: dict[str, Callable[[PolicyInputs, PolicyQAConfig], tuple[PolicyIssue, ...]]] = {
    "P01": check_harmful_content,
    "P02": check_dangerous_instructions,
    "P03": check_misleading_presentation,
    "P04": check_misinformation_framing,
    "P05": check_manipulated_fabricated_media,
    "P06": check_synthetic_media_disclosure,
    "P07": check_asset_rights_provenance,
    "P08": check_privacy,
    "P09": check_spam_metadata,
    "P10": check_repetitive,
    "P11": check_advertiser_suitability,
    "P12": check_title_content_consistency,
    "P13": check_provenance_completeness,
    "P14": check_policy_version_validity,
    "P15": check_prior_gate_consistency,
}