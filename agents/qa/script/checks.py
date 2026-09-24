"""Modular, deterministic Stage-5 QA checks (13 categories).

Every check is a pure function over (script, fact_check_result, claims_by_id,
config) -> list[QAIssue]. Nothing here talks to the network or to an LLM: all
judgements are deterministic heuristics over the script text, the approved
claim records, and the fact-check verdicts. Stage 5 is quality control, NOT
fact finding - unapproved claims are never consulted as fact, and every
issue preserves the audit chain (line -> claim -> source -> evidence).

Checks are registered in `CHECKS` so future rules can be added independently:
write a function with the same signature and append it to the registry.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

from agents.qa.fact_checker.evidence import meaningful_tokens
from agents.qa.fact_checker.models import FactCheckResult
from agents.research.models import FactClaim
from agents.script.models import FACTUAL_PURPOSES, PURPOSE_VALUES, Script, ScriptLine
from agents.qa.script.models import (
    CATEGORY_DIMENSION,
    QACheckRecord,
    QAIssue,
    ScriptQAConfig,
)

GATE_HARD = "hard"
GATE_SOFT = "soft"

_WORD = re.compile(r"[a-z0-9]+")
_PUNCTUATION = re.compile(r"[^a-z0-9]+")
_WORDY = re.compile(r"[a-z]+(?:'t)?")


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _norm(text: str) -> str:
    return " ".join(_WORD.findall(text.lower()))


def _tokens(text: str) -> set[str]:
    return meaningful_tokens(_PUNCTUATION.sub(" ", text.lower()))


def _simple_words(text: str) -> list[str]:
    return _WORDY.findall(text.lower())


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"[.!?]+\s*", text) if part.strip()]


def _ref_claims(line: ScriptLine, claims_by_id: dict[str, FactClaim]) -> list[FactClaim]:
    return [claims_by_id[cid] for cid in line.claim_refs if cid in claims_by_id]


def _coverage(
    ref_claims: list[FactClaim], text: str, threshold: float
) -> float | None:
    """Fraction of the line's meaningful tokens present in its claim records."""
    claim_tokens: set[str] = set()
    for claim in ref_claims:
        claim_tokens |= _tokens(claim.claim_text)
    line_tokens = _tokens(text)
    if not line_tokens or not claim_tokens:
        return None
    return len(line_tokens & claim_tokens) / len(line_tokens)


def _issue(
    category: str,
    severity: str,
    message: str,
    *,
    line_ids=(),  # noqa: ANN002
    claim_ids=(),  # noqa: ANN002
    gate: str = GATE_SOFT,
    check: str = "",
) -> QAIssue:
    return QAIssue(
        category=category,
        severity=severity,
        message=message,
        affected_line_ids=tuple(line_ids),
        affected_claim_ids=tuple(claim_ids),
        gate=gate,
        dimension=CATEGORY_DIMENSION[category],
        check=check,
    ).validate()


# Words that overstate the evidence when the referenced claims never say them.
_CERTAINTY_WORDS = frozenset(
    {
        "proved", "proven", "proves", "confirmed", "conclusively", "definitely",
        "certainly", "guaranteed", "established", "discovered", "demonstrates",
        "demonstrated", "undeniably", "irrefutable", "always", "never",
    }
)

# Words that signal a claim is hedged / epistemically weaker.
_HEDGE_WORDS = frozenset(
    {
        "probably", "likely", "may", "might", "possibly", "could", "maybe",
        "suggests", "suggested", "reportedly", "estimated", "approximately",
        "one", "some", "according", "tends",
    }
)

# "appears"/"seems" are epistemic hedges only when followed by "to"
# ("appears to be"), not when describing appearance ("appears blue").
_FOLLOWING_TO_HEDGES = frozenset({"appears", "seems"})


def _hedge_tokens(text: str) -> set[str]:
    """Hedge tokens present in `text` (appears/seems only when followed by 'to')."""
    words = _simple_words(text)
    hedges = set(words) & _HEDGE_WORDS
    for index, word in enumerate(words):
        if word in _FOLLOWING_TO_HEDGES and index + 1 < len(words) and words[index + 1] == "to":
            hedges.add(word)
    return hedges

# Negation tokens: their appearance in a factual line while absent from the
# referenced claims is treated as a material contradiction of that claim.
_NEGATION_WORDS = frozenset(
    {
        "not", "no", "never", "without", "cannot", "can't", "doesn't",
        "isn't", "aren't", "didn't", "don't", "won't", "wouldn't",
        "couldn't", "shouldn't", "mustn't", "none", "nothing", "nobody",
        "nowhere", "neither", "nor",
    }
)

# Misleading / energetic phrasing (WARNING-level pattern sweep). The hard
# certainty-strengthening gate lives in the uncertainty check; this list is
# deliberately disjoint so the same word is not reported twice at two severities.
_MISLEADING_PHRASES = (
    "the truth is", "everyone knows", "you won't believe", "you wont believe",
    "shocking", "mind-blowing", "mind blowing", "unbelievable", "absolutely",
    "literally", "must watch", "100% guaranteed",
)

# Clickbait framing likely to make a hook untrustworthy.
_CLICKBAIT_PHRASES = (
    "you won't believe", "you wont believe", "everyone knows", "the truth about",
    "the truth is", "mind-blowing", "mind blowing", "shocking", "insane",
    "unbelievable", "must watch", "this will blow your mind",
)

# Inert hook prefixes that add no factual content of their own.
_FRAMING_PREFIXES = (
    "you probably didn't know this, but", "you won't believe", "did you know",
    "have you ever wondered", "here's something", "here is something",
    "believe it or not", "fun fact", "guess what", "ready for this",
)

_ENGAGEMENT_CUES = (
    "follow", "subscribe", "like", "comment", "share", "watch", "more",
    "question", "know", "learn",
)

_ENDING_PURPOSES = ("conclusion", "call_to_action")


# --------------------------------------------------------------------------- #
# 1. Factual fidelity (hard gate)
# --------------------------------------------------------------------------- #


def check_factual_fidelity(
    script: Script,
    fact_check_result: FactCheckResult,
    claims_by_id: dict[str, FactClaim],
    config: ScriptQAConfig,
) -> list[QAIssue]:
    """Factual lines must stay inside their approved claims' wording.

    `meaningful_tokens` coverage is the deterministic proxy for 'the line only
    says what the claims say'. A negation token never said by the claims is a
    material contradiction; otherwise poor coverage is invented/divergent
    wording. Both are HARD gates (reject).
    """
    issues: list[QAIssue] = []
    for line in script.lines:
        if not line.is_factual():
            continue
        refs = _ref_claims(line, claims_by_id)
        if not refs:
            continue  # traceability reports missing refs
        claim_words = set()
        for claim in refs:
            claim_words |= set(_simple_words(claim.claim_text))
        line_words = _simple_words(line.text)
        negated = [w for w in _NEGATION_WORDS if w in line_words and w not in claim_words]
        if negated:
            issues.append(
                _issue(
                    "factual_fidelity",
                    "ERROR",
                    f"line {line.line_id} materially contradicts its approved "
                    f"claim(s) (negation {negated[0]!r} not present in the claims)",
                    line_ids=[line.line_id],
                    claim_ids=line.claim_refs,
                    gate=GATE_HARD,
                    check="factual_fidelity",
                )
            )
            continue
        coverage = _coverage(refs, line.text, config.fidelity_coverage_threshold)
        if coverage is not None and coverage < config.fidelity_coverage_threshold:
            ratio = f"{coverage:.2f}"
            issues.append(
                _issue(
                    "factual_fidelity",
                    "ERROR",
                    f"line {line.line_id} invents wording/details not represented "
                    f"by its approved claims (token coverage {ratio})",
                    line_ids=[line.line_id],
                    claim_ids=line.claim_refs,
                    gate=GATE_HARD,
                    check="factual_fidelity",
                )
            )
    return issues


# --------------------------------------------------------------------------- #
# 2. Claim traceability (hard gates)
# --------------------------------------------------------------------------- #


def check_claim_traceability(
    script: Script,
    fact_check_result: FactCheckResult,
    claims_by_id: dict[str, FactClaim],
    config: ScriptQAConfig,
) -> list[QAIssue]:
    """Every factual line must cite approved claims; no stale/foreign ids."""
    issues: list[QAIssue] = []
    if script.result_id != fact_check_result.result_id:
        issues.append(
            _issue(
                "claim_traceability",
                "ERROR",
                f"script result_id {script.result_id!r} does not match the "
                f"fact-check result {fact_check_result.result_id!r}",
                gate=GATE_HARD,
                check="claim_traceability",
            )
        )
    if fact_check_result.package_verdict == "REJECT":
        issues.append(
            _issue(
                "claim_traceability",
                "ERROR",
                "package was not approved for scripting (package verdict REJECT)",
                gate=GATE_HARD,
                check="claim_traceability",
            )
        )
    if fact_check_result.package_verdict == "REVIEW":
        issues.append(
            _issue(
                "claim_traceability",
                "ERROR",
                "package was not approved for scripting (package verdict REVIEW)",
                gate=GATE_HARD,
                check="claim_traceability",
            )
        )
    if script.provenance is None:
        issues.append(
            _issue(
                "claim_traceability",
                "ERROR",
                "script artifact has no provenance; its origin cannot be verified",
                gate=GATE_HARD,
                check="claim_traceability",
            )
        )

    verdict_by_id = {v.claim_id: v for v in fact_check_result.claim_results}
    approved = set(fact_check_result.approved_claims)
    rejected = set(fact_check_result.rejected_claims)
    review = set(fact_check_result.review_items)

    for line in script.lines:
        if line.is_factual() and not line.claim_refs:
            issues.append(
                _issue(
                    "claim_traceability",
                    "ERROR",
                    f"factual line {line.line_id} has no claim traceability "
                    "(no claim_refs)",
                    line_ids=[line.line_id],
                    gate=GATE_HARD,
                    check="claim_traceability",
                )
            )
        for claim_id in line.claim_refs:
            if claim_id not in verdict_by_id:
                issues.append(
                    _issue(
                        "claim_traceability",
                        "ERROR",
                        f"line {line.line_id} references unknown claim {claim_id}",
                        line_ids=[line.line_id],
                        claim_ids=[claim_id],
                        gate=GATE_HARD,
                        check="claim_traceability",
                    )
                )
            elif claim_id in rejected:
                issues.append(
                    _issue(
                        "claim_traceability",
                        "ERROR",
                        f"line {line.line_id} references REJECTED claim {claim_id}",
                        line_ids=[line.line_id],
                        claim_ids=[claim_id],
                        gate=GATE_HARD,
                        check="claim_traceability",
                    )
                )
            elif claim_id in review:
                issues.append(
                    _issue(
                        "claim_traceability",
                        "ERROR",
                        f"line {line.line_id} references REVIEW (unapproved) "
                        f"claim {claim_id}",
                        line_ids=[line.line_id],
                        claim_ids=[claim_id],
                        gate=GATE_HARD,
                        check="claim_traceability",
                    )
                )
            elif claim_id not in approved:
                issues.append(
                    _issue(
                        "claim_traceability",
                        "ERROR",
                        f"line {line.line_id} references unapproved claim {claim_id}",
                        line_ids=[line.line_id],
                        claim_ids=[claim_id],
                        gate=GATE_HARD,
                        check="claim_traceability",
                    )
                )

    used = script.all_claim_refs()
    if set(script.used_claim_ids) != used:
        issues.append(
            _issue(
                "claim_traceability",
                "ERROR",
                "used_claim_ids must equal the exact set of claim_refs across lines",
                gate=GATE_HARD,
                check="claim_traceability",
            )
        )
    if set(script.unused_approved_claim_ids) & used:
        issues.append(
            _issue(
                "claim_traceability",
                "ERROR",
                "unused_approved_claim_ids overlaps used_claim_ids",
                gate=GATE_HARD,
                check="claim_traceability",
            )
        )
    if not script.package_id.strip() or script.package_id != fact_check_result.package_id:
        issues.append(
            _issue(
                "claim_traceability",
                "ERROR",
                f"script package_id {script.package_id!r} does not bind to the "
                f"fact-check result package {fact_check_result.package_id!r}",
                gate=GATE_HARD,
                check="claim_traceability",
            )
        )
    declared = set(script.used_claim_ids) | set(script.unused_approved_claim_ids)
    if declared != approved:
        issues.append(
            _issue(
                "claim_traceability",
                "ERROR",
                "claim accounting is inconsistent: used_claim_ids | "
                "unused_approved_claim_ids must exactly equal the approved claim set",
                gate=GATE_HARD,
                check="claim_traceability",
            )
        )
    return issues


# --------------------------------------------------------------------------- #
# 3. Claim coverage
# --------------------------------------------------------------------------- #


def check_claim_coverage(
    script: Script,
    fact_check_result: FactCheckResult,
    claims_by_id: dict[str, FactClaim],
    config: ScriptQAConfig,
) -> list[QAIssue]:
    """Track approved vs used vs unused claims; flag omitted critical claims.

    Non-critical unused claims are allowed (a Short selects its best material).
    """
    issues: list[QAIssue] = []
    verdict_by_id = {v.claim_id: v for v in fact_check_result.claim_results}
    approved = list(fact_check_result.approved_claims)
    used = set(script.all_claim_refs())

    def _critical(claim_id: str) -> bool:
        claim = claims_by_id.get(claim_id)
        if claim is not None and claim.critical:
            return True
        verdict = verdict_by_id.get(claim_id)
        return bool(verdict is not None and verdict.critical)

    for claim_id in approved:
        if claim_id in used:
            continue
        if _critical(claim_id):
            severity = "ERROR" if config.critical_unused_is_error else "WARNING"
            issues.append(
                _issue(
                    "claim_coverage",
                    severity,
                    f"critical approved claim {claim_id} is omitted from the script",
                    claim_ids=[claim_id],
                    gate=GATE_SOFT,
                    check="claim_coverage",
                )
            )
        else:
            issues.append(
                _issue(
                    "claim_coverage",
                    "INFO",
                    f"approved claim {claim_id} is not used in the script "
                    "(selection allowed)",
                    claim_ids=[claim_id],
                    gate=GATE_SOFT,
                    check="claim_coverage",
                )
            )
    return issues


# --------------------------------------------------------------------------- #
# 4. Length / duration
# --------------------------------------------------------------------------- #


def check_length_duration(
    script: Script,
    fact_check_result: FactCheckResult,
    claims_by_id: dict[str, FactClaim],
    config: ScriptQAConfig,
) -> list[QAIssue]:
    """Estimated duration against the preferred 30-60s Shorts window.

    Slightly outside the preferred range -> WARNING; clearly unusable
    (below hard min / above hard max) -> hard ERROR.
    """
    issues: list[QAIssue] = []
    if not script.lines:
        return issues  # empty-script gate lives in the structure check
    duration = float(script.estimated_duration_seconds or 0.0)
    if duration < config.hard_min_duration_seconds:
        issues.append(
            _issue(
                "length_duration",
                "ERROR",
                f"estimated duration {duration:.1f}s is below the usable minimum "
                f"of {config.hard_min_duration_seconds:.0f}s",
                gate=GATE_HARD,
                check="length_duration",
            )
        )
    elif duration > config.hard_max_duration_seconds:
        issues.append(
            _issue(
                "length_duration",
                "ERROR",
                f"estimated duration {duration:.1f}s clearly exceeds the usable "
                f"Short-form maximum of {config.hard_max_duration_seconds:.0f}s",
                gate=GATE_HARD,
                check="length_duration",
            )
        )
    elif duration < config.min_duration_seconds:
        issues.append(
            _issue(
                "length_duration",
                "WARNING",
                f"estimated duration {duration:.1f}s is below the preferred "
                f"{config.min_duration_seconds:.0f}s target",
                gate=GATE_SOFT,
                check="length_duration",
            )
        )
    elif duration > config.max_duration_seconds:
        issues.append(
            _issue(
                "length_duration",
                "WARNING",
                f"estimated duration {duration:.1f}s is above the preferred "
                f"{config.max_duration_seconds:.0f}s Shorts target",
                gate=GATE_SOFT,
                check="length_duration",
            )
        )
    return issues


# --------------------------------------------------------------------------- #
# 5. Pacing
# --------------------------------------------------------------------------- #


def check_pacing(
    script: Script,
    fact_check_result: FactCheckResult,
    claims_by_id: dict[str, FactClaim],
    config: ScriptQAConfig,
) -> list[QAIssue]:
    """Deterministic line-level pacing heuristics (no LLM)."""
    issues: list[QAIssue] = []
    literal_lines = list(script.lines)

    for line in literal_lines:
        word_count = len(_words(line.text))
        if word_count > config.hard_max_line_words:
            issues.append(
                _issue(
                    "pacing",
                    "ERROR",
                    f"line {line.line_id} is excessively long for a Short "
                    f"({word_count} words)",
                    line_ids=[line.line_id],
                    gate=GATE_SOFT,
                    check="pacing",
                )
            )
        elif word_count > config.max_line_words:
            issues.append(
                _issue(
                    "pacing",
                    "WARNING",
                    f"line {line.line_id} is long ({word_count} words) "
                    "for a Short",
                    line_ids=[line.line_id],
                    gate=GATE_SOFT,
                    check="pacing",
                )
            )
        sentence_count = len(_sentences(line.text))
        connectors = sum(
            word in {"and", "or", "but", "because", "which", "while", "however"}
            for word in _words(line.text)
        )
        if sentence_count > 2 and word_count > 20:
            issues.append(
                _issue(
                    "pacing",
                    "WARNING",
                    f"line {line.line_id} packs multiple sentences/ideas into "
                    "one line",
                    line_ids=[line.line_id],
                    gate=GATE_SOFT,
                    check="pacing",
                )
            )
        elif connectors >= 5:
            issues.append(
                _issue(
                    "pacing",
                    "WARNING",
                    f"line {line.line_id} chains many ideas with connectors",
                    line_ids=[line.line_id],
                    gate=GATE_SOFT,
                    check="pacing",
                )
            )

    # Uninterrupted explanation block (consecutive factual lines).
    run = 0
    for line in literal_lines:
        if line.purpose in ("fact", "explanation"):
            run += len(_words(line.text))
            if run > config.max_block_words:
                issues.append(
                    _issue(
                        "pacing",
                        "WARNING",
                        f"uninterrupted explanation runs ~{run} words without a break",
                        gate=GATE_SOFT,
                        check="pacing",
                    )
                )
                break
        else:
            run = 0

    # Excessive setup before the first factual line.
    setup_words = 0
    for line in literal_lines:
        if line.purpose in ("fact", "explanation"):
            break
        setup_words += len(_words(line.text))
    else:
        setup_words = 0
    if setup_words > config.max_setup_words:
        issues.append(
            _issue(
                "pacing",
                "WARNING",
                f"excessive setup ({setup_words} words) before the main point",
                gate=GATE_SOFT,
                check="pacing",
            )
        )
    return issues


# --------------------------------------------------------------------------- #
# 6. Clarity
# --------------------------------------------------------------------------- #


def check_clarity(
    script: Script,
    fact_check_result: FactCheckResult,
    claims_by_id: dict[str, FactClaim],
    config: ScriptQAConfig,
) -> list[QAIssue]:
    """Understandability heuristics: long sentences, jargon, parentheticals."""
    issues: list[QAIssue] = []
    previous_purpose = None
    for index, line in enumerate(script.lines):
        long_sentence = max(
            (len(_words(sentence)) for sentence in _sentences(line.text) if sentence),
            default=0,
        )
        if long_sentence > config.max_sentence_words:
            issues.append(
                _issue(
                    "clarity",
                    "WARNING",
                    f"line {line.line_id} has an excessively long sentence "
                    f"({long_sentence} words)",
                    line_ids=[line.line_id],
                    gate=GATE_SOFT,
                    check="clarity",
                )
            )
        acronyms = re.findall(r"\b[A-Z][A-Z]{1,5}\b", line.text)
        for acronym in sorted(set(acronyms)):
            issues.append(
                _issue(
                    "clarity",
                    "WARNING",
                    f"line {line.line_id} uses unexplained acronym {acronym!r}",
                    line_ids=[line.line_id],
                    gate=GATE_SOFT,
                    check="clarity",
                )
            )
        for term in _WORD.findall(line.text):
            if len(term) > 18:
                issues.append(
                    _issue(
                        "clarity",
                        "WARNING",
                        f"line {line.line_id} uses a long, possibly unexplained "
                        f"technical term {term!r}",
                        line_ids=[line.line_id],
                        gate=GATE_SOFT,
                        check="clarity",
                    )
                )
        if line.text.count("(") + line.text.count(")") > 4:
            issues.append(
                _issue(
                    "clarity",
                    "WARNING",
                    f"line {line.line_id} has excessive parentheticals",
                    line_ids=[line.line_id],
                    gate=GATE_SOFT,
                    check="clarity",
                )
            )
        first_word = line.text.strip().split(" ", 1)[0].lower().rstrip(",.;:") if line.text.strip() else ""
        if (
            line.is_factual()
            and index > 0
            and previous_purpose is not None
            and first_word in {"it", "they", "this", "that", "these", "those", "there"}
        ):
            issues.append(
                _issue(
                    "clarity",
                    "WARNING",
                    f"line {line.line_id} opens with an ambiguous reference "
                    f"({first_word!r})",
                    line_ids=[line.line_id],
                    gate=GATE_SOFT,
                    check="clarity",
                )
            )
        previous_purpose = line.purpose
    return issues


# --------------------------------------------------------------------------- #
# 7. Repetition
# --------------------------------------------------------------------------- #


def check_repetition(
    script: Script,
    fact_check_result: FactCheckResult,
    claims_by_id: dict[str, FactClaim],
    config: ScriptQAConfig,
) -> list[QAIssue]:
    """Unnecessary repetition vs intentional hook/conclusion reinforcement."""
    issues: list[QAIssue] = []
    literal_lines = list(script.lines)

    # Identical factual statements repeated in different body lines.
    groups: dict[str, list[str]] = defaultdict(list)
    for line in literal_lines:
        if line.purpose in ("fact", "explanation"):
            groups[_norm(line.text)].append(line.line_id)
    for norm, line_ids in groups.items():
        if len(line_ids) >= 2:
            issues.append(
                _issue(
                    "repetition",
                    "WARNING",
                    f"identical factual statement repeated ({norm!r})",
                    line_ids=line_ids,
                    gate=GATE_SOFT,
                    check="repetition",
                )
            )

    for i in range(len(literal_lines)):
        for j in range(i + 1, len(literal_lines)):
            left, right = literal_lines[i], literal_lines[j]
            norm_left, norm_right = _norm(left.text), _norm(right.text)
            if norm_left == norm_right:
                continue  # exact duplicates handled above (body) or intentional mirrors
            left_words, right_words = _words(left.text), _words(right.text)
            if not left_words or not right_words:
                continue
            overlap = set(left_words) & set(right_words)
            union = set(left_words) | set(right_words)
            similarity = len(overlap) / len(union)
            if similarity < 0.8:
                continue
            both_factual = left.is_factual() and right.is_factual()
            mirror = {left.purpose, right.purpose} <= {"hook", "conclusion"}
            # A conclusion legitimately restates the opening claim; only flag
            # repetition between two non-conclusion body lines.
            if (
                both_factual
                and (j - i) > 1
                and left.purpose not in _ENDING_PURPOSES
                and right.purpose not in _ENDING_PURPOSES
            ):
                issues.append(
                    _issue(
                        "repetition",
                        "WARNING",
                        f"near-duplicate factual statement in {left.line_id} "
                        f"and {right.line_id}",
                        line_ids=[left.line_id, right.line_id],
                        gate=GATE_SOFT,
                        check="repetition",
                    )
                )
            elif mirror:
                issues.append(
                    _issue(
                        "repetition",
                        "INFO",
                        f"hook/conclusion repetition in {left.line_id} and "
                        f"{right.line_id} looks like intentional reinforcement",
                        line_ids=[left.line_id, right.line_id],
                        gate=GATE_SOFT,
                        check="repetition",
                    )
                )
            elif (
                left.purpose == "hook"
                and right.purpose in ("fact", "explanation")
                and (j - i) == 1
            ):
                issues.append(
                    _issue(
                        "repetition",
                        "INFO",
                        f"hook {left.line_id} reuses the opening claim of "
                        f"{right.line_id}; intentional",
                        line_ids=[left.line_id, right.line_id],
                        gate=GATE_SOFT,
                        check="repetition",
                    )
                )
    return issues


# --------------------------------------------------------------------------- #
# 8. Structure
# --------------------------------------------------------------------------- #


def check_structure(
    script: Script,
    fact_check_result: FactCheckResult,
    claims_by_id: dict[str, FactClaim],
    config: ScriptQAConfig,
) -> list[QAIssue]:
    """Opening -> body -> ending coherence. Empty scripts are fatal."""
    issues: list[QAIssue] = []
    if not script.lines:
        issues.append(
            _issue(
                "structure",
                "ERROR",
                "script is empty and structurally unusable",
                gate=GATE_HARD,
                check="structure",
            )
        )
        return issues
    unknown = next(
        (line for line in script.lines if line.purpose not in PURPOSE_VALUES), None
    )
    if unknown is not None:
        issues.append(
            _issue(
                "structure",
                "ERROR",
                f"line {unknown.line_id} uses unsupported purpose "
                f"{unknown.purpose!r} (malformed script)",
                line_ids=[unknown.line_id],
                gate=GATE_HARD,
                check="structure",
            )
        )
    line_ids = [line.line_id for line in script.lines]
    if len(set(line_ids)) != len(line_ids):
        issues.append(
            _issue(
                "structure",
                "ERROR",
                "script contains duplicate line ids (malformed script)",
                gate=GATE_HARD,
                check="structure",
            )
        )
    if not any(line.purpose in ("fact", "explanation") for line in script.lines):
        issues.append(
            _issue(
                "structure",
                "ERROR",
                "script has no main information (fact/explanation) lines",
                gate=GATE_SOFT,
                check="structure",
            )
        )
    if not any(line.purpose in _ENDING_PURPOSES for line in script.lines):
        issues.append(
            _issue(
                "structure",
                "ERROR",
                "script has no ending (conclusion or call-to-action)",
                gate=GATE_SOFT,
                check="structure",
            )
        )
    if script.lines[-1].purpose not in _ENDING_PURPOSES:
        issues.append(
            _issue(
                "structure",
                "WARNING",
                "script does not end on a conclusion or call-to-action",
                gate=GATE_SOFT,
                check="structure",
            )
        )
    if script.lines[0].purpose not in ("hook", "context"):
        issues.append(
            _issue(
                "structure",
                "WARNING",
                "script does not open with a hook or context line",
                gate=GATE_SOFT,
                check="structure",
            )
        )
    return issues


# --------------------------------------------------------------------------- #
# 9. Hook quality
# --------------------------------------------------------------------------- #


def check_hook_quality(
    script: Script,
    fact_check_result: FactCheckResult,
    claims_by_id: dict[str, FactClaim],
    config: ScriptQAConfig,
) -> list[QAIssue]:
    """Opening engagement heuristic: concise, curiosity without unsupported
    claims. Never predicts virality."""
    issues: list[QAIssue] = []
    hook_lines = [line for line in script.lines if line.purpose == "hook"]
    if not hook_lines:
        if config.require_hook:
            issues.append(
                _issue(
                    "hook_quality",
                    "ERROR",
                    "script has no hook line",
                    gate=GATE_SOFT,
                    check="hook_quality",
                )
            )
        return issues
    hook = hook_lines[0]
    word_count = len(_words(hook.text))
    if word_count > config.max_hook_words:
        issues.append(
            _issue(
                "hook_quality",
                "WARNING",
                f"hook is not concise ({word_count} words, limit "
                f"{config.max_hook_words})",
                line_ids=[hook.line_id],
                gate=GATE_SOFT,
                check="hook_quality",
            )
        )
    first_factual = next(
        (line for line in script.lines if line.purpose in ("fact", "explanation")), None
    )
    if first_factual is not None and _norm(hook.text) == _norm(first_factual.text):
        issues.append(
            _issue(
                "hook_quality",
                "WARNING",
                f"hook {hook.line_id} is identical to the first fact line; "
                "weak opening",
                line_ids=[hook.line_id, first_factual.line_id],
                gate=GATE_SOFT,
                check="hook_quality",
            )
        )
    lowered = hook.text.lower()
    for phrase in _CLICKBAIT_PHRASES:
        if phrase in lowered:
            issues.append(
                _issue(
                    "hook_quality",
                    "WARNING",
                    f"hook uses clickbait pattern {phrase!r}",
                    line_ids=[hook.line_id],
                    gate=GATE_SOFT,
                    check="hook_quality",
                )
            )
            break
    if not hook.claim_refs:
        rest = lowered
        for prefix in _FRAMING_PREFIXES:
            rest = rest.replace(prefix, " ")
            rest = rest.replace(prefix.title(), " ")
        if len(_tokens(rest)) >= 5:
            issues.append(
                _issue(
                    "hook_quality",
                    "WARNING",
                    f"hook {hook.line_id} asserts a fact-like statement without "
                    "any claim reference",
                    line_ids=[hook.line_id],
                    gate=GATE_SOFT,
                    check="hook_quality",
                )
            )
    return issues


# --------------------------------------------------------------------------- #
# 10. Uncertainty preservation (hard gate)
# --------------------------------------------------------------------------- #


def check_uncertainty_preservation(
    script: Script,
    fact_check_result: FactCheckResult,
    claims_by_id: dict[str, FactClaim],
    config: ScriptQAConfig,
) -> list[QAIssue]:
    """Hedged claims must keep their hedge; unsupported certainty is fatal."""
    issues: list[QAIssue] = []
    for line in script.lines:
        if not line.is_factual():
            continue
        refs = _ref_claims(line, claims_by_id)
        if not refs:
            continue
        claim_words: set[str] = set()
        claim_hedged = False
        for claim in refs:
            tokens = set(_simple_words(claim.claim_text))
            claim_words |= tokens
            if _hedge_tokens(claim.claim_text):
                claim_hedged = True
        line_words = set(_simple_words(line.text))
        if claim_hedged:
            if not _hedge_tokens(line.text):
                issues.append(
                    _issue(
                        "uncertainty_preservation",
                        "ERROR",
                        f"line {line.line_id} removes the hedge/uncertainty of "
                        "its approved claims",
                        line_ids=[line.line_id],
                        claim_ids=line.claim_refs,
                        gate=GATE_HARD,
                        check="uncertainty_preservation",
                    )
                )
            else:
                issues.append(
                    _issue(
                        "uncertainty_preservation",
                        "INFO",
                        f"line {line.line_id} preserves the hedge of its claims",
                        line_ids=[line.line_id],
                        claim_ids=line.claim_refs,
                        gate=GATE_SOFT,
                        check="uncertainty_preservation",
                    )
                )
        strengthened = next(
            (word for word in _CERTAINTY_WORDS if word in line_words and word not in claim_words),
            None,
        )
        if strengthened:
            issues.append(
                _issue(
                    "uncertainty_preservation",
                    "ERROR",
                    f"line {line.line_id} strengthens certainty with "
                    f"unsupported word {strengthened!r}",
                    line_ids=[line.line_id],
                    claim_ids=line.claim_refs,
                    gate=GATE_HARD,
                    check="uncertainty_preservation",
                )
            )
    return issues


# --------------------------------------------------------------------------- #
# 11. Misleading language
# --------------------------------------------------------------------------- #


def check_misleading_language(
    script: Script,
    fact_check_result: FactCheckResult,
    claims_by_id: dict[str, FactClaim],
    config: ScriptQAConfig,
) -> list[QAIssue]:
    """Pattern sweep for exaggerated phrasing. WARNING-level: an energetic
    phrase is not automatically fatal, but is flagged for review."""
    issues: list[QAIssue] = []
    for line in script.lines:
        lowered = line.text.lower()
        refs = _ref_claims(line, claims_by_id)
        supported = " ".join(claim.claim_text.lower() for claim in refs)
        for phrase in _MISLEADING_PHRASES:
            if phrase in lowered and phrase not in supported:
                issues.append(
                    _issue(
                        "misleading_language",
                        "WARNING",
                        f"line {line.line_id} uses potentially misleading "
                        f"phrasing {phrase!r} that its claims do not support",
                        line_ids=[line.line_id],
                        claim_ids=line.claim_refs,
                        gate=GATE_SOFT,
                        check="misleading_language",
                    )
                )
                break
    return issues


# --------------------------------------------------------------------------- #
# 12. CTA quality
# --------------------------------------------------------------------------- #


def check_cta_quality(
    script: Script,
    fact_check_result: FactCheckResult,
    claims_by_id: dict[str, FactClaim],
    config: ScriptQAConfig,
) -> list[QAIssue]:
    """CTA is optional; when present it should be short, natural, relevant."""
    issues: list[QAIssue] = []
    cta_lines = [line for line in script.lines if line.purpose == "call_to_action"]
    if not cta_lines:
        issues.append(
            _issue(
                "cta_quality",
                "INFO",
                "no call-to-action (optional; a Short does not require one)",
                gate=GATE_SOFT,
                check="cta_quality",
            )
        )
        return issues
    other_norms = {
        _norm(line.text) for line in script.lines if line.purpose != "call_to_action"
    }
    for cta in cta_lines:
        word_count = len(_words(cta.text))
        if word_count > config.max_cta_words:
            issues.append(
                _issue(
                    "cta_quality",
                    "WARNING",
                    f"CTA {cta.line_id} is too long ({word_count} words)",
                    line_ids=[cta.line_id],
                    gate=GATE_SOFT,
                    check="cta_quality",
                )
            )
        if _norm(cta.text) in other_norms:
            issues.append(
                _issue(
                    "cta_quality",
                    "WARNING",
                    f"CTA {cta.line_id} repeats an earlier line",
                    line_ids=[cta.line_id],
                    gate=GATE_SOFT,
                    check="cta_quality",
                )
            )
        lowered = cta.text.lower()
        if not any(cue in lowered for cue in _ENGAGEMENT_CUES) and word_count >= 6:
            issues.append(
                _issue(
                    "cta_quality",
                    "WARNING",
                    f"CTA {cta.line_id} lacks a clear engagement cue",
                    line_ids=[cta.line_id],
                    gate=GATE_SOFT,
                    check="cta_quality",
                )
            )
    return issues


# --------------------------------------------------------------------------- #
# 13. Format / Shorts suitability
# --------------------------------------------------------------------------- #


def check_format_shorts(
    script: Script,
    fact_check_result: FactCheckResult,
    claims_by_id: dict[str, FactClaim],
    config: ScriptQAConfig,
) -> list[QAIssue]:
    """Line counts, giant blocks, and exposition volume for a Short."""
    issues: list[QAIssue] = []
    literal_lines = list(script.lines)
    if not literal_lines:
        return issues
    count = len(literal_lines)
    if count < config.min_lines:
        issues.append(
            _issue(
                "format_shorts",
                "WARNING",
                f"only {count} line(s); a Short needs clear progression "
                f"(min recommended {config.min_lines})",
                gate=GATE_SOFT,
                check="format_shorts",
            )
        )
    elif count > config.max_lines:
        issues.append(
            _issue(
                "format_shorts",
                "WARNING",
                f"{count} lines is a lot for a Short (max recommended "
                f"{config.max_lines})",
                gate=GATE_SOFT,
                check="format_shorts",
            )
        )
    for line in literal_lines:
        if len(line.text) > 160:
            issues.append(
                _issue(
                    "format_shorts",
                    "WARNING",
                    f"line {line.line_id} is a giant block for a Short "
                    f"({len(line.text)} characters)",
                    line_ids=[line.line_id],
                    gate=GATE_SOFT,
                    check="format_shorts",
                )
            )
    word_counts = [len(_words(line.text)) for line in literal_lines]
    total = sum(word_counts)
    if literal_lines and total / len(literal_lines) > 20:
        issues.append(
            _issue(
                "format_shorts",
                "WARNING",
                f"average line is long ({total / len(literal_lines):.1f} "
                f"words/line) for short-form",
                gate=GATE_SOFT,
                check="format_shorts",
            )
        )
    if total > 150:
        issues.append(
            _issue(
                "format_shorts",
                "WARNING",
                f"excessive exposition ({total} words) for a Short",
                gate=GATE_SOFT,
                check="format_shorts",
            )
        )
    return issues


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class QACheck:
    """One registered rule. `hard_gate` is informational (issues carry their
    own per-issue gate); it documents which checks default to fatal."""

    name: str
    category: str
    hard_gate: bool
    fn: Callable[[Script, FactCheckResult, dict[str, FactClaim], ScriptQAConfig], list[QAIssue]]

    @property
    def dimension(self) -> str:
        return CATEGORY_DIMENSION[self.category]


CHECKS: tuple[QACheck, ...] = (
    QACheck("factual_fidelity", "factual_fidelity", True, check_factual_fidelity),
    QACheck("claim_traceability", "claim_traceability", True, check_claim_traceability),
    QACheck("claim_coverage", "claim_coverage", False, check_claim_coverage),
    QACheck("length_duration", "length_duration", True, check_length_duration),
    QACheck("pacing", "pacing", False, check_pacing),
    QACheck("clarity", "clarity", False, check_clarity),
    QACheck("repetition", "repetition", False, check_repetition),
    QACheck("structure", "structure", True, check_structure),
    QACheck("hook_quality", "hook_quality", False, check_hook_quality),
    QACheck("uncertainty_preservation", "uncertainty_preservation", True, check_uncertainty_preservation),
    QACheck("misleading_language", "misleading_language", False, check_misleading_language),
    QACheck("cta_quality", "cta_quality", False, check_cta_quality),
    QACheck("format_shorts", "format_shorts", False, check_format_shorts),
)


def run_checks(
    script: Script,
    fact_check_result: FactCheckResult,
    claims_by_id: dict[str, FactClaim],
    config: ScriptQAConfig,
) -> tuple[list[QAIssue], list[QACheckRecord]]:
    """Run every registered check in order; deterministic issue order.

    The overall issue id numbering is assigned later by the validator so ids
    stay stable (qa_00, qa_01, ...) regardless of content.
    """
    issues: list[QAIssue] = []
    records: list[QACheckRecord] = []
    for check in CHECKS:
        found = check.fn(script, fact_check_result, claims_by_id, config) or []
        issues.extend(found)
        records.append(
            QACheckRecord(
                check=check.name,
                category=check.category,
                dimension=check.dimension,
                hard_gate=check.hard_gate,
                errors=sum(1 for issue in found if issue.severity == "ERROR"),
                warnings=sum(1 for issue in found if issue.severity == "WARNING"),
                infos=sum(1 for issue in found if issue.severity == "INFO"),
            )
        )
    return issues, records