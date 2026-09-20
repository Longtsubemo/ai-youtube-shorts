"""Shared builders/stubs for the Stage-5 (script QA) test suite.

Everything here is synthetic and MOCK - same rule as the Stage-3/4 test
builders: no source stands behind these claims and nothing is asserted true.
The helpers build exact PASS/REVISE/REJECT scenarios offline.
"""

from __future__ import annotations

from agents.qa.script.models import ScriptQAConfig
from agents.research.models import FactClaim, claim_id_for
from fact_check_test_utils import FIXED_AT, make_claim, make_package
from script_test_utils import (
    OCTOPUS_BLUE,
    OCTOPUS_SMARTEST,
    OCTOPUS_SWIM,
    OCTOPUS_THREE_HEARTS,
    build_octopus_pair,
    build_pair,
    make_script,
    make_verdict,
)

__all__ = [
    "FIXED_AT",
    "OCTOPUS_BLUE",
    "OCTOPUS_SMARTEST",
    "OCTOPUS_SWIM",
    "OCTOPUS_THREE_HEARTS",
    "claim_index",
    "claim_ref",
    "generate_and_eval",
    "load_mock_package",
    "load_mock_script_and_result",
    "make_critical_pair",
    "make_octopus_pair",
    "make_script",
    "make_simple_pair",
    "octopus_ok_lines",
    "run_checks_only",
    "run_qa_eval",
]


def claim_index(package) -> dict[str, FactClaim]:
    """Index a ResearchPackage's claim records by claim id."""
    return {claim.claim_id: claim for claim in package.claims}


def run_qa_eval(
    script,
    result,
    package,
    config: ScriptQAConfig | None = None,
    run_id: str = "script-test",
    created_at: str = FIXED_AT,
):
    """Evaluate a script through the real QA entrypoint (network-free)."""
    from agents.qa.script.validator import evaluate_script

    return evaluate_script(
        script,
        result,
        claim_index(package),
        config=config,
        run_id=run_id,
        created_at=created_at,
    )


def run_checks_only(script, result, package, config: ScriptQAConfig | None = None):
    """Run the CHECKS registry directly (no validator wrapping)."""
    from agents.qa.script.checks import CHECKS

    config = config or ScriptQAConfig()
    issues = []
    for check in CHECKS:
        issues.extend(check.fn(script, result, claim_index(package), config) or [])
    return issues


def generate_and_eval(result, package, created_at: str = FIXED_AT):
    """Run Stage-4 generation then Stage-5 QA on a (result, package) pair."""
    from agents.qa.script.validator import evaluate_script
    from agents.script.script_generator import MockScriptGenerator, ScriptConfig

    generation = MockScriptGenerator(config=ScriptConfig()).generate(
        result,
        package.claims,
        config=ScriptConfig(),
        run_id="script-test",
        created_at=created_at,
    )
    assert generation.generated, generation.message
    return evaluate_script(
        generation.script,
        result,
        claim_index(package),
        run_id="script-test",
        created_at=created_at,
    )


def octopus_ok_lines(result) -> list[dict]:
    """A healthy full-script line spec mirroring the mock generator's output."""
    conclusion = "So here's the bottom line: " + OCTOPUS_THREE_HEARTS.split(":", 1)[1]
    return [
        {"text": "You probably didn't know this, but Octopuses have three hearts",
         "purpose": "hook", "claim_refs": (result.approved_claims[0],)},
        {"text": "Why octopuses have three hearts? Let's take a closer look at the answer.",
         "purpose": "context", "claim_refs": ()},
        {"text": OCTOPUS_THREE_HEARTS, "purpose": "fact", "claim_refs": (result.approved_claims[0],)},
        {"text": OCTOPUS_SWIM, "purpose": "explanation", "claim_refs": (result.approved_claims[1],)},
        {"text": OCTOPUS_BLUE, "purpose": "explanation", "claim_refs": (result.approved_claims[2],)},
        {"text": "Here's one more detail worth knowing.", "purpose": "transition", "claim_refs": ()},
        {"text": conclusion, "purpose": "conclusion", "claim_refs": (result.approved_claims[0],)},
        {"text": "If you enjoyed this, follow for more science explainers like this one.",
         "purpose": "call_to_action", "claim_refs": ()},
    ]


def make_octopus_pair(
    approved=("fact", "explanation", "explanation"),
    package_verdict="PASS",
    run_id="script-test",
    created_at=FIXED_AT,
):
    """Octopus (result, package). The two unapproved claims stay REVIEW/REJECT."""
    return build_octopus_pair(
        approved=approved,
        package_verdict=package_verdict,
        run_id=run_id,
        created_at=created_at,
    )


def make_simple_pair(
    claims=(),
    verdicts=None,
    package_verdict="PASS",
    run_id="script-test",
    topic_id="t_test",
    topic_title="Test topic",
):
    """Arbitrary (result, package) from (text, claim_type) pairs."""
    return build_pair(
        claims=claims,
        verdicts=verdicts,
        package_verdict=package_verdict,
        topic_id=topic_id,
        topic_title=topic_title,
        run_id=run_id,
    )


def make_critical_pair(
    run_id="script-test",
    created_at=FIXED_AT,
):
    """Octopus pair where OCTOPUS_SWIM is approved AND critical."""
    from agents.qa.fact_checker.models import FactCheckResult, result_id_for

    claims = [
        make_claim(OCTOPUS_THREE_HEARTS, claim_type="fact"),
        make_claim(OCTOPUS_SWIM, claim_type="explanation", critical=True),
        make_claim(OCTOPUS_BLUE, claim_type="explanation"),
    ]
    package = make_package(
        claims=claims,
        topic_id="t_octopus_hearts",
        topic_title="Why octopuses have three hearts",
    )
    claim_results = tuple(make_verdict(c.claim_id, verdict="PASS", critical=c.critical) for c in claims)
    approved = tuple(c.claim_id for c in claims)
    result = FactCheckResult(
        result_id=result_id_for(run_id, package.package_id),
        topic_id="t_octopus_hearts",
        topic_title="Why octopuses have three hearts",
        package_id=package.package_id,
        claim_results=claim_results,
        package_verdict="PASS",
        review_items=(),
        rejected_claims=(),
        approved_claims=approved,
        created_at=created_at,
        run_id=run_id,
    )
    return result.validate(), package


def load_mock_script_and_result():
    """Load the committed data/mock artifacts used by the CLI/end-to-end tests."""
    from agents.qa.script.stage import load_fact_check_result, load_script

    return load_script(), load_fact_check_result()


def load_mock_package():
    """Rebuild the deterministic mock octopus pack in memory."""
    from agents.research.stage import load_topic_selection, run_research
    from agents.research.provider import MockResearchProvider

    selection = load_topic_selection()
    return run_research(selection, MockResearchProvider(), run_id="mock")


def claim_ref(text: str) -> str:
    """Claim id for a raw claim text (must match a package claim)."""
    return claim_id_for(text)