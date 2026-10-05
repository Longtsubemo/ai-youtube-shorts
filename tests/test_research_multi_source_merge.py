"""Defect 3 fix: duplicate claim text under another source now corroborates.

Root cause
----------
`GeminiResearchProvider._map_candidates` deduplicated identical claim text
across ALL sources with a bare `continue`. Because a claim lives inside its
source record, the *second* occurrence was discarded entirely: the corroborating
source ended up as a package `Source` contributing zero claims, and the claim
itself kept a single `source_id`. Stage 3 then read exactly one source, and
`corroboration_min_sources = 2` could never be met for a critical claim.

Two paths now produce corroboration, and both feed the unchanged
`_resolve_source_ids()`:

  A. explicit - the model names other retrieved URLs in `also_supported_by`
     (pre-existing, unchanged, still not domain-gated)
  B. implicit - the model asserts the identical claim text under a second
     retrieved source (new): claim retained once under the first asserting
     source, second source becomes additional support

Implicit merging additionally requires two different registrable domains, so a
syndicated or mirrored copy of one article cannot manufacture corroboration.

No network, no SDK, no key. Everything below is offline.

Coverage:
   A. merge basics (two/three sources, primary, ordering, determinism)
   B. non-merges (unrelated, merely topical, same registrable domain)
   C. evidence boundary (unretrieved and fabricated URLs)
   D. single-source regressions
   E. the previously claim-less source now binds
   F. Stage 3 gate UNCHANGED (satisfiable, still REVIEWs, still exactly 2)
   G. explicit also_supported_by still intact and NOT domain-gated
   H. trace corroboration counters
   I. no Stage 3 dependency from Stage 2
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from agents.qa.fact_checker.checker import FactChecker
from agents.qa.fact_checker.gates import FactCheckConfig
from agents.research.agent import ResearchAgent
from agents.research.models import ResearchPackage, ResearchPackageError, source_id_for
from agents.research.provider import TopicDescriptor
from providers.gemini.config import GeminiConfig
from providers.gemini.research_provider import (
    GeminiResearchProvider,
    _claim_dedup_key,
    _registrable_domain,
)
from tests.fact_check_test_utils import StaticEvidenceProvider, make_evidence, make_source
from tests.gemini_test_utils import FakeRawClient, make_client

# Distinct registrable domains, so implicit merging is eligible by default.
WIKI = "https://en.wikipedia.org/wiki/Radiotrophic_fungus"
WIKI2 = "https://en.wikipedia.org/wiki/Radiosynthesis_(metabolism)"
SPACE = "https://www.space.com/3845-fungi-thrive-dangerous-radiation.html"
GARDEN = "https://gardeninsider.org/gardening/knowledge/radiation-eating-fungi"

# Same registrable domain, so implicit merging must refuse.
SAME_A = "https://example.com/news/story"
SAME_B = "https://www.example.com/news/story-mirror"
SAME_C = "https://cdn.example.com/news/story-copy"

CLAIM = "Radiosynthesis is the capture of energy from ionizing radiation."
OTHER = "Radiotrophic fungi were studied aboard the International Space Station."
UNRELATED = "Unrelated claims about mollusc anatomy are not supported here."


def selection(topic_id: str = "t_radiotrophic") -> dict:
    return {
        "data": {
            "selected_topic_id": topic_id,
            "selected_topic": {"topic_id": topic_id, "title": "Radiotrophic fungi"},
        }
    }


# --------------------------------------------------------------------------- #
# Provider-level harness (drives the real public `search()`).
# --------------------------------------------------------------------------- #

def _payload(entries, *, tier: str = "high_quality_secondary") -> dict:
    """Model output from (url, claims) where each claim is text or a dict."""
    return {
        "sources": [
            {
                "url": url,
                "title": "",
                "publisher": "",
                "source_type": "web_page",
                "credibility_tier": tier_of(url, tier),
                "claims": [
                    {"claim_text": c} if isinstance(c, str) else c for c in claims
                ],
            }
            for url, claims in entries
        ]
    }


TIER_OVERRIDES: dict[str, str] = {}


def tier_of(url: str, default: str) -> str:
    return TIER_OVERRIDES.get(url, default)


class _TraceableRetriever:
    """MockResearchRetriever plus a minimal additive trace.

    MockResearchRetriever records no trace, so the provider's corroboration
    counters have nowhere to land. This stub supplies only the recording
    surface the provider calls (`record_selection` / `record_model_discarded`);
    retrieval itself is the mock's.
    """

    def __init__(self, sources, trace) -> None:
        from agents.research.retriever import MockResearchRetriever

        self._inner = MockResearchRetriever(sources)
        self._trace = trace
        self.retriever_name = "traceable_stub"

    @property
    def last_trace(self):
        return self._trace

    def retrieve(self, topic):
        return self._inner.retrieve(topic)


class _Trace:
    """The subset of the retrieval trace contract the provider writes to."""

    def __init__(self) -> None:
        self.selection: dict = {}
        self.model_discarded: list = []

    def record_selection(self, payload: dict) -> None:
        self.selection = dict(payload)

    def record_model_discarded(self, **kwargs) -> None:
        self.model_discarded.append(kwargs)


def _retrieved(urls):
    from agents.research.retriever import RetrievedSource

    return [
        RetrievedSource(url=u, title="t", publisher="p", content="body", domain="d")
        for u in urls
    ]


def _run(entries, *, tier: str = "high_quality_secondary", retrieved=None, traceable=False):
    """Drive the provider's real retrieval path over the given model output.

    `retrieved` defaults to every URL in `entries`; pass it explicitly to model
    a URL that retrieval did NOT return.
    """
    urls = [url for url, _ in entries] if retrieved is None else list(retrieved)
    sources = _retrieved(urls)
    fake = FakeRawClient()
    fake.add_result_json(_payload(entries, tier=tier))
    client = make_client(GeminiConfig(api_key="k"), raw_client=fake, sleep=lambda *_: None)
    retriever = _TraceableRetriever(sources, _Trace()) if traceable else _Mock(sources)
    provider = GeminiResearchProvider(client=client, retriever=retriever)
    candidates = provider.search(TopicDescriptor("t_radiotrophic", "Radiotrophic fungi"))
    return candidates, provider


def _Mock(sources):
    from agents.research.retriever import MockResearchRetriever

    return MockResearchRetriever(sources)


def _package(candidates) -> ResearchPackage:
    agent = ResearchAgent(
        provider=type("P", (), {"search": lambda self, t: list(candidates)})(), run_id="merge"
    )
    return agent.research(selection())


def _claim(package: ResearchPackage, text: str = CLAIM):
    for item in package.claims:
        if item.claim_text == text:
            return item
    raise AssertionError(f"claim not found: {text!r}")


def _corroboration(candidates) -> dict:
    """The additive trace block, read from the provider's own counters."""
    return GeminiResearchProvider._corroboration_view(list(candidates), None)


# ---------------------------------------------------------------------- A --
# Merge basics.
# ---------------------------------------------------------------------- #

class TestImplicitMerge:
    def test_identical_claim_under_two_sources_binds_both(self):
        candidates, _ = _run([(WIKI, [CLAIM]), (SPACE, [CLAIM])])
        claim = _claim(_package(candidates))
        assert claim.source_ids == (source_id_for(WIKI), source_id_for(SPACE))

    def test_identical_claim_under_three_sources_binds_three(self):
        candidates, _ = _run([(WIKI, [CLAIM]), (SPACE, [CLAIM]), (GARDEN, [CLAIM])])
        claim = _claim(_package(candidates))
        assert claim.source_ids == (
            source_id_for(WIKI),
            source_id_for(SPACE),
            source_id_for(GARDEN),
        )

    def test_first_asserting_source_remains_primary(self):
        """Space.com listed first, so it owns the claim."""
        candidates, _ = _run([(SPACE, [CLAIM]), (WIKI, [CLAIM])])
        claim = _claim(_package(candidates))
        assert claim.source_ids[0] == source_id_for(SPACE)
        assert claim.source_ids[1] == source_id_for(WIKI)

    def test_source_ordering_is_deterministic(self):
        first, _ = _run([(WIKI, [CLAIM, OTHER]), (SPACE, [CLAIM]), (GARDEN, [OTHER])])
        second, _ = _run([(WIKI, [CLAIM, OTHER]), (SPACE, [CLAIM]), (GARDEN, [OTHER])])
        assert [c.source_ids for c in _package(first).claims] == [
            c.source_ids for c in _package(second).claims
        ]

    def test_case_and_whitespace_variants_are_the_same_claim(self):
        candidates, _ = _run([(WIKI, [CLAIM]), (SPACE, ["  " + CLAIM.upper() + "  "])])
        assert len(_package(candidates).claims) == 1
        assert len(_claim(_package(candidates)).source_ids) == 2

    def test_merged_claim_appears_exactly_once(self):
        candidates, _ = _run([(WIKI, [CLAIM]), (SPACE, [CLAIM])])
        package = _package(candidates)
        assert [c.claim_text for c in package.claims].count(CLAIM) == 1

    def test_distinct_claims_keep_their_own_bindings(self):
        candidates, _ = _run([(WIKI, [CLAIM]), (SPACE, [CLAIM, OTHER])])
        package = _package(candidates)
        assert len(_claim(package, CLAIM).source_ids) == 2
        assert len(_claim(package, OTHER).source_ids) == 1


# ---------------------------------------------------------------------- B --
# Things that must NOT merge.
# ---------------------------------------------------------------------- #

class TestNonMerges:
    def test_unrelated_claim_text_is_not_merged(self):
        candidates, _ = _run([(WIKI, [CLAIM]), (SPACE, [UNRELATED])])
        assert len(_claim(_package(candidates), CLAIM).source_ids) == 1
        assert len(_claim(_package(candidates), UNRELATED).source_ids) == 1

    def test_merely_topical_claim_is_not_merged(self):
        """Same topic, different statement: similarity is not support."""
        candidates, _ = _run([(WIKI, [CLAIM]), (SPACE, ["Radiation is dangerous to humans."])])
        package = _package(candidates)
        for item in package.claims:
            assert len(item.source_ids) == 1

    def test_same_registrable_domain_is_not_merged(self):
        candidates, _ = _run([(SAME_A, [CLAIM]), (SAME_B, [CLAIM])])
        assert len(_claim(_package(candidates)).source_ids) == 1

    def test_same_registrable_domain_subdomain_is_not_merged(self):
        candidates, _ = _run([(SAME_A, [CLAIM]), (SAME_C, [CLAIM])])
        assert len(_claim(_package(candidates)).source_ids) == 1

    def test_different_registrable_domains_are_merged(self):
        candidates, _ = _run([(WIKI, [CLAIM]), (SPACE, [CLAIM])])
        assert len(_claim(_package(candidates)).source_ids) == 2

    def test_multi_label_public_suffix_is_treated_as_one_domain(self):
        """bbc.co.uk and cdn.bbc.co.uk share a registrable domain."""
        candidates, _ = _run(
            [
                ("https://www.bbc.co.uk/news/science", [CLAIM]),
                ("https://cdn.bbc.co.uk/news/science-mirror", [CLAIM]),
            ]
        )
        assert len(_claim(_package(candidates)).source_ids) == 1


class TestRegistrableDomainHelper:
    @pytest.mark.parametrize(
        "url,expected",
        [
            # eTLD+1, so the `en.` sub-domain collapses away.
            ("https://en.wikipedia.org/wiki/X", "wikipedia.org"),
            ("https://www.example.com/a", "example.com"),
            ("https://example.com/b", "example.com"),
            ("https://a.b.example.com/c", "example.com"),
            ("https://EXAMPLE.com/x", "example.com"),
            ("https://example.com:8443/x", "example.com"),
            ("https://www.bbc.co.uk/news", "bbc.co.uk"),
            ("https://cdn.bbc.co.uk/news", "bbc.co.uk"),
            ("https://space.com/a", "space.com"),
            ("", ""),
        ],
    )
    def test_registrable_domain(self, url, expected):
        assert _registrable_domain(url) == expected

    def test_ipv4_literals_are_not_confused_by_a_shared_tail(self):
        """1.2.3.4 and 5.6.3.4 must stay distinct."""
        assert _registrable_domain("https://1.2.3.4/x") != _registrable_domain("https://5.6.3.4/x")

    def test_unknown_public_suffix_falls_back_conservatively(self):
        """An unlisted multi-label suffix compares as the last two labels."""
        assert _registrable_domain("https://a.example.zz") == _registrable_domain(
            "https://b.example.zz"
        )

    def test_dedup_key_normalizes_case_and_whitespace(self):
        assert _claim_dedup_key("  A  B\nC ") == _claim_dedup_key("a b c")


# ---------------------------------------------------------------------- C --
# Evidence boundary must be preserved.
# ---------------------------------------------------------------------- #

class TestEvidenceBoundary:
    def test_source_outside_retrieval_evidence_cannot_be_merged(self):
        """A model-proposed URL not returned by retrieval is dropped, so it can
        neither own a claim nor be merged as corroboration."""
        candidates, _ = _run(
            [(WIKI, [CLAIM]), (SPACE, [CLAIM])], retrieved=[WIKI]
        )
        assert [c.url for c in candidates] == [WIKI]
        assert len(_claim(_package(candidates)).source_ids) == 1

    def test_fabricated_url_is_never_merged(self):
        """The model proposes a URL retrieval never returned; it is dropped
        before merging, so it can never become a supporting source."""
        candidates, _ = _run(
            [(WIKI, [CLAIM]), (SPACE, [CLAIM]), ("https://fabricated.test/x", [CLAIM])],
            retrieved=[WIKI, SPACE],
        )
        claim = _claim(_package(candidates))
        assert len(claim.source_ids) == 2
        assert "https://fabricated.test/x" not in [
            s.url for s in _package(candidates).sources
        ]

    def test_merged_ids_all_resolve_to_real_sources(self):
        candidates, _ = _run([(WIKI, [CLAIM]), (SPACE, [CLAIM])])
        package = _package(candidates)
        known = {s.source_id for s in package.sources}
        assert set(_claim(package).source_ids) <= known

    def test_package_validation_still_rejects_unknown_source_ids(self):
        from agents.research.models import FactClaim

        package = ResearchPackage.build(
            topic_id="t",
            topic_title="t",
            research_status="ready_for_fact_check",
            claims=[
                FactClaim(
                    claim_id="c1",
                    claim_text=CLAIM,
                    claim_type="fact",
                    source_ids=(source_id_for(WIKI), "src_nope0000000"),
                )
            ],
            sources=[make_source(WIKI, tier="primary")],
            run_id="r",
        )
        with pytest.raises(ResearchPackageError):
            package.validate(traceability=True)


# ---------------------------------------------------------------------- D --
# Single-source behavior must be untouched.
# ---------------------------------------------------------------------- #

class TestSingleSourceUnaffected:
    def test_single_source_claim_stays_single(self):
        candidates, _ = _run([(WIKI, [CLAIM])])
        assert _claim(_package(candidates)).source_ids == (source_id_for(WIKI),)

    def test_single_source_package_has_one_claim_one_id(self):
        candidates, _ = _run([(WIKI, [CLAIM, OTHER])])
        package = _package(candidates)
        assert len(package.claims) == 2
        assert all(len(c.source_ids) == 1 for c in package.claims)

    def test_distinct_claims_across_sources_stay_separate(self):
        candidates, _ = _run([(WIKI, [CLAIM]), (SPACE, [OTHER])])
        package = _package(candidates)
        assert len(package.claims) == 2
        assert all(len(c.source_ids) == 1 for c in package.claims)


# ---------------------------------------------------------------------- E --
# The specific regression: a source left with zero claims.
# ---------------------------------------------------------------------- #

class TestPreviouslyClaimlessSource:
    def test_duplicate_source_now_receives_a_binding(self):
        """Before the fix the second source produced no claims at all."""
        candidates, _ = _run([(WIKI, [CLAIM]), (SPACE, [CLAIM])])
        package = _package(candidates)
        assert package.claims_for_source(source_id_for(SPACE)) == [
            _claim(package)
        ] or package.claims_for_source(source_id_for(SPACE))

    def test_both_sources_appear_in_the_package(self):
        candidates, _ = _run([(WIKI, [CLAIM]), (SPACE, [CLAIM])])
        urls = {s.url for s in _package(candidates).sources}
        assert {WIKI, SPACE} <= urls


# ---------------------------------------------------------------------- F --
# Stage 3 must need NO change.
# ---------------------------------------------------------------------- #

class TestStage3GateUnchanged:
    @staticmethod
    def _check(package, claim_text):
        """Run Stage 3 over every bound source, with direct evidence each."""
        provider = StaticEvidenceProvider(
            {
                s.url: [make_evidence(s, claim_text)]
                for s in package.sources
                if s.source_id in _claim(package, claim_text).source_ids
            }
        )
        return FactChecker(provider, run_id="t").check(package)

    def test_two_eligible_sources_satisfy_corroboration(self):
        package = _package(_run([(WIKI, [CLAIM]), (SPACE, [CLAIM])])[0])
        assert len(_claim(package).source_ids) == 2
        result = self._check(package, CLAIM)
        verdict = result.claim_verdict(_claim(package).claim_id)
        assert verdict.verdict == "PASS"
        assert len(verdict.supporting_source_ids) == 2

    def test_non_critical_single_source_claim_still_passes(self):
        """Corroboration is a critical-claim requirement only.

        A non-critical claim backed by one eligible source legitimately passes;
        requiring two here would mean this fix had quietly tightened Stage 3.
        """
        package = _package(_run([(WIKI, [CLAIM])])[0])
        claim = _claim(package)
        assert claim.critical is False
        assert len(claim.source_ids) == 1
        result = self._check(package, CLAIM)
        assert result.claim_verdict(claim.claim_id).verdict == "PASS"

    def test_critical_claim_with_two_eligible_sources_passes(self):
        """The exact shape that previously could not corroborate."""
        package = _package(
            _run(
                [
                    (WIKI, [{"claim_text": CLAIM, "critical": True}]),
                    (SPACE, [{"claim_text": CLAIM, "critical": True}]),
                ]
            )[0]
        )
        claim = _claim(package)
        assert claim.critical is True
        assert len(claim.source_ids) == 2
        result = self._check(package, CLAIM)
        assert result.claim_verdict(claim.claim_id).verdict == "PASS"
        assert result.critical_claim_failures == ()

    def test_critical_claim_with_one_source_still_reviews(self):
        package = _package(_run([(WIKI, [{"claim_text": CLAIM, "critical": True}])])[0])
        claim = _claim(package)
        assert claim.critical is True
        result = self._check(package, CLAIM)
        assert result.claim_verdict(claim.claim_id).verdict == "REVIEW"
        assert claim.claim_id in result.critical_claim_failures

    def test_corroboration_min_sources_is_still_exactly_two(self):
        assert FactCheckConfig().corroboration_min_sources == 2

    def test_gate_module_has_no_provider_dependency(self):
        """Stage 2's fix must not leak into Stage 3's policy."""
        import agents.qa.fact_checker.gates as gates

        source = Path(gates.__file__).read_text(encoding="utf-8")
        assert "research_provider" not in source
        assert "_registrable_domain" not in source


# ---------------------------------------------------------------------- G --
# Explicit path must stay intact.
# ---------------------------------------------------------------------- #

class TestExplicitSupportIntact:
    def test_explicit_also_supported_by_still_binds(self):
        candidates, _ = _run(
            [
                (WIKI, [{"claim_text": CLAIM, "also_supported_by": [SPACE]}]),
                (SPACE, [OTHER]),
            ]
        )
        assert _claim(_package(candidates)).source_ids == (
            source_id_for(WIKI),
            source_id_for(SPACE),
        )

    def test_explicit_path_is_not_domain_gated(self):
        """Same-domain explicit support is still honoured, as before this fix.

        Documented consequence: the explicit path remains a separate
        independence-policy question. This test pins the current behavior so a
        future policy change is deliberate rather than accidental.
        """
        candidates, _ = _run(
            [
                (SAME_A, [{"claim_text": CLAIM, "also_supported_by": [SAME_B]}]),
                (SAME_B, [OTHER]),
            ]
        )
        assert len(_claim(_package(candidates)).source_ids) == 2

    def test_explicit_fabricated_url_is_dropped(self):
        candidates, _ = _run(
            [
                (
                    WIKI,
                    [{"claim_text": CLAIM, "also_supported_by": ["https://fabricated.test/x"]}],
                ),
                (SPACE, [OTHER]),
            ]
        )
        assert _claim(_package(candidates)).source_ids == (source_id_for(WIKI),)

    def test_explicit_and_implicit_paths_combine(self):
        """Explicit declaration plus a merged duplicate on the same claim.

        Explicit support keeps its declared position (immediately after the
        primary); the merged duplicate appends after it. Both survive the same
        evidence gate.
        """
        candidates, _ = _run(
            [
                (WIKI, [{"claim_text": CLAIM, "also_supported_by": [GARDEN]}]),
                (SPACE, [CLAIM]),
                (GARDEN, [OTHER]),
            ]
        )
        assert _claim(_package(candidates)).source_ids == (
            source_id_for(WIKI),
            source_id_for(GARDEN),
            source_id_for(SPACE),
        )


# ---------------------------------------------------------------------- H --
# Additive trace instrumentation.
# ---------------------------------------------------------------------- #

def _trace_block(entries, **kwargs) -> dict:
    _, provider = _run(entries, traceable=True, **kwargs)
    trace = provider.last_retrieval_trace
    assert trace is not None
    return trace.selection["corroboration"]


class TestTraceCorroborationCounters:
    def test_merge_is_recorded(self):
        block = _trace_block([(WIKI, [CLAIM]), (SPACE, [CLAIM])])
        assert block["duplicate_claims"] == 1
        assert block["duplicate_merged"] == 1
        assert block["duplicate_rejected_same_domain"] == 0
        assert block["claims_multi_source"] == 1
        assert block["claims_single_source"] == 0
        assert block["max_sources_per_claim"] == 2

    def test_same_domain_rejection_is_recorded(self):
        block = _trace_block([(SAME_A, [CLAIM]), (SAME_B, [CLAIM])])
        assert block["duplicate_claims"] == 1
        assert block["duplicate_merged"] == 0
        assert block["duplicate_rejected_same_domain"] == 1
        assert block["claims_single_source"] == 1

    def test_under_declaration_is_visible(self):
        """The diagnosis gap: a claim with no extra support at all."""
        block = _trace_block([(WIKI, [CLAIM, OTHER])])
        assert block["explicit_support_declared"] == 0
        assert block["duplicate_claims"] == 0
        assert block["claims_single_source"] == 2
        assert block["claims_multi_source"] == 0
        assert block["max_sources_per_claim"] == 1

    def test_counters_are_all_integers(self):
        block = _trace_block([(WIKI, [CLAIM]), (SPACE, [CLAIM])])
        assert all(isinstance(v, int) for v in block.values())

    def test_trace_records_no_raw_model_text(self):
        block = _trace_block([(WIKI, [CLAIM]), (SPACE, [CLAIM])])
        serialized = json.dumps(block)
        assert CLAIM not in serialized
        assert "http" not in serialized.lower()

    def test_explicit_and_merged_paths_are_counted_separately(self):
        """WIKI explicitly declares SPACE; SPACE also duplicates the claim.

        Both paths are attributed to themselves even though the two URL sets
        overlap, so the counters describe how corroboration was obtained rather
        than just how many distinct URLs survived.
        """
        block = _trace_block(
            [
                (WIKI, [{"claim_text": CLAIM, "also_supported_by": [SPACE]}]),
                (SPACE, [CLAIM]),
            ]
        )
        assert block["explicit_support_declared"] == 1
        assert block["explicit_support_kept"] == 1
        assert block["duplicate_merged"] == 1
        assert block["merged_support_kept"] == 1
        assert block["claims_multi_source"] == 1

    def test_explicit_support_dropped_by_evidence_is_distinguished_from_declared(self):
        """The diagnosis gap this instrumentation exists to close: the model
        declared support, but retrieval never returned that URL."""
        block = _trace_block(
            [
                (WIKI, [{"claim_text": CLAIM, "also_supported_by": ["https://fabricated.test/x"]}]),
                (SPACE, [OTHER]),
            ],
            retrieved=[WIKI, SPACE],
        )
        assert block["explicit_support_declared"] == 1
        assert block["explicit_support_kept"] == 0
        # Both claims (CLAIM on WIKI, OTHER on SPACE) stay single-source.
        assert block["claims_single_source"] == 2
        assert block["claims_multi_source"] == 0
        assert block["max_sources_per_claim"] == 1

    def test_trace_selection_keeps_its_existing_keys(self):
        """Purely additive: nothing that was already recorded disappears."""
        _, provider = _run([(WIKI, [CLAIM])], traceable=True)
        selection = provider.last_retrieval_trace.selection
        for key in (
            "offered",
            "offered_count",
            "model_proposed",
            "model_proposed_count",
            "kept",
            "kept_count",
            "discarded",
            "discarded_count",
            "dropped_by_evidence_gate",
            "dropped_by_evidence_gate_count",
            "model_discarded",
        ):
            assert key in selection, key
        assert "corroboration" in selection


# ---------------------------------------------------------------------- I --
# Stage 2 must not depend on Stage 3.
# ---------------------------------------------------------------------- #

class TestNoStage3Dependency:
    """Stage 2's merge must not reach into Stage 3.

    Scoped to `agents.research.*`, which is the Stage 2 code this change
    actually touches. `providers.gemini` is deliberately NOT probed here: its
    package `__init__` imports `script_provider`, which imports
    `agents.qa.fact_checker.models` for the script stage. That chain is
    pre-existing, unrelated to this fix, and covered by neither this test nor
    this task's scope. The fix adds no import of Stage 3 anywhere, which
    `test_merge_adds_no_new_import` asserts directly on the source.
    """

    @staticmethod
    def _probe(module: str) -> list[str]:
        code = (
            "import sys, json\n"
            f"import {module}\n"
            "print(json.dumps(sorted("
            "m for m in sys.modules if m.startswith('agents.qa')"
            ")))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[1],
        )
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout.strip())

    def test_research_agent_imports_no_stage3_module(self):
        assert self._probe("agents.research.agent") == []

    def test_research_models_import_no_stage3_module(self):
        assert self._probe("agents.research.models") == []

    def test_merge_adds_no_new_import(self):
        """The merge introduced no Stage 3 (or any new) import."""
        source = (
            Path(__file__).resolve().parents[1]
            / "providers"
            / "gemini"
            / "research_provider.py"
        ).read_text(encoding="utf-8")
        imports = [
            line.strip()
            for line in source.splitlines()
            if line.startswith(("import ", "from ")) and "agents.qa" in line
        ]
        assert imports == []

    def test_probe_would_detect_a_real_dependency(self):
        """Negative control: the probe is not vacuously true."""
        assert "agents.qa.fact_checker.gates" in self._probe(
            "agents.qa.fact_checker.gates"
        )


# --------------------------------------------------------------------------- #
# Schema / provenance stability.
# --------------------------------------------------------------------------- #

class TestSchemaAndProvenanceUnchanged:
    def test_research_package_field_set_is_unchanged(self):
        package = _package(_run([(WIKI, [CLAIM]), (SPACE, [CLAIM])])[0])
        assert set(_claim(package).to_dict()) == {
            "claim_id",
            "claim_text",
            "claim_type",
            "source_ids",
            "confidence",
            "verification_status",
            "dates",
            "notes",
            "critical",
        }

    def test_multi_source_serializes_as_a_json_list(self):
        package = _package(_run([(WIKI, [CLAIM]), (SPACE, [CLAIM])])[0])
        payload = package.to_dict()
        entry = next(c for c in payload["data"]["claims"] if c["claim_text"] == CLAIM)
        assert isinstance(entry["source_ids"], list)
        assert len(entry["source_ids"]) == 2
        assert ResearchPackage.from_dict(payload).claims

    def test_tiering_is_untouched_by_merging(self):
        candidates, _ = _run([(WIKI, [CLAIM]), (SPACE, [CLAIM])], tier="primary")
        package = _package(candidates)
        assert all(s.credibility_tier == "primary" for s in package.sources)

    def test_credibility_tiering_untouched_when_one_source_is_discovery(self):
        from agents.research.retriever import MockResearchRetriever, RetrievedSource

        retrieved = [
            RetrievedSource(url=u, title="t", publisher="p", content="c", domain="d")
            for u in (WIKI, SPACE)
        ]
        payload = {
            "sources": [
                {
                    "url": WIKI,
                    "title": "",
                    "publisher": "",
                    "source_type": "web_page",
                    "credibility_tier": "primary",
                    "claims": [{"claim_text": CLAIM}],
                },
                {
                    "url": SPACE,
                    "title": "",
                    "publisher": "",
                    "source_type": "web_page",
                    "credibility_tier": "discovery",
                    "claims": [{"claim_text": CLAIM}],
                },
            ]
        }
        fake = FakeRawClient()
        fake.add_result_json(payload)
        client = make_client(GeminiConfig(api_key="k"), raw_client=fake, sleep=lambda *_: None)
        candidates = GeminiResearchProvider(
            client=client, retriever=MockResearchRetriever(retrieved)
        ).search(TopicDescriptor("t", "x"))
        tiers = {s.url: s.credibility_tier for s in _package(candidates).sources}
        assert tiers[WIKI] == "primary"
        assert tiers[SPACE] == "discovery"

    def test_existing_single_source_artifact_still_loads(self):
        """Legacy payloads are unaffected by the merge."""
        legacy = {
            "package_id": "p1",
            "run_id": "r1",
            "topic_id": "t1",
            "topic_title": "t",
            "research_status": "ready_for_fact_check",
            "claims": [
                {
                    "claim_id": "c1",
                    "claim_text": CLAIM,
                    "claim_type": "fact",
                    "source_ids": [source_id_for(WIKI)],
                    "confidence": 0.5,
                    "verification_status": "provisional",
                    "dates": [],
                    "notes": "",
                    "critical": False,
                }
            ],
            "sources": [
                {
                    "source_id": source_id_for(WIKI),
                    "url": WIKI,
                    "title": "t",
                    "publisher": "p",
                    "source_type": "web_page",
                    "credibility_tier": "primary",
                }
            ],
        }
        package = ResearchPackage.from_dict({"created_at": "2024-01-01T00:00:00+00:00",
                                            "run_id": "r1", "data": legacy})
        package.validate(traceability=True)
        assert package.claims[0].source_ids == (source_id_for(WIKI),)