"""Finance-channel configuration and topic-selection tests.

Covers the finance_mascot Stage 1 scope:
- the six finance pillars are the active configuration,
- obsolete general-interest pillars cannot be emitted or fall back,
- channel identity (promise, audience, tone, mascot) reaches the topic selector,
- candidate selection criteria are curiosity/usefulness/visual/fit and NOT
  source availability, corroboration, or researchability,
- the selected-topic contract exposes topic_id/title/pillar/angle/hook,
- the deterministic mock path stays deterministic,
- the optional Gemini path carries the same contract.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.seo.config import (
    TopicSelectionConfigError,
    load_config,
)
from agents.seo.history import SelectionHistory
from agents.seo.models import TopicCandidate, TopicCandidateError
from agents.seo.scorer import TopicScorer
from agents.seo.selector import TopicSelector
from agents.seo.stage import load_candidates, run_topic_selection
from config.channel import CHANNEL, ChannelConfig

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "topic_selection.yaml"
MOCK_CANDIDATES = ROOT / "data" / "mock" / "topic_candidates.json"

FINANCE_PILLARS = (
    "wealth_engine",
    "silent_wealth_killers",
    "psychology_of_capital",
    "tactical_blueprints",
    "income_expansion",
    "financial_autopsies",
)

OBSOLETE_PILLARS = ("ai", "technology", "science", "space", "curiosity")

PILLAR_LABELS = {
    "wealth_engine": "Wealth Engine (How Money Grows)",
    "silent_wealth_killers": "Silent Wealth Killers (Why Money Doesn't Grow)",
    "psychology_of_capital": "Psychology of Capital (Mindset & Behavior)",
    "tactical_blueprints": "Tactical Blueprints (What You Can Do)",
    "income_expansion": "Income Expansion & Asymmetric Upside (Building Capital)",
    "financial_autopsies": "Financial Autopsies & Real-World Breakdown (Case Studies)",
}


def _write_config(tmp_path: Path, data: dict) -> Path:
    import yaml

    path = tmp_path / "topic_selection.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


class TestFinancePillars:
    def test_config_declares_exactly_the_six_finance_pillars(self):
        config = load_config(CONFIG_PATH)
        assert config.pillars == FINANCE_PILLARS

    def test_each_pillar_has_label_focus_and_guidance(self):
        config = load_config(CONFIG_PATH)
        labels = {pillar.pillar_id: pillar.label for pillar in config.pillar_catalog}
        assert labels == PILLAR_LABELS

    def test_catalog_covers_every_declared_pillar(self):
        config = load_config(CONFIG_PATH)
        assert {p.pillar_id for p in config.pillar_catalog} == set(FINANCE_PILLARS)

    def test_pillar_info_lookup_returns_editorial_definition(self):
        config = load_config(CONFIG_PATH)
        info = config.pillar_info("silent_wealth_killers")
        assert info.label == PILLAR_LABELS["silent_wealth_killers"]
        assert info.focus
        assert info.guidance

    def test_pillar_info_rejects_unknown_pillar(self):
        config = load_config(CONFIG_PATH)
        with pytest.raises(TopicSelectionConfigError):
            config.pillar_info("technology")

    def test_missing_catalog_entry_fails_loudly(self, tmp_path):
        import yaml

        raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        del raw["pillar_catalog"]["wealth_engine"]
        with pytest.raises(TopicSelectionConfigError, match="wealth_engine"):
            load_config(_write_config(tmp_path, raw))

    def test_catalog_entry_for_undeclared_pillar_fails_loudly(self, tmp_path):
        import yaml

        raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        raw["pillar_catalog"]["cooking"] = {
            "label": "Cooking",
            "focus": "food",
            "guidance": "taste",
        }
        with pytest.raises(TopicSelectionConfigError, match="cooking"):
            load_config(_write_config(tmp_path, raw))


class TestObsoletePillarsRemoved:
    def test_no_obsolete_pillar_is_active(self):
        config = load_config(CONFIG_PATH)
        for obsolete in OBSOLETE_PILLARS:
            assert obsolete not in config.pillars

    def test_obsolete_pillar_cannot_be_emitted_as_candidate(self):
        config = load_config(CONFIG_PATH)
        with pytest.raises(TopicCandidateError, match="pillar"):
            TopicCandidate.from_dict(
                _valid_candidate_dict(pillar="science"),
                config.pillars,
                config.topic_types,
            )

    def test_obsolete_topic_type_cannot_be_emitted_as_candidate(self):
        config = load_config(CONFIG_PATH)
        for obsolete in ("recent_discovery", "scientific_concept", "technology_development", "unusual_phenomenon"):
            assert obsolete not in config.topic_types
        with pytest.raises(TopicCandidateError, match="topic_type"):
            TopicCandidate.from_dict(
                _valid_candidate_dict(topic_type="scientific_concept"),
                config.pillars,
                config.topic_types,
            )

    def test_mock_dataset_uses_only_finance_pillars(self, default_config):
        candidates = load_candidates(MOCK_CANDIDATES, default_config)
        assert candidates
        assert {c.pillar for c in candidates} <= set(FINANCE_PILLARS)

    def test_mock_dataset_covers_every_finance_pillar(self, default_config):
        candidates = load_candidates(MOCK_CANDIDATES, default_config)
        assert {c.pillar for c in candidates} == set(FINANCE_PILLARS)

    def test_mock_dataset_topic_types_are_finance_types(self, default_config):
        candidates = load_candidates(MOCK_CANDIDATES, default_config)
        for candidate in candidates:
            assert candidate.topic_type in default_config.topic_types


def _valid_candidate_dict(**overrides) -> dict:
    data = {
        "topic_id": "t_example",
        "title": "Example money topic",
        "description": "An example description.",
        "pillar": "wealth_engine",
        "topic_type": "evergreen",
        "angle": "A framing angle.",
        "hook_direction": "An opening moment.",
        "signals": {"interest": 0.7, "trend_momentum": 0.4, "competition": 0.5, "freshness": 0.3},
        "content_scores": {
            "curiosity": 0.8,
            "explainer_potential": 0.8,
            "visual_potential": 0.7,
            "audience_fit": 0.8,
        },
    }
    data.update(overrides)
    return data


class TestChannelReachesSelector:
    def test_channel_identity_is_finance_mascot(self):
        config = load_config(CONFIG_PATH)
        assert config.channel.channel_id == "finance_mascot"
        assert config.channel.channel_promise == "Understand money. Grow wealth."
        assert config.channel.language == "English"
        assert config.channel.tone == "Friendly + educational"

    def test_channel_audience_is_present_and_finance_scoped(self):
        audience = load_config(CONFIG_PATH).channel.audience
        assert "English-speaking" in audience
        assert "money" in audience.lower() and "wealth" in audience.lower()

    def test_mascot_is_a_non_speaking_financial_detective(self):
        mascot = load_config(CONFIG_PATH).channel.mascot
        assert mascot.name == "fox"
        assert mascot.role == "financial_detective"
        assert mascot.speaks is False

    def test_mascot_asset_is_configured_but_not_shipped(self):
        mascot = load_config(CONFIG_PATH).channel.mascot
        assert mascot.asset_path
        assert mascot.asset_exists is False
        assert not (ROOT / mascot.asset_path).exists(), "Stage 1 must not create the mascot asset"

    def test_channel_config_is_reusable_standalone(self):
        channel = ChannelConfig()
        assert channel.channel_id == "finance_mascot"
        assert channel.to_dict()["mascot"]["role"] == "financial_detective"
        assert channel.mascot_asset_path == channel.mascot.asset_path

    def test_module_level_channel_matches_config_loaded_channel(self):
        assert load_config(CONFIG_PATH).channel == CHANNEL

    def test_channel_is_recorded_in_selection_artifact(self, default_config, mock_candidates):
        selection = run_topic_selection(
            mock_candidates, SelectionHistory(), default_config, run_id="finance_run"
        )
        scoring = selection.to_dict()["data"]["scoring_configuration"]
        assert scoring["channel"]["channel_id"] == "finance_mascot"
        assert scoring["channel"]["audience"]
        assert len(scoring["pillar_catalog"]) == len(FINANCE_PILLARS)


class TestSelectionCriteria:
    def test_weights_favor_curiosity_usefulness_visual_and_fit(self):
        weights = load_config(CONFIG_PATH).scoring.weights
        for key in ("curiosity", "explainer_potential", "visual_potential", "audience_fit"):
            assert weights[key] > 0.0

    def test_scoring_has_no_source_or_evidence_terms(self):
        weights = load_config(CONFIG_PATH).scoring.weights
        forbidden = (
            "source",
            "evidence",
            "corroboration",
            "researchability",
            "fact",
            "verifiability",
            "citation",
        )
        for key in weights:
            assert not any(term in key.lower() for term in forbidden), key

    def test_higher_curiosity_candidate_outranks_lower(self, candidate_factory, config_factory):
        config = config_factory(
            weights={
                "interest": 0.0,
                "trend_momentum": 0.0,
                "competition": 0.0,
                "curiosity": 1.0,
                "explainer_potential": 0.0,
                "visual_potential": 0.0,
                "audience_fit": 0.0,
            }
        )
        low = candidate_factory(topic_id="t_low", content={
            "curiosity": 0.2, "explainer_potential": 0.9, "visual_potential": 0.9, "audience_fit": 0.9})
        high = candidate_factory(topic_id="t_high", content={
            "curiosity": 0.95, "explainer_potential": 0.1, "visual_potential": 0.1, "audience_fit": 0.1})
        scorer = TopicScorer(config)
        selection = TopicSelector(config, run_id="curiosity").select(
            [scorer.score(low), scorer.score(high)], SelectionHistory()
        )
        assert selection.selected_topic_id == "t_high"

    def test_higher_usefulness_candidate_outranks_lower(self, candidate_factory, config_factory):
        config = config_factory(
            weights={
                "interest": 0.0,
                "trend_momentum": 0.0,
                "competition": 0.0,
                "curiosity": 0.0,
                "explainer_potential": 1.0,
                "visual_potential": 0.0,
                "audience_fit": 0.0,
            }
        )
        low = candidate_factory(topic_id="t_low", content={
            "curiosity": 0.9, "explainer_potential": 0.2, "visual_potential": 0.9, "audience_fit": 0.9})
        high = candidate_factory(topic_id="t_high", content={
            "curiosity": 0.1, "explainer_potential": 0.95, "visual_potential": 0.1, "audience_fit": 0.1})
        scorer = TopicScorer(config)
        selection = TopicSelector(config, run_id="usefulness").select(
            [scorer.score(low), scorer.score(high)], SelectionHistory()
        )
        assert selection.selected_topic_id == "t_high"

    def test_visual_potential_can_decide_the_winner(self, candidate_factory, config_factory):
        config = config_factory(
            weights={
                "interest": 0.0,
                "trend_momentum": 0.0,
                "competition": 0.0,
                "curiosity": 0.0,
                "explainer_potential": 0.0,
                "visual_potential": 1.0,
                "audience_fit": 0.0,
            }
        )
        low = candidate_factory(topic_id="t_low", content={
            "curiosity": 0.9, "explainer_potential": 0.9, "visual_potential": 0.1, "audience_fit": 0.9})
        high = candidate_factory(topic_id="t_high", content={
            "curiosity": 0.1, "explainer_potential": 0.1, "visual_potential": 0.95, "audience_fit": 0.1})
        scorer = TopicScorer(config)
        selection = TopicSelector(config, run_id="visual").select(
            [scorer.score(low), scorer.score(high)], SelectionHistory()
        )
        assert selection.selected_topic_id == "t_high"


class TestSelectedTopicContract:
    REQUIRED = ("topic_id", "title", "pillar", "angle", "hook_direction")

    def test_selected_topic_exposes_required_fields(self, default_config, mock_candidates):
        selection = run_topic_selection(
            mock_candidates, SelectionHistory(), default_config, run_id="contract_run"
        )
        selected = selection.to_dict()["data"]["selected_topic"]
        for field in self.REQUIRED:
            assert field in selected, f"missing selected-topic field: {field}"
            assert isinstance(selected[field], str) and selected[field].strip()

    def test_selected_topic_pillar_is_a_finance_pillar(self, default_config, mock_candidates):
        selection = run_topic_selection(
            mock_candidates, SelectionHistory(), default_config, run_id="pillar_run"
        )
        assert selection.to_dict()["data"]["selected_topic"]["pillar"] in FINANCE_PILLARS

    def test_every_ranked_candidate_exposes_angle_and_hook(self, default_config, mock_candidates):
        selection = run_topic_selection(
            mock_candidates, SelectionHistory(), default_config, run_id="ranked_run"
        )
        for candidate in selection.to_dict()["data"]["ranked_candidates"]:
            assert candidate["angle"].strip()
            assert candidate["hook_direction"].strip()

    def test_missing_angle_is_rejected(self, default_config):
        data = _valid_candidate_dict()
        del data["angle"]
        with pytest.raises(TopicCandidateError, match="angle"):
            TopicCandidate.from_dict(data, default_config.pillars, default_config.topic_types)

    def test_blank_hook_direction_is_rejected(self, default_config):
        data = _valid_candidate_dict()
        data["hook_direction"] = "   "
        with pytest.raises(TopicCandidateError, match="hook_direction"):
            TopicCandidate.from_dict(data, default_config.pillars, default_config.topic_types)

    def test_angle_and_hook_round_trip(self, default_config):
        original = TopicCandidate.from_dict(
            _valid_candidate_dict(), default_config.pillars, default_config.topic_types
        )
        rebuilt = TopicCandidate.from_dict(
            json.loads(json.dumps(original.to_dict())),
            default_config.pillars,
            default_config.topic_types,
        )
        assert rebuilt == original
        assert rebuilt.angle == original.angle
        assert rebuilt.hook_direction == original.hook_direction


class TestDeterministicMockPath:
    def test_mock_selection_is_reproducible(self, default_config, mock_candidates):
        def run(run_id):
            return run_topic_selection(
                [TopicCandidate.from_dict(c.to_dict(), default_config.pillars, default_config.topic_types)
                 for c in mock_candidates],
                SelectionHistory(),
                default_config,
                run_id=run_id,
                seed=default_config.selection.seed,
                created_at="2026-10-05T00:00:00+00:00",
            ).to_dict()["data"]

        first = run("determinism_a")
        second = run("determinism_b")
        assert first["selected_topic_id"] == second["selected_topic_id"]
        assert [c["topic_id"] for c in first["ranked_candidates"]] == [
            c["topic_id"] for c in second["ranked_candidates"]
        ]
        assert [c["total_score"] for c in first["ranked_candidates"]] == pytest.approx(
            [c["total_score"] for c in second["ranked_candidates"]]
        )

    def test_mock_provenance_marks_internal_mock_source(self, default_config, mock_candidates):
        selection = run_topic_selection(
            mock_candidates, SelectionHistory(), default_config, run_id="prov_run"
        )
        provenance = selection.to_dict()["data"]["selected_topic"]["provenance"]
        assert provenance["source"] == "internal_mock"
        assert provenance["provider"] is None

    def test_mock_path_makes_no_network_call(self, default_config, mock_candidates):
        from agents.seo.provider import MockTopicCandidateProvider, build_topic_candidate_provider

        provider = build_topic_candidate_provider(None, config=default_config)
        assert isinstance(provider, MockTopicCandidateProvider)
        assert provider.list_candidates() == load_candidates(MOCK_CANDIDATES, default_config)


class TestOptionalGeminiPath:
    """Schema + prompt wiring only. No network, no API key."""

    def _provider(self, config):
        from providers.gemini.topic_provider import GeminiTopicProvider
        from tests.gemini_test_utils import FakeRawClient, make_client

        client = make_client(_gemini_config(), raw_client=FakeRawClient())
        return GeminiTopicProvider(config=config, client=client)

    def test_schema_requires_angle_and_hook(self):
        from providers.gemini.topic_provider import TopicCandidateSchema

        fields = set(TopicCandidateSchema.model_fields)
        assert {"angle", "hook_direction"} <= fields

    def test_gemini_candidates_carry_angle_and_hook(self, default_config):
        from tests.gemini_test_utils import FakeRawClient, make_client, valid_candidates_payload
        from providers.gemini.topic_provider import GeminiTopicProvider

        fake = FakeRawClient()
        fake.add_result_json(valid_candidates_payload(count=3))
        client = make_client(_gemini_config(), raw_client=fake)
        provider = GeminiTopicProvider(config=default_config, client=client)
        for candidate in provider.list_candidates():
            assert candidate.angle.strip()
            assert candidate.hook_direction.strip()
            assert candidate.pillar in FINANCE_PILLARS

    def test_prompt_renders_finance_pillars_and_channel(self, default_config):
        rendered = self._provider(default_config)._render_prompt()
        for pillar in FINANCE_PILLARS:
            assert pillar in rendered
        assert "finance_mascot" in rendered
        assert "Understand money. Grow wealth." in rendered
        assert "financial_detective" in rendered

    def test_prompt_references_no_unresolved_placeholders(self, default_config):
        import re

        rendered = self._provider(default_config)._render_prompt()
        # Template tokens are bare {snake_case} words; the JSON output example
        # legitimately contains quoted keys such as {"interest": 0.0}.
        leftover = sorted(set(re.findall(r"\{[a-z_]+\}", rendered)))
        assert leftover == []

    def test_prompt_has_no_fact_checking_stage_requirements(self, default_config):
        rendered = self._provider(default_config)._render_prompt().lower()
        assert "fact verification" not in rendered
        assert "no fact-verification stage" in rendered

    def test_system_prompt_carries_channel_identity(self, default_config):
        system = self._provider(default_config)._system_prompt()
        assert "finance_mascot" in system
        assert "Understand money. Grow wealth." in system
        assert "fox financial detective" in system

    def test_prompt_path_version_matches_declared_version(self):
        from providers.gemini.topic_provider import DEFAULT_PROMPT_PATH, PROMPT_VERSION

        assert PROMPT_VERSION == "seo/topic_candidates-v2"
        assert DEFAULT_PROMPT_PATH.name == "topic_candidates-v2.txt"
        assert DEFAULT_PROMPT_PATH.is_file()

    def test_removed_prompt_template_is_gone(self):
        assert not (ROOT / "prompts" / "seo" / "topic_candidates-v1.txt").exists()


def _gemini_config():
    from providers.gemini.config import GeminiConfig

    return GeminiConfig(api_key="test-key")