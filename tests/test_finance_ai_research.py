"""Stage 2 (AI Research) tests for the finance_mascot pipeline.

Covers the required areas:
 1. valid topic_selection.json -> valid research.json
 2. research output schema
 3. topic_id propagation
 4. Gemini structured response mapping (offline fake client only)
 5. missing/invalid topic input handling
 6. deterministic mock research
 7. provider abstraction
 8. no source/evidence fields are required
 9. numeric claims need no fabricated values
10. CLI writes the flat artifact at data/<run_id>/research.json

No test here requires GEMINI_API_KEY, makes a network call, or touches Stages
3-5.
"""

from __future__ import annotations

import io
import json
import re
import sys
from pathlib import Path

import pytest

from agents.research.ai_research import (
    ResearchTopicError,
    build_research_topic,
    generate_ai_research,
    load_topic_selection,
    run_ai_research,
    write_ai_research,
)
from agents.research.ai_research_models import (
    RESEARCH_FIELDS,
    AIResearch,
    AIResearchError,
    ResearchTopic,
)
from agents.research.provider import (
    AIResearchProviderError,
    MockAIResearchProvider,
    build_ai_research_provider,
)
from agents.seo.config import load_config
from agents.seo.history import SelectionHistory
from agents.seo.selector import TopicSelector
from agents.seo.scorer import TopicScorer
from agents.seo.stage import load_candidates, run_topic_selection
from config.channel import CHANNEL
from tests.gemini_test_utils import FakeRawClient, make_client

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "topic_selection.yaml"
MOCK_CANDIDATES = ROOT / "data" / "mock" / "topic_candidates.json"
PROMPT_PATH = ROOT / "prompts" / "research" / "topic_research-v1.txt"

FINANCE_PILLARS = (
    "wealth_engine",
    "silent_wealth_killers",
    "psychology_of_capital",
    "tactical_blueprints",
    "income_expansion",
    "financial_autopsies",
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _stage2_tests_never_reach_gemini(monkeypatch):
    """Hermetic guarantee for every test in this module.

    The Stage 2 CLI calls ``load_dotenv()``, which would otherwise re-populate
    ``GEMINI_API_KEY`` from the developer's local ``.env`` even after a test
    deleted it - letting a "provider error" test build a REAL Gemini client and
    make a live API call. Disabling dotenv here makes that impossible.
    """
    import agents.research.ai_research_cli as cli

    monkeypatch.setattr(cli, "load_dotenv", None, raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)


@pytest.fixture(scope="module")
def default_config():
    return load_config(CONFIG_PATH)


@pytest.fixture
def mock_candidates(default_config):
    return load_candidates(MOCK_CANDIDATES, default_config)


@pytest.fixture
def topic(default_config, mock_candidates) -> ResearchTopic:
    """A real ResearchTopic derived from a real Stage 1 run."""
    scored = [TopicScorer(default_config).score(c, SelectionHistory()) for c in mock_candidates]
    selection = TopicSelector(default_config, run_id="stage2_fixture").select(scored, SelectionHistory())
    return build_research_topic(selection)


@pytest.fixture
def selection(default_config, mock_candidates):
    return run_topic_selection(
        mock_candidates, SelectionHistory(), default_config, run_id="stage2_run"
    )


@pytest.fixture
def selection_file(tmp_path, selection) -> Path:
    path = tmp_path / "topic_selection.json"
    path.write_text(json.dumps(selection.to_dict(), indent=2), encoding="utf-8")
    return path


def _gemini_config():
    from providers.gemini.config import GeminiConfig

    return GeminiConfig(api_key="test-key")


def _gemini_provider(default_config, fake):
    from providers.gemini.research_provider import GeminiAIResearchProvider

    return GeminiAIResearchProvider(config=default_config, client=make_client(_gemini_config(), raw_client=fake))


def _research_payload(**overrides) -> dict:
    payload = {
        "key_concepts": ["Compounding", "Time horizon"],
        "explanation": "Money grows by earning a return on earlier returns.",
        "examples": ["A monthly contribution left alone for decades."],
        "numbers": [],
        "interesting_angles": ["Reveal that time matters more than size."],
        "common_misconceptions": ["That starting amount decides the outcome."],
        "story_ideas": ["Split screen: two balances at the same age."],
    }
    payload.update(overrides)
    return payload


# --------------------------------------------------------------------------- #
# 1. valid topic_selection.json -> valid research.json
# --------------------------------------------------------------------------- #


class TestStage1ToStage2:
    def test_selection_file_produces_research_artifact(self, default_config, selection_file):
        selection = load_topic_selection(selection_file)
        research = generate_ai_research(selection, run_id="stage2_run")
        assert isinstance(research, AIResearch)
        assert research.topic_id == selection.selected_topic_id
        assert research.explanation.strip()

    def test_research_written_to_json_round_trips(self, default_config, selection_file, tmp_path):
        selection = load_topic_selection(selection_file)
        research = generate_ai_research(selection, run_id="stage2_run")
        out = write_ai_research(research, tmp_path / "research.json")
        assert out.is_file()
        reloaded = AIResearch.from_json(out.read_text(encoding="utf-8"))
        assert reloaded.to_dict() == research.to_dict()

    def test_research_topic_carries_stage1_fields(self, selection):
        topic = build_research_topic(selection)
        assert topic.topic_id == selection.selected_topic_id
        assert topic.title.strip()
        assert topic.pillar in FINANCE_PILLARS
        assert topic.angle.strip()
        assert topic.hook_direction.strip()

    def test_run_id_propagates_into_research(self, selection):
        research = generate_ai_research(selection, run_id="propagate_me")
        assert research.run_id == "propagate_me"
        assert research.to_dict()["run_id"] == "propagate_me"


# --------------------------------------------------------------------------- #
# 2. research output schema
# --------------------------------------------------------------------------- #


class TestResearchSchema:
    def test_artifact_is_structured_json_with_all_content_fields(self, selection):
        data = generate_ai_research(selection, run_id="schema_run").to_dict()
        for field in ("topic_id", *RESEARCH_FIELDS):
            assert field in data, f"missing research field: {field}"
        assert isinstance(data["key_concepts"], list)
        assert isinstance(data["explanation"], str)
        for field in ("examples", "numbers", "interesting_angles", "common_misconceptions", "story_ideas"):
            assert isinstance(data[field], list), field

    def test_provenance_fields_present(self, selection):
        data = generate_ai_research(selection, run_id="prov_run").to_dict()
        assert set(data["produced_by"]) == {"prompt_version", "provider", "model"}
        assert data["created_at"]
        assert data["pillar"]

    def test_research_fields_tuple_matches_contract(self):
        assert RESEARCH_FIELDS == (
            "key_concepts",
            "explanation",
            "examples",
            "numbers",
            "interesting_angles",
            "common_misconceptions",
            "story_ideas",
        )

    def test_validation_requires_topic_id_and_explanation(self):
        with pytest.raises(AIResearchError, match="topic_id"):
            AIResearch(topic_id="", explanation="something").validate()
        with pytest.raises(AIResearchError, match="explanation"):
            AIResearch(topic_id="t1", explanation="").validate()

    def test_validation_accepts_material_with_empty_optional_lists(self):
        research = AIResearch(topic_id="t1", explanation="A plain explanation.")
        assert research.validate() is research
        assert research.numbers == []

    def test_research_topic_dataclass_round_trip(self, topic):
        data = topic.to_dict()
        assert data["topic_id"] == topic.topic_id
        for field in ("title", "pillar", "angle", "hook_direction"):
            assert field in data


# --------------------------------------------------------------------------- #
# 3. topic_id propagation
# --------------------------------------------------------------------------- #


class TestTopicIdPropagation:
    def test_mock_provider_propagates_topic_id(self, topic):
        assert MockAIResearchProvider().generate(topic).topic_id == topic.topic_id

    def test_gemini_provider_propagates_topic_id(self, default_config, topic):
        fake = FakeRawClient()
        fake.add_result_json(_research_payload())
        research = _gemini_provider(default_config, fake).generate(topic)
        assert research.topic_id == topic.topic_id

    def test_run_ai_research_propagates_topic_id(self, selection, topic):
        research = run_ai_research(selection, MockAIResearchProvider(), run_id="tid")
        assert research.topic_id == topic.topic_id
        assert research.to_dict()["topic_id"] == topic.topic_id

    def test_topic_id_survives_artifact_round_trip(self, selection, tmp_path):
        research = generate_ai_research(selection, run_id="roundtrip")
        path = write_ai_research(research, tmp_path / "research.json")
        assert AIResearch.from_json(path.read_text(encoding="utf-8")).topic_id == research.topic_id


# --------------------------------------------------------------------------- #
# 4. Gemini structured response mapping (offline)
# --------------------------------------------------------------------------- #


class TestGeminiStructuredMapping:
    def test_structured_payload_maps_to_artifact(self, default_config, topic):
        fake = FakeRawClient()
        fake.add_result_json(_research_payload())
        research = _gemini_provider(default_config, fake).generate(topic)
        assert research.key_concepts == ["Compounding", "Time horizon"]
        assert research.explanation.startswith("Money grows")
        assert research.examples == ["A monthly contribution left alone for decades."]
        assert research.interesting_angles == ["Reveal that time matters more than size."]
        assert research.common_misconceptions == ["That starting amount decides the outcome."]
        assert research.story_ideas == ["Split screen: two balances at the same age."]

    def test_gemini_stamps_provenance(self, default_config, topic):
        from providers.gemini.research_provider import PROMPT_VERSION

        fake = FakeRawClient()
        fake.add_result_json(_research_payload())
        research = _gemini_provider(default_config, fake).generate(topic)
        assert research.produced_by == {
            "prompt_version": PROMPT_VERSION,
            "provider": "gemini",
            "model": research.produced_by["model"],
        }
        assert research.produced_by["model"]

    def test_gemini_schema_is_a_pydantic_model(self):
        from providers.gemini.research_provider import AIResearchSchema

        assert issubclass(AIResearchSchema, __import__("pydantic").BaseModel)

    def test_gemini_schema_accepts_empty_payload(self):
        from providers.gemini.research_provider import AIResearchSchema

        parsed = AIResearchSchema.model_validate({})
        assert parsed.numbers == []
        assert parsed.explanation == ""

    def test_gemini_uses_structured_call_not_grounded_search(self, default_config, topic):
        """The provider must call generate_structured, never the
        search-grounded generate_research."""
        calls: list[str] = []

        class _RecordingClient:
            model = "recording-model"

            def generate_structured(self, prompt, response_model, *, system_prompt=None, **kw):
                calls.append("generate_structured")
                return response_model.model_validate(_research_payload())

            def generate_research(self, *a, **kw):  # pragma: no cover - must not be called
                calls.append("generate_research")
                raise AssertionError("grounded search must not be used in Stage 2")

        from providers.gemini.research_provider import GeminiAIResearchProvider

        provider = GeminiAIResearchProvider(config=default_config, client=_RecordingClient())
        research = provider.generate(topic)
        assert calls == ["generate_structured"]
        assert research.explanation

    def test_gemini_malformed_json_raises_provider_error(self, default_config, topic):
        from providers.gemini.errors import GeminiMalformedResponseError

        fake = FakeRawClient()
        fake.add_result("not json at all")
        with pytest.raises(GeminiMalformedResponseError):
            _gemini_provider(default_config, fake).generate(topic)

    def test_gemini_wrong_types_raise_schema_error(self, default_config, topic):
        from providers.gemini.errors import GeminiSchemaValidationError

        fake = FakeRawClient()
        fake.add_result_json({"key_concepts": "not-a-list"})
        with pytest.raises(GeminiSchemaValidationError):
            _gemini_provider(default_config, fake).generate(topic)

    def test_gemini_prompt_includes_topic_and_channel(self, default_config, topic):
        fake = FakeRawClient()
        rendered = _gemini_provider(default_config, fake)._render_prompt(topic)
        assert topic.title in rendered
        assert topic.pillar in rendered
        assert CHANNEL.channel_id in rendered
        assert CHANNEL.channel_promise in rendered
        assert sorted(set(re.findall(r"\{[a-z_]+\}", rendered))) == []

    def test_gemini_prompt_forbids_web_sources_and_fabricated_numbers(self, default_config, topic):
        fake = FakeRawClient()
        rendered = _gemini_provider(default_config, fake)._render_prompt(topic).lower()
        assert "no web research" in rendered
        assert "no citations" in rendered
        assert "do not fabricate precise figures" in rendered
        assert "leave `numbers` as an empty list" in rendered

    def test_gemini_system_prompt_carries_channel_and_mascot(self, default_config, topic):
        fake = FakeRawClient()
        system = _gemini_provider(default_config, fake)._system_prompt()
        assert CHANNEL.channel_promise in system
        assert "financial_detective" in system
        assert "never speaks" in system

    def test_prompt_file_exists(self):
        assert PROMPT_PATH.is_file()


# --------------------------------------------------------------------------- #
# 5. missing / invalid topic input handling
# --------------------------------------------------------------------------- #


class TestInvalidTopicInput:
    def _selection_with(self, **selected_overrides):
        from agents.seo.result import TopicSelection

        selected = {
            "topic_id": "t_x",
            "title": "A topic",
            "pillar": "wealth_engine",
            "angle": "An angle.",
            "hook_direction": "A hook.",
            "topic_type": "evergreen",
            "description": "Desc.",
        }
        selected.update(selected_overrides)
        return TopicSelection(
            selection_id="ts_1",
            selected_topic_id=selected.get("topic_id", ""),
            selected_topic=selected,
            ranked_candidates=[selected],
            selection_mode="cold_start",
            historical_data_available=False,
            selection_reason="test",
            scoring_configuration={},
            created_at="2026-10-05T00:00:00+00:00",
            run_id="invalid_input",
        )

    @pytest.mark.parametrize(
        "field", ["topic_id", "title", "pillar", "angle", "hook_direction"]
    )
    def test_missing_required_field_raises(self, field):
        selection = self._selection_with(**{field: ""})
        with pytest.raises(ResearchTopicError, match=field):
            build_research_topic(selection)

    def test_blank_whitespace_field_raises(self):
        selection = self._selection_with(title="   ")
        with pytest.raises(ResearchTopicError, match="title"):
            build_research_topic(selection)

    def test_missing_selected_topic_raises(self):
        from agents.seo.result import TopicSelection

        selection = TopicSelection(
            selection_id="ts_2",
            selected_topic_id="t_x",
            selected_topic={},
            ranked_candidates=[],
            selection_mode="cold_start",
            historical_data_available=False,
            selection_reason="test",
            scoring_configuration={},
            created_at="2026-10-05T00:00:00+00:00",
        )
        with pytest.raises(ResearchTopicError, match="no selected_topic"):
            build_research_topic(selection)

    def test_mismatched_topic_id_raises(self):
        selection = self._selection_with()
        selection.selected_topic_id = "t_other"
        with pytest.raises(ResearchTopicError, match="does not match"):
            build_research_topic(selection)

    def test_missing_selection_file_is_cli_error_not_traceback(self, tmp_path, monkeypatch, capsys):
        from agents.research.ai_research_cli import main

        monkeypatch.setattr(
            sys, "argv", ["ai_research_cli", "--run-id", "does_not_exist_stage2"]
        )
        assert main() == 2
        assert "Missing Stage 1 artifact" in capsys.readouterr().out

    def test_missing_stage1_artifact_writes_no_research(self, tmp_path, monkeypatch):
        from agents.research.ai_research_cli import main

        monkeypatch.setattr(sys, "argv", ["ai_research_cli", "--run-id", "nope_stage2"])
        assert main() == 2
        assert not (ROOT / "data" / "nope_stage2" / "research.json").exists()


# --------------------------------------------------------------------------- #
# 6. deterministic mock research
# --------------------------------------------------------------------------- #


class TestDeterministicMock:
    def test_same_topic_same_bytes(self, topic):
        provider = MockAIResearchProvider()
        first = provider.generate(topic).to_dict()
        second = provider.generate(topic).to_dict()
        first.pop("created_at")
        second.pop("created_at")
        assert first == second

    def test_different_topics_differ(self, default_config, mock_candidates, topic):
        scored = [TopicScorer(default_config).score(c, SelectionHistory()) for c in mock_candidates]
        selection = TopicSelector(default_config, run_id="det").select(scored, SelectionHistory())
        ranked = selection.ranked_candidates
        other_topic = ResearchTopic(
            topic_id=ranked[1]["topic_id"],
            title=ranked[1]["title"],
            pillar=ranked[1]["pillar"],
            angle=ranked[1]["angle"],
            hook_direction=ranked[1]["hook_direction"],
        )
        provider = MockAIResearchProvider()
        assert provider.generate(topic).to_dict() != provider.generate(other_topic).to_dict()

    def test_mock_marks_itself_as_mock(self, topic):
        research = MockAIResearchProvider().generate(topic)
        assert research.produced_by["provider"] == "mock"
        assert research.produced_by["prompt_version"] is None

    def test_mock_numbers_are_empty_never_fabricated(self, topic):
        assert MockAIResearchProvider().generate(topic).numbers == []

    def test_mock_output_is_pillar_aligned(self, default_config, mock_candidates):
        provider = MockAIResearchProvider(config=default_config)
        scored = [TopicScorer(default_config).score(c, SelectionHistory()) for c in mock_candidates]
        selection = TopicSelector(default_config, run_id="align").select(scored, SelectionHistory())
        for ranked in selection.ranked_candidates:
            built = ResearchTopic(
                topic_id=ranked["topic_id"],
                title=ranked["title"],
                pillar=ranked["pillar"],
                angle=ranked["angle"],
                hook_direction=ranked["hook_direction"],
            )
            label = default_config.pillar_info(built.pillar).label
            assert label in provider.generate(built).explanation

    def test_default_generate_uses_mock_without_key(self, monkeypatch, selection):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        research = generate_ai_research(selection, run_id="nomock_key")
        assert research.produced_by["provider"] == "mock"


# --------------------------------------------------------------------------- #
# 7. provider abstraction
# --------------------------------------------------------------------------- #


class TestProviderAbstraction:
    def test_default_provider_is_mock(self, default_config):
        provider = build_ai_research_provider(None, config=default_config)
        assert isinstance(provider, MockAIResearchProvider)

    def test_explicit_mock_provider(self, default_config):
        assert isinstance(
            build_ai_research_provider("mock", config=default_config), MockAIResearchProvider
        )

    def test_unknown_provider_raises(self, default_config):
        with pytest.raises(AIResearchProviderError):
            build_ai_research_provider("openai", config=default_config)

    def test_gemini_provider_requires_api_key(self, default_config):
        from providers.gemini.errors import GeminiConfigurationError

        with pytest.raises(GeminiConfigurationError):
            build_ai_research_provider("gemini", config=default_config, env={})

    def test_gemini_provider_not_silently_falls_back_to_mock(self, default_config, monkeypatch):
        from providers.gemini.errors import GeminiProviderError

        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with pytest.raises(GeminiProviderError):
            build_ai_research_provider("gemini", config=default_config, env={})

    def test_provider_is_replaceable_by_any_object(self, selection):
        class FakeProvider:
            provider_name = "fake"

            def generate(self, topic: ResearchTopic) -> AIResearch:
                return AIResearch(topic_id=topic.topic_id, explanation="custom provider output")

        research = run_ai_research(selection, FakeProvider(), run_id="replaceable")
        assert research.explanation == "custom provider output"
        assert research.topic_id == selection.selected_topic_id

    def test_gemini_provider_satisfies_protocol(self, default_config):
        from providers.gemini.research_provider import GeminiAIResearchProvider

        fake = FakeRawClient()
        provider = _gemini_provider(default_config, fake)
        assert isinstance(provider, GeminiAIResearchProvider)
        assert provider.provider_name == "gemini"
        assert callable(provider.generate)

    def test_only_requested_provider_is_imported(self, default_config):
        """The mock path must not pull in the Gemini provider module."""
        import sys

        sys.modules.pop("providers.gemini.research_provider", None)
        build_ai_research_provider("mock", config=default_config)
        assert "providers.gemini.research_provider" not in sys.modules


# --------------------------------------------------------------------------- #
# 8. no source/evidence fields required
# --------------------------------------------------------------------------- #


class TestNoSourceMachinery:
    def test_contract_has_no_source_or_evidence_fields(self):
        forbidden = ("source", "evidence", "citation", "url", "reference", "corroboration", "source_url")
        for field in ("topic_id", *RESEARCH_FIELDS, "pillar", "run_id", "created_at", "produced_by"):
            assert not any(bad in field.lower() for bad in forbidden), field

    def test_artifact_has_no_source_or_evidence_keys(self, selection):
        data = generate_ai_research(selection, run_id="nosource").to_dict()
        for key in data:
            assert not any(bad in key.lower() for bad in ("source", "evidence", "citation", "url")), key

    def test_research_validates_without_any_provenance(self):
        research = AIResearch(topic_id="t1", explanation="Plain explanation.")
        assert research.validate() is research
        assert research.produced_by == {"prompt_version": None, "provider": None, "model": None}

    def test_research_module_has_no_retrieval_imports(self):
        source = (ROOT / "agents" / "research" / "ai_research.py").read_text(encoding="utf-8")
        for banned in ("httpx", "requests", "beautifulsoup", "BeautifulSoup", "firecrawl", "searxng"):
            assert banned.lower() not in source.lower(), banned

    def test_provider_module_has_no_retrieval_imports(self):
        source = (ROOT / "agents" / "research" / "provider.py").read_text(encoding="utf-8")
        for banned in ("httpx", "requests", "beautifulsoup", "firecrawl", "searxng"):
            assert banned.lower() not in source.lower(), banned

    def test_gemini_research_provider_does_not_call_grounded_search(self):
        """No call to the search-grounded client method, and no grounding
        tool keywords in the request path."""
        source = (ROOT / "providers" / "gemini" / "research_provider.py").read_text(encoding="utf-8")
        assert "self._client.generate_research" not in source
        assert "self._client.generate_structured" in source
        assert "google_search" not in source
        assert "url_context" not in source


# --------------------------------------------------------------------------- #
# 9. numeric claims need no fabricated values
# --------------------------------------------------------------------------- #


class TestNumbersOptional:
    def test_empty_numbers_validates(self):
        research = AIResearch(topic_id="t1", explanation="Explains the mechanism.")
        assert research.validate() is research

    def test_gemini_can_return_empty_numbers(self, default_config, topic):
        fake = FakeRawClient()
        fake.add_result_json(_research_payload(numbers=[]))
        research = _gemini_provider(default_config, fake).generate(topic)
        assert research.numbers == []

    def test_gemini_can_omit_numbers_key_entirely(self, default_config, topic):
        payload = _research_payload()
        del payload["numbers"]
        fake = FakeRawClient()
        fake.add_result_json(payload)
        research = _gemini_provider(default_config, fake).generate(topic)
        assert research.numbers == []

    def test_gemini_can_omit_every_list_field(self, default_config, topic):
        fake = FakeRawClient()
        fake.add_result_json({"explanation": "Only the explanation is required."})
        research = _gemini_provider(default_config, fake).generate(topic)
        assert research.numbers == []
        assert research.explanation

    def test_approximate_numbers_are_allowed(self, default_config, topic):
        fake = FakeRawClient()
        fake.add_result_json(
            _research_payload(numbers=["Roughly 7% average annual return (illustrative)."])
        )
        research = _gemini_provider(default_config, fake).generate(topic)
        assert len(research.numbers) == 1

    def test_schema_does_not_require_a_minimum_count(self):
        from providers.gemini.research_provider import AIResearchSchema

        parsed = AIResearchSchema.model_validate({"numbers": []})
        assert parsed.numbers == []


# --------------------------------------------------------------------------- #
# 10. CLI writes the flat artifact at data/<run_id>/research.json
# --------------------------------------------------------------------------- #


class TestStage2CLI:
    def _install_argv(self, monkeypatch, *argv):
        monkeypatch.setattr(sys, "argv", ["ai_research_cli", *argv])

    def _stub_state_save(self, monkeypatch):
        from pipeline.state import PipelineState

        monkeypatch.setattr(PipelineState, "save", lambda self, path: Path(path))

    def _run_cli(self, monkeypatch, tmp_path, run_id, *extra, selection=None, capsys=None):
        """Run the Stage 2 CLI against a real data/<run_id>/ layout."""
        import shutil

        from agents.research.ai_research_cli import main

        self._install_argv(monkeypatch, "--run-id", run_id, *extra)
        self._stub_state_save(monkeypatch)

        run_dir = Path("data") / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        if selection is not None:
            (run_dir / "topic_selection.json").write_text(
                json.dumps(selection.to_dict(), indent=2), encoding="utf-8"
            )
        try:
            code = main()
            captured = capsys.readouterr().out if capsys is not None else ""
            return code, captured, run_dir
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)

    def test_cli_out_flag_content(self, monkeypatch, tmp_path, selection):
        out = tmp_path / "custom_research.json"
        code, _, _ = self._run_cli(
            monkeypatch, tmp_path, "cli_stage2_out", "--out", str(out), selection=selection
        )
        assert code == 0
        assert out.is_file()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["topic_id"] == selection.selected_topic_id
        assert data["explanation"].strip()
        assert data["produced_by"]["provider"] == "mock"

    def test_cli_default_path_content_verified(self, monkeypatch, capsys, selection):
        """Verify the default-path file exists at write time, then clean up."""
        import shutil

        from agents.research.ai_research_cli import main

        self._install_argv(monkeypatch, "--run-id", "cli_default_stage2b")
        self._stub_state_save(monkeypatch)
        run_dir = Path("data") / "cli_default_stage2b"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "topic_selection.json").write_text(
            json.dumps(selection.to_dict(), indent=2), encoding="utf-8"
        )
        try:
            assert main() == 0
            expected = run_dir / "research.json"
            assert expected.is_file(), "CLI must write data/<run_id>/research.json"
            data = json.loads(expected.read_text(encoding="utf-8"))
            assert data["topic_id"] == selection.selected_topic_id
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)

    def test_cli_uses_mock_provider_by_default(self, monkeypatch, capsys, selection):
        code, captured, _ = self._run_cli(
            monkeypatch, None, "cli_prov_stage2", selection=selection, capsys=capsys
        )
        assert code == 0
        assert "provider=mock" in captured

    def test_cli_gemini_without_key_errors_and_writes_nothing(self, monkeypatch, capsys, selection):
        """A valid Stage 1 artifact exists, so the CLI reaches provider
        construction and must fail loudly on the missing key - writing nothing."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        code, captured, _ = self._run_cli(
            monkeypatch,
            None,
            "cli_gem_stage2",
            "--provider",
            "gemini",
            selection=selection,
            capsys=capsys,
        )
        assert code == 2
        assert "GEMINI_API_KEY" in captured
        assert "provider error" in captured

    def test_cli_returns_2_when_stage1_artifact_missing(self, monkeypatch, capsys):
        self._install_argv(monkeypatch, "--run-id", "absent_stage2_run")
        from agents.research.ai_research_cli import main

        assert main() == 2
        assert "Missing Stage 1 artifact" in capsys.readouterr().out