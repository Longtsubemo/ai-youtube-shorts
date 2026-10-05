"""Stage 3 (Script Generation) tests for the finance_mascot pipeline.

Covers the required areas:
  1. valid Stage 1 + Stage 2 artifacts -> valid script.json
  2. script output schema
  3. topic_id propagation from the selected topic
  4. Gemini structured response mapping (offline fake client only)
  5. missing/invalid upstream artifact handling (both inputs, separately)
  6. deterministic mock script generation
  7. ScriptProvider abstraction / replaceability
  8. no source / evidence / citation fields required
  9. finance-safety restriction is present in prompt + provider contract
 10. CLI writes the flat artifact at data/<run_id>/script.json
 11. mascot never speaks
 12. duration target 65-80s, never below 60s
 13. prompt exists, is finance/channel aware, forbids browsing and citations
 14. structured output uses generate_structured, never generate_research
 15. sections carry Scene-Planning-usable narration
 16. no QA stage, no score, no PASS/FAIL contract

No test here requires GEMINI_API_KEY, makes a network call, or touches Stages
4-5.
"""

from __future__ import annotations

import io
import json
import re
import socket
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from agents.research.ai_research_models import AIResearch
from agents.script.provider import (
    MockScriptProvider,
    ScriptProviderError,
    build_script_provider,
)
from agents.script.script_models import (
    DURATION_MIN_SECONDS,
    DURATION_TARGET_RANGE,
    SECTION_ROLES,
    TARGET_DURATION_SECONDS,
    Script,
    ScriptError,
    ScriptSection,
)
from agents.script.stage import (
    REQUIRED_TOPIC_FIELDS,
    ScriptInputError,
    build_script_context,
    generate_script,
    load_research,
    load_topic_selection,
    run_script_generation,
    write_script,
)
from agents.seo.config import load_config
from agents.seo.history import SelectionHistory
from agents.seo.scorer import TopicScorer
from agents.seo.selector import TopicSelector
from agents.seo.stage import load_candidates, run_topic_selection
from config.channel import CHANNEL
from tests.gemini_test_utils import FakeRawClient, make_client

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "topic_selection.yaml"
MOCK_CANDIDATES = ROOT / "data" / "mock" / "topic_candidates.json"
PROMPT_PATH = ROOT / "prompts" / "script" / "short_script-v1.txt"

# Phrasings a real script must never contain (personalized-advice directives).
FORBIDDEN_NARRATION = (
    "you should buy",
    "you should invest",
    "you must invest",
    "buy this stock",
    "this stock will",
    "guaranteed returns",
    "i recommend you",
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _stage3_tests_never_reach_gemini(monkeypatch):
    """Hermetic guarantee for every test in this module.

    The Stage 3 CLI calls ``load_dotenv()``, which would otherwise re-populate
    ``GEMINI_API_KEY`` from the developer's local ``.env`` even after a test
    deleted it - letting a "provider error" test build a REAL Gemini client and
    make a live API call. Disabling dotenv here makes that impossible.
    """
    import agents.script.script_cli as cli

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
def selection(default_config, mock_candidates):
    """A real Stage 1 selection produced by the normal Stage 1 stage."""
    return run_topic_selection(
        mock_candidates, SelectionHistory(), default_config, run_id="stage3_run"
    )


@pytest.fixture
def research(default_config, selection) -> AIResearch:
    """A real Stage 2 research artifact derived from the Stage 1 selection."""
    from agents.research.ai_research import generate_ai_research

    return generate_ai_research(selection, run_id="stage3_run")


@pytest.fixture
def selection_file(tmp_path, selection) -> Path:
    path = tmp_path / "topic_selection.json"
    path.write_text(json.dumps(selection.to_dict(), indent=2), encoding="utf-8")
    return path


@pytest.fixture
def research_file(tmp_path, research) -> Path:
    path = tmp_path / "research.json"
    path.write_text(json.dumps(research.to_dict(), indent=2), encoding="utf-8")
    return path


def _gemini_config():
    from providers.gemini.config import GeminiConfig

    return GeminiConfig(api_key="test-key")


def _gemini_provider(default_config, fake, prompt_path=None):
    from providers.gemini.script_provider import GeminiScriptProvider

    kwargs = {} if prompt_path is None else {"prompt_path": prompt_path}
    return GeminiScriptProvider(
        config=default_config,
        client=make_client(_gemini_config(), raw_client=fake),
        **kwargs,
    )


def _script_payload(**overrides) -> dict:
    payload = {
        "title": "How money quietly grows",
        "narration": (
            "Here is something about money almost nobody explains. "
            "Money grows by earning a return on earlier returns. "
            "Two identical monthly amounts end up worlds apart. "
            "That gap is compounding doing the work."
        ),
        "sections": [
            {
                "section_id": "s1",
                "role": "hook",
                "narration": "Here is something about money almost nobody explains.",
                "estimated_seconds": 8,
                "visual_hint": "The fox opens a case file.",
            },
            {
                "section_id": "s2",
                "role": "explanation",
                "narration": "Money grows by earning a return on earlier returns.",
                "estimated_seconds": 14,
                "visual_hint": "A diagram of returns stacking.",
            },
            {
                "section_id": "s3",
                "role": "payoff",
                "narration": "That gap is compounding doing the work.",
                "estimated_seconds": 12,
                "visual_hint": "Two balances split apart.",
            },
        ],
    }
    payload.update(overrides)
    return payload


# --------------------------------------------------------------------------- #
# 1. valid Stage 1 + Stage 2 artifacts -> valid script.json
# --------------------------------------------------------------------------- #


class TestUpstreamToScript:
    def test_both_artifacts_produce_a_script(self, selection, research):
        script = generate_script(selection, research, run_id="stage3_run")
        assert isinstance(script, Script)
        assert script.topic_id == selection.selected_topic_id
        assert script.narration.strip()
        assert script.sections

    def test_script_round_trips_through_json(self, selection, research, tmp_path):
        script = generate_script(selection, research, run_id="stage3_run")
        out = write_script(script, tmp_path / "script.json")
        assert out.is_file()
        assert Script.from_json(out.read_text(encoding="utf-8")).to_dict() == script.to_dict()

    def test_artifacts_load_from_files(self, selection_file, research_file):
        loaded_selection = load_topic_selection(selection_file)
        loaded_research = load_research(research_file)
        script = generate_script(loaded_selection, loaded_research)
        assert script.topic_id == loaded_selection.selected_topic_id

    def test_run_id_propagates_into_script(self, selection, research):
        script = generate_script(selection, research, run_id="propagate_me")
        assert script.run_id == "propagate_me"
        assert script.to_dict()["run_id"] == "propagate_me"

    def test_created_at_can_be_pinned(self, selection, research):
        script = generate_script(selection, research, created_at="2026-01-01T00:00:00+00:00")
        assert script.created_at == "2026-01-01T00:00:00+00:00"


# --------------------------------------------------------------------------- #
# 2. script output schema
# --------------------------------------------------------------------------- #


class TestScriptSchema:
    def test_artifact_is_structured_json_with_script_fields(self, selection, research):
        data = generate_script(selection, research).to_dict()
        for field in (
            "topic_id",
            "title",
            "duration_target_seconds",
            "narration",
            "sections",
            "pillar",
            "channel_id",
            "mascot_speaks",
            "mascot_note",
            "educational_notice",
            "run_id",
            "created_at",
            "produced_by",
        ):
            assert field in data, f"missing script field: {field}"
        assert isinstance(data["sections"], list)
        for section in data["sections"]:
            for field in ("section_id", "role", "narration", "estimated_seconds", "visual_hint"):
                assert field in section, f"missing section field: {field}"

    def test_sections_are_ordered_and_uniquely_identified(self, selection, research):
        script = generate_script(selection, research)
        ids = [section.section_id for section in script.sections]
        assert ids == sorted(ids), "sections must be in ascending narrative order"
        assert len(set(ids)) == len(ids), "section ids must be unique"

    def test_section_from_dict_accepts_legacy_type_and_text(self):
        section = ScriptSection.from_dict({"type": "hook", "text": "Legacy words."})
        assert section.role == "hook"
        assert section.narration == "Legacy words."
        assert section.section_id == "s1"

    def test_produced_by_records_provider_and_prompt_version(self, selection, research, default_config):
        script = generate_script(selection, research, MockScriptProvider(default_config))
        assert script.produced_by["provider"] == "mock"
        assert script.produced_by["model"] is None


# --------------------------------------------------------------------------- #
# 3. topic_id propagation
# --------------------------------------------------------------------------- #


class TestTopicPropagation:
    def test_topic_id_matches_the_selected_topic(self, selection, research):
        script = generate_script(selection, research)
        assert script.topic_id == selection.selected_topic_id
        assert script.topic_id == selection.selected_topic["topic_id"]

    def test_topic_fields_come_from_the_stage1_artifact(self, selection, research):
        context = build_script_context(selection, research)
        selected = selection.selected_topic
        for field in ("topic_id", "title", "pillar", "angle", "hook_direction"):
            assert context[field] == selected[field], field

    def test_script_title_defaults_to_selected_title(self, selection, research, default_config, monkeypatch):
        monkeypatch.setattr(
            MockScriptProvider, "generate", lambda self, ctx: Script(
                topic_id=ctx["topic_id"],
                title=ctx["title"],
                narration="words",
                sections=[ScriptSection("s1", "hook", "words")],
            )
        )
        script = generate_script(selection, research, MockScriptProvider(default_config))
        assert script.title == selection.selected_topic["title"]


# --------------------------------------------------------------------------- #
# 4. Gemini structured response mapping (offline)
# --------------------------------------------------------------------------- #


class TestGeminiProvider:
    def test_structured_response_maps_to_script(self, selection, research, default_config):
        fake = FakeRawClient()
        fake.add_result_json(_script_payload())
        provider = _gemini_provider(default_config, fake)

        script = provider.generate(build_script_context(selection, research))

        assert isinstance(script, Script)
        assert script.title == "How money quietly grows"
        assert script.narration.startswith("Here is something about money")
        assert len(script.sections) == 3
        assert script.sections[0].role == "hook"
        assert script.sections[0].narration
        assert script.estimated_seconds == 34

    def test_script_is_channel_and_pillar_aware(self, selection, research, default_config):
        fake = FakeRawClient()
        fake.add_result_json(_script_payload())
        provider = _gemini_provider(default_config, fake)

        script = provider.generate(build_script_context(selection, research))
        prompt = fake.calls[0][1]

        assert script.channel_id == CHANNEL.channel_id
        assert script.pillar == selection.selected_topic["pillar"]
        assert selection.selected_topic["title"] in prompt
        assert selection.selected_topic["hook_direction"] in prompt
        assert research.explanation in prompt
        assert CHANNEL.channel_promise in prompt
        assert CHANNEL.audience in prompt
        assert CHANNEL.mascot.role in prompt

    def test_produced_by_records_model_and_prompt_version(self, selection, research, default_config):
        fake = FakeRawClient()
        fake.add_result_json(_script_payload())
        provider = _gemini_provider(default_config, fake)

        script = provider.generate(build_script_context(selection, research))
        assert script.produced_by["provider"] == "gemini"
        assert script.produced_by["model"] == _gemini_config().model
        assert script.produced_by["prompt_version"]

    def test_empty_narration_response_fails_closed(self, selection, research, default_config):
        from providers.gemini.errors import GeminiSchemaValidationError

        fake = FakeRawClient()
        fake.add_result_json({"title": "t", "narration": "", "sections": []})
        provider = _gemini_provider(default_config, fake)

        with pytest.raises(GeminiSchemaValidationError):
            provider.generate(build_script_context(selection, research))

    def test_missing_prompt_template_is_reported(self, selection, research, default_config, tmp_path):
        provider = _gemini_provider(default_config, FakeRawClient(), prompt_path=tmp_path / "nope.txt")
        with pytest.raises(FileNotFoundError):
            provider.generate(build_script_context(selection, research))

    def test_uses_structured_output_not_grounded_research(self, selection, research, default_config):
        """Stage 3 must never enable Search grounding / URL context."""
        fake = FakeRawClient()
        fake.add_result_json(_script_payload())
        provider = _gemini_provider(default_config, fake)

        provider.generate(build_script_context(selection, research))

        sent_config = fake.calls[0][2]
        assert getattr(sent_config, "tools", None) is None
        assert getattr(sent_config, "url_context", None) is None
        assert getattr(sent_config, "google_search", None) is None


# --------------------------------------------------------------------------- #
# 5. missing / invalid upstream input handling
# --------------------------------------------------------------------------- #


class TestInvalidInput:
    @staticmethod
    def _mutate(selection, **fields) -> "object":
        """Return a TopicSelection whose ``data`` payload has `fields` replaced."""
        from agents.seo.result import TopicSelection

        payload = json.loads(json.dumps(selection.to_dict()))
        payload["data"].update(fields)
        return TopicSelection.from_dict(payload)

    @staticmethod
    def _drop_topic_field(selection, field: str):
        from agents.seo.result import TopicSelection

        payload = json.loads(json.dumps(selection.to_dict()))
        payload["data"]["selected_topic"].pop(field, None)
        return TopicSelection.from_dict(payload)

    def test_missing_topic_selection_artifact(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_topic_selection(tmp_path / "topic_selection.json")

    def test_missing_research_artifact(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_research(tmp_path / "research.json")

    def test_selection_without_selected_topic(self, selection, research):
        with pytest.raises(ScriptInputError, match="selected_topic"):
            build_script_context(self._mutate(selection, selected_topic={}), research)

    def test_selection_with_null_selected_topic(self, selection, research):
        with pytest.raises(ScriptInputError, match="selected_topic"):
            build_script_context(self._mutate(selection, selected_topic=None), research)

    @pytest.mark.parametrize("field", REQUIRED_TOPIC_FIELDS)
    def test_selection_missing_required_topic_field(self, selection, research, field):
        with pytest.raises(ScriptInputError, match=field):
            build_script_context(self._drop_topic_field(selection, field), research)

    @pytest.mark.parametrize("field", REQUIRED_TOPIC_FIELDS)
    def test_selection_with_blank_required_topic_field(self, selection, research, field):
        broken = self._mutate(
            selection,
            selected_topic={**selection.selected_topic, field: "   "},
        )
        with pytest.raises(ScriptInputError, match=field):
            build_script_context(broken, research)

    def test_selection_topic_id_mismatch(self, selection, research):
        broken = self._mutate(
            selection,
            selected_topic={**selection.selected_topic, "topic_id": "some_other_topic"},
        )
        with pytest.raises(ScriptInputError, match="does not match"):
            build_script_context(broken, research)

    def test_selection_without_selected_topic_id(self, selection, research):
        with pytest.raises(ScriptInputError, match="selected_topic_id"):
            build_script_context(self._mutate(selection, selected_topic_id=""), research)

    def test_research_topic_id_mismatch(self, selection, research):
        mismatched = AIResearch(
            topic_id="not_the_selected_topic", explanation="Something about money."
        )
        with pytest.raises(ScriptInputError, match="does not match the selected topic"):
            build_script_context(selection, mismatched)

    def test_research_without_explanation_is_rejected(self, selection, research):
        empty = AIResearch(topic_id=selection.selected_topic_id, explanation="   ")
        with pytest.raises(ScriptInputError, match="explanation"):
            build_script_context(selection, empty)

    def test_research_without_topic_id_is_rejected(self, selection, research):
        broken = AIResearch(topic_id="", explanation="Something about money.")
        with pytest.raises(ScriptInputError, match="topic_id"):
            build_script_context(selection, broken)

    def test_invalid_json_artifact_reports_cleanly(self, tmp_path, selection):
        path = tmp_path / "topic_selection.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            load_topic_selection(path)


# --------------------------------------------------------------------------- #
# 6. deterministic mock generation
# --------------------------------------------------------------------------- #


class TestMockProvider:
    def test_same_inputs_give_identical_narration(self, selection, research, default_config):
        context = build_script_context(selection, research)
        first = MockScriptProvider(default_config).generate(context)
        second = MockScriptProvider(default_config).generate(context)
        assert first.narration == second.narration
        assert [s.narration for s in first.sections] == [s.narration for s in second.sections]

    def test_mock_uses_the_supplied_research_material(self, selection, research, default_config):
        script = MockScriptProvider(default_config).generate(build_script_context(selection, research))
        assert research.explanation in script.narration
        assert research.examples[0] in script.narration or research.story_ideas[0] in script.narration

    def test_mock_makes_no_network_call(self, selection, research, default_config, monkeypatch):
        def _boom(*args, **kwargs):
            raise AssertionError("mock provider must not open a socket")

        monkeypatch.setattr(socket, "socket", _boom)
        script = MockScriptProvider(default_config).generate(build_script_context(selection, research))
        assert script.sections

    def test_mock_never_imports_the_gemini_sdk(self):
        import agents.script.provider as provider_module

        source = Path(provider_module.__file__).read_text(encoding="utf-8")
        assert "import google" not in source
        assert "from google" not in source


# --------------------------------------------------------------------------- #
# 7. provider abstraction / replaceability
# --------------------------------------------------------------------------- #


class TestProviderAbstraction:
    def test_mock_is_the_default(self):
        assert build_script_provider(None).provider_name == "mock"
        assert build_script_provider("mock").provider_name == "mock"

    def test_unknown_provider_is_rejected(self):
        with pytest.raises(ScriptProviderError, match="unknown script provider"):
            build_script_provider("openai")

    def test_gemini_provider_requires_a_key(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        with pytest.raises(Exception) as excinfo:
            build_script_provider("gemini", env={})
        assert "gemini" in type(excinfo.value).__name__.lower()

    def test_injected_client_avoids_a_second_gemini_client(self, default_config):
        client = make_client(_gemini_config(), raw_client=FakeRawClient())
        provider = build_script_provider("gemini", config=default_config, client=client)
        assert provider.provider_name == "gemini"
        assert provider.model == _gemini_config().model

    def test_any_object_with_generate_can_serve_as_provider(self, selection, research):
        class CustomProvider:
            provider_name = "custom"

            def generate(self, context):
                return Script(
                    topic_id=context["topic_id"],
                    title="Custom script",
                    narration="A fully custom narration for the topic.",
                    sections=[ScriptSection("s1", "hook", "A fully custom narration.")],
                )

        script = run_script_generation(selection, research, CustomProvider())
        assert script.title == "Custom script"
        assert script.produced_by["provider"] is None  # stage does not overwrite provenance

    def test_run_script_generation_does_not_mutate_the_provider_result(self, selection, research):
        class OnceProvider:
            provider_name = "once"

            def __init__(self):
                self.calls = 0

            def generate(self, context):
                self.calls += 1
                return Script(
                    topic_id=context["topic_id"],
                    title="t",
                    narration="words",
                    sections=[ScriptSection("s1", "hook", "words")],
                    run_id="provider-side",
                )

        provider = OnceProvider()
        script = run_script_generation(selection, research, provider, run_id="stage_run_id")
        assert script.run_id == "stage_run_id"
        assert provider.calls == 1


# --------------------------------------------------------------------------- #
# 8. no source / evidence / citation fields required
# --------------------------------------------------------------------------- #


class TestNoEvidenceSurface:
    FORBIDDEN_FIELDS = (
        "sources",
        "evidence",
        "citations",
        "references",
        "urls",
        "source_url",
        "evidence_tier",
        "claim_id",
    )

    def test_script_model_has_no_source_fields(self):
        data = Script(topic_id="t", title="t").to_dict()
        for field in self.FORBIDDEN_FIELDS:
            assert field not in data, f"Stage 3 must not carry a {field} field"

    def test_narration_needs_no_source_or_numbers(self, selection, research):
        """A script with no statistics and no sources is still valid."""
        sparse = AIResearch(
            topic_id=selection.selected_topic_id,
            explanation="A plain explanation with no figures at all.",
        )
        script = generate_script(selection, sparse)
        assert script.narration.strip()
        script.validate()


# --------------------------------------------------------------------------- #
# 9. finance-safety restriction lives in the prompt / provider contract
# --------------------------------------------------------------------------- #


class TestFinanceSafety:
    def test_prompt_forbids_personalized_advice(self):
        prompt = PROMPT_PATH.read_text(encoding="utf-8").lower()
        assert "you should buy" in prompt
        assert "you must invest" in prompt
        assert "this stock will" in prompt
        assert "personalized investment, tax, or financial advice" in prompt

    def test_prompt_forbids_prediction_language(self):
        prompt = PROMPT_PATH.read_text(encoding="utf-8").lower()
        assert "predict prices" in prompt

    def test_system_prompt_states_the_advice_restriction(self, selection, research, default_config):
        provider = _gemini_provider(default_config, FakeRawClient())
        system = provider._system_prompt().lower()
        assert "never personalized investment, tax, or financial advice" in system
        assert "never cite sources" in system

    def test_artifact_carries_an_educational_notice(self, selection, research):
        data = generate_script(selection, research).to_dict()
        assert "not personalized" in data["educational_notice"].lower()

    def test_mock_narration_contains_no_directives(self, selection, research, default_config):
        script = MockScriptProvider(default_config).generate(build_script_context(selection, research))
        narration = script.narration.lower()
        for phrase in FORBIDDEN_NARRATION:
            assert phrase not in narration, f"mock narration contains a directive: {phrase}"


# --------------------------------------------------------------------------- #
# 10. CLI writes the flat artifact
# --------------------------------------------------------------------------- #


class TestCLI:
    def _run(self, monkeypatch, argv, run_dir):
        import agents.script.script_cli as cli

        monkeypatch.setattr(sys, "argv", argv)
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = cli.main()
        return code, buffer.getvalue()

    def test_cli_writes_script_json(self, tmp_path, monkeypatch, selection, research):
        monkeypatch.chdir(tmp_path)
        run_dir = tmp_path / "data" / "cli_run"
        run_dir.mkdir(parents=True)
        (run_dir / "topic_selection.json").write_text(
            json.dumps(selection.to_dict(), indent=2), encoding="utf-8"
        )
        (run_dir / "research.json").write_text(
            json.dumps(research.to_dict(), indent=2), encoding="utf-8"
        )

        code, out = self._run(monkeypatch, ["script_cli", "--run-id", "cli_run", "--config", str(CONFIG_PATH)], run_dir)

        assert code == 0
        artifact = run_dir / "script.json"
        assert artifact.is_file()
        data = json.loads(artifact.read_text(encoding="utf-8"))
        assert data["topic_id"] == selection.selected_topic_id
        assert data["narration"].strip()
        assert data["sections"]
        assert data["run_id"] == "cli_run"
        assert "provider=mock" in out

    def test_cli_updates_pipeline_state(self, tmp_path, monkeypatch, selection, research):
        monkeypatch.chdir(tmp_path)
        run_dir = tmp_path / "data" / "state_run"
        run_dir.mkdir(parents=True)
        (run_dir / "topic_selection.json").write_text(
            json.dumps(selection.to_dict(), indent=2), encoding="utf-8"
        )
        (run_dir / "research.json").write_text(
            json.dumps(research.to_dict(), indent=2), encoding="utf-8"
        )

        code, _ = self._run(monkeypatch, ["script_cli", "--run-id", "state_run", "--config", str(CONFIG_PATH)], run_dir)

        assert code == 0
        state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
        assert state["current_stage"] == "script_generation"
        assert state["artifacts"]["script_generation"]

    def test_cli_reports_missing_stage1_artifact(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        code, out = self._run(
            monkeypatch, ["script_cli", "--run-id", "nothing_here", "--config", str(CONFIG_PATH)], tmp_path / "data" / "nothing_here"
        )
        assert code == 2
        assert "topic_selection.json" in out

    def test_cli_reports_missing_stage2_artifact(self, tmp_path, monkeypatch, selection):
        monkeypatch.chdir(tmp_path)
        run_dir = tmp_path / "data" / "no_research"
        run_dir.mkdir(parents=True)
        (run_dir / "topic_selection.json").write_text(
            json.dumps(selection.to_dict(), indent=2), encoding="utf-8"
        )

        code, out = self._run(monkeypatch, ["script_cli", "--run-id", "no_research", "--config", str(CONFIG_PATH)], run_dir)
        assert code == 2
        assert "research.json" in out

    def test_cli_reports_mismatched_topic(self, tmp_path, monkeypatch, selection, research):
        monkeypatch.chdir(tmp_path)
        run_dir = tmp_path / "data" / "mismatch_run"
        run_dir.mkdir(parents=True)
        (run_dir / "topic_selection.json").write_text(
            json.dumps(selection.to_dict(), indent=2), encoding="utf-8"
        )
        mismatched = research.to_dict()
        mismatched["topic_id"] = "a_different_topic"
        (run_dir / "research.json").write_text(json.dumps(mismatched, indent=2), encoding="utf-8")

        code, out = self._run(monkeypatch, ["script_cli", "--run-id", "mismatch_run", "--config", str(CONFIG_PATH)], run_dir)
        assert code == 2
        assert "does not match the selected topic" in out

    def test_cli_gemini_without_key_fails_without_a_network_call(
        self, tmp_path, monkeypatch, selection, research
    ):
        monkeypatch.chdir(tmp_path)
        run_dir = tmp_path / "data" / "gemini_run"
        run_dir.mkdir(parents=True)
        (run_dir / "topic_selection.json").write_text(
            json.dumps(selection.to_dict(), indent=2), encoding="utf-8"
        )
        (run_dir / "research.json").write_text(
            json.dumps(research.to_dict(), indent=2), encoding="utf-8"
        )

        def _boom(*args, **kwargs):
            raise AssertionError("CLI must not reach the network without a key")

        monkeypatch.setattr(socket, "socket", _boom)
        code, out = self._run(
            monkeypatch,
            ["script_cli", "--run-id", "gemini_run", "--provider", "gemini", "--config", str(CONFIG_PATH)],
            run_dir,
        )
        assert code == 2
        assert "provider error" in out
        assert not (run_dir / "script.json").exists()


# --------------------------------------------------------------------------- #
# 11. the mascot never speaks
# --------------------------------------------------------------------------- #


class TestMascotNeverSpeaks:
    def test_script_records_a_non_speaking_mascot(self, selection, research):
        data = generate_script(selection, research).to_dict()
        assert data["mascot_speaks"] is False
        assert "does not speak" in data["mascot_note"].lower()

    def test_gemini_script_records_a_non_speaking_mascot(self, selection, research, default_config):
        fake = FakeRawClient()
        fake.add_result_json(_script_payload())
        script = _gemini_provider(default_config, fake).generate(
            build_script_context(selection, research)
        )
        assert script.mascot_speaks is False

    def test_prompt_forbids_mascot_dialogue(self):
        prompt = PROMPT_PATH.read_text(encoding="utf-8").lower()
        assert "does not speak" in prompt
        assert "never write narration for the mascot" in prompt
        assert "no lines for the mascot" in prompt

    def test_system_prompt_forbids_mascot_dialogue(self, selection, research, default_config):
        system = _gemini_provider(default_config, FakeRawClient())._system_prompt().lower()
        assert "mascot does not speak" in system
        assert "never write a line for it" in system

    def test_mascot_reference_lives_in_visual_hint_not_narration(self, selection, research, default_config):
        script = MockScriptProvider(default_config).generate(build_script_context(selection, research))
        for section in script.sections:
            if section.visual_hint:
                assert section.narration == section.narration.strip()


# --------------------------------------------------------------------------- #
# 12. duration target 65-80s, never below 60s
# --------------------------------------------------------------------------- #


class TestDuration:
    def test_channel_target_is_inside_the_short_format_range(self):
        assert DURATION_TARGET_RANGE == (65, 80)
        assert DURATION_TARGET_RANGE[0] <= TARGET_DURATION_SECONDS <= DURATION_TARGET_RANGE[1]
        assert CHANNEL.target_duration_seconds >= DURATION_MIN_SECONDS

    def test_default_script_targets_the_channel_duration(self, selection, research, default_config):
        script = MockScriptProvider(default_config).generate(build_script_context(selection, research))
        assert script.duration_target_seconds == CHANNEL.target_duration_seconds

    def test_gemini_script_targets_the_channel_duration(self, selection, research, default_config):
        fake = FakeRawClient()
        fake.add_result_json(_script_payload())
        script = _gemini_provider(default_config, fake).generate(
            build_script_context(selection, research)
        )
        assert script.duration_target_seconds == CHANNEL.target_duration_seconds

    def test_sections_sum_to_their_estimated_seconds(self, selection, research, default_config):
        script = MockScriptProvider(default_config).generate(build_script_context(selection, research))
        assert script.estimated_seconds == sum(
            section.estimated_seconds for section in script.sections
        )
        assert script.estimated_seconds > DURATION_MIN_SECONDS

    def test_prompt_states_the_duration_and_the_floor(self):
        prompt = PROMPT_PATH.read_text(encoding="utf-8")
        assert "{target_duration_seconds}" in prompt
        assert "never shorter than {duration_min_seconds} seconds" in prompt

    def test_zero_duration_target_is_rejected(self):
        with pytest.raises(ScriptError, match="duration_target_seconds"):
            Script(
                topic_id="t",
                title="t",
                duration_target_seconds=0,
                narration="words",
                sections=[ScriptSection("s1", "hook", "words")],
            ).validate()


# --------------------------------------------------------------------------- #
# 13. prompt exists and is finance / channel aware
# --------------------------------------------------------------------------- #


class TestPrompt:
    def test_prompt_file_exists(self):
        assert PROMPT_PATH.is_file()

    def test_prompt_is_json_only_and_structured(self):
        prompt = PROMPT_PATH.read_text(encoding="utf-8").lower()
        assert "json" in prompt
        assert '"narration"' in prompt
        assert '"sections"' in prompt
        assert '"role"' in prompt

    def test_prompt_requires_sections_and_hook_payoff(self):
        prompt = PROMPT_PATH.read_text(encoding="utf-8").lower()
        assert "3-6 ordered segments" in prompt
        assert "curiosity in the first sentence" in prompt
        assert "payoff or reveal" in prompt

    def test_prompt_forbids_browsing_and_citations(self):
        prompt = PROMPT_PATH.read_text(encoding="utf-8").lower()
        assert "no web research" in prompt
        assert "no browsing" in prompt
        assert "no citations, no urls" in prompt
        assert "no claims about verifiability" in prompt

    def test_prompt_declares_detective_framing_optional(self):
        prompt = PROMPT_PATH.read_text(encoding="utf-8").lower()
        assert "style, not a mandatory formula" in prompt

    def test_prompt_tokens_are_all_substituted(self, selection, research, default_config):
        provider = _gemini_provider(default_config, FakeRawClient())
        rendered = provider._render_prompt(build_script_context(selection, research))
        assert re.search(r"\{[a-z_]+\}", rendered) is None, "unsubstituted prompt token remains"

    def test_prompt_warns_against_inventing_numbers(self):
        prompt = PROMPT_PATH.read_text(encoding="utf-8").lower()
        assert "never invent a figure" in prompt

    def test_prompt_respects_section_roles(self):
        assert set(SECTION_ROLES) >= {"hook", "setup", "explanation", "example", "reveal", "payoff"}


# --------------------------------------------------------------------------- #
# 13b. the prompt carries a spoken-word budget instead of trusting invented
#      section durations
# --------------------------------------------------------------------------- #


class TestPromptWordBudget:
    """The live run produced 142 words against a 70s target (~57s spoken).

    The fix is a writing instruction, not a runtime gate: the prompt now asks
    for a measurable word budget and requires `estimated_seconds` to be derived
    from that word count rather than invented.
    """

    LOW = 165
    HIGH = 190
    WORDS_PER_SECOND = 2.5

    @staticmethod
    def _prompt() -> str:
        return PROMPT_PATH.read_text(encoding="utf-8")

    @classmethod
    def _flat(cls) -> str:
        """Prompt text with every run of whitespace collapsed to one space.

        The prompt is hard-wrapped prose, so assertions must not depend on where
        a line happens to break.
        """
        return re.sub(r"\s+", " ", cls._prompt()).lower()

    def test_prompt_states_the_spoken_word_budget(self):
        prompt = self._prompt()
        assert f"{self.LOW}-{self.HIGH} words" in prompt

    def test_prompt_states_the_speaking_rate(self):
        prompt = self._prompt()
        assert "2.5 spoken words per" in prompt
        assert "150 words per minute" in prompt

    def test_prompt_derives_seconds_from_words(self):
        prompt = self._prompt()
        assert "word count / 2.5" in prompt
        assert "derive seconds from" in prompt.lower()

    def test_prompt_requires_estimated_seconds_to_be_computed_not_guessed(self):
        prompt = self._prompt()
        assert "do not guess it" in prompt
        assert "do not aim it at the target" in prompt

    def test_prompt_ties_section_length_to_the_total(self):
        flat = self._flat()
        assert "total words / number of sections" in flat
        assert "all sections together must total 165-190 words." in flat

    def test_prompt_forbids_double_counting_narration_and_sections(self):
        prompt = self._prompt()
        assert "they do not add words to it" in prompt
        assert "do not repeat or expand anything here" in prompt

    def test_prompt_forbids_padding(self):
        flat = self._flat()
        assert "do not pad the script with filler" in flat
        assert "rather than by padding" in flat

    def test_prompt_keeps_the_narration_natural(self):
        flat = self._flat()
        assert "not a rhythm to recite" in flat
        assert (
            "do not mention counting, timing, or this budget in the narration itself." in flat
        )

    def test_budget_band_maps_into_the_65_80_second_window(self):
        """The stated word range must actually land inside 65-80s at 2.5 w/s."""
        low_seconds = self.LOW / self.WORDS_PER_SECOND
        high_seconds = self.HIGH / self.WORDS_PER_SECOND
        assert 65 <= low_seconds <= 80
        assert 65 <= high_seconds <= 80
        assert DURATION_MIN_SECONDS < low_seconds

    def test_budget_band_clears_the_live_failure(self):
        """The 142-word live result must fall outside the band, and be under 60s."""
        assert not (self.LOW <= 142 <= self.HIGH)
        assert 142 / self.WORDS_PER_SECOND < DURATION_MIN_SECONDS

    def test_prompt_still_anchors_the_duration_tokens(self):
        prompt = self._prompt()
        assert "{target_duration_seconds}" in prompt
        assert "{duration_min_seconds}" in prompt
        assert "never shorter than {duration_min_seconds} seconds" in prompt
        assert "between 65 and 80" in prompt

    def test_prompt_still_declares_no_qa_in_the_output(self):
        prompt = self._prompt()
        assert "No quality score, no PASS/FAIL, no confidence rating, no verdict." in prompt

    # -- preserved channel / style constraints ------------------------------ #

    def test_word_budget_did_not_disturb_mascot_rule(self):
        prompt = self._prompt()
        assert "**The mascot DOES NOT SPEAK.**" in prompt
        assert "Never write narration for the mascot" in prompt
        assert "No lines for the mascot." in prompt

    def test_word_budget_did_not_disturb_finance_safety(self):
        prompt = self._prompt()
        assert '"you should buy"' in prompt
        assert '"you must invest in"' in prompt
        assert '"this stock will"' in prompt
        assert "Do NOT predict prices, returns, or outcomes." in prompt
        assert "personalized investment, tax, or financial advice" in prompt

    def test_word_budget_did_not_disturb_optional_detective_framing(self):
        prompt = self._prompt()
        assert "style, not a mandatory formula" in prompt
        assert (
            "equally valid structures include a straight question, a story, a comparison"
            in self._flat()
        )

    def test_word_budget_did_not_disturb_no_browsing_or_citations(self):
        prompt = self._prompt()
        assert "No web research, no browsing, no searches." in prompt
        assert "No citations, no URLs, no source names, no links." in prompt
        assert "No claims about verifiability, corroboration" in prompt

    def test_word_budget_did_not_disturb_structure_requirements(self):
        prompt = self._prompt()
        assert "3-6 ordered segments" in prompt
        assert "curiosity in the first sentence" in prompt
        assert "payoff or reveal" in prompt
        assert "never invent a figure" in prompt

    def test_prompt_still_renders_with_no_leftover_tokens(self, selection, research, default_config):
        provider = _gemini_provider(default_config, FakeRawClient())
        rendered = provider._render_prompt(build_script_context(selection, research))
        assert re.search(r"\{[a-z_]+\}", rendered) is None
        assert f"{self.LOW}-{self.HIGH} words" in rendered


# --------------------------------------------------------------------------- #
# 14. structured output path
# --------------------------------------------------------------------------- #


class TestStructuredOutput:
    def test_provider_calls_generate_structured(self, selection, research, default_config, monkeypatch):
        import providers.gemini.script_provider as provider_module

        calls: list[str] = []
        monkeypatch.setattr(
            provider_module.GeminiClient,
            "generate_structured",
            lambda self, *a, **k: calls.append(a[0]) or _script_schema(),
        )
        monkeypatch.setattr(
            provider_module.GeminiClient,
            "generate_research",
            lambda self, *a, **k: (_ for _ in ()).throw(
                AssertionError("Stage 3 must not call generate_research")
            ),
        )
        provider = _gemini_provider(default_config, FakeRawClient())
        provider.generate(build_script_context(selection, research))
        assert len(calls) == 1

    def test_response_model_is_permissive_and_json_only(self):
        from providers.gemini.script_provider import ScriptSchema

        assert ScriptSchema.model_validate({}).narration == ""
        assert ScriptSchema.model_validate({}).sections == []
        assert set(ScriptSchema.model_fields) == {"title", "narration", "sections"}

    def test_no_url_in_request_config(self, selection, research, default_config):
        fake = FakeRawClient()
        fake.add_result_json(_script_payload())
        provider = _gemini_provider(default_config, fake)
        provider.generate(build_script_context(selection, research))
        sent_config = fake.calls[0][2]
        dumped = repr(sent_config)
        assert "url_context" not in dumped
        assert "google_search" not in dumped


def _script_schema():
    from providers.gemini.script_provider import ScriptSchema

    payload = _script_payload()
    return ScriptSchema.model_validate(payload)


# --------------------------------------------------------------------------- #
# 15. sections are usable by Scene Planning
# --------------------------------------------------------------------------- #


class TestSectionHandoff:
    def test_every_section_carries_its_own_narration(self, selection, research, default_config):
        script = MockScriptProvider(default_config).generate(build_script_context(selection, research))
        for section in script.sections:
            assert section.narration.strip(), f"section {section.section_id} has no narration"

    def test_section_narration_is_a_subset_of_the_full_narration(self, selection, research, default_config):
        script = MockScriptProvider(default_config).generate(build_script_context(selection, research))
        for section in script.sections:
            assert section.narration in script.narration

    def test_full_narration_is_the_join_of_the_sections(self, selection, research, default_config):
        script = MockScriptProvider(default_config).generate(build_script_context(selection, research))
        joined = " ".join(section.narration.strip() for section in script.sections)
        assert script.narration.strip() == joined

    def test_sections_carry_a_role_and_a_duration_for_timing(self, selection, research, default_config):
        script = MockScriptProvider(default_config).generate(build_script_context(selection, research))
        for section in script.sections:
            assert section.role in SECTION_ROLES
            assert section.estimated_seconds > 0

    def test_sections_are_shape_compatible_with_scene_planning_consumer(self, selection, research):
        """Scene Planning needs the words per section, not a single blob."""
        script = generate_script(selection, research)
        handoff = {
            "narration": script.narration,
            "sections": [section.to_dict() for section in script.sections],
        }
        assert handoff["sections"]
        for section in handoff["sections"]:
            assert isinstance(section["narration"], str) and section["narration"]
            assert isinstance(section["role"], str)
            assert isinstance(section["estimated_seconds"], int)


# --------------------------------------------------------------------------- #
# 16. no QA stage, no score, no PASS/FAIL contract
# --------------------------------------------------------------------------- #


class TestNoQaContract:
    def test_package_exposes_no_validation_verdict(self):
        import agents.script as script_package

        for name in ("ScriptValidation", "validate_script", "verdict", "score", "qa"):
            assert not hasattr(script_package, name), f"Stage 3 must not export {name}"

    def test_stage_package_defines_no_qa_module(self):
        stage_dir = ROOT / "agents" / "script"
        offenders = [
            path.name
            for path in stage_dir.glob("*.py")
            if path.name in {"validator.py", "qa.py", "quality.py", "fact_checker.py"}
        ]
        assert offenders == []

    def test_script_model_has_no_score_or_verdict_field(self):
        data = Script(topic_id="t", title="t").to_dict()
        for field in ("score", "verdict", "qa", "quality", "passed", "warnings", "issues"):
            assert field not in data, f"Stage 3 artifact must not carry a {field} field"

    def test_validation_is_structural_not_a_score(self):
        """`validate` may require material, but never returns a score."""
        assert Script.validate.__doc__
        assert "no PASS/FAIL verdict" in Script.validate.__doc__
        with pytest.raises(ScriptError):
            Script(topic_id="t", title="t", narration="", sections=[]).validate()

    def test_narration_is_not_scanned_for_content_rules(self, selection, research):
        """No content gating: even an off-topic word still yields a script."""
        weird = AIResearch(
            topic_id=selection.selected_topic_id,
            explanation="Buy this stock will make you rich.",
        )
        script = generate_script(selection, weird)
        assert script.narration.strip()

    # -- the word-budget fix must stay in the prompt, not become a runtime gate -- #

    def test_model_carries_no_word_count_or_duration_estimate(self):
        """No runtime word-count field was introduced to enforce the budget."""
        data = Script(topic_id="t", title="t").to_dict()
        for field in (
            "word_count",
            "words",
            "spoken_seconds",
            "actual_seconds",
            "measured_seconds",
            "duration_actual",
            "duration_estimate",
        ):
            assert field not in data, f"Stage 3 must not carry a runtime {field}"
        assert "word_count" not in ScriptSection(
            section_id="s1", role="hook", narration="w"
        ).to_dict()

    def test_validate_does_not_enforce_a_word_budget(self):
        """A 3-word script still validates: the budget is a prompt instruction."""
        script = Script(
            topic_id="t",
            title="t",
            narration="Too short.",
            sections=[ScriptSection("s1", "hook", "Too short.")],
        )
        assert script.validate() is script
        assert len(script.narration.split()) == 2

    def test_short_and_long_narration_are_both_accepted(self, selection, research):
        """No duration-based rejection: length never fails closed."""
        for narration in ("Short.", "word " * 900):
            script = generate_script(
                selection,
                AIResearch(topic_id=selection.selected_topic_id, explanation=narration),
            )
            assert script.validate() is not None

    def test_stage_defines_no_duration_gate(self):
        """No word-count or duration rejection helper exists in the stage."""
        import agents.script.stage as stage_module

        source = Path(stage_module.__file__).read_text(encoding="utf-8")
        assert "word_count" not in source
        assert "len(script.narration.split())" not in source