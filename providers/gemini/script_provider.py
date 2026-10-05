"""Stage-3 Gemini script provider.

Turns the Stage 1 topic plus Stage 2 research into a finished spoken Shorts
script using the existing Gemini client/config infrastructure - there is no
second client. It calls ``generate_structured`` (NOT ``generate_research``), so
no Google Search grounding and no URL context are enabled and nothing is
retrieved.

The response schema is deliberately permissive: only ``title``, ``narration``
and ``sections`` are required, and every field of a section defaults when
absent. There is no score, no verdict and no validation gate beyond "the
narration is not empty".
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from agents.research.ai_research_models import AIResearch
from agents.script.script_models import (
    EDUCATIONAL_NOTICE,
    MASCOT_NOTE,
    MASCOT_SPEAKS,
    TARGET_DURATION_SECONDS,
    Script,
    ScriptSection,
)
from agents.seo.config import TopicSelectionConfig
from providers.gemini.client import GeminiClient
from providers.gemini.errors import GeminiSchemaValidationError

PROMPT_VERSION = "script/short_script-v1"
DEFAULT_PROMPT_PATH = Path("prompts") / "script" / "short_script-v1.txt"
GENERATED_BY = "gemini_script_provider_v1"


class ScriptSectionSchema(BaseModel):
    """One narration segment as returned by the model."""

    section_id: str = ""
    role: str = ""
    narration: str = Field(default="")
    estimated_seconds: int = 0
    visual_hint: str = ""


class ScriptSchema(BaseModel):
    """Structured script response for one selected topic."""

    title: str = Field(default="")
    narration: str = Field(default="")
    sections: list[ScriptSectionSchema] = Field(default_factory=list)


class GeminiScriptProvider:
    """Script provider backed by Gemini structured output."""

    provider_name = "gemini"

    def __init__(
        self,
        config: TopicSelectionConfig | None,
        client: GeminiClient,
        *,
        prompt_path: str | Path = DEFAULT_PROMPT_PATH,
    ) -> None:
        self._config = config
        self._client = client
        self._prompt_path = Path(prompt_path)

    # -- metadata --------------------------------------------------------- #

    @property
    def model(self) -> str:
        return self._client.model

    @property
    def prompt_version(self) -> str:
        return PROMPT_VERSION

    @property
    def produced_by(self) -> dict:
        """Header provenance for the script artifact."""
        return {
            "prompt_version": PROMPT_VERSION,
            "provider": self.provider_name,
            "model": self.model,
        }

    # -- protocol --------------------------------------------------------- #

    def generate(self, context: dict) -> Script:
        schema = self._client.generate_structured(
            self._render_prompt(context),
            ScriptSchema,
            system_prompt=self._system_prompt(),
        )
        try:
            sections = [
                ScriptSection(
                    section_id=item.section_id or f"s{index + 1}",
                    role=item.role,
                    narration=item.narration,
                    estimated_seconds=item.estimated_seconds,
                    visual_hint=item.visual_hint,
                )
                for index, item in enumerate(schema.sections)
            ]
            script = Script(
                topic_id=str(context["topic_id"]),
                title=schema.title or str(context.get("title") or ""),
                duration_target_seconds=TARGET_DURATION_SECONDS,
                narration=schema.narration,
                sections=sections,
                pillar=str(context.get("pillar") or ""),
                channel_id=self._channel_id(),
                produced_by=self.produced_by,
                educational_notice=EDUCATIONAL_NOTICE,
                mascot_note=MASCOT_NOTE,
                mascot_speaks=MASCOT_SPEAKS,
            )
            return script.validate()
        except (TypeError, ValueError) as exc:
            raise GeminiSchemaValidationError(
                f"Gemini script failed Stage-3 validation: {exc}"
            ) from exc

    # -- internals --------------------------------------------------------- #

    def _channel_id(self) -> str:
        if self._config is not None:
            return self._config.channel.channel_id
        return "finance_mascot"

    def _system_prompt(self) -> str:
        channel = self._config.channel if self._config else None
        promise = channel.channel_promise if channel else "Understand money. Grow wealth."
        mascot_role = channel.mascot.role if channel else "financial_detective"
        return (
            "You write spoken narration for a short-form finance video. "
            f"The channel promise is: {promise} "
            "You answer from your own knowledge only: never browse, never cite "
            "sources, and never include URLs. "
            f"The on-screen character is a {mascot_role}. The mascot DOES NOT SPEAK "
            "- it appears visually alongside the narration, so never write a line "
            "for it. This is general financial education, never personalized "
            "investment, tax, or financial advice. Return JSON only."
        )

    def _render_prompt(self, context: dict) -> str:
        path = Path(self._prompt_path)
        if not path.is_file():
            raise FileNotFoundError(f"prompt template not found: {path}")

        channel = self._config.channel if self._config else None
        research: AIResearch = context["research"]
        pillar_info = None
        if self._config is not None:
            try:
                pillar_info = self._config.pillar_info(str(context.get("pillar") or ""))
            except Exception:
                pillar_info = None

        def _bullets(items: list[str], empty: str = "- (none supplied)") -> list[str]:
            values = [f"- {item}" for item in (items or []) if str(item).strip()]
            return values or [empty]

        research_block = "\n".join(
            [
                "## key_concepts",
                *_bullets(research.key_concepts),
                "",
                "## explanation",
                research.explanation or "(none supplied)",
                "",
                "## examples",
                *_bullets(research.examples),
                "",
                "## numbers",
                *_bullets(research.numbers, "- (none supplied - do NOT invent figures)"),
                "",
                "## interesting_angles",
                *_bullets(research.interesting_angles),
                "",
                "## common_misconceptions",
                *_bullets(research.common_misconceptions),
                "",
                "## story_ideas",
                *_bullets(research.story_ideas),
            ]
        )

        replacements = {
            "{title}": str(context.get("title") or ""),
            "{pillar}": str(context.get("pillar") or ""),
            "{pillar_label}": pillar_info.label if pillar_info else str(context.get("pillar") or ""),
            "{pillar_focus}": pillar_info.focus if pillar_info else str(context.get("pillar") or ""),
            "{pillar_guidance}": pillar_info.guidance if pillar_info else "",
            "{angle}": str(context.get("angle") or "(none supplied)"),
            "{hook_direction}": str(context.get("hook_direction") or "(none supplied)"),
            "{topic_type}": str(context.get("topic_type") or "(none supplied)"),
            "{topic_description}": str(context.get("description") or "(none supplied)"),
            "{channel_id}": channel.channel_id if channel else "finance_mascot",
            "{channel_promise}": channel.channel_promise if channel else "Understand money. Grow wealth.",
            "{audience}": channel.audience if channel else "",
            "{tone}": channel.tone if channel else "",
            "{target_duration_seconds}": str(channel.target_duration_seconds) if channel else "70",
            "{duration_min_seconds}": "60",
            "{mascot_role}": channel.mascot.role if channel else "financial_detective",
            "{research_material}": research_block,
        }

        rendered = path.read_text(encoding="utf-8")
        for token, value in replacements.items():
            rendered = rendered.replace(token, value)
        return rendered