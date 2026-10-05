"""Stage-4 Gemini Scene Planner provider."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from agents.script.script_models import Script
from agents.scene_planner.scene_models import (
    MASCOT_ASSET_PATH,
    MASCOT_SPEAKS,
    Mascot,
    Scene,
    ScenePlan,
    ScenePlanningError,
)
from agents.seo.config import TopicSelectionConfig
from providers.gemini.client import GeminiClient
from providers.gemini.errors import GeminiSchemaValidationError

PROMPT_VERSION = "scene_planner/short_scene-v1"
DEFAULT_PROMPT_PATH = Path("prompts") / "scene_planner" / "short_scene-v1.txt"


class MascotSchema(BaseModel):
    present: bool = True
    action: str = ""
    asset: str = MASCOT_ASSET_PATH


class SceneSchema(BaseModel):
    scene_id: str = ""
    duration_seconds: int = 0
    narration: str = Field(default="")
    source_section_id: str = ""
    visual_description: str = ""
    camera: str = ""
    composition: str = ""
    environment: str = ""
    generation_prompt: str = ""
    required_assets: list[str] = Field(default_factory=list)
    transition_hint: str = ""
    mascot: MascotSchema = Field(default_factory=MascotSchema)


class ScenePlanSchema(BaseModel):
    title: str = ""
    scenes: list[SceneSchema] = Field(default_factory=list)


class GeminiScenePlannerProvider:
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

    @property
    def model(self) -> str:
        return self._client.model

    @property
    def produced_by(self) -> dict:
        return {
            "prompt_version": PROMPT_VERSION,
            "provider": self.provider_name,
            "model": self.model,
        }

    def _channel_id(self) -> str:
        if self._config is not None:
            return self._config.channel.channel_id
        return "finance_mascot"

    def generate(self, context: dict) -> ScenePlan:
        script: Script = context["script"]
        schema = self._client.generate_structured(
            self._render_prompt(script),
            ScenePlanSchema,
            system_prompt=self._system_prompt(),
        )
        try:
            scenes = []
            for idx, item in enumerate(schema.scenes):
                scenes.append(
                    Scene(
                        scene_id=item.scene_id or f"sc{idx + 1:02d}",
                        order=idx,
                        duration_seconds=max(1, item.duration_seconds),
                        narration=item.narration,
                        source_section_id=item.source_section_id,
                        visual_description=item.visual_description,
                        camera=item.camera,
                        composition=item.composition,
                        environment=item.environment,
                        generation_prompt=item.generation_prompt,
                        required_assets=[a for a in item.required_assets if a],
                        transition_hint=item.transition_hint,
                        mascot=Mascot(
                            present=item.mascot.present,
                            action=item.mascot.action,
                            asset=item.mascot.asset or MASCOT_ASSET_PATH,
                        ),
                    )
                )
            plan = ScenePlan(
                topic_id=script.topic_id,
                title=schema.title or script.title,
                target_duration_seconds=script.duration_target_seconds,
                total_scene_duration_seconds=sum(s.duration_seconds for s in scenes),
                scenes=scenes,
                channel_id=self._channel_id(),
                mascot_asset_path=MASCOT_ASSET_PATH,
                produced_by=self.produced_by,
            )
            return plan.validate()
        except Exception as exc:  # noqa: BLE001
            raise GeminiSchemaValidationError(f"Gemini scene plan validation failed: {exc}") from exc

    def _system_prompt(self) -> str:
        channel = self._config.channel if self._config else None
        promise = channel.channel_promise if channel else "Understand money. Grow wealth."
        mascot_role = channel.mascot.role if channel else "financial_detective"
        return (
            "You create visual scenes from spoken narration. "
            f"Channel promise: {promise}. "
            "Answer from your own knowledge. Never browse, never cite sources, never include URLs. "
            f"Mascot role: {mascot_role}. The mascot DOES NOT SPEAK. "
            "Create visuals only; never rewrite the spoken narration. "
            "This is general financial education, never personalized investment advice. "
            "Return JSON only."
        )

    def _render_prompt(self, script: Script) -> str:
        path = Path(self._prompt_path)
        if not path.is_file():
            raise FileNotFoundError(f"prompt template not found: {path}")

        sections = "\n\n".join(
            [
                f"SECTION {i+1}: {s.section_id} / {s.role} / est={s.estimated_seconds}s\n{s.narration}"
                for i, s in enumerate(script.sections)
            ]
        )
        rendered = path.read_text(encoding="utf-8")
        replacements = {
            "{channel_id}": self._channel_id(),
            "{topic_id}": script.topic_id,
            "{title}": script.title,
            "{pillar}": script.pillar,
            "{target_duration_seconds}": str(script.duration_target_seconds),
            "{full_narration}": script.narration,
            "{sections}": sections,
            "{mascot_asset_path}": MASCOT_ASSET_PATH,
        }
        for token, value in replacements.items():
            rendered = rendered.replace(token, value)
        return rendered
