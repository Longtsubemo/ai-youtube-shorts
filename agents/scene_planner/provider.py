"""ScenePlannerProvider abstraction and deterministic mock."""

from __future__ import annotations

import os
from typing import Mapping, Protocol

from agents.script.script_models import Script
from agents.scene_planner.scene_models import (
    MASCOT_ASSET_PATH,
    Mascot,
    Scene,
    ScenePlan,
    ScenePlanningError,
)
from agents.seo.config import TopicSelectionConfig


class ScenePlannerProviderError(ValueError):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ScenePlannerProvider(Protocol):
    provider_name: str

    def generate(self, context: dict) -> ScenePlan: ...


# Deterministic, scene-specific visual subjects for the mock planner. Each
# yields a compact SDXL-compatible generation_prompt: subject, supporting
# objects, simple composition/environment - no narration, no mascot, no
# readable text, no render-mode/camera words, no channel style block (the
# image worker prepends CHANNEL.image_style later).
_ROLE_SUBJECTS = {
    "hook": "Magnifying glass hovering over scattered coins on a wooden detective desk",
    "setup": "Stack of paperwork and a closed ledger on a warm wooden desk",
    "explanation": "Balance scale holding coins on both pans against a plain paper backdrop",
    "example": "Open briefcase with stacked coins and folded documents on a wooden desk",
    "reveal": "Lifted sheet of paper uncovering hidden coins on a wooden desk",
    "payoff": "Upward line curve drawn in ink on paper with coins stacked beside it",
    "closing": "Closed ledger, fountain pen and stacked coins on a warm wooden desk",
}
_DEFAULT_SUBJECT = "Coins, folded documents and a magnifying glass on a wooden desk"
_PROMPT_TAIL = "Muted paper background, simple spacious composition."


def _mock_generation_prompt(role: str) -> str:
    subject = _ROLE_SUBJECTS.get(role, _DEFAULT_SUBJECT)
    return f"{subject}. {_PROMPT_TAIL}"


class MockScenePlannerProvider:
    provider_name = "mock"

    def __init__(self, config: TopicSelectionConfig | None = None) -> None:
        self._config = config

    @property
    def produced_by(self) -> dict:
        return {"prompt_version": None, "provider": "mock", "model": None}

    def generate(self, context: dict) -> ScenePlan:
        script: Script = context["script"]
        scenes: list[Scene] = []
        for idx, section in enumerate(script.sections):
            scene_id = f"sc{idx + 1:02d}"
            scenes.append(
                Scene(
                    scene_id=scene_id,
                    order=idx,
                    duration_seconds=max(4, section.estimated_seconds or 7),
                    narration=section.narration,
                    source_section_id=section.section_id,
                    visual_description=f"Visual interpretation of {section.role} for {script.title}",
                    camera="medium shot",
                    composition="clean composition",
                    environment="minimal background",
                    generation_prompt=_mock_generation_prompt(section.role),
                    required_assets=[MASCOT_ASSET_PATH],
                    mascot=Mascot(present=True, action=f"{section.role}_action", asset=MASCOT_ASSET_PATH),
                )
            )
        return ScenePlan(
            topic_id=script.topic_id,
            title=script.title,
            target_duration_seconds=script.duration_target_seconds,
            total_scene_duration_seconds=sum(s.duration_seconds for s in scenes),
            scenes=scenes,
            channel_id=(self._config.channel.channel_id if self._config else (script.channel_id or "finance_mascot")),
            mascot_asset_path=MASCOT_ASSET_PATH,
            produced_by=self.produced_by,
        )


def build_scene_planner_provider(
    name: str | None,
    *,
    config: TopicSelectionConfig | None = None,
    env: Mapping[str, str] | None = None,
    client: object | None = None,
) -> ScenePlannerProvider:
    if name in (None, "", "mock"):
        return MockScenePlannerProvider(config)

    if name == "gemini":
        from providers.gemini.client import GeminiClient
        from providers.gemini.config import GeminiConfig
        from providers.gemini.scene_planner_provider import GeminiScenePlannerProvider

        if client is None:
            gemini_config = GeminiConfig.from_env(env if env is not None else os.environ)
            client = GeminiClient(gemini_config)
        return GeminiScenePlannerProvider(config=config, client=client)

    raise ScenePlannerProviderError(f"unknown scene planner provider: {name!r}")
