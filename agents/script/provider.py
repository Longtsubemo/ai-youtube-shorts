"""ScriptProvider abstraction and its deterministic mock implementation.

The package-based script-generation path (docs/ARCHITECTURE.md, user-facing
"Stage 3" = repo Stage.SCRIPT_GENERATION): a `ResearchPackage` is turned into
a validated 30-60s Short script directly, without waiting on a fact-check
result. The deterministic validator is the final gate for every provider, so
neither the mock nor a real (Gemini) provider can publish a script that
invents, strengthens, or drops hedges.

A real implementation (e.g. `GeminiScriptProvider` in `providers/gemini/`)
drops in behind the same protocol without touching the stage logic. The mock
provider serves deterministic planning only; it makes no network requests.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Protocol

from agents.research.models import FactClaim, ResearchPackage
from agents.script.models import PRODUCED_BY, Script, ScriptLine, line_id_for, script_id_for
from agents.script.script_generator import (
    STATUS_BLOCKED,
    STATUS_GENERATED,
    STATUS_INSUFFICIENT,
    MockScriptGenerator,
    ScriptConfig,
    ScriptGenerationResult,
)
from agents.script.validator import validate_script_from_package
from agents.seo.models import Provenance, utc_now


class ScriptProviderError(ValueError):
    """Raised when a script provider cannot fulfill a generation request."""


class ScriptProvider(Protocol):
    """Anything that can turn a ResearchPackage into a validated Script.

    The provider consumes ONLY the claims in the package and must obey the
    same deterministic bounds a mock does: no invented facts, no strengthened
    certainty, preserved hedges, claim_refs that all resolve to package
    claims, and a script that passes `validate_script_from_package`.
    """

    def generate(
        self,
        package: ResearchPackage,
        config: ScriptConfig | None = None,
        run_id: str = "",
        created_at: str = "",
    ) -> ScriptGenerationResult: ...


class MockScriptProvider:
    """Deterministic, network-free script provider over a ResearchPackage.

    Uses the same planning logic as `MockScriptGenerator`: every factual line
    is a (near) verbatim package claim, framing lines are inert boilerplate,
    and the deterministic package validator is the final gate. Without a
    fact-check result there is no approved/rejected split, so the whole
    package claim set is the working set and `unused_approved_claim_ids` holds
    the package claims the script left out.
    """

    def __init__(self, config: ScriptConfig | None = None) -> None:
        self._generator = MockScriptGenerator()
        self.config = (config or ScriptConfig()).validate()

    def generate(
        self,
        package: ResearchPackage,
        config: ScriptConfig | None = None,
        run_id: str = "",
        created_at: str = "",
    ) -> ScriptGenerationResult:
        config = (config or self.config).validate()
        package = package.validate()
        claims = list(package.claims)

        if len(claims) < config.min_approved_claims:
            return ScriptGenerationResult(
                status=STATUS_INSUFFICIENT,
                message=(
                    f"insufficient content: {len(claims)} package claim(s) "
                    f"< required minimum {config.min_approved_claims}"
                ),
                blocked_reasons=("too_few_approved_claims",),
            )

        placeholder_lines = self._generator._plan(claims, config, package.topic_title)
        estimated_seconds = sum(
            len(line.text.split()) / config.words_per_minute * 60
            for line in placeholder_lines
        )
        if estimated_seconds < config.min_duration_seconds:
            return ScriptGenerationResult(
                status=STATUS_INSUFFICIENT,
                message=(
                    f"insufficient content: script would run "
                    f"{estimated_seconds:.1f}s, below the "
                    f"{config.min_duration_seconds:.0f}s target"
                ),
                blocked_reasons=("too_short",),
            )

        lines = [
            ScriptLine(
                line_id=line.line_id,
                text=line.text,
                claim_refs=line.claim_refs,
                purpose=line.purpose,
                estimated_duration_seconds=round(
                    len(line.text.split()) / config.words_per_minute * 60, 2
                ),
            )
            for line in placeholder_lines
        ]
        total_seconds = round(sum(line.estimated_duration_seconds for line in lines), 1)
        used = {claim_id for line in lines for claim_id in line.claim_refs}

        run_id = run_id or package.run_id
        created_at = created_at or utc_now()
        script = Script(
            script_id=script_id_for(run_id, package.topic_id),
            topic_id=package.topic_id,
            topic_title=package.topic_title,
            title=self._title(package, claims),
            hook=lines[0].text,
            lines=tuple(lines),
            estimated_duration_seconds=total_seconds,
            used_claim_ids=tuple(sorted(used)),
            unused_approved_claim_ids=tuple(
                sorted({claim.claim_id for claim in package.claims} - used)
            ),
            result_id="",
            package_id=package.package_id,
            provenance=Provenance(
                generated_by="script_agent_v1",
                source="MockScriptProvider",
                created_at=created_at,
                provider=None,
                model="MockScriptProvider",
            ),
            created_at=created_at,
            run_id=run_id,
            produced_by=dict(PRODUCED_BY),
        )
        script.validate()

        validation = validate_script_from_package(script, package, config)
        if validation.valid:
            return ScriptGenerationResult(
                status=STATUS_GENERATED,
                script=script,
                validation=validation,
                message="generated",
            )
        return ScriptGenerationResult(
            status=STATUS_BLOCKED,
            script=script,
            validation=validation,
            message="generated script failed validation",
            blocked_reasons=validation.errors,
        )

    @staticmethod
    def _title(package: ResearchPackage, claims: list[FactClaim]) -> str:
        title = (package.topic_title or claims[0].claim_text).strip()
        if len(title) > 70:
            title = title.split("?")[0][:70].strip() or claims[0].claim_text[:70]
        return title[:70]


def build_script_provider(
    name: str | None,
    *,
    env: Mapping[str, str] | None = None,
    config: ScriptConfig | None = None,
) -> ScriptProvider:
    """Return the script provider selected by `name` (default: mock).

    Only the requested provider is imported, so a mock/default run never
    needs or imports the Gemini SDK. Providing `gemini` requires the
    GEMINI_API_KEY environment variable; a missing key raises
    GeminiConfigurationError rather than silently failing over to mock.
    """
    if name in (None, "", "mock"):
        return MockScriptProvider(config=config)

    if name == "gemini":
        from providers.gemini.client import GeminiClient
        from providers.gemini.config import GeminiConfig
        from providers.gemini.script_provider import GeminiScriptProvider

        gemini_config = GeminiConfig.from_env(env if env is not None else os.environ)
        client = GeminiClient(gemini_config)
        return GeminiScriptProvider(client=client)

    raise ScriptProviderError(f"unknown script provider: {name!r}")