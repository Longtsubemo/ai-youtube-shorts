"""Stage-4 Gemini script provider.

Implements the `ScriptProvider` protocol from `agents/script/provider.py`
using the provider foundation in `providers/gemini/client.py`.

Flow (full pipeline — `fact_check_result` supplied):

    ResearchPackage + FactCheckResult
        -> whitelist-join `result.approved_claims` against the package
           (rejected / REVIEW / unknown claims can never enter)
        -> prompt renders ONLY those approved claims (ids + texts + evidence)
        -> Gemini structured output: title + lines with purpose/claim_refs
        -> deterministic mapping into the existing `Script` contract
           (durations derived from word counts, used/unused tracking, provenance)
        -> `Script.result_id` = `FactCheckResult.result_id`
        -> deterministic fact-check validation (`validate_script`) is the final
           gate: result_id mismatch, non-approved refs, wording outside token
           coverage, dropped hedges, or strengthened certainty all fail closed
           as a `blocked` generation, never a persisted artifact.

Omitting `fact_check_result` selects the isolated preview path: the whole
package is the working set and `result_id` stays empty. That path exists for
standalone Stage-4 testing only; the CLI gates it behind an explicit opt-in so
a normal pipeline run can never silently produce an unbound script.

Evidence/rules boundary (identical to the other Gemini providers):

- The model is NOT a fact authority: the ONLY factual input is the approved
  claim list in the prompt. A line may only reference approved claim ids;
  anything else fails validation.
- Fact Checking, not Gemini, decides what is approved: the provider receives
  the approved set and never re-evaluates or re-researches a verdict.
- Preserved hedges: a hedged claim must not lose its hedge, and no claim may
  be upgraded to certainty (the deterministic validator enforces this).
- Malformed model output (bad JSON, wrong purpose Literal, missing lines,
  empty text) fails as a `GeminiProviderError`, never a raw SDK exception.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

from pydantic import BaseModel, Field, TypeAdapter

from agents.qa.fact_checker.models import FactCheckResult
from agents.research.models import FactClaim, ResearchPackage
from agents.script.models import Script, ScriptError, ScriptLine, line_id_for, script_id_for
from agents.script.script_generator import (
    STATUS_BLOCKED,
    STATUS_GENERATED,
    STATUS_INSUFFICIENT,
    ScriptConfig,
    ScriptGenerationResult,
)
from agents.script.stage import approved_claims, bind_fact_check_result, fact_check_gate
from agents.script.validator import validate_script, validate_script_from_package
from agents.seo.models import Provenance, utc_now
from providers.gemini.client import GeminiClient
from providers.gemini.errors import GeminiMalformedResponseError, GeminiSchemaValidationError

PROMPT_VERSION = "script/script-v1"
DEFAULT_PROMPT_PATH = Path("prompts") / "script" / "script-v1.txt"
GENERATED_BY = "gemini_script_provider_v1"

_SYSTEM_PROMPT = (
    "You are a script writer for a YouTube Shorts explainer channel. You write "
    "ONLY from the claims provided in the prompt; your generated text is NOT "
    "evidence and must never introduce new facts, statistics, quotes, or "
    "sources. Never strengthen a claim and never drop its hedge. Return JSON "
    "only."
)


class _ScriptLineSchema(BaseModel):
    text: str = Field(min_length=1)
    purpose: Literal["hook", "context", "explanation", "fact", "transition", "conclusion", "call_to_action"]
    claim_refs: list[str] = Field(default_factory=list)


class ScriptResultSchema(BaseModel):
    """The response schema mirroring the existing Script line contract."""

    title: str = Field(min_length=1)
    lines: list[_ScriptLineSchema] = Field(min_length=1)


# Convenience re-export for tests.
ScriptResultAdapter = TypeAdapter(ScriptResultSchema)


class GeminiScriptProvider:
    """Script provider backed by Gemini structured output.

    `produced_by`/`provenance` record the provider, model, and prompt version
    so a persisted script is fully traceable to the exact Gemini run that
    produced it.
    """

    provider_name = "gemini"

    def __init__(
        self,
        client: GeminiClient,
        *,
        prompt_path: str | Path = DEFAULT_PROMPT_PATH,
        malformed_retries: int = 1,
    ) -> None:
        self._client = client
        self._prompt_path = Path(prompt_path)
        self._malformed_retries = max(0, int(malformed_retries))

    # -- metadata --------------------------------------------------------- #

    @property
    def model(self) -> str:
        return self._client.model

    @property
    def prompt_version(self) -> str:
        return PROMPT_VERSION

    @property
    def produced_by(self) -> dict:
        """Header provenance for the Script envelope."""
        return {
            "prompt_version": PROMPT_VERSION,
            "provider": self.provider_name,
            "model": self.model,
        }

    # -- protocol --------------------------------------------------------- #

    def generate(
        self,
        package: ResearchPackage,
        config: ScriptConfig | None = None,
        run_id: str = "",
        created_at: str = "",
        fact_check_result: FactCheckResult | None = None,
    ) -> ScriptGenerationResult:
        config = (config or ScriptConfig()).validate()
        package = package.validate()

        if fact_check_result is not None:
            # Stage 3 is the source of truth for what may be scripted.
            fact_check_result = bind_fact_check_result(package, fact_check_result)
            gate = fact_check_gate(fact_check_result, config)
            if gate is not None:
                return gate
            claims = list(approved_claims(fact_check_result, package))
            if len(claims) < config.min_approved_claims:
                return ScriptGenerationResult(
                    status=STATUS_INSUFFICIENT,
                    message=(
                        f"insufficient approved content: {len(claims)} approved "
                        f"claim(s) < required minimum {config.min_approved_claims}"
                    ),
                    blocked_reasons=("too_few_approved_claims",),
                )
            result_id = fact_check_result.result_id
        else:
            # Isolated preview path: the whole package is the working set.
            claims = list(package.claims)
            result_id = ""

        prompt = self._render_prompt(package, config, claims)
        schema = self._request_schema(prompt)
        try:
            script = self._to_script(schema, package, claims, result_id, config, run_id, created_at)
        except ScriptError as exc:
            raise GeminiSchemaValidationError(
                f"Gemini script failed Stage-4 validation: {exc}"
            ) from exc

        if fact_check_result is not None:
            validation = validate_script(script, fact_check_result, claims, config)
        else:
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

    # -- internals --------------------------------------------------------- #

    def _request_schema(self, prompt: str) -> ScriptResultSchema:
        """Ask the model for the script JSON.

        A non-JSON/truncated response (GeminiMalformedResponseError) is
        retried a bounded number of times: it is usually a transient short
        output, not a deterministic failure. Schema errors are NOT retried —
        they reflect a prompt/schema mismatch and would only fail again.
        """
        last_error: GeminiMalformedResponseError | None = None
        for _ in range(self._malformed_retries + 1):
            try:
                return self._client.generate_structured(
                    prompt,
                    ScriptResultSchema,
                    system_prompt=_SYSTEM_PROMPT,
                )
            except GeminiMalformedResponseError as exc:
                last_error = exc
        assert last_error is not None  # pragma: no cover - loop always ran once
        raise last_error

    def _system_prompt(self) -> str:
        return _SYSTEM_PROMPT

    def _render_prompt(
        self, package: ResearchPackage, config: ScriptConfig, claims: Sequence[FactClaim]
    ) -> str:
        path = Path(self._prompt_path)
        if not path.is_file():
            raise FileNotFoundError(f"prompt template not found: {path}")
        max_words = round(config.max_duration_seconds / 60 * config.words_per_minute)
        return (
            path.read_text(encoding="utf-8")
            .replace("{topic_id}", package.topic_id)
            .replace("{topic_title}", package.topic_title)
            .replace("{claims}", self._render_claims(package, claims))
            .replace("{max_words}", str(max_words))
        )

    @staticmethod
    def _render_claims(package: ResearchPackage, claims: Sequence[FactClaim]) -> str:
        """Render the permitted claims as a deterministic, bounded text block.

        `claims` is the provider's permitted working set: the approved claims on
        the full-pipeline path, or the whole package on the preview path. Only
        these ids ever reach the model, so it cannot script a rejected or REVIEW
        claim even if it wanted to.
        """
        tier_by_id = {source.source_id: source.credibility_tier for source in package.sources}
        lines: list[str] = []
        for claim in claims:
            tiers = sorted({tier_by_id[sid] for sid in claim.source_ids if sid in tier_by_id})
            meta = f"{claim.claim_type}/{claim.verification_status}"
            if tiers:
                meta += f"/{'/'.join(tiers)}"
            lines.append(f"[{claim.claim_id}] ({meta}) {claim.claim_text}")
        return "\n".join(lines)

    def _to_script(
        self,
        schema: ScriptResultSchema,
        package: ResearchPackage,
        claims: Sequence[FactClaim],
        result_id: str,
        config: ScriptConfig,
        run_id: str,
        created_at: str,
    ) -> Script:
        lines = []
        for index, item in enumerate(schema.lines):
            text = item.text.strip()
            lines.append(
                ScriptLine(
                    line_id=line_id_for(index),
                    text=text,
                    claim_refs=tuple(item.claim_refs),
                    purpose=item.purpose,
                    estimated_duration_seconds=round(
                        len(text.split()) / config.words_per_minute * 60, 2
                    ),
                ).validate()
            )
        total_seconds = round(sum(line.estimated_duration_seconds for line in lines), 1)
        used = {claim_id for line in lines for claim_id in line.claim_refs}

        run_id = run_id or package.run_id
        created_at = created_at or utc_now()
        script = Script(
            script_id=script_id_for(run_id, package.topic_id),
            topic_id=package.topic_id,
            topic_title=package.topic_title,
            title=schema.title.strip()[:70],
            hook=lines[0].text,
            lines=tuple(lines),
            estimated_duration_seconds=total_seconds,
            used_claim_ids=tuple(sorted(used)),
            unused_approved_claim_ids=tuple(
                sorted({claim.claim_id for claim in claims} - used)
            ),
            result_id=result_id,
            package_id=package.package_id,
            provenance=Provenance(
                generated_by=GENERATED_BY,
                source=self.__class__.__name__,
                created_at=created_at,
                provider=self.provider_name,
                model=self.model,
                prompt_version=PROMPT_VERSION,
            ),
            created_at=created_at,
            run_id=run_id,
            produced_by=dict(self.produced_by),
        )
        return script.validate()