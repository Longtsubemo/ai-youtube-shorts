"""Stage-4 script generator: turns ONLY approved claims into a Short script.

The generator's only factual input is `FactCheckResult.approved_claims`
(joined against the research package). It applies defense in depth: even if a
caller passes the entire package, the generator re-filters to approved claim
ids, so rejected / REVIEW / unapproved claims can never be materialized as
fact. A future LLM-backed generator plugs in behind the same protocol and the
same `ScriptGenerationResult` contract and remains bounded to the same input.

Runtime shape (target: a 30-60s YouTube Short at ~150 words/min):

  hook        first approved claim, curiosity-framed but fully supported
  context     non-factual transition from the topic title
  fact        verbatim approved claims (one line each)
  transition  non-factual "one more detail" bridge (only with 3+ claims)
  conclusion  restatement of the first claim's core, with refs
  cta         non-factual call to action

Every factual line keeps the exact wording (and hedges) of its claim; nothing
is invented, strengthened, or reworded into higher certainty. Duration is
estimated deterministically from word counts and a configurable speaking rate.
"""

from __future__ import annotations

import argparse
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - python-dotenv is a declared dep
    load_dotenv = None

from agents.qa.fact_checker.models import FactCheckResult
from agents.script.models import (
    PRODUCED_BY,
    Script,
    ScriptLine,
    line_id_for,
    script_id_for,
)
from agents.script.validator import ScriptValidation, validate_script
from agents.seo.models import Provenance, utc_now
from agents.research.models import FactClaim, ResearchPackage
from pipeline.state import PipelineState, Stage

# --------------------------------------------------------------------------- #
# Statuses
# --------------------------------------------------------------------------- #

STATUS_GENERATED = "generated"
STATUS_BLOCKED = "blocked"
STATUS_INSUFFICIENT = "insufficient_content"


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ScriptConfig:
    """Deterministic script-configuration knobs (mirrors FactCheckConfig)."""

    words_per_minute: int = 150
    min_duration_seconds: float = 30.0
    max_duration_seconds: float = 60.0
    min_approved_claims: int = 2
    allow_review_content: bool = False
    semantic_checks: bool = True
    semantic_coverage_threshold: float = 0.5

    def validate(self) -> "ScriptConfig":
        if self.words_per_minute <= 0:
            raise ValueError("words_per_minute must be > 0")
        if not 0 < self.min_duration_seconds <= self.max_duration_seconds:
            raise ValueError("duration bounds must satisfy 0 < min <= max")
        if self.semantic_coverage_threshold <= 0:
            raise ValueError("semantic_coverage_threshold must be > 0")
        return self


@dataclass(frozen=True)
class ScriptGenerationResult:
    """Outcome of a generation attempt (generated / blocked / insufficient)."""

    status: str
    script: Script | None = None
    validation: ScriptValidation | None = None
    message: str = ""
    blocked_reasons: tuple[str, ...] = ()

    @property
    def generated(self) -> bool:
        return self.status == STATUS_GENERATED

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "message": self.message,
            "blocked_reasons": list(self.blocked_reasons),
        }


# --------------------------------------------------------------------------- #
# Generator protocol
# --------------------------------------------------------------------------- #


class ScriptGenerator(Protocol):
    """Anything that can turn approved claims into a validated script.

    `claims` may be the full research package; the generator MUST filter to
    `fact_check_result.approved_claims` first and operate on those records
    only. This is the boundary a future LLM-backed generator inherits.
    """

    def generate(
        self,
        fact_check_result: FactCheckResult,
        claims: Sequence[FactClaim],
        config: ScriptConfig | None = None,
        run_id: str = "",
        created_at: str = "",
    ) -> ScriptGenerationResult: ...


# --------------------------------------------------------------------------- #
# Deterministic mock generator
# --------------------------------------------------------------------------- #


def _first_sentence(text: str) -> str:
    sentence = re.split(r"(?<=[.:;!?])\s+", re.sub(r"[:;]\s*", ": ", text.strip()))[0]
    return sentence.rstrip(":;")


def _after_colon(text: str) -> str:
    """Return the part after the first colon (a claim's concrete payload)."""
    if ":" in text:
        return text.split(":", 1)[1].strip()
    return text


def _duration_seconds(text: str, words_per_minute: int) -> float:
    return round(len(text.split()) / words_per_minute * 60, 2)


class MockScriptGenerator:
    """Deterministic, network-free generator that rewrites approved claims.

    No external knowledge is used anywhere: every factual line is a (near)
    verbatim claim, every framing line is inert boilerplate, and claim_refs
    keep the audit chain script line -> claim -> source -> evidence intact.
    """

    def __init__(self, config: ScriptConfig | None = None) -> None:
        self.config = (config or ScriptConfig()).validate()

    def generate(
        self,
        fact_check_result: FactCheckResult,
        claims: Sequence[FactClaim],
        config: ScriptConfig | None = None,
        run_id: str = "",
        created_at: str = "",
    ) -> ScriptGenerationResult:
        config = (config or self.config).validate()
        result = fact_check_result.validate()
        claims_by_id = {claim.claim_id: claim for claim in claims}

        # Defensive boundary: only approved claim ids are ever materialized.
        missing = [cid for cid in result.approved_claims if cid not in claims_by_id]
        if missing:
            return ScriptGenerationResult(
                status=STATUS_BLOCKED,
                message="cannot generate: approved claim(s) missing from claim records",
                blocked_reasons=tuple(f"missing claim record: {cid}" for cid in missing),
            )
        approved = [claims_by_id[cid] for cid in result.approved_claims]

        # Gate 1: the package as a whole did not survive fact checking.
        if result.package_verdict == "REJECT":
            return ScriptGenerationResult(
                status=STATUS_BLOCKED,
                message="package verdict REJECT: no validated content to script",
                blocked_reasons=("package_verdict=REJECT",),
            )
        if result.package_verdict == "REVIEW" and not config.allow_review_content:
            return ScriptGenerationResult(
                status=STATUS_BLOCKED,
                message="package verdict REVIEW: review claims blocked by default",
                blocked_reasons=("package_verdict=REVIEW",),
            )

        # Gate 2: not enough approved material to fill a Short.
        if len(approved) < config.min_approved_claims:
            return ScriptGenerationResult(
                status=STATUS_INSUFFICIENT,
                message=(
                    f"insufficient approved content: {len(approved)} approved "
                    f"claim(s) < required minimum {config.min_approved_claims}"
                ),
                blocked_reasons=("too_few_approved_claims",),
            )

        # Gate 3: approved material is too thin to reach the duration target.
        placeholder_lines = self._plan(approved, config, result.topic_title)
        estimated_seconds = sum(
            _duration_seconds(line.text, config.words_per_minute)
            for line in placeholder_lines
        )
        if estimated_seconds < config.min_duration_seconds:
            return ScriptGenerationResult(
                status=STATUS_INSUFFICIENT,
                message=(
                    f"insufficient approved content: script would run "
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
                estimated_duration_seconds=_duration_seconds(
                    line.text, config.words_per_minute
                ),
            )
            for line in placeholder_lines
        ]
        total_seconds = round(sum(line.estimated_duration_seconds for line in lines), 1)
        used = {claim_id for line in lines for claim_id in line.claim_refs}

        run_id = run_id or result.run_id
        created_at = created_at or utc_now()
        script = Script(
            script_id=script_id_for(run_id, result.topic_id),
            topic_id=result.topic_id,
            topic_title=result.topic_title,
            title=self._title(result, approved),
            hook=lines[0].text,
            lines=tuple(lines),
            estimated_duration_seconds=total_seconds,
            used_claim_ids=tuple(sorted(used)),
            unused_approved_claim_ids=tuple(
                sorted(set(result.approved_claims) - used)
            ),
            result_id=result.result_id,
            package_id=result.package_id,
            provenance=Provenance(
                generated_by="script_agent_v1",
                source="MockScriptGenerator",
                created_at=created_at,
                provider=None,
                model="MockScriptGenerator",
            ),
            created_at=created_at,
            run_id=run_id,
            produced_by=dict(PRODUCED_BY),
        )
        script.validate()

        validation = validate_script(
            script,
            result,
            approved,
            config if config.semantic_checks else (
                ScriptConfig(
                    words_per_minute=config.words_per_minute,
                    min_duration_seconds=config.min_duration_seconds,
                    max_duration_seconds=config.max_duration_seconds,
                    min_approved_claims=config.min_approved_claims,
                    allow_review_content=config.allow_review_content,
                    semantic_checks=False,
                    semantic_coverage_threshold=config.semantic_coverage_threshold,
                )
            ),
        )
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

    # -- planning helpers (all deterministic) ------------------------------- #

    def _plan(
        self,
        approved: list[FactClaim],
        config: ScriptConfig,
        topic_title: str,
    ) -> list[ScriptLine]:
        """Build ordered ScriptLine objects from the approved claims.

        `estimated_duration_seconds` is intentionally 0.0 here; the caller
        fills real durations after the too-short gate. Design uses FactClaim
        directly so the plan can never reach beyond approved records.
        """
        primary = approved[0]
        lines: list[ScriptLine] = []

        hook_text = "You probably didn't know this, but " + _first_sentence(
            primary.claim_text
        )
        lines.append(ScriptLine(line_id_for(0), hook_text, (primary.claim_id,), "hook"))

        context_text = (
            f"{topic_title}? Let's take a closer look at the answer."
            if topic_title
            else "Let's take a closer look at the answer."
        )
        lines.append(ScriptLine(line_id_for(1), context_text, (), "context"))

        for no, claim in enumerate(approved):
            lines.append(
                ScriptLine(
                    line_id_for(2 + no),
                    claim.claim_text,
                    (claim.claim_id,),
                    "explanation" if claim.claim_type == "explanation" else "fact",
                )
            )

        cursor = 2 + len(approved)
        if len(approved) >= 3:
            lines.append(
                ScriptLine(
                    line_id_for(cursor),
                    "Here's one more detail worth knowing.",
                    (),
                    "transition",
                )
            )
            cursor += 1

        lines.append(
            ScriptLine(
                line_id_for(cursor),
                "So here's the bottom line: " + _after_colon(primary.claim_text),
                (primary.claim_id,),
                "conclusion",
            )
        )
        cursor += 1
        lines.append(
            ScriptLine(
                line_id_for(cursor),
                "If you enjoyed this, follow for more science explainers like this one.",
                (),
                "call_to_action",
            )
        )
        return lines

    def _title(self, result: FactCheckResult, approved: list[FactClaim]) -> str:
        title = (result.topic_title or approved[0].claim_text).strip()
        if len(title) > 70:
            title = title.split("?")[0][:70].strip() or approved[0].claim_text[:70]
        return title[:70]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _build_mock_package():
    """Rebuild the deterministic mock package for the mock/octopus demo."""
    from agents.research.stage import load_topic_selection, run_research
    from agents.research.provider import MockResearchProvider

    selection = load_topic_selection()
    return run_research(selection, MockResearchProvider(), run_id="mock")


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m agents.script.script_generator",
        description="Generate a validated Short script (mock default, or Gemini over a ResearchPackage).",
    )
    parser.add_argument(
        "--result",
        type=Path,
        default=None,
        help="FactCheckResult artifact JSON; REQUIRED with --research unless --preview "
             "(default for the no---research path: data/mock/fact_check_result.json)",
    )
    parser.add_argument(
        "--provider",
        choices=("mock", "gemini"),
        default="mock",
        help="script provider (default: mock; gemini requires GEMINI_API_KEY and --research)",
    )
    parser.add_argument(
        "--research",
        type=Path,
        default=None,
        help="ResearchPackage artifact JSON to generate from (required for --provider gemini)",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="run id for the output artifact (default: from --research, else generated)",
    )
    parser.add_argument(
        "--allow-review-content",
        action="store_true",
        help="allow generation when the package verdict is REVIEW",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="ISOLATED PREVIEW ONLY: generate from --research with no FactCheckResult, "
             "so the whole package is the working set and script.result_id stays empty. "
             "Never use this for a pipeline run.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output root under which <run_id>/scripts/script.json is written",
    )
    args = parser.parse_args()

    if args.research is not None:
        return _run_from_package(args)
    if args.provider == "gemini":
        print("[script_generator] --provider gemini requires --research <research_package.json>")
        return 2
    return _run_from_fact_check(args)


def _run_from_fact_check(args) -> int:
    from agents.script.stage import (
        DEFAULT_FACT_CHECK_RESULT_PATH,
        run_script_generation,
    )

    result_path = args.result or DEFAULT_FACT_CHECK_RESULT_PATH
    if not result_path.is_file():
        raise SystemExit(f"[script_generator] FactCheckResult not found: {result_path}")
    result = FactCheckResult.from_json(result_path.read_text(encoding="utf-8"))
    package = _build_mock_package()
    run_id = args.run_id or uuid.uuid4().hex[:12]
    config = ScriptConfig(allow_review_content=args.allow_review_content)

    generation = run_script_generation(result, package, config=config, run_id=run_id)
    _print_result(generation, run_id, args.provider)
    if not generation.generated:
        return 0 if generation.status == STATUS_INSUFFICIENT else 1

    out_root = args.out or Path("data")
    out_path = _write_artifact(generation.script, run_id, out_root)
    print(f"[script_generator] script written to {out_path}")
    print(f"[script_generator] run state written to {out_root / run_id / 'state.json'}")
    return 0


def _run_from_package(args) -> int:
    """Full-pipeline Stage 4: generate from a package BOUND to Stage 3.

    The FactCheckResult is mandatory here. It supplies the approved-claim
    whitelist and the `result_id` the Script must inherit; without it the
    provider would have no verdict to honor, so the only way to run this path
    unbound is the explicit `--preview` opt-in used for isolated Stage-4
    testing. A missing/invalid result fails before any artifact is written.
    """
    from agents.script.provider import ScriptProviderError, build_script_provider
    from agents.script.stage import DEFAULT_FACT_CHECK_RESULT_PATH

    if args.preview and args.result is not None:
        print("[script_generator] --preview and --result are mutually exclusive")
        return 2

    fact_check_result = None
    if not args.preview:
        result_path = args.result
        if result_path is None:
            print(
                "[script_generator] --research requires --result <fact_check_result.json> "
                "so the script is bound to Stage-3 approved claims. "
                "Use --preview only for isolated Stage-4 testing "
                f"(result would otherwise default to {DEFAULT_FACT_CHECK_RESULT_PATH})."
            )
            return 2
        if not result_path.is_file():
            print(f"[script_generator] provider error: FactCheckResult not found: {result_path}")
            return 2

    if load_dotenv is not None:
        load_dotenv()  # honour GEMINI_API_KEY/GEMINI_MODEL from .env (ignored by git)

    if not args.research.is_file():
        raise SystemExit(f"[script_generator] ResearchPackage not found: {args.research}")
    package = ResearchPackage.from_json(args.research.read_text(encoding="utf-8"))

    if fact_check_result is None and not args.preview:
        try:
            fact_check_result = FactCheckResult.from_json(result_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[script_generator] provider error: invalid FactCheckResult: {exc}")
            return 2

    run_id = args.run_id or package.run_id
    config = ScriptConfig(allow_review_content=args.allow_review_content)
    try:
        provider = build_script_provider(args.provider)
        generation = provider.generate(
            package,
            config=config,
            run_id=run_id,
            fact_check_result=fact_check_result,
        )
    except ScriptProviderError as exc:
        print(f"[script_generator] provider error: {exc}")
        return 2
    except Exception as exc:
        print(f"[script_generator] provider error: {type(exc).__name__}: {exc}")
        return 2

    if args.provider == "gemini" and generation.script is not None:
        generation.script.produced_by = dict(provider.produced_by)
    _print_result(generation, run_id, args.provider)
    if not generation.generated:
        return 0 if generation.status == STATUS_INSUFFICIENT else 1

    out_root = args.out or Path("data")
    out_path = _write_artifact(generation.script, run_id, out_root)
    print(f"[script_generator] script written to {out_path}")
    print(f"[script_generator] run state written to {out_root / run_id / 'state.json'}")
    return 0


def _print_result(generation: ScriptGenerationResult, run_id: str, provider: str) -> None:
    print(f"[script_generator] run_id={run_id}")
    print(f"[script_generator] provider={provider}")
    print(f"[script_generator] status={generation.status}")
    print(f"[script_generator] message: {generation.message}")
    if generation.blocked_reasons:
        print("[script_generator] blocked by:")
        for reason in generation.blocked_reasons:
            print(f"  - {reason}")
    if not generation.generated:
        return

    script = generation.script
    print(f"[script_generator] produced_by={script.produced_by.get('model')}")
    print(f"[script_generator] script_id={script.script_id}")
    print(f"[script_generator] title: {script.title}")
    print(f"[script_generator] estimated duration: {script.estimated_duration_seconds}s")
    print(f"[script_generator] claims used: {', '.join(script.used_claim_ids) or '-'}")
    print(
        "[script_generator] claims unused: "
        f"{', '.join(script.unused_approved_claim_ids) or '-'}"
    )
    print("[script_generator] lines:")
    for line in script.lines:
        refs = ",".join(line.claim_refs) or "-"
        print(
            f"  [{line.line_id}] ({line.purpose:>14}) {line.estimated_duration_seconds:>5.1f}s "
            f"refs={refs}  {line.text}"
        )


def _write_artifact(script: Script, run_id: str, out_root: Path) -> Path:
    out_path = out_root / run_id / "scripts" / "script.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(script.to_json(), encoding="utf-8")

    state_path = out_root / run_id / "state.json"
    try:
        state = PipelineState.load(state_path)
    except FileNotFoundError:
        state = PipelineState(run_id=run_id)
    state.advance_to(Stage.SCRIPT_GENERATION)
    state.add_artifact(Stage.SCRIPT_GENERATION, out_path)
    state.save(state_path)
    return out_path


if __name__ == "__main__":
    raise SystemExit(main())