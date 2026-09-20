"""Stage 5 - Script QA: quality control over the Stage-4 Script artifact.

Docs: docs/DATA_CONTRACTS.md §8.1, docs/PROJECT_PROGRESS.md Stage 5.
CLI: python -m agents.qa.script.

NOTE: this package imports only models + validator here. The checks, scorer,
and stage runner live in submodules imported by path so that
`python -m agents.qa.script` runs without a double-import RuntimeWarning
(same convention as agents.script).
"""

from __future__ import annotations

from agents.qa.script.models import (
    CATEGORY_DIMENSION,
    QA_CATEGORIES,
    QA_DIMENSIONS,
    QA_SEVERITIES,
    QA_VERDICTS,
    SCHEMA,
    SCHEMA_VERSION,
    STAGE,
    ClaimCoverage,
    QACheckRecord,
    QAIssue,
    ScriptQAConfig,
    ScriptQAError,
    ScriptQAResult,
    qa_result_id_for,
)
from agents.qa.script.validator import (
    VERDICT_PASS,
    VERDICT_REJECT,
    VERDICT_REVISE,
    aggregate_verdict,
    evaluate_script,
)

__all__ = [
    "CATEGORY_DIMENSION",
    "QA_CATEGORIES",
    "QA_DIMENSIONS",
    "QA_SEVERITIES",
    "QA_VERDICTS",
    "SCHEMA",
    "SCHEMA_VERSION",
    "STAGE",
    "ClaimCoverage",
    "QACheckRecord",
    "QAIssue",
    "ScriptQAConfig",
    "ScriptQAError",
    "ScriptQAResult",
    "VERDICT_PASS",
    "VERDICT_REJECT",
    "VERDICT_REVISE",
    "aggregate_verdict",
    "evaluate_script",
    "qa_result_id_for",
]