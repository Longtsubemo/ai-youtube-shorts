"""Stage 9 - Policy/Safety QA: pre-publication gate over the artifact chain.

Docs: docs/DATA_CONTRACTS.md §11, docs/PROJECT_PROGRESS.md Stage 9,
docs/POLICY_AND_SAFETY.md.
CLI: python -m agents.qa.policy.

NOTE: this package imports only models + validator here (same convention as
agents.qa.script and agents.qa.video): the checks, registry, classifier
boundary, and stage runner live in submodules imported by path so
`python -m agents.qa.policy` runs without a double-import RuntimeWarning.

Stage 9 is POLICY / SAFETY QA, a GATE -- it does NOT implement Stage 10
(human approval) or Stage 11 (YouTube upload/scheduling). It is deterministic,
network-free, GPU-free, API-free, and is NOT a YouTube compliance guarantee.
The artifact header `stage` label is "policy_qa"; pipeline STATE is advanced
to the existing generic Stage.QUALITY_CHECK (`STATE_STAGE` in models.py) --
no Stage enum member was added or renamed.
"""

from __future__ import annotations

from agents.qa.policy.models import (
    POLICY_QA_CONFIG_VERSION,
    POLICY_GATES,
    POLICY_SEVERITIES,
    POLICY_VERDICTS,
    SCHEMA,
    SCHEMA_VERSION,
    STAGE,
    STATE_STAGE,
    PolicyCheckRecord,
    PolicyInputs,
    PolicyIssue,
    PolicyQAConfig,
    PolicyQAError,
    PolicyQAResult,
    RiskSummary,
    policy_result_id_for,
)
from agents.qa.policy.validator import (
    VERDICT_PASS,
    VERDICT_REJECT,
    VERDICT_REVIEW,
    aggregate_verdict,
    evaluate_policy,
)

__all__ = [
    "POLICY_QA_CONFIG_VERSION",
    "POLICY_GATES",
    "POLICY_SEVERITIES",
    "POLICY_VERDICTS",
    "SCHEMA",
    "SCHEMA_VERSION",
    "STAGE",
    "STATE_STAGE",
    "VERDICT_PASS",
    "VERDICT_REJECT",
    "VERDICT_REVIEW",
    "PolicyCheckRecord",
    "PolicyInputs",
    "PolicyIssue",
    "PolicyQAConfig",
    "PolicyQAError",
    "PolicyQAResult",
    "RiskSummary",
    "aggregate_verdict",
    "evaluate_policy",
    "policy_result_id_for",
]