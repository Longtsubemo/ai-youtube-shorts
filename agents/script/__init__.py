"""Stage 4 - Script Generation: approved claims -> validated Short script.

Docs: docs/DATA_CONTRACTS.md §8, docs/PROJECT_PROGRESS.md Stage 4.
CLI: python -m agents.script.script_generator (or python -m agents.script).

NOTE: this package imports only models/validator here. The generator, config,
statuses, and stage runner live in `agents.script.script_generator` / 
`agents.script.stage` and are imported by submodule path so that
`python -m agents.script.script_generator` runs without a double-import
RuntimeWarning.
"""

from __future__ import annotations

from agents.script.models import (
    FACTUAL_PURPOSES,
    PURPOSE_VALUES,
    SCHEMA,
    SCHEMA_VERSION,
    STAGE,
    Script,
    ScriptError,
    ScriptLine,
    line_id_for,
    script_id_for,
)
from agents.script.validator import ScriptValidation, validate_script

__all__ = [
    "FACTUAL_PURPOSES",
    "PURPOSE_VALUES",
    "SCHEMA",
    "SCHEMA_VERSION",
    "STAGE",
    "Script",
    "ScriptError",
    "ScriptLine",
    "ScriptValidation",
    "line_id_for",
    "script_id_for",
    "validate_script",
]