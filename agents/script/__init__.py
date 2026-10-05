"""Stage 3: Script Generation.

Reads ``research.json`` and writes ``script.json``.

CLI: ``python -m agents.script.script_cli --run-id <run_id>``
"""

from agents.script.script_models import Script, ScriptSection
from agents.script.stage_simple import write_script
from agents.script.validator import ScriptValidation, validate_script

__all__ = [
    "Script",
    "ScriptSection",
    "ScriptValidation",
    "validate_script",
    "write_script",
]