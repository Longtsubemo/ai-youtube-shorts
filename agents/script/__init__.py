"""Stage 3: Script Generation.

Reads the Stage 1 ``topic_selection.json`` artifact and the Stage 2
``research.json`` artifact, and writes ``script.json`` - a finished spoken
YouTube Shorts script.

CLI: ``python -m agents.script.script_cli --run-id <run_id>``

This stage is a content stage only: no retrieval, no sources, no QA score, and
no PASS/FAIL verdict.
"""

from agents.script.provider import (
    MockScriptProvider,
    ScriptProvider,
    ScriptProviderError,
    build_script_provider,
)
from agents.script.script_models import (
    DURATION_MIN_SECONDS,
    DURATION_TARGET_RANGE,
    EDUCATIONAL_NOTICE,
    MASCOT_NOTE,
    MASCOT_SPEAKS,
    SECTION_ROLES,
    TARGET_DURATION_SECONDS,
    Script,
    ScriptError,
    ScriptSection,
)
from agents.script.stage import (
    REQUIRED_TOPIC_FIELDS,
    ScriptInputError,
    build_script_context,
    generate_script,
    load_research,
    load_topic_selection,
    run_script_generation,
    write_script,
)

__all__ = [
    "DURATION_MIN_SECONDS",
    "DURATION_TARGET_RANGE",
    "EDUCATIONAL_NOTICE",
    "MASCOT_NOTE",
    "MASCOT_SPEAKS",
    "MockScriptProvider",
    "REQUIRED_TOPIC_FIELDS",
    "SECTION_ROLES",
    "Script",
    "ScriptError",
    "ScriptInputError",
    "ScriptProvider",
    "ScriptProviderError",
    "ScriptSection",
    "TARGET_DURATION_SECONDS",
    "build_script_context",
    "build_script_provider",
    "generate_script",
    "load_research",
    "load_topic_selection",
    "run_script_generation",
    "write_script",
]