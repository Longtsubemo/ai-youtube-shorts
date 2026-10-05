"""Simplified script generation."""

from __future__ import annotations

import json
from pathlib import Path

from agents.script.script_models import Script


def write_script(script: Script, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(script.to_dict(), indent=2), encoding="utf-8")
    return out_path

