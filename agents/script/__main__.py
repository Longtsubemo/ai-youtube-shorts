"""Convenience entrypoint: `python -m agents.script` runs the Stage-4 CLI."""

from agents.script.script_generator import main

if __name__ == "__main__":
    raise SystemExit(main())