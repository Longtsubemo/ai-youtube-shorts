"""Stage 3: Script Generation.

Reads the Stage 1 ``topic_selection.json`` artifact and the Stage 2
``research.json`` artifact, and writes ``script.json`` - a finished spoken
YouTube Shorts script.

Gemini's own knowledge carries the material through from Stage 2; this stage
only shapes it into narration. There is no retrieval, no source harvesting, no
corroboration, and no QA score or PASS/FAIL verdict here.

The provider boundary is `ScriptProvider`, so the Gemini implementation stays
replaceable and the deterministic mock keeps tests key-free.
"""

from __future__ import annotations

import hashlib

import json
from dataclasses import replace
from pathlib import Path

from agents.research.ai_research_models import AIResearch
from agents.script.script_models import (
    TARGET_DURATION_SECONDS,
    Script,
    ScriptError,
    ScriptSection,
)
from agents.seo.result import TopicSelection

REQUIRED_TOPIC_FIELDS = ("topic_id", "title", "pillar", "angle", "hook_direction")


class ScriptInputError(ValueError):
    """Raised when a Stage 1 / Stage 2 artifact cannot supply a usable topic.

    An input-shape check, not a content-quality gate.
    """


def load_topic_selection(path: str | Path) -> TopicSelection:
    """Load the Stage 1 ``topic_selection.json`` artifact."""
    return TopicSelection.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def load_research(path: str | Path) -> AIResearch:
    """Load the Stage 2 ``research.json`` artifact."""
    return AIResearch.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def write_script(script: Script, out_path: str | Path) -> Path:
    """Write the Stage 3 ``script.json`` artifact."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(script.to_dict(), indent=2), encoding="utf-8")
    return out_path


# Rotated per topic so the channel's videos do not share one structure
# (YouTube's inauthentic-content policy targets templated, repetitive uploads).
NARRATIVE_FORMATS: tuple[str, ...] = tuple(
    f"{name}: {idea}\nOpening move (the first sentence): {opening}\n"
    f"Section roles, in this order: {', '.join(beats)}"
    for name, idea, opening, beats in (
        ("Myth-buster", "take a popular belief about the topic apart and end on what is actually true.",
         "state the myth as if it were true.", ("myth", "crack", "truth", "proof", "takeaway")),
        ("Case file", "tell it as a short detective case that ends by naming the culprit.",
         "present a puzzling clue or a crime scene.", ("clue", "investigation", "twist", "culprit")),
        ("Story", "follow one ordinary, fictional person; the concept emerges from what happens to them.",
         "introduce the person by name in a specific moment.", ("character", "choice", "consequence", "lesson")),
        ("Comparison", "put two people, choices or paths side by side and show why they end up far apart.",
         "name both sides in the first sentence.", ("side_a", "side_b", "divergence", "outcome", "takeaway")),
        ("Counterintuitive reveal", "open with a claim that sounds wrong, then show the mechanism that makes it right.",
         "the surprising claim itself, stated flatly.", ("claim", "doubt", "mechanism", "reveal")),
        ("What-if chain", "ask one 'what happens if' question and follow its consequences to the end.",
         "the 'what happens if' question itself.", ("question", "first_effect", "second_effect", "end_state")),
        ("Breakdown", "take one everyday thing apart piece by piece and show the hidden part that matters.",
         "name the everyday thing and promise to open it up.", ("object", "layer_one", "layer_two", "hidden_part", "takeaway")),
    )
)


def narrative_format_for(topic_id: str) -> str:
    """Deterministic per topic, so reruns of a topic keep their format."""
    # shortcut: hash-based, so two consecutive topics can land on the same format; use the
    # topic history to rotate if that becomes visible on the channel.
    digest = hashlib.sha256(topic_id.encode("utf-8")).digest()
    return NARRATIVE_FORMATS[int.from_bytes(digest[:4], "big") % len(NARRATIVE_FORMATS)]


def build_script_context(selection: TopicSelection, research: AIResearch) -> dict:
    """Combine the Stage 1 and Stage 2 artifacts into one prompt input.

    The topic fields come from the real Stage 1 selection artifact and the
    substance comes from the real Stage 2 research artifact. No third input
    format is invented.
    """
    selected = selection.selected_topic
    if not isinstance(selected, dict) or not selected:
        raise ScriptInputError("topic_selection.json has no selected_topic")

    missing = [f for f in REQUIRED_TOPIC_FIELDS if not str(selected.get(f) or "").strip()]
    if missing:
        raise ScriptInputError("selected_topic missing required field(s): " + ", ".join(missing))

    if not str(selection.selected_topic_id or "").strip():
        raise ScriptInputError("topic_selection.json has no selected_topic_id")
    if str(selected["topic_id"]).strip() != str(selection.selected_topic_id).strip():
        raise ScriptInputError(
            "selected_topic.topic_id does not match selected_topic_id: "
            f"{selected['topic_id']!r} != {selection.selected_topic_id!r}"
        )

    if not str(research.topic_id or "").strip():
        raise ScriptInputError("research.json has no topic_id")
    if not str(research.explanation or "").strip():
        raise ScriptInputError("research.json has no explanation to build a script from")
    if research.topic_id.strip() != selection.selected_topic_id.strip():
        raise ScriptInputError(
            "research.topic_id does not match the selected topic: "
            f"{research.topic_id!r} != {selection.selected_topic_id!r}"
        )

    return {
        "topic_id": str(selected["topic_id"]).strip(),
        "title": str(selected["title"]).strip(),
        "pillar": str(selected["pillar"]).strip(),
        "angle": str(selected["angle"]).strip(),
        "hook_direction": str(selected["hook_direction"]).strip(),
        "topic_type": str(selected.get("topic_type") or "").strip(),
        "description": str(selected.get("description") or "").strip(),
        "research": research,
        "run_id": selection.run_id,
        "narrative_format": narrative_format_for(str(selected["topic_id"]).strip()),
    }


def run_script_generation(
    selection: TopicSelection,
    research: AIResearch,
    provider,
    *,
    run_id: str = "",
    created_at: str | None = None,
) -> Script:
    """Generate the spoken script from the topic + research via `provider`.

    The stage owns artifact concerns (run id, timestamp, structural
    validation); the provider owns how the script is written.
    """
    context = build_script_context(selection, research)
    script = provider.generate(context)
    updates: dict = {"run_id": run_id or context["run_id"]}
    if created_at is not None:
        updates["created_at"] = created_at
    # Script is frozen; rebuild rather than mutate.
    return replace(script, **updates).validate()


def generate_script(
    selection: TopicSelection,
    research: AIResearch,
    provider=None,
    *,
    run_id: str = "",
    created_at: str | None = None,
) -> Script:
    """Produce the script artifact, defaulting to the deterministic mock."""
    if provider is None:
        from agents.script.provider import MockScriptProvider

        provider = MockScriptProvider()
    return run_script_generation(
        selection, research, provider, run_id=run_id, created_at=created_at
    )


__all__ = [
    "REQUIRED_TOPIC_FIELDS",
    "TARGET_DURATION_SECONDS",
    "Script",
    "ScriptError",
    "ScriptInputError",
    "ScriptSection",
    "build_script_context",
    "generate_script",
    "load_research",
    "load_topic_selection",
    "run_script_generation",
    "write_script",
]