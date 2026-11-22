"""PipelineState.load_or_create: state.json accumulates across stages."""

from __future__ import annotations

from pipeline.state import PipelineState, Stage


def _record(path, stage, artifact):
    state = PipelineState.load_or_create(path, "run1")
    state.advance_to(stage)
    state.add_artifact(stage, artifact)
    state.save(path)


def test_stages_accumulate_and_reruns_do_not_duplicate(tmp_path):
    path = tmp_path / "state.json"
    _record(path, Stage.TOPIC_SELECTION, "topic_selection.json")
    created = PipelineState.load(path).created_at
    _record(path, Stage.AI_RESEARCH, "research.json")
    _record(path, Stage.AI_RESEARCH, "research.json")

    state = PipelineState.load(path)
    assert state.artifacts == {
        "topic_selection": ["topic_selection.json"],
        "ai_research": ["research.json"],
    }
    assert state.current_stage is Stage.AI_RESEARCH
    assert state.created_at == created


def test_unreadable_state_starts_fresh(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("{not json", encoding="utf-8")
    state = PipelineState.load_or_create(path, "run1")
    assert state.run_id == "run1" and state.artifacts == {}
