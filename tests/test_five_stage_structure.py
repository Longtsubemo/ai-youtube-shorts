def test_five_stage_enum():
    from pipeline import state
    stages = [s.value for s in state.Stage]
    assert len(stages) == 5
    assert "topic_selection" in stages
    assert "ai_research" in stages
    assert "script_generation" in stages
    assert "scene_planning" in stages
    assert "video_generation" in stages

