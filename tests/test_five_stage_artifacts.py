def test_artifact_paths():
    from agents.research.ai_research_models import AIResearch
    from agents.script.script_models import Script
    from agents.scene_planner.scene_models import ScenePlan
    r = AIResearch(topic_id="t1")
    assert r.to_dict()["topic_id"] == "t1"
    s = Script(topic_id="t1", title="t")
    assert s.to_dict()["title"] == "t"
    p = ScenePlan(topic_id="t1")
    assert p.to_dict()["topic_id"] == "t1"

