def test_channel_config():
    from config import channel
    assert channel.CHANNEL.channel_id == "finance_mascot"


def test_research_output_contract():
    from agents.research.ai_research_models import AIResearch
    r = AIResearch(topic_id="t1")
    d = r.to_dict()
    assert d["topic_id"] == "t1"


def test_script_output_contract():
    from agents.script.script_models import Script
    s = Script(topic_id="t1", title="t")
    d = s.to_dict()
    assert d["title"] == "t"


def test_scene_plan_output_contract():
    from agents.scene_planner.scene_models import ScenePlan
    p = ScenePlan(topic_id="t1")
    d = p.to_dict()
    assert d["topic_id"] == "t1"


def test_mascot_asset_path():
    from config import channel
    assert "finance_mascot" in channel.CHANNEL.mascot_asset_path

