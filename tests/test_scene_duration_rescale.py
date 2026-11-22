"""Stage 4 rescales model scene durations to the script's timing budget."""

from __future__ import annotations

from dataclasses import replace

import pytest

from agents.scene_planner.provider import MockScenePlannerProvider
from agents.scene_planner.stage import rescale_durations, run_scene_planning
from tests.test_scene_planner_generation_prompts import build_script


@pytest.mark.parametrize(
    "durations, budget",
    [
        ([8, 8, 8, 11, 10, 10, 8, 8, 7, 12], 72),  # the real 90s-for-72s case
        ([5, 5, 5], 30),
        ([20, 1, 1], 9),
        ([3, 3, 3, 3], 3),  # budget below one second per scene
        ([7], 70),
    ],
)
def test_rescale_hits_budget_with_one_second_floor(durations, budget):
    out = rescale_durations(durations, budget)
    assert len(out) == len(durations)
    assert all(d >= 1 for d in out)
    assert sum(out) == max(budget, len(durations))


def test_rescale_keeps_order_and_leaves_matching_plans_alone():
    out = rescale_durations([4, 12, 8], 48)
    assert out[1] > out[2] > out[0]
    assert rescale_durations([4, 12, 8], 24) == [4, 12, 8]
    assert rescale_durations([4, 12, 8], 0) == [4, 12, 8]


class _PaddedProvider(MockScenePlannerProvider):
    """Mock plan with every scene stretched, like the observed Gemini drift."""

    def generate(self, context):
        plan = super().generate(context)
        return replace(plan, scenes=[replace(s, duration_seconds=s.duration_seconds * 2) for s in plan.scenes])


def test_run_scene_planning_fits_plan_to_script_estimate():
    script = build_script()
    plan = run_scene_planning(script, _PaddedProvider(), run_id="rescale-test")
    budget = script.estimated_seconds or script.duration_target_seconds
    assert plan.estimated_seconds == budget
    assert plan.total_scene_duration_seconds == budget
