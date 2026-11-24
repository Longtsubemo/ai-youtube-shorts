"""Stage 4 rescales model scene durations to the script's timing budget."""

from __future__ import annotations

from dataclasses import replace

import pytest

from agents.scene_planner.provider import MockScenePlannerProvider
from agents.scene_planner.stage import (
    NARRATION_WORDS_PER_SECOND,
    narration_seconds,
    rescale_durations,
    run_scene_planning,
)
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


def test_run_scene_planning_times_scenes_from_narration_pace():
    script = build_script()
    plan = run_scene_planning(script, _PaddedProvider(), run_id="rescale-test")
    budget = narration_seconds(script.narration)
    assert budget == max(1, round(len(script.narration.split()) / NARRATION_WORDS_PER_SECOND))
    assert plan.estimated_seconds == plan.total_scene_duration_seconds == budget
    # the model's padded durations are ignored; longer narration gets a longer scene
    words = [len(sc.narration.split()) for sc in plan.scenes]
    longest = max(range(len(words)), key=words.__getitem__)
    assert plan.scenes[longest].duration_seconds == max(sc.duration_seconds for sc in plan.scenes)


def test_narration_seconds_matches_measured_voice():
    assert narration_seconds(" ".join(["w"] * 154)) == 59  # measured: 59.15s
    assert narration_seconds("") == 1
