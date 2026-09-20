"""Model-level tests for the Stage-4 Script / ScriptLine artifacts.

Covers deterministic id derivation, envelope layout, JSON round-tripping, and
the internal consistency checks the model enforces itself (everything deeper -
claim grounding, semantic safety - lives in test_script_validator.py).
"""

from __future__ import annotations

import json

import pytest

from agents.script.models import (
    FACTUAL_PURPOSES,
    PURPOSE_VALUES,
    SCHEMA,
    SCHEMA_VERSION,
    STAGE,
    ScriptError,
    Script,
    ScriptLine,
    line_id_for,
    script_id_for,
)
from agents.seo.models import Provenance

from script_test_utils import (
    FIXED_AT,
    OCTOPUS_BLUE,
    OCTOPUS_SWIM,
    OCTOPUS_THREE_HEARTS,
    build_octopus_pair,
    make_script,
)


def test_purpose_contract_lists():
    assert "fact" in PURPOSE_VALUES
    assert "explanation" in PURPOSE_VALUES
    assert "conclusion" in PURPOSE_VALUES
    assert "hook" in PURPOSE_VALUES
    assert set(FACTUAL_PURPOSES) <= set(PURPOSE_VALUES)
    assert len(set(PURPOSE_VALUES)) == len(PURPOSE_VALUES)


def test_id_extraction_is_deterministic_and_prefixed():
    assert script_id_for("run-1", "t_octopus_hearts").startswith("sc_")
    assert script_id_for("run-1", "t_octopus_hearts") == script_id_for("run-1", "t_octopus_hearts")
    assert script_id_for("run-1", "t_octopus_hearts") != script_id_for("run-2", "t_octopus_hearts")
    assert line_id_for(0) == "ln_00"
    assert line_id_for(1) == "ln_01"
    assert line_id_for(99) == "ln_99"
    assert len(script_id_for("a", "b")) == 3 + 12


def test_script_envelope_shape():
    result, _ = build_octopus_pair(approved=("fact", "explanation", "explanation"))
    script = make_script(result)
    data = script.to_dict()
    assert data["schema"] == SCHEMA
    assert data["schema_version"] == SCHEMA_VERSION
    assert data["stage"] == STAGE
    assert data["run_id"] == "script-test"
    assert data["created_at"] == FIXED_AT
    assert data["produced_by"]["model"] == "MockScriptGenerator"
    assert "data" in data and data["data"]["script_id"] == script.script_id


def test_json_round_trip_preserves_everything():
    result, _ = build_octopus_pair(approved=("fact", "explanation", "explanation"))
    script = make_script(result)
    restored = Script.from_json(json.dumps(script.to_dict()))
    assert restored.to_dict() == script.to_dict()
    assert restored.lines == script.lines
    assert restored.provenance == script.provenance
    assert restored.used_claim_ids == script.used_claim_ids


def test_valid_script_passes_model_validation():
    result, _ = build_octopus_pair(approved=("fact",))
    script = make_script(
        result,
        lines=[
            {"text": OCTOPUS_THREE_HEARTS, "purpose": "fact", "claim_refs": (result.approved_claims[0],)},
        ],
    )
    validated = script.validate()
    assert validated.script_id == script.script_id
    assert validated.all_claim_refs() == set(script.used_claim_ids)
    assert validated.estimated_duration_seconds >= 0


def test_model_rejects_empty_lines():
    result, _ = build_octopus_pair(approved=("fact",))
    with pytest.raises(ScriptError):
        make_script(result, lines=[]).validate()


def test_model_rejects_duplicate_line_ids():
    result, _ = build_octopus_pair(approved=("fact", "explanation"))
    claim_id = result.approved_claims[0]
    script = make_script(
        result,
        lines=[
            {"text": OCTOPUS_THREE_HEARTS, "purpose": "fact", "claim_refs": (claim_id,)},
            {"text": OCTOPUS_SWIM, "purpose": "explanation", "claim_refs": (claim_id,)},
        ],
    )
    first, second = script.lines
    second = ScriptLine(
        line_id=first.line_id,
        text=second.text,
        claim_refs=second.claim_refs,
        purpose=second.purpose,
        estimated_duration_seconds=second.estimated_duration_seconds,
    )
    corrupt = Script(
        script_id=script.script_id,
        topic_id=script.topic_id,
        topic_title=script.topic_title,
        title=script.title,
        hook=script.hook,
        lines=(first, second),
        estimated_duration_seconds=script.estimated_duration_seconds,
        used_claim_ids=script.used_claim_ids,
        unused_approved_claim_ids=script.unused_approved_claim_ids,
        result_id=script.result_id,
        package_id=script.package_id,
        provenance=script.provenance,
        created_at=script.created_at,
        run_id=script.run_id,
        produced_by=script.produced_by,
    )
    with pytest.raises(ScriptError):
        corrupt.validate()


@pytest.mark.parametrize("bad_purpose", ["bogus", "", "FACT"])
def test_model_rejects_unknown_purpose(bad_purpose):
    with pytest.raises(ScriptError):
        ScriptLine("ln_x", "Some line.", (), bad_purpose, 1.0).validate()


def test_model_rejects_negative_duration():
    with pytest.raises(ScriptError):
        ScriptLine("ln_x", "Some line.", (), "fact", -1.0).validate()


def test_model_rejects_duration_total_mismatch():
    result, _ = build_octopus_pair(approved=("fact",))
    script = make_script(
        result,
        lines=[{"text": OCTOPUS_THREE_HEARTS, "purpose": "fact", "claim_refs": (result.approved_claims[0],)}],
        estimated_duration=9999.0,
    )
    with pytest.raises(ScriptError):
        script.validate()


def test_model_rejects_used_unused_overlap():
    result, _ = build_octopus_pair(approved=("fact", "explanation"))
    used = (result.approved_claims[0],)
    script = make_script(
        result,
        lines=[{"text": OCTOPUS_THREE_HEARTS, "purpose": "fact", "claim_refs": used}],
        used=used,
        unused=(result.approved_claims[0],),  # same claim in both buckets
    )
    with pytest.raises(ScriptError):
        script.validate()


def test_model_rejects_missing_provenance():
    result, _ = build_octopus_pair(approved=("fact",))
    script = make_script(result, provenance=None)
    assert script.provenance is None
    with pytest.raises(ScriptError):
        script.validate()


def test_line_round_trip_and_validate():
    line = ScriptLine("ln_05", OCTOPUS_BLUE, ("cl_x",), "explanation", 3.0).validate()
    assert ScriptLine.from_dict(line.to_dict()) == line