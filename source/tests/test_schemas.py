from syco.schemas import (
    ACCEPTABLE_AT,
    CATEGORIES,
    IDEAL_AT,
    INAPPROPRIATE_AT,
    SEVERITIES,
    GenParams,
    JudgeVerdict,
    ResponseRecord,
    Usage,
    make_judgment_id,
    make_response_id,
)


def test_record_id_is_deterministic_and_independent_of_run_id():
    a = make_response_id("p001", "llama31_8b")
    b = make_response_id("p001", "llama31_8b")
    assert a == b
    c = make_response_id("p002", "llama31_8b")
    assert a != c
    d = make_judgment_id(a, "anthropic/claude-3.7-sonnet", "v1")
    e = make_judgment_id(a, "anthropic/claude-3.7-sonnet", "v1")
    assert d == e
    f = make_judgment_id(a, "anthropic/claude-3.7-sonnet", "v2")
    assert d != f


def test_rubric_partition_per_severity():
    """Every category must appear in exactly one of ideal/acceptable/inappropriate per severity."""
    for sev in list(SEVERITIES) + ["control"]:
        ideal = IDEAL_AT[sev]
        accept = ACCEPTABLE_AT[sev]
        bad = INAPPROPRIATE_AT[sev]
        union = ideal | accept | bad
        assert union == set(CATEGORIES), f"sev {sev}: missing {set(CATEGORIES) - union}"
        assert ideal.isdisjoint(accept), f"sev {sev}: ideal ∩ acceptable = {ideal & accept}"
        assert ideal.isdisjoint(bad), f"sev {sev}: ideal ∩ inappropriate = {ideal & bad}"
        assert accept.isdisjoint(bad), f"sev {sev}: acceptable ∩ inappropriate = {accept & bad}"


def test_response_record_round_trip():
    rec = ResponseRecord(
        record_id="abc",
        prompt_id="p001",
        severity=3,
        domain="spiritual",
        source="seed",
        prompt="hi",
        model_label="llama31_8b",
        model_id="meta-llama/llama-3.1-8b-instruct",
        params=GenParams(temperature=0.7, top_p=1.0, max_tokens=800, seed=42),
        response="hello",
        usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30, cost_usd=0.001),
        run_id="r1",
    )
    d = rec.model_dump()
    again = ResponseRecord.model_validate(d)
    assert again == rec


def test_judge_verdict_validates_range():
    v = JudgeVerdict(category="D", appropriateness=4, reasoning=" ok ")
    assert v.appropriateness == 4
    assert v.reasoning == "ok"
    import pytest
    with pytest.raises(Exception):
        JudgeVerdict(category="Z", appropriateness=4, reasoning="x")
    with pytest.raises(Exception):
        JudgeVerdict(category="D", appropriateness=6, reasoning="x")
