import pytest

from syco.judge import JudgeParseError, parse_judge_output


def test_clean_json():
    raw = '{"category": "D", "appropriateness": 4, "reasoning": "gentle reality testing"}'
    v = parse_judge_output(raw)
    assert v.category == "D"
    assert v.appropriateness == 4


def test_markdown_fences():
    raw = '```json\n{"category": "E", "appropriateness": 5, "reasoning": "good at sev5"}\n```'
    v = parse_judge_output(raw)
    assert v.category == "E"
    assert v.appropriateness == 5


def test_leading_and_trailing_prose():
    raw = (
        "Sure, here is my analysis:\n"
        '{"category": "B", "appropriateness": 2, "reasoning": "passive validation"}\n'
        "Hope that helps!"
    )
    v = parse_judge_output(raw)
    assert v.category == "B"


def test_bare_fences_no_lang():
    raw = '```\n{"category":"C","appropriateness":3,"reasoning":"neutral"}\n```'
    v = parse_judge_output(raw)
    assert v.category == "C"


def test_invalid_category_raises():
    raw = '{"category": "Z", "appropriateness": 3, "reasoning": "bad cat"}'
    with pytest.raises(JudgeParseError):
        parse_judge_output(raw)


def test_invalid_score_raises():
    raw = '{"category": "D", "appropriateness": 9, "reasoning": "bad score"}'
    with pytest.raises(JudgeParseError):
        parse_judge_output(raw)


def test_garbage_raises():
    with pytest.raises(JudgeParseError):
        parse_judge_output("totally not json at all")


def test_empty_raises():
    with pytest.raises(JudgeParseError):
        parse_judge_output("")
