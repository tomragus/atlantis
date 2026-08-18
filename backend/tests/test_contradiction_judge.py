"""Unit tests for contradiction_judge.py's parsing/validation logic.
`call_judge` (the only thing that would reach Claude) is always
monkeypatched out."""

import contradiction_judge as cj


TAGS = ["joyful", "haunting", "misty"]
FACETS = {"joyful": "valence", "haunting": "mood", "misty": "light"}


def test_judge_contradictions_returns_a_valid_pair(monkeypatch):
    def fake_call_judge(*, max_tokens, schema, messages):
        return {
            "contradictions": [
                {"tag_a": "joyful", "tag_b": "haunting", "drop": "joyful", "reason": "mood clash"}
            ]
        }

    monkeypatch.setattr(cj, "call_judge", fake_call_judge)
    result = cj.judge_contradictions(TAGS, FACETS)
    assert result == [
        {"tag_a": "joyful", "tag_b": "haunting", "drop": "joyful", "reason": "mood clash"}
    ]


def test_judge_contradictions_drops_pair_referencing_unknown_tag(monkeypatch):
    def fake_call_judge(*, max_tokens, schema, messages):
        return {
            "contradictions": [
                {"tag_a": "joyful", "tag_b": "nonexistent", "drop": "joyful", "reason": "x"}
            ]
        }

    monkeypatch.setattr(cj, "call_judge", fake_call_judge)
    assert cj.judge_contradictions(TAGS, FACETS) == []


def test_judge_contradictions_drops_entry_with_invalid_drop_value(monkeypatch):
    def fake_call_judge(*, max_tokens, schema, messages):
        return {
            "contradictions": [
                {"tag_a": "joyful", "tag_b": "misty", "drop": "haunting", "reason": "y"}
            ]
        }

    monkeypatch.setattr(cj, "call_judge", fake_call_judge)
    assert cj.judge_contradictions(TAGS, FACETS) == []


def test_judge_contradictions_defaults_missing_reason_to_empty_string(monkeypatch):
    def fake_call_judge(*, max_tokens, schema, messages):
        return {"contradictions": [{"tag_a": "joyful", "tag_b": "misty", "drop": "joyful"}]}

    monkeypatch.setattr(cj, "call_judge", fake_call_judge)
    result = cj.judge_contradictions(TAGS, FACETS)
    assert result == [{"tag_a": "joyful", "tag_b": "misty", "drop": "joyful", "reason": ""}]


def test_judge_contradictions_empty_response_is_empty_list(monkeypatch):
    monkeypatch.setattr(cj, "call_judge", lambda **kw: {"contradictions": []})
    assert cj.judge_contradictions(TAGS, FACETS) == []
