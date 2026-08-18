"""Unit test for judge_client.call_judge's request/parse plumbing, shared by
both judge passes. The Anthropic client's `.messages.create` is monkeypatched
so this never makes a real network call to Claude."""

import json
from types import SimpleNamespace

import judge_client


def test_call_judge_parses_json_out_of_the_text_block(monkeypatch):
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=json.dumps({"ok": True, "n": 3}))]
        )

    monkeypatch.setattr(judge_client._client.messages, "create", fake_create)

    result = judge_client.call_judge(
        max_tokens=100,
        schema={"type": "object"},
        messages=[{"role": "user", "content": "hi"}],
    )

    assert result == {"ok": True, "n": 3}
    assert captured["model"] == judge_client.JUDGE_MODEL
    assert captured["max_tokens"] == 100
    assert captured["output_config"] == {"format": {"type": "json_schema", "schema": {"type": "object"}}}


def test_call_judge_skips_non_text_blocks(monkeypatch):
    def fake_create(**kwargs):
        return SimpleNamespace(
            content=[
                SimpleNamespace(type="other", text=None),
                SimpleNamespace(type="text", text=json.dumps({"ok": True})),
            ]
        )

    monkeypatch.setattr(judge_client._client.messages, "create", fake_create)
    result = judge_client.call_judge(max_tokens=10, schema={}, messages=[])
    assert result == {"ok": True}
