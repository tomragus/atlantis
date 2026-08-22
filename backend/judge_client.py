"""Shared Claude Sonnet 5 structured-output call, used by both judge passes
(judge_1.py's per-tag mood check and judge_2.py's cross-facet
contradiction check) so the client setup and call/parse pattern lives in one
place instead of two copies that could silently drift apart.
"""

import json

import anthropic

JUDGE_MODEL = "claude-sonnet-5"
_client = anthropic.Anthropic()


def call_judge(*, max_tokens: int, schema: dict, messages: list[dict]) -> dict:
    """Sends one structured-output request to Claude Sonnet 5 and returns the
    parsed JSON body.

    Every judge call goes through the `output_config.format: json_schema`
    path (never freeform text), so this is the one place that knows how to
    pull the JSON back out of the response.
    """
    response = _client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=max_tokens,
        output_config={"format": {"type": "json_schema", "schema": schema}},
        messages=messages,
    )
    text_block = next(b for b in response.content if b.type == "text")
    return json.loads(text_block.text)
