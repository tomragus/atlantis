"""Unit tests for judge.py's parsing/fallback logic. `call_judge` (the only
thing that would reach Claude) is always monkeypatched out."""

import judge
from conftest import png_bytes


def test_judge_image_reports_flagged_and_unflagged_tags(monkeypatch):
    def fake_call_judge(*, max_tokens, schema, messages):
        return {
            "verdicts": [
                {"tag": "joyful", "flagged": True, "reason": "scene is grim"},
                {"tag": "gritty", "flagged": False, "reason": ""},
            ]
        }

    monkeypatch.setattr(judge, "call_judge", fake_call_judge)
    result = judge.judge_image(png_bytes(), ["joyful", "gritty"])

    by_tag = {v["tag"]: v for v in result}
    assert by_tag["joyful"] == {"tag": "joyful", "flagged": True, "reason": "scene is grim"}
    assert by_tag["gritty"] == {"tag": "gritty", "flagged": False, "reason": ""}


def test_judge_image_falls_back_to_unflagged_for_missing_verdict(monkeypatch):
    def fake_call_judge(*, max_tokens, schema, messages):
        return {"verdicts": [{"tag": "gritty", "flagged": False, "reason": ""}]}
        # "misty" is silently missing from the model's response.

    monkeypatch.setattr(judge, "call_judge", fake_call_judge)
    result = judge.judge_image(png_bytes(), ["gritty", "misty"])

    by_tag = {v["tag"]: v for v in result}
    assert by_tag["misty"] == {"tag": "misty", "flagged": False, "reason": ""}


def test_judge_image_ignores_response_items_without_a_tag(monkeypatch):
    def fake_call_judge(*, max_tokens, schema, messages):
        return {"verdicts": [{"flagged": True, "reason": "no tag field"}]}

    monkeypatch.setattr(judge, "call_judge", fake_call_judge)
    result = judge.judge_image(png_bytes(), ["gritty"])

    assert result == [{"tag": "gritty", "flagged": False, "reason": ""}]


def test_judge_image_preserves_input_tag_order(monkeypatch):
    def fake_call_judge(*, max_tokens, schema, messages):
        return {
            "verdicts": [
                {"tag": "gritty", "flagged": False, "reason": ""},
                {"tag": "misty", "flagged": False, "reason": ""},
            ]
        }

    monkeypatch.setattr(judge, "call_judge", fake_call_judge)
    result = judge.judge_image(png_bytes(), ["misty", "gritty"])
    assert [v["tag"] for v in result] == ["misty", "gritty"]


def test_resize_shrinks_images_above_the_max_side():
    from PIL import Image
    import io

    big = png_bytes(size=(2000, 1000))
    resized = judge._resize(big, 1024, "JPEG")
    out = Image.open(io.BytesIO(resized))
    assert max(out.size) <= 1024


def test_resize_leaves_small_images_unchanged_dimensions():
    from PIL import Image
    import io

    small = png_bytes(size=(16, 16))
    resized = judge._resize(small, 1024, "JPEG")
    out = Image.open(io.BytesIO(resized))
    assert out.size == (16, 16)
