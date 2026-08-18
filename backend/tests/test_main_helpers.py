"""Unit tests for main.py's non-network helpers: filename sanitization and
the tagging-summary top-20 backfill."""

from datetime import datetime, timezone

import main


def test_safe_for_filename_strips_disallowed_characters():
    # "." isn't in the kept set either -- callers pass the extension separately.
    assert main.safe_for_filename('sun/set:"beach"?.png') == "sunsetbeachpng"


def test_safe_for_filename_keeps_hyphens_and_allowed_punctuation():
    assert main.safe_for_filename("dream-pop (v2) & friends'") == "dream-pop (v2) & friends'"


def test_safe_for_filename_empty_input_falls_back_to_untitled():
    assert main.safe_for_filename("///:::") == "untitled"


def test_image_base_name_strips_extension_and_directories():
    assert main.image_base_name("../../etc/sunset.png") == "sunset"


def test_image_base_name_none_falls_back_to_untitled():
    assert main.image_base_name(None) == "untitled"
    assert main.image_base_name("") == "untitled"


def test_audio_filename_format():
    stamp = datetime(2026, 8, 17, tzinfo=timezone.utc)
    assert main.audio_filename("sunset-beach", "clip", 1, stamp) == "sunset-beach-clip-1-08-17-2026.mp3"


def test_build_tagging_summary_always_includes_final_tags_even_if_low_scoring():
    ranked = {
        "mood": [("haunting", 0.9), ("wistful", 0.1)],
        "color": [("crimson", 0.8)],
    }
    facet_of = {"haunting": "mood", "wistful": "mood", "crimson": "color"}
    final_tags = ["wistful"]  # deliberately the lower-scoring mood candidate
    final_scores = {"wistful": 0.1}

    summary = main.build_tagging_summary(ranked, facet_of, [], [], final_tags, final_scores)

    tags_in_top20 = {entry["tag"] for entry in summary["top20"]}
    assert "wistful" in tags_in_top20
    assert summary["final"] == [{"tag": "wistful", "facet": "mood", "score": 0.1}]


def test_build_tagging_summary_caps_top20_at_20_entries():
    ranked = {"facet": [(f"word{i}", float(i)) for i in range(30)]}
    facet_of = {f"word{i}": "facet" for i in range(30)}

    summary = main.build_tagging_summary(ranked, facet_of, [], [], [], {})

    assert len(summary["top20"]) == 20
    # Best-scoring 20 of the 30 candidates, sorted descending.
    scores = [entry["score"] for entry in summary["top20"]]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == 29.0


def test_build_tagging_summary_top20_has_no_duplicate_tags():
    ranked = {"facet": [("shared", 0.5), ("other", 0.4)]}
    facet_of = {"shared": "facet", "other": "facet"}
    final_tags = ["shared"]
    final_scores = {"shared": 0.5}

    summary = main.build_tagging_summary(ranked, facet_of, [], [], final_tags, final_scores)

    tags = [entry["tag"] for entry in summary["top20"]]
    assert len(tags) == len(set(tags))
