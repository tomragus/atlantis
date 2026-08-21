"""Unit tests for main.py's non-network helpers: filename sanitization and
the tagging summary."""

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


def test_build_tagging_summary_builds_final_tags_with_facet_and_score():
    facet_of = {"haunting": "mood", "wistful": "mood", "crimson": "color"}
    final_tags = ["wistful", "crimson"]
    final_scores = {"wistful": 0.1, "crimson": 0.8}

    summary = main.build_tagging_summary(facet_of, [], [], final_tags, final_scores)

    assert summary["final"] == [
        {"tag": "wistful", "facet": "mood", "score": 0.1},
        {"tag": "crimson", "facet": "color", "score": 0.8},
    ]
    assert summary["judge"] == []
    assert summary["contradictions"] == []
