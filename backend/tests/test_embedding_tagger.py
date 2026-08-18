"""Unit tests for embedding_tagger.py.

Two layers:
- Pure dict/list logic (select_tags, word_to_facet, drop_flagged_tags,
  _ordered_by_score) needs no model at all -- tested directly against
  hand-built `ranked` structures.
- The per-facet scoring/ranking math in tags_by_facet and the PCA layout in
  embedding_projection are tested against the `fake_siglip_state` fixture
  (synthetic vectors, see conftest.py) so the real torch math runs without
  ever loading or downloading an actual model.
"""

import pytest

import embedding_tagger
from vocabulary import ADJECTIVES, QUOTAS


def test_select_tags_takes_top_n_per_facet_quota():
    # QUOTAS["color"] == 2, so the third-ranked color candidate is dropped.
    ranked = {
        "color": [("crimson", 0.9), ("azure", 0.8), ("mint", 0.1)],
        "mood": [("haunting", 0.95)],
    }
    tags, scores = embedding_tagger.select_tags(ranked)
    assert tags.count("crimson") == 1 and tags.count("azure") == 1
    assert "mint" not in tags


def test_select_tags_orders_combined_list_by_score_descending():
    ranked = {
        "color": [("crimson", 0.2), ("azure", 0.1)],
        "mood": [("haunting", 0.95)],
    }
    tags, scores = embedding_tagger.select_tags(ranked)
    assert tags == ["haunting", "crimson", "azure"]
    assert scores == {"haunting": 0.95, "crimson": 0.2, "azure": 0.1}


def test_word_to_facet_flattens_ranked_structure():
    ranked = {
        "color": [("crimson", 0.9), ("azure", 0.8)],
        "mood": [("haunting", 0.95)],
    }
    assert embedding_tagger.word_to_facet(ranked) == {
        "crimson": "color",
        "azure": "color",
        "haunting": "mood",
    }


def test_drop_flagged_tags_removes_only_flagged_entries():
    tags = ["crimson", "azure", "haunting"]
    scores = {"crimson": 0.9, "azure": 0.8, "haunting": 0.95}
    new_tags, new_scores = embedding_tagger.drop_flagged_tags(tags, scores, {"azure"})
    assert new_tags == ["crimson", "haunting"]
    assert new_scores == {"crimson": 0.9, "haunting": 0.95}


def test_drop_flagged_tags_no_op_when_nothing_flagged():
    tags = ["crimson", "azure"]
    scores = {"crimson": 0.9, "azure": 0.8}
    new_tags, new_scores = embedding_tagger.drop_flagged_tags(tags, scores, {"not-present"})
    assert new_tags == tags
    assert new_scores == scores


def test_drop_flagged_tags_can_empty_the_list():
    tags = ["crimson"]
    scores = {"crimson": 0.9}
    new_tags, new_scores = embedding_tagger.drop_flagged_tags(tags, scores, {"crimson"})
    assert new_tags == []
    assert new_scores == {}


def test_ordered_by_score_is_stable_on_ties():
    tags = ["b", "a", "c"]
    scores = {"b": 1.0, "a": 1.0, "c": 2.0}
    ordered_tags, ordered_scores = embedding_tagger._ordered_by_score(tags, scores)
    assert ordered_tags == ["c", "b", "a"]  # c first (higher score), b/a keep input order
    assert ordered_scores == {"c": 2.0, "b": 1.0, "a": 1.0}


def test_tags_by_facet_ranks_the_exact_match_first(fake_siglip_state):
    target_word = ADJECTIVES["mood"][3]
    image_vector = fake_siglip_state.text_vectors["mood"][3].unsqueeze(0)

    ranked = embedding_tagger.tags_by_facet(b"unused", algorithm="siglip", image_vector=image_vector)

    assert ranked["mood"][0][0] == target_word
    assert ranked["mood"][0][1] == max(score for _, score in ranked["mood"])
    # Every facet is present and fully ranked (not just the quota's worth).
    assert set(ranked) == set(ADJECTIVES)
    assert len(ranked["color"]) == len(ADJECTIVES["color"])


def test_tags_by_facet_siglip_scores_are_probabilities(fake_siglip_state):
    image_vector = fake_siglip_state.text_vectors["mood"][0].unsqueeze(0)
    ranked = embedding_tagger.tags_by_facet(b"unused", algorithm="siglip", image_vector=image_vector)
    for _, score in ranked["color"]:
        assert 0.0 <= score <= 1.0


def test_tags_by_facet_non_siglip_uses_raw_cosine_not_sigmoid(monkeypatch):
    from conftest import make_fake_algorithm_state

    state = make_fake_algorithm_state()
    monkeypatch.setitem(embedding_tagger._state, "clip", state)

    target_word = ADJECTIVES["color"][0]
    image_vector = state.text_vectors["color"][0].unsqueeze(0)
    ranked = embedding_tagger.tags_by_facet(b"unused", algorithm="clip", image_vector=image_vector)

    # Exact self-match: cosine similarity is ~1.0, not squashed through a
    # sigmoid the way SigLIP's score would be.
    assert ranked["color"][0][0] == target_word
    assert ranked["color"][0][1] == pytest.approx(1.0, abs=1e-5)


def test_select_tags_respects_full_vocabulary_quota_sum(fake_siglip_state):
    image_vector = fake_siglip_state.text_vectors["mood"][0].unsqueeze(0)
    ranked = embedding_tagger.tags_by_facet(b"unused", algorithm="siglip", image_vector=image_vector)
    tags, scores = embedding_tagger.select_tags(ranked)
    assert len(tags) == sum(QUOTAS.values())
    assert len(set(tags)) == len(tags)  # no duplicates


def test_embedding_projection_shape_and_model_id(fake_siglip_state):
    points = [(ADJECTIVES["mood"][i], "mood") for i in range(4)]
    image_vector = fake_siglip_state.text_vectors["mood"][0].unsqueeze(0)

    projection = embedding_tagger.embedding_projection(
        b"unused", points, algorithm="siglip", image_vector=image_vector
    )

    assert projection["model"] == "google/siglip2-base-patch16-224"
    assert projection["source_dimensions"] == 8
    assert len(projection["points"]) == len(points)
    assert set(projection["image"]) == {"x", "y"}
    for point, (word, facet) in zip(projection["points"], points):
        assert point["tag"] == word
        assert point["facet"] == facet
