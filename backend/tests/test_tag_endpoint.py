"""Pipeline-stage tests for POST /tag: embedding scoring -> quota selection ->
vision judge -> contradiction judge -> prompt assembly, run end to end
through the real FastAPI route.

The embedding model is swapped for `fake_siglip_state` (see conftest.py) so
the real ranking/quota/projection code runs against synthetic vectors. The
two judge passes are monkeypatched at their call site in main.py's imports
(`judge.judge_image`, `contradiction_judge.judge_contradictions`) so no
Claude call is ever made.

IMPORTANT: like /generate (see test_generate_endpoint.py), /tag mixes a
`File(...)` parameter (image) with plain `str` parameters (genre,
algorithm), so FastAPI resolves genre/algorithm as query parameters, not
form fields -- the frontend already calls it that way (see
`tagOneAlgorithm` in frontend/app/page.tsx). Every request below passes
them via `params=`, never `data=`; sending them as form data would silently
fall back to the defaults instead of actually exercising validation (or, for
an unmocked algorithm, risk loading a real, undownloaded model).
"""

from conftest import png_bytes


def _patch_embed_image(monkeypatch, fake_vector, algorithm="siglip"):
    import embedding_tagger

    monkeypatch.setattr(
        embedding_tagger,
        "embed_image",
        lambda image_bytes, algorithm=algorithm: fake_vector,
    )


def _no_flags(monkeypatch):
    import contradiction_judge
    import judge

    monkeypatch.setattr(
        judge, "judge_image", lambda image_bytes, tags: [
            {"tag": t, "flagged": False, "reason": ""} for t in tags
        ],
    )
    monkeypatch.setattr(
        contradiction_judge, "judge_contradictions", lambda tags, facet_of: [],
    )


def test_tag_rejects_unsupported_content_type(client):
    resp = client.post("/tag", files={"image": ("x.txt", b"not an image", "text/plain")})
    assert resp.status_code == 400


def test_tag_rejects_neither_image_nor_sample_image(client):
    resp = client.post("/tag")
    assert resp.status_code == 400


def test_tag_rejects_unknown_sample_image(client):
    resp = client.post("/tag", params={"sample_image": "not-a-real-image.png"})
    assert resp.status_code == 400


def test_tag_rejects_both_image_and_sample_image(client):
    # An upload must never be silently discarded in favor of a sample image
    # the caller didn't actually ask to use instead.
    resp = client.post(
        "/tag",
        files={"image": ("x.png", png_bytes(), "image/png")},
        params={"sample_image": "beach-1.png"},
    )
    assert resp.status_code == 400


def test_tag_sample_image_reads_bytes_from_disk(client, monkeypatch, fake_siglip_state):
    import main

    fake_vector = fake_siglip_state.text_vectors["mood"][0].unsqueeze(0)
    _patch_embed_image(monkeypatch, fake_vector)
    _no_flags(monkeypatch)

    real_name = next(iter(sorted(p.name for p in main.TEST_IMAGES_DIR.glob("*.png"))))

    resp = client.post("/tag", params={"sample_image": real_name})
    assert resp.status_code == 200


def test_tag_rejects_unknown_genre(client, monkeypatch, fake_siglip_state):
    fake_vector = fake_siglip_state.text_vectors["mood"][0].unsqueeze(0)
    _patch_embed_image(monkeypatch, fake_vector)
    _no_flags(monkeypatch)

    resp = client.post(
        "/tag",
        files={"image": ("x.png", png_bytes(), "image/png")},
        params={"genre": "not-a-real-genre"},
    )
    assert resp.status_code == 400


def test_tag_rejects_unknown_algorithm(client):
    # Deliberately no fake state/mocks: validate_algorithm runs before any
    # embedding code, so a correctly-passed bad algorithm must 400 out
    # before ever reaching embed_image -- if it didn't, this would try to
    # load a real, undownloaded model.
    resp = client.post(
        "/tag",
        files={"image": ("x.png", png_bytes(), "image/png")},
        params={"algorithm": "not-a-real-algorithm"},
    )
    assert resp.status_code == 400


def test_tag_happy_path(client, monkeypatch, fake_siglip_state):
    from vocabulary import QUOTAS

    fake_vector = fake_siglip_state.text_vectors["mood"][0].unsqueeze(0)
    _patch_embed_image(monkeypatch, fake_vector)
    _no_flags(monkeypatch)

    resp = client.post(
        "/tag",
        files={"image": ("x.png", png_bytes(), "image/png")},
        params={"genre": "jazz", "algorithm": "siglip"},
    )
    assert resp.status_code == 200
    body = resp.json()

    assert body["algorithm"] == "siglip"
    # Genre leads the prompt, tags follow.
    assert body["prompt"].startswith("jazz, ")
    # No genre picked: the tag list alone.
    expected_tag_count = sum(QUOTAS.values())
    assert len(body["tags"]) == expected_tag_count
    assert set(body["tag_scores"]) == set(body["tags"])
    assert len(body["tagging"]["final"]) == expected_tag_count
    assert body["tagging"]["embedding"]["model"] == "google/siglip2-base-patch16-224"
    assert body["tagging"]["judges_enabled"] is True


def test_tag_without_genre_omits_it_from_prompt(client, monkeypatch, fake_siglip_state):
    fake_vector = fake_siglip_state.text_vectors["mood"][0].unsqueeze(0)
    _patch_embed_image(monkeypatch, fake_vector)
    _no_flags(monkeypatch)

    resp = client.post("/tag", files={"image": ("x.png", png_bytes(), "image/png")})
    body = resp.json()
    assert body["prompt"] == ", ".join(body["tags"])


def test_tag_drops_tags_flagged_by_vision_judge(client, monkeypatch, fake_siglip_state):
    import contradiction_judge
    import embedding_tagger
    import judge

    fake_vector = fake_siglip_state.text_vectors["mood"][0].unsqueeze(0)
    _patch_embed_image(monkeypatch, fake_vector)

    ranked = embedding_tagger.tags_by_facet(b"unused", algorithm="siglip", image_vector=fake_vector)
    expected_tags, _ = embedding_tagger.select_tags(ranked)
    tag_to_drop = expected_tags[0]

    def fake_judge_image(image_bytes, tags):
        return [
            {"tag": t, "flagged": t == tag_to_drop, "reason": "bad fit" if t == tag_to_drop else ""}
            for t in tags
        ]

    monkeypatch.setattr(judge, "judge_image", fake_judge_image)
    monkeypatch.setattr(contradiction_judge, "judge_contradictions", lambda tags, facet_of: [])

    resp = client.post("/tag", files={"image": ("x.png", png_bytes(), "image/png")})
    body = resp.json()

    assert tag_to_drop not in body["tags"]
    assert tag_to_drop not in body["prompt"]
    assert len(body["tags"]) == len(expected_tags) - 1


def test_tag_drops_tag_flagged_by_contradiction_judge(client, monkeypatch, fake_siglip_state):
    import contradiction_judge
    import embedding_tagger
    import judge

    fake_vector = fake_siglip_state.text_vectors["mood"][0].unsqueeze(0)
    _patch_embed_image(monkeypatch, fake_vector)

    ranked = embedding_tagger.tags_by_facet(b"unused", algorithm="siglip", image_vector=fake_vector)
    expected_tags, _ = embedding_tagger.select_tags(ranked)
    tag_a, tag_b = expected_tags[0], expected_tags[1]

    monkeypatch.setattr(
        judge, "judge_image", lambda image_bytes, tags: [
            {"tag": t, "flagged": False, "reason": ""} for t in tags
        ],
    )
    monkeypatch.setattr(
        contradiction_judge,
        "judge_contradictions",
        lambda tags, facet_of: [
            {"tag_a": tag_a, "tag_b": tag_b, "drop": tag_a, "reason": "mood clash"}
        ],
    )

    resp = client.post("/tag", files={"image": ("x.png", png_bytes(), "image/png")})
    body = resp.json()

    assert tag_a not in body["tags"]
    assert tag_b in body["tags"]
    assert body["tagging"]["contradictions"] == [
        {"tag_a": tag_a, "tag_b": tag_b, "drop": tag_a, "reason": "mood clash"}
    ]


def test_tag_falls_back_when_vision_judge_raises(client, monkeypatch, fake_siglip_state):
    import contradiction_judge
    import embedding_tagger
    import judge

    fake_vector = fake_siglip_state.text_vectors["mood"][0].unsqueeze(0)
    _patch_embed_image(monkeypatch, fake_vector)

    ranked = embedding_tagger.tags_by_facet(b"unused", algorithm="siglip", image_vector=fake_vector)
    expected_tags, _ = embedding_tagger.select_tags(ranked)

    def raising_judge(image_bytes, tags):
        raise RuntimeError("rate limited")

    monkeypatch.setattr(judge, "judge_image", raising_judge)
    monkeypatch.setattr(contradiction_judge, "judge_contradictions", lambda tags, facet_of: [])

    resp = client.post("/tag", files={"image": ("x.png", png_bytes(), "image/png")})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body["tags"]) == set(expected_tags)
    assert body["tagging"]["judge"] == []


def test_tag_falls_back_when_contradiction_judge_raises(client, monkeypatch, fake_siglip_state):
    import contradiction_judge
    import embedding_tagger
    import judge

    fake_vector = fake_siglip_state.text_vectors["mood"][0].unsqueeze(0)
    _patch_embed_image(monkeypatch, fake_vector)

    ranked = embedding_tagger.tags_by_facet(b"unused", algorithm="siglip", image_vector=fake_vector)
    expected_tags, _ = embedding_tagger.select_tags(ranked)

    monkeypatch.setattr(
        judge, "judge_image", lambda image_bytes, tags: [
            {"tag": t, "flagged": False, "reason": ""} for t in tags
        ],
    )

    def raising_contradictions(tags, facet_of):
        raise RuntimeError("timeout")

    monkeypatch.setattr(contradiction_judge, "judge_contradictions", raising_contradictions)

    resp = client.post("/tag", files={"image": ("x.png", png_bytes(), "image/png")})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body["tags"]) == set(expected_tags)
    assert body["tagging"]["contradictions"] == []


def test_tag_judges_false_skips_both_judge_calls_entirely(client, monkeypatch, fake_siglip_state):
    import contradiction_judge
    import embedding_tagger
    import judge

    fake_vector = fake_siglip_state.text_vectors["mood"][0].unsqueeze(0)
    _patch_embed_image(monkeypatch, fake_vector)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("judges=False must skip this call entirely, not just ignore its result")

    monkeypatch.setattr(judge, "judge_image", fail_if_called)
    monkeypatch.setattr(contradiction_judge, "judge_contradictions", fail_if_called)

    ranked = embedding_tagger.tags_by_facet(b"unused", algorithm="siglip", image_vector=fake_vector)
    expected_tags, _ = embedding_tagger.select_tags(ranked)

    resp = client.post(
        "/tag",
        files={"image": ("x.png", png_bytes(), "image/png")},
        params={"judges": "false"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["tags"] == expected_tags
    assert body["tagging"]["judge"] == []
    assert body["tagging"]["contradictions"] == []
    assert body["tagging"]["judges_enabled"] is False


def test_tag_judges_true_by_default(client, monkeypatch, fake_siglip_state):
    fake_vector = fake_siglip_state.text_vectors["mood"][0].unsqueeze(0)
    _patch_embed_image(monkeypatch, fake_vector)

    calls = []

    def spy_judge_image(image_bytes, tags):
        calls.append("judge_image")
        return [{"tag": t, "flagged": False, "reason": ""} for t in tags]

    def spy_judge_contradictions(tags, facet_of):
        calls.append("judge_contradictions")
        return []

    import contradiction_judge
    import judge

    monkeypatch.setattr(judge, "judge_image", spy_judge_image)
    monkeypatch.setattr(contradiction_judge, "judge_contradictions", spy_judge_contradictions)

    resp = client.post("/tag", files={"image": ("x.png", png_bytes(), "image/png")})
    assert resp.status_code == 200
    assert calls == ["judge_image", "judge_contradictions"]
