"""Pipeline-stage tests for POST /generate: takes a settled /tag response and
renders it via Suno (through apiframe) + downloads audio -- no re-tagging.

`request_song` is monkeypatched directly (its own polling logic against
apiframe is covered at a finer grain in test_apiframe.py) and
`requests.get` (the audio download) is faked, so no network call ever
reaches apiframe or Suno. The `_block_real_network` autouse fixture in
conftest.py is a second safety net: if a test's monkeypatching turns out to
be incomplete, the request fails loudly instead of silently hitting a real
endpoint.

IMPORTANT: `/generate` mixes `Form(...)` parameters (image_filename, tags,
tag_scores, tagging, prompt) with plain `str` parameters (genre, algorithm).
FastAPI resolves the latter as query parameters, not form fields, in that
combination -- the frontend already sends them that way (see
`handleGenerate` in frontend/app/page.tsx). Sending genre/algorithm as form
`data` here would silently be ignored in favor of their defaults, which is
exactly what happened the first time these tests were written -- so every
request below passes them via `params=`, not `data=`.
"""

import json

import pytest

from conftest import FakeResponse


def _generate_query(**overrides):
    query = {"genre": "jazz", "algorithm": "siglip"}
    query.update(overrides)
    return query


def _generate_form(**overrides):
    form = {
        "image_filename": "sunset.png",
        "tags": json.dumps(["gritty", "warm"]),
        "tag_scores": json.dumps({"gritty": 0.5, "warm": 0.6}),
        "tagging": json.dumps({"top20": [], "judge": [], "contradictions": [], "final": []}),
        "prompt": "jazz, gritty, warm",
    }
    form.update(overrides)
    return form


@pytest.fixture
def isolated_audio_dir(tmp_path, monkeypatch):
    import main

    monkeypatch.setattr(main, "AUDIO_DIR", tmp_path)
    return tmp_path


def _completed_job(tracks):
    return {"id": "job123", "status": "COMPLETED", "result": {"tracks": tracks}}


def test_generate_rejects_unknown_genre(client):
    resp = client.post(
        "/generate",
        params=_generate_query(genre="not-a-real-genre"),
        data=_generate_form(),
    )
    assert resp.status_code == 400


def test_generate_rejects_unknown_algorithm(client):
    resp = client.post(
        "/generate",
        params=_generate_query(algorithm="not-a-real-algorithm"),
        data=_generate_form(),
    )
    assert resp.status_code == 400


def test_generate_happy_path_downloads_both_takes(client, monkeypatch, isolated_audio_dir):
    import main

    job = _completed_job(
        [
            {"title": "Take One", "duration": 120, "audioUrl": "https://example.com/a.mp3"},
            {"title": "Take Two", "duration": 118, "audioUrl": "https://example.com/b.mp3"},
        ]
    )
    monkeypatch.setattr(main, "request_song", lambda prompt: job)
    monkeypatch.setattr(main.requests, "get", lambda url, timeout=None: FakeResponse(content=b"fake-audio-bytes"))

    resp = client.post("/generate", params=_generate_query(), data=_generate_form())
    assert resp.status_code == 200
    body = resp.json()

    assert body["job_id"] == "job123"
    assert body["algorithm"] == "siglip"
    assert body["tags"] == ["gritty", "warm"]
    assert len(body["tracks"]) == 2
    assert {t.stat().st_size for t in isolated_audio_dir.glob("*.mp3")} == {len(b"fake-audio-bytes")}


def test_generate_prompt_passed_through_verbatim(client, monkeypatch, isolated_audio_dir):
    import main

    captured_prompts = []

    def fake_request_song(prompt):
        captured_prompts.append(prompt)
        return _completed_job([{"title": "T", "duration": 1, "audioUrl": "https://x/a.mp3"}])

    monkeypatch.setattr(main, "request_song", fake_request_song)
    monkeypatch.setattr(main.requests, "get", lambda url, timeout=None: FakeResponse(content=b"x"))

    client.post(
        "/generate",
        params=_generate_query(),
        data=_generate_form(prompt="jazz, gritty, warm"),
    )
    assert captured_prompts == ["jazz, gritty, warm"]


def test_generate_raises_when_every_download_fails(client, monkeypatch, isolated_audio_dir):
    import main

    job = _completed_job([{"title": "T", "duration": 1, "audioUrl": "https://x/a.mp3"}])
    monkeypatch.setattr(main, "request_song", lambda prompt: job)
    monkeypatch.setattr(main.requests, "get", lambda url, timeout=None: FakeResponse(ok=False))

    resp = client.post("/generate", params=_generate_query(), data=_generate_form())
    assert resp.status_code == 502


def test_wipe_history_removes_files(client, isolated_audio_dir):
    (isolated_audio_dir / "a.mp3").write_bytes(b"x" * 10)
    (isolated_audio_dir / "b.mp3").write_bytes(b"y" * 20)

    resp = client.delete("/history")
    assert resp.status_code == 200
    body = resp.json()
    assert body["files_removed"] == 2
    assert body["megabytes_freed"] == pytest.approx(0.0, abs=1e-6)
    assert list(isolated_audio_dir.iterdir()) == []
