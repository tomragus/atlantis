"""Narrower unit tests for the apiframe/Suno stage: apiframe_call's error
passthrough, wait_for_job's COMPLETED/FAILED/timeout branches, request_song's
submit-then-poll shape, and save_audio's collision fallback + partial-failure
handling.

`requests.request`/`requests.get` are always faked -- nothing here reaches
apiframe or Suno. `FakeClock` decouples wait_for_job's polling loop from real
wall-clock time so a timeout test doesn't take 300 real seconds.
"""

from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi import HTTPException

from conftest import FakeClock, FakeResponse


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    monkeypatch.setenv("APIFRAME_API_KEY", "test-key")


@pytest.fixture
def fake_clock(monkeypatch):
    import main

    clock = FakeClock()
    monkeypatch.setattr(main.time, "sleep", clock.sleep)
    monkeypatch.setattr(main.time, "monotonic", clock.monotonic)
    return clock


def test_apiframe_call_raises_without_api_key(monkeypatch):
    import main

    monkeypatch.delenv("APIFRAME_API_KEY", raising=False)
    with pytest.raises(HTTPException) as exc_info:
        main.apiframe_call("GET", "/jobs/x")
    assert exc_info.value.status_code == 500


def test_apiframe_call_passes_upstream_error_through(monkeypatch):
    import main

    monkeypatch.setattr(
        main.requests, "request",
        lambda method, url, headers=None, timeout=None, **kw: FakeResponse(
            ok=False, status_code=403, text="Forbidden: bad key"
        ),
    )
    with pytest.raises(HTTPException) as exc_info:
        main.apiframe_call("GET", "/jobs/x")
    assert exc_info.value.status_code == 502
    assert "Forbidden: bad key" in exc_info.value.detail


def test_wait_for_job_returns_on_completed(fake_clock, monkeypatch):
    import main

    monkeypatch.setattr(
        main.requests, "request",
        lambda *a, **k: FakeResponse(json_data={"status": "COMPLETED", "id": "j1"}),
    )
    result = main.wait_for_job("j1", "song")
    assert result["status"] == "COMPLETED"


def test_wait_for_job_raises_on_failed(fake_clock, monkeypatch):
    import main

    monkeypatch.setattr(
        main.requests, "request",
        lambda *a, **k: FakeResponse(json_data={"status": "FAILED", "error": "bad prompt"}),
    )
    with pytest.raises(HTTPException) as exc_info:
        main.wait_for_job("j1", "song")
    assert exc_info.value.status_code == 502
    assert "bad prompt" in exc_info.value.detail


def test_wait_for_job_times_out(fake_clock, monkeypatch):
    import main

    monkeypatch.setattr(
        main.requests, "request",
        lambda *a, **k: FakeResponse(json_data={"status": "PROCESSING"}),
    )
    with pytest.raises(HTTPException) as exc_info:
        main.wait_for_job("j1", "song")
    assert exc_info.value.status_code == 504


def test_request_song_submits_prompt_then_polls(fake_clock, monkeypatch):
    import main

    calls = []

    def fake_request(method, url, headers=None, timeout=None, json=None, **kw):
        calls.append((method, url, json))
        if url.endswith("/music/generate"):
            return FakeResponse(json_data={"jobId": "job-42"})
        return FakeResponse(json_data={"status": "COMPLETED", "id": "job-42", "result": {"tracks": []}})

    monkeypatch.setattr(main.requests, "request", fake_request)

    job = main.request_song("jazz, warm, gritty")
    assert job["id"] == "job-42"
    submit_call = calls[0]
    assert submit_call[2]["prompt"] == "jazz, warm, gritty"
    assert submit_call[2]["sunoParams"]["instrumental"] is True


class _FixedDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 8, 17, 12, 0, 0, tzinfo=timezone.utc)


def test_save_audio_falls_back_on_filename_collision(monkeypatch, tmp_path):
    import main

    monkeypatch.setattr(main, "AUDIO_DIR", tmp_path)
    monkeypatch.setattr(main, "datetime", _FixedDateTime)

    colliding_name = main.audio_filename("sunset", "siglip", 1, _FixedDateTime.now(timezone.utc))
    (tmp_path / colliding_name).write_bytes(b"pre-existing")

    monkeypatch.setattr(main.requests, "get", lambda url, timeout=None: FakeResponse(content=b"new-bytes"))

    job = {"result": {"tracks": [{"title": "T", "duration": 1, "audioUrl": "https://x/a.mp3"}]}}
    tracks = main.save_audio("sunset.png", job, "siglip")

    assert len(tracks) == 1
    saved_name = Path(tracks[0]["audioFile"]).name
    assert saved_name != colliding_name
    assert (tmp_path / colliding_name).read_bytes() == b"pre-existing"
    assert (tmp_path / saved_name).read_bytes() == b"new-bytes"


def test_save_audio_skips_failed_downloads_but_keeps_the_rest(monkeypatch, tmp_path):
    import main

    monkeypatch.setattr(main, "AUDIO_DIR", tmp_path)

    responses = {
        "https://x/ok.mp3": FakeResponse(content=b"good"),
        "https://x/bad.mp3": FakeResponse(ok=False),
    }
    monkeypatch.setattr(main.requests, "get", lambda url, timeout=None: responses[url])

    job = {
        "result": {
            "tracks": [
                {"title": "Good", "duration": 1, "audioUrl": "https://x/ok.mp3"},
                {"title": "Bad", "duration": 1, "audioUrl": "https://x/bad.mp3"},
            ]
        }
    }
    tracks = main.save_audio("sunset.png", job, "siglip")
    assert len(tracks) == 1
    assert tracks[0]["title"] == "Good"


def test_save_audio_raises_when_every_download_fails(monkeypatch, tmp_path):
    import main

    monkeypatch.setattr(main, "AUDIO_DIR", tmp_path)
    monkeypatch.setattr(main.requests, "get", lambda url, timeout=None: FakeResponse(ok=False))

    job = {"result": {"tracks": [{"title": "T", "duration": 1, "audioUrl": "https://x/a.mp3"}]}}
    with pytest.raises(HTTPException) as exc_info:
        main.save_audio("sunset.png", job, "siglip")
    assert exc_info.value.status_code == 502
