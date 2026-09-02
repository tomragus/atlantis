"""Runs the real app, over real HTTP, with Suno/Anthropic-costing calls
stubbed out -- what loadtest.py drives.

Deliberately a separate runner rather than an env-var flag inside main.py
itself (e.g. MOCK_SUNO=1): that would mean shipping a test-only code path in
the app the deployed container also carries. Patching main.request_song/
main.save_audio here, then serving the same app object over uvicorn same as
`uvicorn main:app` would, gets the real thing under real concurrency/
threading with zero footprint in the app's own source.

Usage: python3 loadtest_server.py [--port 8098]
"""

import argparse
import threading
import time
from unittest.mock import patch

import main

# Stands in for apiframe's own real cap -- imported rather than a second
# hardcoded 5, so this mock can never silently drift from what main.py
# itself enforces.
_MAX_REAL_CONCURRENCY = main.MAX_CONCURRENT_GENERATIONS

_concurrency_lock = threading.Lock()
_concurrent_jobs = 0


def fake_request_song(prompt: str) -> dict:
    """Stands in for the real Suno submit-and-poll round trip. Also stands
    in for apiframe's own concurrency cap: tracks how many fake jobs are
    simultaneously "in flight" and raises if more than _MAX_REAL_CONCURRENCY
    are ever concurrent -- exactly the failure the real apiframe API would
    return on a 6th submission (see PLAN-generation-queue.md). If main.py's
    queue is doing its job, this should never trip; if it does, that's a
    real bug in the queue, not a flaky test, since the whole point of the
    queue is to keep this exact call site from ever being reached more than
    _MAX_REAL_CONCURRENCY times at once.

    The 1.5s sleep (not the old 0.3s) is deliberately long enough that a
    load test with more than 5 concurrent visitors reliably creates real
    contention for the cap -- too short a sleep and slots free before the
    next request even lands, and the cap never gets meaningfully exercised.
    """
    global _concurrent_jobs
    with _concurrency_lock:
        _concurrent_jobs += 1
        current = _concurrent_jobs
        if current > _MAX_REAL_CONCURRENCY:
            _concurrent_jobs -= 1
            raise RuntimeError(
                f"apiframe concurrency cap violated: {current} concurrent jobs "
                f"(max {_MAX_REAL_CONCURRENCY}) -- main.py's queue should have "
                "prevented this from ever happening"
            )
    try:
        time.sleep(1.5)
        return {"id": "loadtest-fake-job"}
    finally:
        with _concurrency_lock:
            _concurrent_jobs -= 1


def fake_save_audio(token, run_id, original_filename, job, algorithm) -> list[dict]:
    return [
        {
            "title": "loadtest track",
            "duration": 30,
            "audioUrl": "http://example.invalid/fake.mp3",
            "ref": f"loadtest/{run_id}/fake.mp3",
            "url": "/files/loadtest-fake.mp3",
        }
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8098)
    args = parser.parse_args()

    with patch.object(main, "request_song", fake_request_song), patch.object(
        main, "save_audio", fake_save_audio
    ):
        import uvicorn

        # Single worker: the point is measuring how one Cloud Run instance's
        # capacity scales with concurrency, matching production's one-
        # process-per-instance model (see main.py's own threading.Lock
        # design, which assumes exactly that).
        uvicorn.run(main.app, host="127.0.0.1", port=args.port, log_level="warning")
