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
import time
from unittest.mock import patch

import main


def fake_request_song(prompt: str) -> dict:
    """Stands in for the real Suno submit-and-poll round trip. A short
    sleep, not zero, so the load test's latency numbers reflect a request
    that's actually I/O-bound for a moment (as /generate always is) rather
    than measuring pure Python overhead."""
    time.sleep(0.3)
    return {"id": "loadtest-fake-job"}


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
