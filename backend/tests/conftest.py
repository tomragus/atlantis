"""Shared fixtures for the backend test suite.

Nothing in here (or in any test) may reach apiframe/Suno or Claude -- those
calls are always monkeypatched out at their module boundary
(`requests.request`/`requests.get` in main.py, `call_judge` in judge_1.py/
judge_2.py). The embedding models are handled the same way:
`fake_algorithm_state` swaps in small synthetic vectors so the real
ranking/quota/projection math in embedding_tagger.py still runs, without
downloading or loading any actual model weights.
"""

import io
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def png_bytes(size=(16, 16), color=(120, 50, 200)) -> bytes:
    """A tiny in-memory PNG, real enough for PIL/judge._resize to decode."""
    image = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def make_fake_algorithm_state(dim: int = 8, seed: int = 0):
    """A ready `_AlgorithmState` with small synthetic per-word vectors instead
    of a loaded model, keyed by the real vocabulary shape so the real
    per-facet ranking code exercises its actual data shape.
    """
    from embedding_tagger import _AlgorithmState
    from vocabulary import ADJECTIVES

    generator = torch.Generator().manual_seed(seed)
    text_vectors = {
        facet: torch.nn.functional.normalize(
            torch.randn(len(words), dim, generator=generator), dim=-1
        )
        for facet, words in ADJECTIVES.items()
    }
    return _AlgorithmState(
        ready=True,
        text_vectors=text_vectors,
        logit_scale=10.0,
        logit_bias=-5.0,
    )


@pytest.fixture
def fake_siglip_state(monkeypatch):
    """Installs a fake, ready SigLIP state so `_load` is a no-op and
    `tags_by_facet`/`embedding_projection` run their real math against
    synthetic vectors instead of a downloaded model.
    """
    import embedding_tagger

    state = make_fake_algorithm_state()
    monkeypatch.setitem(embedding_tagger._state, "siglip", state)
    return state


@pytest.fixture
def client():
    import main

    from fastapi.testclient import TestClient

    return TestClient(main.app)


@pytest.fixture(autouse=True)
def _block_real_network(monkeypatch):
    """Safety net, applied to every test: any code path that reaches real
    `requests.request`/`requests.get` or the real Anthropic client without a
    test-local monkeypatch fails loudly instead of silently making a real
    (and, for apiframe, billed) network call. A test that needs a canned
    response monkeypatches these same targets itself, which overrides this
    guard for that test.
    """
    import requests as requests_module

    import judge_client

    def _blocked(*args, **kwargs):
        raise RuntimeError(
            "Test attempted a real network call (requests.request/get or the "
            "Anthropic client) without monkeypatching it first."
        )

    monkeypatch.setattr(requests_module, "request", _blocked)
    monkeypatch.setattr(requests_module, "get", _blocked)
    monkeypatch.setattr(judge_client._client.messages, "create", _blocked)


class FakeClock:
    """A controllable stand-in for the `time` module, so wait_for_job's
    polling loop advances instantly instead of over real wall-clock time.
    """

    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class FakeResponse:
    """A minimal stand-in for `requests.Response`, covering the attributes
    main.py actually reads (`.ok`, `.status_code`, `.text`, `.json()`,
    `.content`).
    """

    def __init__(self, ok=True, status_code=200, text="", json_data=None, content=b""):
        self.ok = ok
        self.status_code = status_code
        self.text = text
        self._json = json_data if json_data is not None else {}
        self.content = content

    def json(self):
        return self._json
