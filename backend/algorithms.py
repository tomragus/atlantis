"""The embedding algorithms the tagger can run, and the picker's copy for each.

Kept out of `embedding_tagger.py` deliberately, mirroring that file's own split from
`genres.py`: `main.py`'s `/generate` route needs to validate which algorithm produced
the tags it's saving to history, but never needs to load a single model to do that.
Keeping this list and its validator here, with no heavy imports, means validating an
`algorithm` string doesn't drag torch/transformers into a route that has no other
reason to import them -- the same "import main stays fast" reasoning `main.py`
already applies to its own deferred imports.
"""

from fastapi import HTTPException

ALGORITHMS = ["siglip", "clip", "align", "metaclip", "coca"]
DEFAULT_ALGORITHM = "siglip"

# Display label + one-line blurb per algorithm, for the frontend's algorithm-picker
# card (GET /algorithms in main.py). The blurb calls out each algorithm's scoring
# scale explicitly -- only SigLIP has a trained calibration head turning cosine
# similarity into a 0-1 probability (see embedding_tagger.py's module docstring); the
# rest score with raw cosine similarity on their own native scale -- so the UI copy
# can stay algorithm-aware without repeating that reasoning per view.
ALGORITHM_INFO = {
    "siglip": {
        "label": "SigLIP",
        "blurb": "Today's default. Sigmoid-calibrated match probability (0-1).",
    },
    "clip": {
        "label": "CLIP",
        "blurb": "OpenAI's original joint embedding. Raw cosine similarity, its own scale.",
    },
    "align": {
        "label": "ALIGN",
        "blurb": "Trained on noisy web alt-text at huge scale. Raw cosine similarity.",
    },
    "metaclip": {
        "label": "MetaCLIP",
        "blurb": "Meta's reproduction of CLIP's data curation. Raw cosine similarity.",
    },
    "coca": {
        "label": "CoCa",
        "blurb": "Also captions the image in free text, shown alongside its tags.",
    },
}


def validate_algorithm(algorithm: str) -> None:
    """Raises if `algorithm` isn't one of ALGORITHMS. Mirrors genres.validate_genre."""
    if algorithm not in ALGORITHMS:
        raise HTTPException(
            400, f"Unknown algorithm {algorithm!r}. See GET /algorithms."
        )
