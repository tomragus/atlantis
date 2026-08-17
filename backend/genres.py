"""Genres the user can pick from, prepended to the tags in the Suno prompt.

Kept out of `vocabulary.py`: that file is scored against the image by SigLIP, and a
genre is neither visually detectable nor something the tagger should choose. This is
a menu the user picks from, and nothing here is ever embedded.

Ordered by family rather than alphabetically, because the picker is a knob you spin
to browse.
"""

from fastapi import HTTPException

GENRES = [
    # rock and guitar
    "rock",
    "psychedelic rock",
    "prog rock",
    "math rock",
    "metal",
    "heavy metal",
    "thrash metal",
    # pop
    "dream pop",
    "synth-pop",
    # electronic
    "idm",
    "house",
    "techno",
    "trance",
    # jazz
    "jazz",
    "jazz fusion",
    # cinematic
    "orchestral",
    # world
    "reggae",
]

# The label is what reads well on a knob; the value is what steers Suno.
PROMPT_TERMS = {
    "house": "deep house",
}


def prompt_term(genre: str) -> str:
    """The text that goes into the Suno prompt for a picked genre."""
    return PROMPT_TERMS.get(genre, genre)


def validate_genre(genre: str) -> None:
    """Raises if `genre` is non-empty and not one of GENRES.

    Empty is always fine -- it means no genre was picked.
    """
    if genre and genre not in GENRES:
        raise HTTPException(400, f"Unknown genre {genre!r}. See GET /genres.")
