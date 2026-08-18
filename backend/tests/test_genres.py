import pytest
from fastapi import HTTPException

import genres


def test_validate_genre_accepts_empty_string():
    genres.validate_genre("")  # no genre picked -- must not raise


def test_validate_genre_accepts_known_genre():
    genres.validate_genre(genres.GENRES[0])


def test_validate_genre_rejects_unknown_genre():
    with pytest.raises(HTTPException) as exc_info:
        genres.validate_genre("not-a-real-genre")
    assert exc_info.value.status_code == 400


def test_prompt_term_uses_override_when_present():
    assert genres.prompt_term("house") == "deep house"


def test_prompt_term_falls_back_to_the_label_itself():
    assert genres.prompt_term("jazz") == "jazz"
