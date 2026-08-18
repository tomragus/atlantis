import pytest
from fastapi import HTTPException

import algorithms


def test_validate_algorithm_accepts_every_known_algorithm():
    for algorithm in algorithms.ALGORITHMS:
        algorithms.validate_algorithm(algorithm)  # must not raise


def test_validate_algorithm_rejects_unknown_algorithm():
    with pytest.raises(HTTPException) as exc_info:
        algorithms.validate_algorithm("not-a-real-algorithm")
    assert exc_info.value.status_code == 400


def test_every_algorithm_has_display_info():
    for algorithm in algorithms.ALGORITHMS:
        info = algorithms.ALGORITHM_INFO[algorithm]
        assert info["label"]
        assert info["blurb"]


def test_default_algorithm_is_a_known_algorithm():
    assert algorithms.DEFAULT_ALGORITHM in algorithms.ALGORITHMS
