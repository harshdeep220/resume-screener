"""
test_ai_scorer.py — Unit tests for the AI scoring module.
"""

from unittest.mock import patch, MagicMock

import pytest

from src.ai_scorer import (
    _make_cache_key,
    _parse_ai_response,
    truncate_at_sentence,
    score_resume,
)


# ── Cache key tests ──────────────────────────────────────────────────


def test_cache_key_includes_jd():
    """Same resume text, different JD text should produce different cache keys."""
    resume = "Python developer with Django experience"
    key1 = _make_cache_key("Backend Python developer", resume)
    key2 = _make_cache_key("Frontend React developer", resume)

    assert key1 != key2


def test_cache_key_deterministic():
    """Same inputs should always produce the same cache key."""
    jd = "Python developer"
    resume = "Django expert"

    key1 = _make_cache_key(jd, resume)
    key2 = _make_cache_key(jd, resume)

    assert key1 == key2


# ── Response parsing tests ───────────────────────────────────────────


def test_parse_valid_json_response():
    """Well-formed JSON should be parsed correctly."""
    response = '{"score": 8, "rationale": "Strong Python and Django match."}'
    result = _parse_ai_response(response)

    assert result["score"] == 8
    assert result["rationale"] == "Strong Python and Django match."


def test_parse_json_in_code_block():
    """JSON inside markdown code blocks should be extracted."""
    response = '```json\n{"score": 7, "rationale": "Good fit."}\n```'
    result = _parse_ai_response(response)

    assert result["score"] == 7
    assert result["rationale"] == "Good fit."


def test_parse_malformed_response():
    """Non-JSON response should return fallback score=5."""
    result = _parse_ai_response("I think this candidate is pretty good, maybe a 7.")

    assert result["score"] == 5
    assert "error" in result["rationale"].lower() or "API" in result["rationale"]


def test_parse_score_clamping():
    """Scores outside 0–10 should be clamped."""
    result = _parse_ai_response('{"score": 15, "rationale": "Amazing!"}')
    assert result["score"] == 10

    result = _parse_ai_response('{"score": -3, "rationale": "Terrible!"}')
    assert result["score"] == 0


# ── Truncation tests ────────────────────────────────────────────────


def test_text_truncation_within_limit():
    """Short text should not be truncated."""
    text = "This is a short text."
    assert truncate_at_sentence(text, 100) == text


def test_text_truncation_at_sentence_boundary():
    """Long text should be truncated at the last sentence boundary."""
    text = "First sentence. Second sentence. Third sentence is very long and goes on."
    result = truncate_at_sentence(text, 40)

    assert len(result) <= 40
    assert result.endswith(".")


def test_text_truncation_no_period():
    """Text without periods should be hard-truncated at max_chars."""
    text = "A" * 5000
    result = truncate_at_sentence(text, 2000)
    assert len(result) == 2000


# ── Cache hit test ───────────────────────────────────────────────────


def test_cache_hit_skips_api():
    """If the cache contains the key, the API should not be called."""
    jd = "Python developer"
    resume = "Django expert with Python skills"
    cache_key = _make_cache_key(jd, resume)

    cache = {
        cache_key: {"score": 9, "rationale": "Excellent match from cache."}
    }

    result = score_resume(jd, resume, _cache=cache)

    assert result["score"] == 9
    assert result["rationale"] == "Excellent match from cache."


# ── API retry test ───────────────────────────────────────────────────


@patch("src.ai_scorer.os.getenv", return_value="fake-api-key")
def test_api_retry_on_failure(mock_getenv):
    """API should retry on failure and succeed on the 3rd attempt."""
    mock_client = MagicMock()

    # Fail twice, succeed on third
    mock_client.models.generate_content.side_effect = [
        Exception("Rate limit"),
        Exception("Timeout"),
        MagicMock(text='{"score": 7, "rationale": "Good candidate."}'),
    ]

    result = score_resume(
        "Python developer",
        "Django expert with 5 years of experience",
        api_delay=0.01,  # fast for tests
        max_retries=3,
        _cache={},
        _client=mock_client,
    )

    assert result["score"] == 7
    assert mock_client.models.generate_content.call_count == 3
