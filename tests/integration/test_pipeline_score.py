"""
test_pipeline_score.py — Integration tests for AI scoring pipeline (mocked API).
"""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.extractor import extract_text
from src.jd_parser import parse_jd
from src.resume_parser import parse_resume
from src.ai_scorer import score_resume, _make_cache_key


FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


@patch("src.ai_scorer.os.getenv", return_value="fake-api-key")
def test_single_resume_ai_scoring(mock_getenv):
    """Mock Gemini client; feed 1 JD + 1 resume; assert full {score, rationale} dict."""
    # Mock the client
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = MagicMock(
        text='{"score": 8, "rationale": "Strong Python and API skills match the JD well."}'
    )

    # Load fixture data
    _, jd_text = extract_text(FIXTURES_DIR / "sample_jd.txt")
    _, resume_text = extract_text(FIXTURES_DIR / "resumes" / "alice_chen.txt")

    result = score_resume(
        jd_text=jd_text,
        resume_text=resume_text,
        api_delay=0.01,
        _cache={},
        _client=mock_client,
    )

    assert result["score"] == 8
    assert "Python" in result["rationale"] or "match" in result["rationale"].lower()
    assert mock_client.models.generate_content.call_count == 1


@patch("src.ai_scorer.os.getenv", return_value="fake-api-key")
def test_cache_written_after_call(mock_getenv):
    """Cache dict should contain the result after a successful API call."""
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = MagicMock(
        text='{"score": 6, "rationale": "Decent match."}'
    )

    _, jd_text = extract_text(FIXTURES_DIR / "sample_jd.txt")
    _, resume_text = extract_text(FIXTURES_DIR / "resumes" / "bob_martinez.txt")

    cache = {}
    result = score_resume(
        jd_text=jd_text,
        resume_text=resume_text,
        api_delay=0.01,
        _cache=cache,
        _client=mock_client,
    )

    # Verify cache was populated
    cache_key = _make_cache_key(jd_text, resume_text)
    assert cache_key in cache
    assert cache[cache_key]["score"] == 6
