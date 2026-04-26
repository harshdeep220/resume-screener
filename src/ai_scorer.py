"""
ai_scorer.py — Gemini API integration for semantic resume scoring.

Makes a single Gemini API call per resume with a structured prompt.
Includes caching (keyed on JD + resume text hash), rate limiting,
exponential backoff, and graceful fallback on failure.

Uses the new google-genai SDK (replaces deprecated google-generativeai).
"""

import hashlib
import json
import logging
import re
import time
from pathlib import Path

from google import genai
from dotenv import load_dotenv
import os

logger = logging.getLogger(__name__)

# Load API key from .env
load_dotenv()

# Cache file path
_CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
_CACHE_FILE = _CACHE_DIR / "scores_cache.json"

# Default fallback score when API fails
_FALLBACK_SCORE = 5
_FALLBACK_RATIONALE = "[API error — manual review required]"

# Prompt template
_PROMPT_TEMPLATE = """You are an expert HR analyst. Given the job description and resume below,
score the candidate's relevance from 0 to 10 (integer) and provide a one-sentence rationale.

Return ONLY valid JSON in this exact format:
{{"score": <int 0-10>, "rationale": "<one sentence>"}}

--- JOB DESCRIPTION ---
{jd_text}

--- RESUME ---
{resume_text}"""


def _load_cache() -> dict:
    """Load the scores cache from disk.

    Returns:
        Dict mapping cache keys to score dicts.
    """
    if _CACHE_FILE.exists():
        try:
            with open(_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            logger.warning("Cache file corrupted — starting fresh.")
    return {}


def _save_cache(cache: dict) -> None:
    """Save the scores cache to disk.

    Args:
        cache: Dict mapping cache keys to score dicts.
    """
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def _make_cache_key(jd_text: str, resume_text: str) -> str:
    """Generate a cache key from JD + resume text (Fix #1).

    Both inputs are included in the hash so the same resume produces
    different scores against different JDs.

    Args:
        jd_text: Job description text.
        resume_text: Resume text.

    Returns:
        SHA-256 hex digest.
    """
    combined = jd_text + resume_text
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def truncate_at_sentence(text: str, max_chars: int) -> str:
    """Truncate text at the last sentence boundary within the limit.

    Per architecture Fix #6, truncation happens at the last period (.)
    within the character limit to avoid cutting mid-sentence.

    Args:
        text: Text to truncate.
        max_chars: Maximum character count.

    Returns:
        Truncated text.
    """
    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars]

    # Find the last sentence boundary
    last_period = truncated.rfind(".")
    if last_period > max_chars * 0.5:  # Only use if we keep at least half
        return truncated[: last_period + 1]

    return truncated


def _parse_ai_response(response_text: str) -> dict:
    """Parse the AI response into a score dict.

    Handles well-formed JSON and attempts to extract JSON from
    markdown code blocks or mixed text.

    Args:
        response_text: Raw response from the API.

    Returns:
        Dict with 'score' (int) and 'rationale' (str).
    """
    # Try direct JSON parse
    try:
        result = json.loads(response_text.strip())
        score = int(result.get("score", _FALLBACK_SCORE))
        rationale = str(result.get("rationale", _FALLBACK_RATIONALE))
        # Clamp score to valid range
        score = max(0, min(10, score))
        return {"score": score, "rationale": rationale}
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Try extracting JSON from markdown code blocks
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group(1))
            score = int(result.get("score", _FALLBACK_SCORE))
            rationale = str(result.get("rationale", _FALLBACK_RATIONALE))
            score = max(0, min(10, score))
            return {"score": score, "rationale": rationale}
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Try finding a JSON object anywhere in the text
    json_match = re.search(r"\{[^{}]*\"score\"[^{}]*\}", response_text)
    if json_match:
        try:
            result = json.loads(json_match.group(0))
            score = int(result.get("score", _FALLBACK_SCORE))
            rationale = str(result.get("rationale", _FALLBACK_RATIONALE))
            score = max(0, min(10, score))
            return {"score": score, "rationale": rationale}
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    logger.warning("Failed to parse AI response — using fallback score.")
    return {"score": _FALLBACK_SCORE, "rationale": _FALLBACK_RATIONALE}


def _create_client() -> genai.Client:
    """Create and return a configured Gemini API client.

    Returns:
        A google.genai.Client instance.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    return genai.Client(api_key=api_key)


def score_resume(
    jd_text: str,
    resume_text: str,
    model_name: str = "gemini-3.1-flash-lite-preview",
    jd_max_chars: int = 2000,
    resume_max_chars: int = 2000,
    api_delay: float = 2.0,
    max_retries: int = 3,
    _cache: dict | None = None,
    _client: genai.Client | None = None,
) -> dict:
    """Score a single resume against a JD using the Gemini API.

    Includes caching, rate limiting, and exponential backoff (Fix #3).

    Args:
        jd_text: Full JD text.
        resume_text: Full resume text.
        model_name: Gemini model to use.
        jd_max_chars: Max characters for JD in the prompt.
        resume_max_chars: Max characters for resume in the prompt.
        api_delay: Seconds to wait between API calls.
        max_retries: Maximum retry attempts on failure.
        _cache: Optional injected cache dict (for testing).
        _client: Optional injected genai.Client (for testing).

    Returns:
        Dict with 'score' (int 0–10) and 'rationale' (str).
    """
    # Check cache
    cache_key = _make_cache_key(jd_text, resume_text)
    cache = _cache if _cache is not None else _load_cache()

    if cache_key in cache:
        logger.debug("Cache hit — skipping API call.")
        return cache[cache_key]

    # Truncate texts for the prompt (Fix #6)
    jd_truncated = truncate_at_sentence(jd_text, jd_max_chars)
    resume_truncated = truncate_at_sentence(resume_text, resume_max_chars)

    prompt = _PROMPT_TEMPLATE.format(
        jd_text=jd_truncated, resume_text=resume_truncated
    )

    # Configure API client
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "your_key_here":
        logger.error("GOOGLE_API_KEY not set — using fallback score.")
        return {"score": _FALLBACK_SCORE, "rationale": "[No API key configured]"}

    client = _client if _client is not None else genai.Client(api_key=api_key)

    # Retry with exponential backoff (Fix #3)
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            result = _parse_ai_response(response.text)

            # Save to cache
            cache[cache_key] = result
            if _cache is None:
                _save_cache(cache)

            return result

        except Exception as e:
            wait_time = api_delay * (2 ** attempt)
            logger.warning(
                "API call failed (attempt %d/%d): %s — retrying in %.1fs",
                attempt + 1,
                max_retries,
                e,
                wait_time,
            )
            time.sleep(wait_time)

    # All retries exhausted — fallback
    logger.error("All %d API retries failed — using neutral fallback score.", max_retries)
    fallback = {"score": _FALLBACK_SCORE, "rationale": _FALLBACK_RATIONALE}

    # Cache the fallback too so we don't keep retrying
    cache[cache_key] = fallback
    if _cache is None:
        _save_cache(cache)

    return fallback


def score_resumes_batch(
    jd_text: str,
    resume_texts: list[str],
    model_name: str = "gemini-3.1-flash-lite-preview",
    jd_max_chars: int = 2000,
    resume_max_chars: int = 2000,
    api_delay: float = 2.0,
    max_retries: int = 3,
) -> list[dict]:
    """Score multiple resumes against a JD.

    Processes sequentially with a configurable delay between calls
    to respect rate limits.

    Args:
        jd_text: Full JD text.
        resume_texts: List of full resume texts.
        model_name: Gemini model to use.
        jd_max_chars: Max characters for JD in the prompt.
        resume_max_chars: Max characters for resume in the prompt.
        api_delay: Seconds to wait between API calls.
        max_retries: Maximum retry attempts per call.

    Returns:
        List of score dicts, one per resume.
    """
    cache = _load_cache()

    # Create a single client for the batch
    api_key = os.getenv("GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key) if api_key else None

    results = []

    for i, resume_text in enumerate(resume_texts):
        result = score_resume(
            jd_text=jd_text,
            resume_text=resume_text,
            model_name=model_name,
            jd_max_chars=jd_max_chars,
            resume_max_chars=resume_max_chars,
            api_delay=api_delay,
            max_retries=max_retries,
            _cache=cache,
            _client=client,
        )
        results.append(result)

        # Rate-limit delay between calls (skip after the last one)
        if i < len(resume_texts) - 1:
            time.sleep(api_delay)

    # Save cache once after the full batch
    _save_cache(cache)

    return results
