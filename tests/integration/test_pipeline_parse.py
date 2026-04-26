"""
test_pipeline_parse.py — Integration tests for text extraction + parsing pipeline.
"""

from pathlib import Path

import pytest

from src.extractor import extract_text
from src.jd_parser import parse_jd
from src.resume_parser import parse_resume
from src.nlp_engine import score_resumes


FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
RESUMES_DIR = FIXTURES_DIR / "resumes"


def test_full_extraction_batch():
    """Load all fixture resumes — assert 10 (filename, text) tuples with no exceptions."""
    resume_files = sorted(RESUMES_DIR.glob("*.txt"))
    assert len(resume_files) == 10

    results = []
    for rf in resume_files:
        filename, text = extract_text(rf)
        results.append((filename, text))

    assert len(results) == 10
    for filename, text in results:
        assert filename  # non-empty filename
        assert text  # non-empty text (all fixtures are valid)


def test_full_parse_pipeline():
    """Full batch: 1 JD + 10 resumes → all parsed profiles + NLP scores."""
    # Extract JD
    _, jd_text = extract_text(FIXTURES_DIR / "sample_jd.txt")
    assert jd_text

    # Parse JD
    jd_profile = parse_jd(jd_text)
    assert jd_profile.title
    assert len(jd_profile.required_skills) > 0

    # Extract and parse resumes
    resume_files = sorted(RESUMES_DIR.glob("*.txt"))
    resume_profiles = []
    for rf in resume_files:
        filename, text = extract_text(rf)
        assert text, f"Extraction failed for {filename}"
        profile = parse_resume(text, filename)
        resume_profiles.append(profile)

    assert len(resume_profiles) == 10

    # Verify all profiles have required fields
    for rp in resume_profiles:
        assert rp.filename
        assert rp.candidate_name
        assert rp.raw_text

    # Compute NLP scores
    nlp_results = score_resumes(
        jd_skills=jd_profile.required_skills,
        jd_text=jd_profile.raw_text,
        resume_skills_list=[rp.skills for rp in resume_profiles],
        resume_texts=[rp.raw_text for rp in resume_profiles],
    )

    assert len(nlp_results) == 10

    for nr in nlp_results:
        assert 0.0 <= nr.nlp_score <= 10.0


def test_ranking_matches_expectations():
    """Top-scoring candidates should be Python-heavy resumes, not frontend/marketing."""
    # Extract JD
    _, jd_text = extract_text(FIXTURES_DIR / "sample_jd.txt")
    jd_profile = parse_jd(jd_text)

    # Extract and parse resumes
    resume_files = sorted(RESUMES_DIR.glob("*.txt"))
    profiles = []
    for rf in resume_files:
        filename, text = extract_text(rf)
        profiles.append(parse_resume(text, filename))

    # NLP score
    results = score_resumes(
        jd_skills=jd_profile.required_skills,
        jd_text=jd_profile.raw_text,
        resume_skills_list=[rp.skills for rp in profiles],
        resume_texts=[rp.raw_text for rp in profiles],
    )

    # Pair filenames with scores and sort
    scored = sorted(
        zip([rp.filename for rp in profiles], results),
        key=lambda x: x[1].nlp_score,
        reverse=True,
    )

    # The top candidate by NLP should be one of the Python-heavy resumes
    top_filenames = [s[0] for s in scored[:3]]
    python_resumes = {"alice_chen.txt", "iris_nakamura.txt", "grace_liu.txt", "bob_martinez.txt"}

    # At least 2 of the top 3 should be Python-heavy
    overlap = set(top_filenames) & python_resumes
    assert len(overlap) >= 2, f"Top 3 NLP: {top_filenames}, expected overlap with {python_resumes}"
