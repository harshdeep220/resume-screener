"""
test_parsers.py — Unit tests for JD and Resume parsers.
"""

from pathlib import Path

import pytest

from src.extractor import extract_text
from src.jd_parser import parse_jd
from src.resume_parser import parse_resume


# ── Fixtures ─────────────────────────────────────────────────────────

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


@pytest.fixture
def jd_text():
    """Load and extract text from the sample JD fixture."""
    _, text = extract_text(FIXTURES_DIR / "sample_jd.txt")
    return text


@pytest.fixture
def alice_text():
    """Load and extract text from Alice's resume."""
    _, text = extract_text(FIXTURES_DIR / "resumes" / "alice_chen.txt")
    return text


@pytest.fixture
def carol_text():
    """Load and extract text from Carol's resume (no section headers match)."""
    _, text = extract_text(FIXTURES_DIR / "resumes" / "carol_williams.txt")
    return text


# ── JD Parser tests ─────────────────────────────────────────────────


def test_jd_parser_extracts_known_skills(jd_text):
    """JD with explicit skill requirements should have them in required_skills."""
    profile = parse_jd(jd_text)

    # These skills are explicitly listed in the sample JD
    expected_skills = {"python", "docker", "kubernetes", "pytest", "git"}
    for skill in expected_skills:
        assert skill in profile.required_skills, f"Missing skill: {skill}"


def test_jd_parser_extracts_title(jd_text):
    """First non-empty line should be extracted as the title."""
    profile = parse_jd(jd_text)
    assert profile.title  # non-empty
    assert "Python" in profile.title or "Senior" in profile.title


def test_jd_parser_has_keywords(jd_text):
    """Keywords list should be non-empty after lemmatisation."""
    profile = parse_jd(jd_text)
    assert len(profile.keywords) > 0


def test_jd_parser_preserves_raw_text(jd_text):
    """raw_text should be the full cleaned text."""
    profile = parse_jd(jd_text)
    assert profile.raw_text == jd_text


def test_jd_parser_unknown_jd():
    """JD with no taxonomy matches should return empty required_skills, not error."""
    profile = parse_jd("Looking for a passionate individual excited about quantum feng shui.")
    assert isinstance(profile.required_skills, set)
    # May be empty or have incidental matches — key is no exception


def test_jd_parser_empty_text():
    """Empty JD should produce a valid but empty profile."""
    profile = parse_jd("")
    assert profile.title == ""
    assert profile.required_skills == set()
    assert profile.keywords == []
    assert profile.raw_text == ""


# ── Resume Parser tests ─────────────────────────────────────────────


def test_resume_parser_sections(alice_text):
    """Resume with standard headers should have sections detected."""
    profile = parse_resume(alice_text, "alice_chen.txt")

    # Alice's resume has Summary, Experience, Education, Skills headers
    assert len(profile.sections) > 0
    section_keys = {k.lower() for k in profile.sections.keys()}
    assert "experience" in section_keys or "skills" in section_keys


def test_resume_parser_no_headers():
    """Resume with no standard headers should not raise an exception."""
    text = "John Doe\njohn@email.com\nI am a developer who knows Python and Django."
    profile = parse_resume(text, "no_headers.txt")

    assert profile.filename == "no_headers.txt"
    assert profile.raw_text == text
    assert isinstance(profile.sections, dict)
    # sections may be empty — that's fine, raw_text is used


def test_candidate_name_extraction(alice_text):
    """First non-empty line should be the candidate name."""
    profile = parse_resume(alice_text, "alice_chen.txt")
    assert profile.candidate_name == "Alice Chen"


def test_candidate_name_extraction_varied():
    """Name extraction should work for various formats."""
    text = "Bob Martinez\nEmail: bob@mail.com\nSummary:\nDeveloper."
    profile = parse_resume(text, "bob.txt")
    assert profile.candidate_name == "Bob Martinez"


def test_resume_parser_extracts_skills(alice_text):
    """Resume parser should find skills using the taxonomy."""
    profile = parse_resume(alice_text, "alice_chen.txt")

    expected = {"python", "django", "docker"}
    for skill in expected:
        assert skill in profile.skills, f"Missing skill: {skill}"


def test_resume_parser_has_keywords(alice_text):
    """Resume keywords should be non-empty."""
    profile = parse_resume(alice_text, "alice_chen.txt")
    assert len(profile.keywords) > 0


def test_resume_parser_empty_text():
    """Empty resume text should produce a valid but empty profile."""
    profile = parse_resume("", "empty.txt")
    assert profile.filename == "empty.txt"
    assert profile.candidate_name == ""
    assert profile.skills == set()
