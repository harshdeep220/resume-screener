"""
test_extractor.py — Unit tests for the text extraction module.
"""

import logging
from pathlib import Path

import pytest

from src.extractor import clean_text, extract_text


# ── Fixtures ─────────────────────────────────────────────────────────

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
RESUMES_DIR = FIXTURES_DIR / "resumes"


# ── clean_text tests ─────────────────────────────────────────────────


def test_clean_text_strips_noise():
    """Control characters and excessive whitespace should be removed."""
    dirty = "Hello\x00\x01 World\x0b\x0c  \t\t  foo  \n\n\n\n\nbar"
    result = clean_text(dirty)

    assert "\x00" not in result
    assert "\x01" not in result
    assert "\x0b" not in result
    assert "  " not in result  # no double spaces
    assert "\n\n\n" not in result  # max 2 consecutive newlines
    assert "Hello" in result
    assert "World" in result
    assert "foo" in result
    assert "bar" in result


def test_clean_text_preserves_structure():
    """Single newlines and normal spacing should be preserved."""
    text = "Line one\nLine two\nLine three"
    result = clean_text(text)
    assert result == "Line one\nLine two\nLine three"


def test_clean_text_unicode_normalization():
    """Unicode should be normalised to NFC form."""
    # é as combining characters (NFD)
    nfd = "re\u0301sume\u0301"
    result = clean_text(nfd)
    # After NFC normalisation, it should be precomposed
    assert "r\u00e9sum\u00e9" in result


def test_clean_text_empty_string():
    """Empty input should return empty output."""
    assert clean_text("") == ""


# ── TXT extraction tests ────────────────────────────────────────────


def test_txt_extraction():
    """Plain text files should be extracted and cleaned."""
    txt_file = RESUMES_DIR / "alice_chen.txt"
    filename, text = extract_text(txt_file)

    assert filename == "alice_chen.txt"
    assert text  # non-empty
    assert "Alice Chen" in text
    assert "Python" in text
    assert "Django" in text


def test_txt_extraction_roundtrip():
    """Extracted text should contain all key content from the source."""
    txt_file = RESUMES_DIR / "bob_martinez.txt"
    filename, text = extract_text(txt_file)

    assert filename == "bob_martinez.txt"
    assert "Bob Martinez" in text
    assert "Python" in text
    assert "React" in text


# ── Unsupported format test ──────────────────────────────────────────


def test_unsupported_format_returns_empty(tmp_path):
    """Unsupported file formats should return empty text with a warning."""
    unsupported = tmp_path / "resume.xlsx"
    unsupported.write_text("some data")

    filename, text = extract_text(unsupported)

    assert filename == "resume.xlsx"
    assert text == ""


# ── Scanned PDF detection ───────────────────────────────────────────


def test_short_text_warning(tmp_path, caplog):
    """Files with very little text should trigger a warning."""
    short_file = tmp_path / "short.txt"
    short_file.write_text("Hi")  # only 2 chars, below _MIN_TEXT_LENGTH

    with caplog.at_level(logging.WARNING):
        filename, text = extract_text(short_file)

    assert filename == "short.txt"
    assert text == ""  # too short — treated as empty
    assert "too short" in caplog.text or "appears to be" in caplog.text


# ── Missing file handling ───────────────────────────────────────────


def test_missing_file_returns_empty(tmp_path):
    """Non-existent files should return empty text without crashing."""
    fake = tmp_path / "does_not_exist.txt"
    filename, text = extract_text(fake)

    assert filename == "does_not_exist.txt"
    assert text == ""
