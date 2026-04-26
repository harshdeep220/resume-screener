"""
resume_parser.py — Parse resume text into a structured ResumeProfile.

Extracts candidate name, skills (via taxonomy matching), sections
(via header detection), keywords, and preserves raw text.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from src.jd_parser import SKILLS_TAXONOMY, _extract_skills, _extract_keywords

logger = logging.getLogger(__name__)

# Standard section headers found in resumes (case-insensitive matching)
_SECTION_HEADERS = [
    "summary",
    "objective",
    "experience",
    "work experience",
    "professional experience",
    "employment history",
    "education",
    "skills",
    "technical skills",
    "projects",
    "certifications",
    "certificates",
    "awards",
    "publications",
    "languages",
    "interests",
    "hobbies",
    "references",
    "achievements",
    "volunteer",
    "training",
]


@dataclass
class ResumeProfile:
    """Structured representation of a resume."""

    filename: str = ""
    candidate_name: str = ""
    skills: set = field(default_factory=set)
    keywords: list = field(default_factory=list)
    sections: dict = field(default_factory=dict)
    raw_text: str = ""


def _extract_candidate_name(text: str) -> str:
    """Extract candidate name using the first non-empty line heuristic.

    Args:
        text: Cleaned resume text.

    Returns:
        The first non-empty line as the probable candidate name.
    """
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _detect_sections(text: str) -> dict[str, str]:
    """Detect and extract resume sections by matching header keywords.

    Scans each line for known section headers. Content between two headers
    is assigned to the preceding header. If no headers are found, returns
    an empty dict (the caller should use full raw_text as fallback).

    Args:
        text: Cleaned resume text.

    Returns:
        Dict mapping lowercase section names to their text content.
    """
    lines = text.split("\n")
    sections = {}
    current_section = None
    current_lines = []

    # Build a regex pattern for section header detection
    header_pattern = re.compile(
        r"^\s*(?:" + "|".join(re.escape(h) for h in _SECTION_HEADERS) + r")\s*:?\s*$",
        re.IGNORECASE,
    )

    for line in lines:
        stripped = line.strip()

        # Check if this line is a section header
        if header_pattern.match(stripped):
            # Save the previous section
            if current_section is not None:
                sections[current_section] = "\n".join(current_lines).strip()

            # Start a new section
            current_section = stripped.lower().rstrip(":")
            current_lines = []
        elif current_section is not None:
            current_lines.append(line)

    # Save the last section
    if current_section is not None:
        sections[current_section] = "\n".join(current_lines).strip()

    return sections


def parse_resume(text: str, filename: str = "") -> ResumeProfile:
    """Parse a resume text into a ResumeProfile.

    Section detection is opportunistic — if no known headers are found,
    the full text is used as a single block. This is a deliberate fallback,
    not a failure mode (per architecture Fix #4).

    Args:
        text: Cleaned resume text.
        filename: Original filename (for identification).

    Returns:
        A ResumeProfile dataclass.
    """
    sections = _detect_sections(text)

    if not sections:
        logger.debug(
            "[%s] No standard section headers found — using full text.",
            filename or "unknown",
        )

    return ResumeProfile(
        filename=filename,
        candidate_name=_extract_candidate_name(text),
        skills=_extract_skills(text),
        keywords=_extract_keywords(text),
        sections=sections,
        raw_text=text,
    )
