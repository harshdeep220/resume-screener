"""
jd_parser.py — Parse job description text into a structured JDProfile.

Extracts required skills (via taxonomy matching), keywords (via spaCy
lemmatisation), and preserves the raw cleaned text for downstream AI scoring.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import spacy

logger = logging.getLogger(__name__)

# Load spaCy model (small English model — tokeniser + lemmatiser only)
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# Path to the skills taxonomy file
_TAXONOMY_PATH = Path(__file__).resolve().parent.parent / "data" / "skills_taxonomy.json"


def _load_taxonomy() -> set[str]:
    """Load the skills taxonomy from the JSON file.

    Returns:
        A set of lowercase skill strings.
    """
    try:
        with open(_TAXONOMY_PATH, "r", encoding="utf-8") as f:
            skills = json.load(f)
        return {s.lower().strip() for s in skills if isinstance(s, str)}
    except FileNotFoundError:
        logger.error("Skills taxonomy not found at %s", _TAXONOMY_PATH)
        return set()
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in skills taxonomy: %s", e)
        return set()


# Load taxonomy once at module level
SKILLS_TAXONOMY = _load_taxonomy()


@dataclass
class JDProfile:
    """Structured representation of a job description."""

    title: str = ""
    required_skills: set = field(default_factory=set)
    keywords: list = field(default_factory=list)
    raw_text: str = ""


def _extract_title(text: str) -> str:
    """Extract job title from the first non-empty line (H1 heuristic).

    Args:
        text: Cleaned JD text.

    Returns:
        The first non-empty line as the probable title.
    """
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _extract_skills(text: str) -> set[str]:
    """Match tokens in the text against the skills taxonomy.

    Uses spaCy lemmatisation for normalisation. Multi-word skills are
    matched by checking bigrams and trigrams against the taxonomy.

    Args:
        text: Cleaned text to extract skills from.

    Returns:
        Set of matched skill strings (lowercase).
    """
    doc = nlp(text.lower())
    found_skills = set()

    # Single-token matching (lemmatised + raw form)
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        # Check both lemma and raw lowercase text to handle cases where
        # spaCy's lemmatiser produces incorrect forms (e.g. "kubernetes" → "kubernete")
        if token.lemma_ in SKILLS_TAXONOMY:
            found_skills.add(token.lemma_)
        if token.text in SKILLS_TAXONOMY:
            found_skills.add(token.text)

    # Multi-word matching (bigrams and trigrams on raw lowered text)
    words = text.lower().split()
    for n in (2, 3):
        for i in range(len(words) - n + 1):
            ngram = " ".join(words[i : i + n])
            if ngram in SKILLS_TAXONOMY:
                found_skills.add(ngram)

    return found_skills


def _extract_keywords(text: str) -> list[str]:
    """Extract meaningful lemmatised tokens from text.

    Filters out stop words, punctuation, and very short tokens.

    Args:
        text: Cleaned text.

    Returns:
        List of lemmatised keyword strings.
    """
    doc = nlp(text.lower())
    keywords = []
    for token in doc:
        if (
            not token.is_stop
            and not token.is_punct
            and not token.is_space
            and len(token.lemma_) > 1
        ):
            keywords.append(token.lemma_)
    return keywords


def parse_jd(text: str) -> JDProfile:
    """Parse a job description text into a JDProfile.

    Args:
        text: Cleaned job description text.

    Returns:
        A JDProfile dataclass with title, required_skills, keywords, and raw_text.
    """
    return JDProfile(
        title=_extract_title(text),
        required_skills=_extract_skills(text),
        keywords=_extract_keywords(text),
        raw_text=text,
    )
