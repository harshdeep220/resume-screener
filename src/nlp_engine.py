"""
nlp_engine.py — Deterministic NLP scoring without API calls.

Computes two sub-scores:
1. Skill set overlap (Recall similarity, normalised 0–10)
2. BM25 text retrieval score (batch-normalized 0–10)

Final nlp_score = 0.5 × skill_overlap_score + 0.5 × bm25_score
"""

import logging
from dataclasses import dataclass
import re

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


@dataclass
class NLPResult:
    """Result of NLP scoring for a single resume."""

    skill_overlap_score: float  # 0.0 – 10.0
    bm25_score: float           # 0.0 – 10.0
    nlp_score: float            # 0.0 – 10.0  (blended)
    skill_matches: set          # skills found in both JD and resume
    skill_gaps: set             # JD skills not found in resume


def compute_skill_overlap(jd_skills: set, resume_skills: set) -> tuple[float, set, set]:
    """Compute Jaccard similarity between JD and resume skill sets.

    Normalised to 0–10 scale per architecture Fix #2.

    Args:
        jd_skills: Set of required skills from the JD.
        resume_skills: Set of skills found in the resume.

    Returns:
        Tuple of (score_0_to_10, matched_skills, gap_skills).
    """
    if not jd_skills:
        # No skills to match against — return neutral score
        return 5.0, set(), set()

    matches = jd_skills & resume_skills
    gaps = jd_skills - resume_skills

    # Calculate Recall metric (matches / required skills) instead of Jaccard
    # This prevents penalizing candidates for having extra skills.
    recall = len(matches) / len(jd_skills)

    # Normalise to 0–10
    score = recall * 10.0

    return score, matches, gaps


def compute_bm25_scores(
    jd_text: str,
    resume_texts: list[str],
) -> list[float]:
    """Compute BM25 scores for resumes using the JD as a query.

    BM25 handles document length normalization better than TF-IDF Cosine Similarity,
    preventing large resumes from being unfairly penalized.
    The raw scores are min-max normalized to a 0.0 - 10.0 scale across the batch.

    Args:
        jd_text: Cleaned JD text.
        resume_texts: List of cleaned resume texts.

    Returns:
        List of normalized BM25 scores (0.0–10.0), one per resume.
    """
    if not resume_texts:
        return []

    # Simple whitespace/punctuation tokenization
    def tokenize(text):
        return re.findall(r'\b\w+\b', text.lower())

    tokenized_corpus = [tokenize(doc) for doc in resume_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = tokenize(jd_text)
    raw_scores = bm25.get_scores(tokenized_query)

    # Min-max normalization across the batch
    if len(raw_scores) == 0:
        return []
        
    min_score = min(raw_scores)
    max_score = max(raw_scores)

    if max_score == min_score:
        # Avoid division by zero if all scores are identical
        return [10.0] * len(raw_scores)

    # Normalize to 0-10
    normalized_scores = []
    for score in raw_scores:
        norm = ((score - min_score) / (max_score - min_score)) * 10.0
        normalized_scores.append(round(norm, 2))

    return normalized_scores


def score_resumes(
    jd_skills: set,
    jd_text: str,
    resume_skills_list: list[set],
    resume_texts: list[str],
    max_features: int = 500,
) -> list[NLPResult]:
    """Score all resumes against a JD using NLP methods.

    Args:
        jd_skills: Set of required skills from the JD.
        jd_text: Cleaned JD text.
        resume_skills_list: List of skill sets, one per resume.
        resume_texts: List of cleaned resume texts.
        max_features: TF-IDF max features.

    Returns:
        List of NLPResult objects, one per resume.
    """
    # BM25 scores (computed in batch)
    bm25_scores = compute_bm25_scores(jd_text, resume_texts)

    results = []
    for i, (resume_skills, bm25_score) in enumerate(
        zip(resume_skills_list, bm25_scores)
    ):
        skill_score, matches, gaps = compute_skill_overlap(jd_skills, resume_skills)

        # Blend: 50% skill overlap + 50% BM25
        nlp_score = round(0.5 * skill_score + 0.5 * bm25_score, 2)

        results.append(
            NLPResult(
                skill_overlap_score=round(skill_score, 2),
                bm25_score=round(bm25_score, 2),
                nlp_score=nlp_score,
                skill_matches=matches,
                skill_gaps=gaps,
            )
        )

    return results
