"""
nlp_engine.py — Deterministic NLP scoring without API calls.

Computes two sub-scores:
1. Skill set overlap (Jaccard similarity, normalised 0–10)
2. TF-IDF cosine similarity (fitted on full corpus, normalised 0–10)

Final nlp_score = 0.5 × skill_overlap_score + 0.5 × tfidf_score
"""

import logging
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class NLPResult:
    """Result of NLP scoring for a single resume."""

    skill_overlap_score: float  # 0.0 – 10.0
    tfidf_score: float          # 0.0 – 10.0
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


def compute_tfidf_scores(
    jd_text: str,
    resume_texts: list[str],
    max_features: int = 500,
) -> list[float]:
    """Compute TF-IDF cosine similarity between JD and each resume.

    The vectoriser is fit once on the combined corpus (JD + all resumes)
    so that IDF weights are meaningful across the batch (Fix #7).

    Args:
        jd_text: Cleaned JD text.
        resume_texts: List of cleaned resume texts.
        max_features: Maximum number of TF-IDF features.

    Returns:
        List of cosine similarity scores (0.0–10.0), one per resume.
    """
    if not resume_texts:
        return []

    # Fit on full corpus: JD at index 0, resumes at indices 1..N
    corpus = [jd_text] + resume_texts

    vectoriser = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        sublinear_tf=True,
    )

    try:
        tfidf_matrix = vectoriser.fit_transform(corpus)
    except ValueError:
        # Empty corpus or all stop words
        logger.warning("TF-IDF vectorisation failed — returning neutral scores.")
        return [5.0] * len(resume_texts)

    # JD vector is row 0; resume vectors are rows 1..N
    jd_vector = tfidf_matrix[0:1]
    resume_vectors = tfidf_matrix[1:]

    similarities = cosine_similarity(jd_vector, resume_vectors).flatten()

    # Normalise to 0–10
    scores = [round(sim * 10.0, 2) for sim in similarities]

    return scores


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
    # TF-IDF scores (computed in batch)
    tfidf_scores = compute_tfidf_scores(jd_text, resume_texts, max_features)

    results = []
    for i, (resume_skills, tfidf_score) in enumerate(
        zip(resume_skills_list, tfidf_scores)
    ):
        skill_score, matches, gaps = compute_skill_overlap(jd_skills, resume_skills)

        # Blend: 50% skill overlap + 50% TF-IDF
        nlp_score = round(0.5 * skill_score + 0.5 * tfidf_score, 2)

        results.append(
            NLPResult(
                skill_overlap_score=round(skill_score, 2),
                tfidf_score=round(tfidf_score, 2),
                nlp_score=nlp_score,
                skill_matches=matches,
                skill_gaps=gaps,
            )
        )

    return results
