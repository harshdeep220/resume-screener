"""
scoring_engine.py — Weighted score blending and candidate ranking.

Blends NLP and AI scores with configurable weights, sorts candidates
by final score (descending), with alphabetical tiebreaking.
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CandidateResult:
    """Final scored result for one candidate."""

    filename: str = ""
    candidate_name: str = ""
    nlp_score: float = 0.0       # 0.0 – 10.0
    ai_score: float = 0.0        # 0.0 – 10.0
    final_score: float = 0.0     # 0.0 – 10.0
    rationale: str = ""
    skill_matches: set = field(default_factory=set)
    skill_gaps: set = field(default_factory=set)


def compute_final_scores(
    filenames: list[str],
    candidate_names: list[str],
    nlp_scores: list[float],
    ai_scores: list[float],
    ai_rationales: list[str],
    skill_matches_list: list[set],
    skill_gaps_list: list[set],
    nlp_weight: float = 0.4,
    ai_weight: float = 0.6,
) -> list[CandidateResult]:
    """Compute weighted final scores and return a ranked list.

    Args:
        filenames: List of resume filenames.
        candidate_names: List of extracted candidate names.
        nlp_scores: List of NLP scores (0–10).
        ai_scores: List of AI scores (0–10).
        ai_rationales: List of AI rationale strings.
        skill_matches_list: List of matched skill sets.
        skill_gaps_list: List of gap skill sets.
        nlp_weight: Weight for NLP score (default 0.4).
        ai_weight: Weight for AI score (default 0.6).

    Returns:
        List of CandidateResult, sorted by final_score descending.
        Ties are broken alphabetically by filename.
    """
    results = []

    for i in range(len(filenames)):
        final = round(
            nlp_weight * nlp_scores[i] + ai_weight * ai_scores[i], 2
        )

        results.append(
            CandidateResult(
                filename=filenames[i],
                candidate_name=candidate_names[i],
                nlp_score=round(nlp_scores[i], 2),
                ai_score=round(ai_scores[i], 2),
                final_score=final,
                rationale=ai_rationales[i],
                skill_matches=skill_matches_list[i],
                skill_gaps=skill_gaps_list[i],
            )
        )

    # Sort: descending by final_score, then alphabetical by filename for ties
    results.sort(key=lambda r: (-r.final_score, r.filename))

    return results
