"""
test_scoring_engine.py — Unit tests for the scoring engine.
"""

import pytest

from src.scoring_engine import compute_final_scores


# ── Weighted blend tests ─────────────────────────────────────────────


def test_weighted_blend_correct():
    """Final score should be the correct weighted blend of NLP and AI scores."""
    results = compute_final_scores(
        filenames=["a.txt"],
        candidate_names=["Alice"],
        nlp_scores=[6.0],
        ai_scores=[8.0],
        ai_rationales=["Good fit."],
        skill_matches_list=[{"python"}],
        skill_gaps_list=[{"docker"}],
        nlp_weight=0.4,
        ai_weight=0.6,
    )

    assert len(results) == 1
    # 0.4 * 6.0 + 0.6 * 8.0 = 2.4 + 4.8 = 7.2
    assert results[0].final_score == 7.2


def test_weighted_blend_custom_weights():
    """Custom weights should produce the correct final score."""
    results = compute_final_scores(
        filenames=["a.txt"],
        candidate_names=["Alice"],
        nlp_scores=[5.0],
        ai_scores=[10.0],
        ai_rationales=["Excellent."],
        skill_matches_list=[set()],
        skill_gaps_list=[set()],
        nlp_weight=0.3,
        ai_weight=0.7,
    )

    # 0.3 * 5.0 + 0.7 * 10.0 = 1.5 + 7.0 = 8.5
    assert results[0].final_score == 8.5


# ── Sorting tests ───────────────────────────────────────────────────


def test_sort_order():
    """Results should be sorted descending by final_score."""
    results = compute_final_scores(
        filenames=["c.txt", "a.txt", "e.txt", "b.txt", "d.txt"],
        candidate_names=["C", "A", "E", "B", "D"],
        nlp_scores=[2.0, 8.0, 4.0, 6.0, 10.0],
        ai_scores=[3.0, 9.0, 5.0, 7.0, 1.0],
        ai_rationales=[""] * 5,
        skill_matches_list=[set()] * 5,
        skill_gaps_list=[set()] * 5,
        nlp_weight=0.4,
        ai_weight=0.6,
    )

    scores = [r.final_score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_tie_breaking():
    """Candidates with identical final_score should be ordered alphabetically by filename."""
    results = compute_final_scores(
        filenames=["charlie.txt", "alice.txt"],
        candidate_names=["Charlie", "Alice"],
        nlp_scores=[5.0, 5.0],
        ai_scores=[5.0, 5.0],
        ai_rationales=["Same.", "Same."],
        skill_matches_list=[set(), set()],
        skill_gaps_list=[set(), set()],
    )

    assert results[0].filename == "alice.txt"
    assert results[1].filename == "charlie.txt"


# ── Edge case tests ──────────────────────────────────────────────────


def test_single_candidate():
    """Single candidate should not crash and should have rank 1."""
    results = compute_final_scores(
        filenames=["solo.txt"],
        candidate_names=["Solo"],
        nlp_scores=[7.0],
        ai_scores=[8.0],
        ai_rationales=["Good."],
        skill_matches_list=[{"python"}],
        skill_gaps_list=[set()],
    )

    assert len(results) == 1
    assert results[0].filename == "solo.txt"


def test_zero_scores():
    """All-zero scores should produce valid ranking without confusion."""
    results = compute_final_scores(
        filenames=["a.txt", "b.txt"],
        candidate_names=["A", "B"],
        nlp_scores=[0.0, 0.0],
        ai_scores=[0.0, 0.0],
        ai_rationales=["", ""],
        skill_matches_list=[set(), set()],
        skill_gaps_list=[set(), set()],
    )

    assert len(results) == 2
    for r in results:
        assert r.final_score == 0.0
    # Alphabetical tiebreak
    assert results[0].filename == "a.txt"
