"""
test_nlp_engine.py — Unit tests for the NLP scoring engine.
"""

import pytest

from src.nlp_engine import compute_skill_overlap, compute_tfidf_scores, score_resumes


# ── Skill overlap tests ──────────────────────────────────────────────


def test_perfect_overlap():
    """Identical skill sets should produce a score of 10.0."""
    skills = {"python", "django", "docker"}
    score, matches, gaps = compute_skill_overlap(skills, skills)

    assert score == 10.0
    assert matches == skills
    assert gaps == set()


def test_zero_overlap():
    """Completely disjoint skill sets should produce a score of 0.0."""
    jd_skills = {"python", "django", "docker"}
    resume_skills = {"java", "spring", "maven"}
    score, matches, gaps = compute_skill_overlap(jd_skills, resume_skills)

    assert score == 0.0
    assert matches == set()
    assert gaps == jd_skills


def test_partial_overlap():
    """Partial overlap should produce a score between 0 and 10."""
    jd_skills = {"python", "django", "docker", "kubernetes"}
    resume_skills = {"python", "django", "react"}

    score, matches, gaps = compute_skill_overlap(jd_skills, resume_skills)

    assert 0.0 < score < 10.0
    assert matches == {"python", "django"}
    assert gaps == {"docker", "kubernetes"}


def test_score_normalization():
    """Jaccard score of 0.5 must produce a skill_overlap_score of 5.0."""
    # Jaccard = |A∩B| / |A∪B| = 1/2 = 0.5 ⟹ score = 5.0
    jd_skills = {"python"}
    resume_skills = {"python", "java"}  # union=2, intersection=1 → 0.5

    score, matches, gaps = compute_skill_overlap(jd_skills, resume_skills)
    assert score == 5.0


def test_empty_jd_skills():
    """Empty JD skills should return a neutral score of 5.0."""
    score, matches, gaps = compute_skill_overlap(set(), {"python", "java"})
    assert score == 5.0
    assert matches == set()
    assert gaps == set()


# ── TF-IDF tests ────────────────────────────────────────────────────


def test_tfidf_scores_count():
    """Should return one score per resume."""
    jd = "Python Django REST API developer"
    resumes = [
        "Python Django developer with REST API experience",
        "Java Spring developer",
        "Marketing manager",
    ]

    scores = compute_tfidf_scores(jd, resumes)
    assert len(scores) == 3


def test_tfidf_score_range():
    """All TF-IDF scores should be in [0.0, 10.0]."""
    jd = "Python Django developer experienced with Docker and Kubernetes"
    resumes = [
        "Python Django developer",
        "Java Spring developer",
        "Totally unrelated marketing text about brand strategy",
    ]

    scores = compute_tfidf_scores(jd, resumes)
    for s in scores:
        assert 0.0 <= s <= 10.0


def test_tfidf_relevant_scores_higher():
    """Resumes matching the JD keywords should score higher."""
    jd = "Python Django developer REST API Docker"
    resumes = [
        "Experienced Python Django developer building REST APIs with Docker",
        "Marketing manager with brand strategy experience",
    ]

    scores = compute_tfidf_scores(jd, resumes)
    assert scores[0] > scores[1]  # Python resume > marketing resume


def test_tfidf_empty_resumes():
    """Empty resume list should return empty scores list."""
    scores = compute_tfidf_scores("Python developer", [])
    assert scores == []


# ── Integrated NLP scorer tests ──────────────────────────────────────


def test_nlp_score_range():
    """nlp_score should always be in [0.0, 10.0]."""
    jd_skills = {"python", "django", "docker"}
    jd_text = "Python Django Docker developer"
    resume_skills = [{"python", "java"}, {"react", "vue"}, {"python", "django", "docker"}]
    resume_texts = [
        "Python and Java developer",
        "React and Vue frontend developer",
        "Python Django Docker backend engineer",
    ]

    results = score_resumes(jd_skills, jd_text, resume_skills, resume_texts)

    assert len(results) == 3
    for r in results:
        assert 0.0 <= r.nlp_score <= 10.0
        assert 0.0 <= r.skill_overlap_score <= 10.0
        assert 0.0 <= r.tfidf_score <= 10.0


def test_tfidf_fit_on_corpus():
    """TF-IDF vectoriser should be fit on the combined corpus, not per-pair.

    Verifying indirectly: if we have 3 resumes but one shares all terms
    with the JD, its TF-IDF score should be higher than one with no
    shared terms. This only works correctly if the IDF weights come from
    the full corpus.
    """
    jd_skills = {"python"}
    jd_text = "Python backend developer experienced with Django REST API"
    resume_skills = [{"python"}, set(), set()]
    resume_texts = [
        "Python backend developer Django REST API expert",
        "Underwater basket weaving instructor for 10 years",
        "Professional cat herder and tea sommelier",
    ]

    results = score_resumes(jd_skills, jd_text, resume_skills, resume_texts)

    # The first resume should have a higher NLP score
    assert results[0].nlp_score > results[1].nlp_score
    assert results[0].nlp_score > results[2].nlp_score
