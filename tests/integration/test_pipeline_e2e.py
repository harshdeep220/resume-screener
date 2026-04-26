"""
test_pipeline_e2e.py — End-to-end integration test with mocked Gemini API.
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.extractor import extract_text
from src.jd_parser import parse_jd
from src.resume_parser import parse_resume
from src.nlp_engine import score_resumes
from src.ai_scorer import score_resume
from src.scoring_engine import compute_final_scores
from src.output import export_csv, export_json


FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
RESUMES_DIR = FIXTURES_DIR / "resumes"


def _mock_ai_score(jd_text, resume_text, **kwargs):
    """Deterministic mock AI scorer based on keyword overlap."""
    keywords = ["python", "django", "fastapi", "docker", "kubernetes", "aws", "postgresql"]
    text_lower = resume_text.lower()
    matches = sum(1 for kw in keywords if kw in text_lower)
    score = min(10, int(matches * 10 / len(keywords) + 0.5))
    return {"score": score, "rationale": f"Matched {matches}/{len(keywords)} key terms."}


@patch("src.ai_scorer.genai")
@patch("src.ai_scorer.os.getenv", return_value="fake-api-key")
def test_full_e2e_pipeline(mock_getenv, mock_genai, tmp_path):
    """Full end-to-end: extract → parse → NLP score → AI score → rank → export."""
    # ── Extract JD ───────────────────────────────────────────────
    _, jd_text = extract_text(FIXTURES_DIR / "sample_jd.txt")
    assert jd_text

    # ── Extract resumes ──────────────────────────────────────────
    resume_files = sorted(RESUMES_DIR.glob("*.txt"))
    resumes = []
    for rf in resume_files:
        filename, text = extract_text(rf)
        if text:
            resumes.append((filename, text))

    assert len(resumes) == 10

    # ── Parse ────────────────────────────────────────────────────
    jd_profile = parse_jd(jd_text)
    resume_profiles = [parse_resume(text, fname) for fname, text in resumes]

    # ── NLP scoring ──────────────────────────────────────────────
    nlp_results = score_resumes(
        jd_skills=jd_profile.required_skills,
        jd_text=jd_profile.raw_text,
        resume_skills_list=[rp.skills for rp in resume_profiles],
        resume_texts=[rp.raw_text for rp in resume_profiles],
    )

    assert len(nlp_results) == 10

    # ── AI scoring (mocked) ──────────────────────────────────────
    ai_results = []
    for rp in resume_profiles:
        result = _mock_ai_score(jd_profile.raw_text, rp.raw_text)
        ai_results.append(result)

    assert len(ai_results) == 10

    # ── Rank ─────────────────────────────────────────────────────
    ranked = compute_final_scores(
        filenames=[rp.filename for rp in resume_profiles],
        candidate_names=[rp.candidate_name for rp in resume_profiles],
        nlp_scores=[nr.nlp_score for nr in nlp_results],
        ai_scores=[float(ar["score"]) for ar in ai_results],
        ai_rationales=[ar["rationale"] for ar in ai_results],
        skill_matches_list=[nr.skill_matches for nr in nlp_results],
        skill_gaps_list=[nr.skill_gaps for nr in nlp_results],
    )

    # ── Assertions ───────────────────────────────────────────────
    assert len(ranked) == 10

    # All candidates produced
    filenames = {r.filename for r in ranked}
    assert len(filenames) == 10

    # Scores are in valid range
    for r in ranked:
        assert 0.0 <= r.final_score <= 10.0
        assert 0.0 <= r.nlp_score <= 10.0
        assert 0.0 <= r.ai_score <= 10.0

    # Sorted descending
    scores = [r.final_score for r in ranked]
    assert scores == sorted(scores, reverse=True)

    # Top candidates should be Python-heavy
    top3 = [r.filename for r in ranked[:3]]
    python_resumes = {"iris_nakamura.txt", "alice_chen.txt", "grace_liu.txt", "bob_martinez.txt"}
    overlap = set(top3) & python_resumes
    assert len(overlap) >= 2, f"Top 3: {top3}"

    # ── Export ────────────────────────────────────────────────────
    # Temporarily patch export path
    import src.output as output_mod
    original_dir = output_mod._OUTPUT_DIR
    output_mod._OUTPUT_DIR = tmp_path

    try:
        csv_path = export_csv(ranked)
        json_path = export_json(ranked)

        # CSV assertions
        assert csv_path.exists()
        assert csv_path.suffix == ".csv"
        csv_content = csv_path.read_text()
        assert "rank" in csv_content
        assert "candidate_name" in csv_content
        assert "final_score" in csv_content

        # JSON assertions
        assert json_path.exists()
        assert json_path.suffix == ".json"
        json_data = json.loads(json_path.read_text())
        assert isinstance(json_data, list)
        assert len(json_data) == 10
        assert json_data[0]["rank"] == 1

    finally:
        output_mod._OUTPUT_DIR = original_dir


def test_no_exceptions_for_any_resume():
    """The entire pipeline should never crash for any fixture resume."""
    _, jd_text = extract_text(FIXTURES_DIR / "sample_jd.txt")
    jd_profile = parse_jd(jd_text)

    resume_files = sorted(RESUMES_DIR.glob("*.txt"))
    for rf in resume_files:
        filename, text = extract_text(rf)
        assert text, f"Failed to extract: {filename}"

        profile = parse_resume(text, filename)
        assert profile.filename == filename
        assert profile.raw_text
