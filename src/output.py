"""
output.py — Terminal display and file export for ranked results.

Produces a Rich terminal table with colour-coded scores, plus
CSV and JSON exports to the output/ directory.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

from src.scoring_engine import CandidateResult

logger = logging.getLogger(__name__)

console = Console()

_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def _score_colour(score: float) -> str:
    """Return a Rich colour string based on score value.

    Args:
        score: Score from 0.0 to 10.0.

    Returns:
        Rich colour name.
    """
    if score >= 8.0:
        return "bold green"
    elif score >= 6.0:
        return "green"
    elif score >= 4.0:
        return "yellow"
    elif score >= 2.0:
        return "red"
    else:
        return "bold red"


def display_results(
    results: list[CandidateResult],
    top_n: int | None = None,
    title: str = "Resume Screening Results",
) -> None:
    """Display ranked results in a Rich terminal table.

    Args:
        results: Sorted list of CandidateResult objects.
        top_n: If set, only show the top N candidates.
        title: Table title.
    """
    table = Table(title=title, show_lines=True, header_style="bold cyan")
    table.add_column("Rank", justify="center", style="bold", width=5)
    table.add_column("Candidate", min_width=20)
    table.add_column("File", min_width=15, style="dim")
    table.add_column("NLP", justify="center", width=6)
    table.add_column("AI", justify="center", width=6)
    table.add_column("Final", justify="center", width=6)
    table.add_column("Rationale", min_width=30)

    display_list = results[:top_n] if top_n else results

    for rank, r in enumerate(display_list, start=1):
        table.add_row(
            str(rank),
            r.candidate_name or "—",
            r.filename,
            f"[{_score_colour(r.nlp_score)}]{r.nlp_score:.1f}[/]",
            f"[{_score_colour(r.ai_score)}]{r.ai_score:.1f}[/]",
            f"[{_score_colour(r.final_score)}]{r.final_score:.1f}[/]",
            r.rationale or "—",
        )

    console.print()
    console.print(table)
    console.print()

    if top_n and len(results) > top_n:
        console.print(
            f"  [dim]Showing top {top_n} of {len(results)} candidates. "
            f"Remove --top to see all.[/dim]"
        )
        console.print()


def export_csv(results: list[CandidateResult]) -> Path:
    """Export results to a timestamped CSV file.

    Args:
        results: List of CandidateResult objects.

    Returns:
        Path to the created CSV file.
    """
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = _OUTPUT_DIR / f"results_{timestamp}.csv"

    rows = []
    for rank, r in enumerate(results, start=1):
        rows.append(
            {
                "rank": rank,
                "candidate_name": r.candidate_name,
                "filename": r.filename,
                "nlp_score": r.nlp_score,
                "ai_score": r.ai_score,
                "final_score": r.final_score,
                "rationale": r.rationale,
                "skill_matches": ", ".join(sorted(r.skill_matches)),
                "skill_gaps": ", ".join(sorted(r.skill_gaps)),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False, encoding="utf-8")
    logger.info("CSV exported to %s", filepath)

    return filepath


def export_json(results: list[CandidateResult]) -> Path:
    """Export results to a timestamped JSON file.

    Args:
        results: List of CandidateResult objects.

    Returns:
        Path to the created JSON file.
    """
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = _OUTPUT_DIR / f"results_{timestamp}.json"

    rows = []
    for rank, r in enumerate(results, start=1):
        rows.append(
            {
                "rank": rank,
                "candidate_name": r.candidate_name,
                "filename": r.filename,
                "nlp_score": r.nlp_score,
                "ai_score": r.ai_score,
                "final_score": r.final_score,
                "rationale": r.rationale,
                "skill_matches": sorted(r.skill_matches),
                "skill_gaps": sorted(r.skill_gaps),
            }
        )

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    logger.info("JSON exported to %s", filepath)

    return filepath
