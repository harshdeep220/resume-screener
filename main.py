"""
main.py — CLI entry point for the AI-Powered Resume Screener.

Usage:
    python main.py --jd path/to/job_description.pdf --resumes path/to/resumes/
    python main.py --jd data/jd.txt --resumes data/resumes/ --top 5
    python main.py --jd data/jd.txt --resumes data/resumes/ --weights nlp=0.3,ai=0.7
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.logging import RichHandler

from src.extractor import extract_text
from src.jd_parser import parse_jd
from src.resume_parser import parse_resume
from src.nlp_engine import score_resumes
from src.ai_scorer import score_resumes_batch
from src.scoring_engine import compute_final_scores
from src.output import display_results, export_csv, export_json

console = Console()

# Config file path
_CONFIG_PATH = Path(__file__).resolve().parent / "config.json"


def _load_config() -> dict:
    """Load configuration from config.json."""
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        console.print("[yellow]⚠ config.json not found — using defaults.[/yellow]")
        return {}


def _parse_weights(weight_str: str) -> tuple[float, float]:
    """Parse a weight string like 'nlp=0.3,ai=0.7'.

    Args:
        weight_str: Comma-separated key=value pairs.

    Returns:
        Tuple of (nlp_weight, ai_weight).

    Raises:
        ValueError: If format is invalid or weights don't sum to ~1.0.
    """
    parts = dict(item.split("=") for item in weight_str.split(","))
    nlp_w = float(parts.get("nlp", 0.4))
    ai_w = float(parts.get("ai", 0.6))

    if abs(nlp_w + ai_w - 1.0) > 0.01:
        raise ValueError(
            f"Weights must sum to 1.0, got nlp={nlp_w} + ai={ai_w} = {nlp_w + ai_w}"
        )

    return nlp_w, ai_w


def _collect_resume_files(resume_dir: Path) -> list[Path]:
    """Collect all supported resume files from a directory.

    Args:
        resume_dir: Path to the directory containing resumes.

    Returns:
        Sorted list of Path objects for supported file types.
    """
    supported = {".pdf", ".docx", ".txt"}
    files = [
        f for f in resume_dir.iterdir()
        if f.is_file() and f.suffix.lower() in supported
    ]
    return sorted(files)


def main():
    """Main entry point — orchestrates the full screening pipeline."""

    # ── Argument parsing ──────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="AI-Powered Resume Screener — rank candidates against a job description.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --jd data/jd.txt --resumes data/resumes/
  python main.py --jd data/jd.pdf --resumes data/resumes/ --top 5
  python main.py --jd data/jd.docx --resumes data/resumes/ --weights nlp=0.3,ai=0.7
        """,
    )
    parser.add_argument(
        "--jd", required=True, type=Path,
        help="Path to the job description file (.pdf, .docx, or .txt)",
    )
    parser.add_argument(
        "--resumes", required=True, type=Path,
        help="Path to the directory containing resume files",
    )
    parser.add_argument(
        "--top", type=int, default=None,
        help="Show only the top N candidates (default: show all)",
    )
    parser.add_argument(
        "--weights", type=str, default=None,
        help="Override scoring weights, e.g. 'nlp=0.3,ai=0.7'",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Custom output directory (default: ./output/)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # ── Logging setup ─────────────────────────────────────────────────
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_time=False, show_path=False)],
    )
    logger = logging.getLogger(__name__)

    # ── Load config ───────────────────────────────────────────────────
    config = _load_config()

    nlp_weight = config.get("nlp_weight", 0.4)
    ai_weight = config.get("ai_weight", 0.6)
    if args.weights:
        try:
            nlp_weight, ai_weight = _parse_weights(args.weights)
        except ValueError as e:
            console.print(f"[red]✖ Invalid weights: {e}[/red]")
            sys.exit(1)

    model_name = config.get("model", "gemini-1.5-flash")
    api_delay = config.get("api_delay_seconds", 2)
    jd_max_chars = config.get("jd_max_chars", 2000)
    resume_max_chars = config.get("resume_max_chars", 2000)
    tfidf_max_features = config.get("tfidf_max_features", 500)

    # ── Validate inputs ──────────────────────────────────────────────
    if not args.jd.exists():
        console.print(f"[red]✖ JD file not found: {args.jd}[/red]")
        sys.exit(1)

    if not args.resumes.is_dir():
        console.print(f"[red]✖ Resumes directory not found: {args.resumes}[/red]")
        sys.exit(1)

    resume_files = _collect_resume_files(args.resumes)
    if not resume_files:
        console.print(f"[red]✖ No supported files found in {args.resumes}[/red]")
        sys.exit(1)

    console.print(f"\n[bold cyan]📋 AI-Powered Resume Screener[/bold cyan]")
    console.print(f"   JD: {args.jd}")
    console.print(f"   Resumes: {len(resume_files)} files in {args.resumes}")
    console.print(f"   Weights: NLP={nlp_weight}, AI={ai_weight}")
    console.print(f"   Model: {model_name}\n")

    # ── Phase 1: Extract text ────────────────────────────────────────
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Extracting text..."),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("Extracting", total=len(resume_files) + 1)

        # Extract JD
        jd_filename, jd_text = extract_text(args.jd)
        progress.advance(task)

        if not jd_text:
            console.print(f"[red]✖ Failed to extract text from JD: {args.jd}[/red]")
            sys.exit(1)

        # Extract resumes
        resumes = []  # list of (filename, text)
        skipped = []
        for rf in resume_files:
            filename, text = extract_text(rf)
            if text:
                resumes.append((filename, text))
            else:
                skipped.append(filename)
            progress.advance(task)

    if skipped:
        console.print(f"[yellow]⚠ Skipped {len(skipped)} files: {', '.join(skipped)}[/yellow]")

    if not resumes:
        console.print("[red]✖ No resumes could be processed.[/red]")
        sys.exit(1)

    console.print(f"   ✓ Extracted text from {len(resumes)} resumes\n")

    # ── Phase 2: Parse ───────────────────────────────────────────────
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Parsing documents..."),
        console=console,
    ) as progress:
        task = progress.add_task("Parsing", total=len(resumes) + 1)

        # Parse JD
        jd_profile = parse_jd(jd_text)
        progress.advance(task)

        logger.info(
            "JD: '%s' — %d required skills detected",
            jd_profile.title,
            len(jd_profile.required_skills),
        )

        # Parse resumes
        resume_profiles = []
        for filename, text in resumes:
            profile = parse_resume(text, filename)
            resume_profiles.append(profile)
            progress.advance(task)

    console.print(f"   ✓ Parsed {len(resume_profiles)} resumes\n")

    # ── Phase 3: NLP scoring ─────────────────────────────────────────
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Computing NLP scores..."),
        console=console,
    ) as progress:
        task = progress.add_task("NLP", total=1)

        nlp_results = score_resumes(
            jd_skills=jd_profile.required_skills,
            jd_text=jd_profile.raw_text,
            resume_skills_list=[rp.skills for rp in resume_profiles],
            resume_texts=[rp.raw_text for rp in resume_profiles],
            max_features=tfidf_max_features,
        )
        progress.advance(task)

    console.print(f"   ✓ NLP scores computed\n")

    # ── Phase 4: AI scoring ──────────────────────────────────────────
    console.print("[bold blue]🤖 AI scoring via Gemini...[/bold blue]")

    with Progress(
        SpinnerColumn(),
        TextColumn(f"[bold blue]Scoring with {model_name}..."),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("AI Scoring", total=len(resume_profiles))

        ai_results = []
        from src.ai_scorer import score_resume, _save_cache
        from google import genai
        import time

        # Create a single client for the batch
        api_key = os.getenv("GOOGLE_API_KEY")
        client = genai.Client(api_key=api_key) if api_key and api_key != "your_key_here" else None

        cache_dict = {}
        for i, rp in enumerate(resume_profiles):
            result = score_resume(
                jd_text=jd_profile.raw_text,
                resume_text=rp.raw_text,
                model_name=model_name,
                jd_max_chars=jd_max_chars,
                resume_max_chars=resume_max_chars,
                api_delay=api_delay,
                _cache=cache_dict,
                _client=client,
            )
            ai_results.append(result)
            progress.advance(task)

            # Rate-limit delay between calls
            if i < len(resume_profiles) - 1:
                time.sleep(api_delay)

        # Save cache after batch
        _save_cache(cache_dict)

    console.print(f"   ✓ AI scores computed\n")

    # ── Phase 5: Rank ────────────────────────────────────────────────
    ranked = compute_final_scores(
        filenames=[rp.filename for rp in resume_profiles],
        candidate_names=[rp.candidate_name for rp in resume_profiles],
        nlp_scores=[nr.nlp_score for nr in nlp_results],
        ai_scores=[float(ar["score"]) for ar in ai_results],
        ai_rationales=[ar["rationale"] for ar in ai_results],
        skill_matches_list=[nr.skill_matches for nr in nlp_results],
        skill_gaps_list=[nr.skill_gaps for nr in nlp_results],
        nlp_weight=nlp_weight,
        ai_weight=ai_weight,
    )

    # ── Phase 6: Output ──────────────────────────────────────────────
    display_results(ranked, top_n=args.top, title=f"Results for: {jd_profile.title}")

    csv_path = export_csv(ranked)
    json_path = export_json(ranked)

    console.print(f"[green]📁 CSV exported:[/green]  {csv_path}")
    console.print(f"[green]📁 JSON exported:[/green] {json_path}")
    console.print()


if __name__ == "__main__":
    main()
