# 🎯 AI-Powered Resume Screener

An automated HR tool that scores and ranks job applicants against a job description using **NLP keyword analysis** and **Google Gemini AI** semantic scoring.

Feed it a JD and a folder of resumes → get a ranked shortlist with scores, rationales, and exportable results.

## ✨ Features

- **Multi-format support** — PDF, DOCX, and TXT files for both JDs and resumes
- **Two-layer scoring** — Fast NLP (TF-IDF + skill overlap) + Gemini AI semantic analysis
- **500+ skill taxonomy** — Deterministic skill matching across tech, frameworks, and domains
- **Smart caching** — API results are cached so re-runs are free
- **Rate-limit aware** — Configurable delays + exponential backoff for the free tier
- **Rich CLI output** — Colour-coded terminal tables with progress bars
- **Export** — Timestamped CSV and JSON files in `output/`

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- A [Google AI Studio API key](https://aistudio.google.com) (free)

### Install

```bash
git clone <your-repo>
cd resume_screener
python -m venv .venv
.venv\Scripts\activate           # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Configure

```bash
# Create .env file with your API key
echo GOOGLE_API_KEY=your_key_here > .env
```

Optionally edit `config.json` to adjust model, weights, or rate-limit delay.

### Run

```bash
# Basic usage
python main.py --jd path/to/job_description.pdf --resumes path/to/resumes/

# Show only top 5 candidates
python main.py --jd path/to/jd.txt --resumes path/to/resumes/ --top 5

# Override scoring weights
python main.py --jd path/to/jd.docx --resumes path/to/resumes/ --weights nlp=0.3,ai=0.7

# Verbose logging
python main.py --jd path/to/jd.txt --resumes path/to/resumes/ -v
```

## 📊 How It Works

```
JD + Resumes → Extract Text → Parse Skills → NLP Score → AI Score → Rank → Output
```

| Stage | What it does |
|---|---|
| **Extract** | PDF/DOCX/TXT → clean UTF-8 text |
| **Parse** | Skills taxonomy matching + section detection |
| **NLP Score** | TF-IDF cosine similarity + Jaccard skill overlap (0–10) |
| **AI Score** | Gemini API semantic relevance scoring (0–10) |
| **Final Score** | Weighted blend: `0.4 × NLP + 0.6 × AI` (configurable) |

## ⚙️ Configuration

Edit `config.json`:

```json
{
  "model": "gemini-1.5-flash",
  "nlp_weight": 0.4,
  "ai_weight": 0.6,
  "api_delay_seconds": 2,
  "jd_max_chars": 2000,
  "resume_max_chars": 2000,
  "tfidf_max_features": 500
}
```

## 🧪 Testing

```bash
# Unit tests (fast, no API calls)
pytest tests/unit/ -v

# Integration tests (mocked API)
pytest tests/integration/ -v

# All tests with coverage
pytest tests/ --cov=src --cov-report=term
```

## 📁 Project Structure

```
resume_screener/
├── main.py                    # CLI entry point
├── config.json                # Scoring weights and settings
├── requirements.txt
├── .env                       # GOOGLE_API_KEY (gitignored)
├── src/
│   ├── extractor.py           # PDF/DOCX/TXT → clean text
│   ├── jd_parser.py           # JD → JDProfile
│   ├── resume_parser.py       # Resume → ResumeProfile
│   ├── nlp_engine.py          # TF-IDF + skill overlap scoring
│   ├── ai_scorer.py           # Gemini API integration
│   ├── scoring_engine.py      # Weighted blend + ranking
│   └── output.py              # Rich table + CSV/JSON export
├── data/
│   └── skills_taxonomy.json   # ~500 curated skills
├── cache/                     # Auto-generated API cache
├── output/                    # Auto-generated exports
└── tests/
    ├── unit/                  # Fast unit tests
    ├── integration/           # Pipeline integration tests
    └── fixtures/              # Sample JD + 10 resumes
```

## ⚠️ Known Limitations

| Issue | Mitigation |
|---|---|
| Scanned PDFs return no text | Warning logged, candidate excluded with note |
| Free-tier rate limit (15 RPM) | 2s delay between calls; 50 resumes ≈ 2 min |
| Skills taxonomy is static | Editable JSON — add skills before a run |
| No OCR support | Can add `pytesseract` later as a drop-in |

## 📄 License

MIT
