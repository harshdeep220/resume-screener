# Project Report: AI-Powered Resume Screener Pro

## 1. Project Overview
**Resume Screener Pro** is an automated HR utility designed to parse, analyze, and rank candidate resumes against a specific Job Description (JD). It bridges the gap between deterministic NLP analysis and generative AI reasoning, providing HR professionals with a ranked shortlist of candidates accompanied by detailed, explainable rationales.

Originally developed as a CLI application, it has been fully migrated into a modern, responsive web application powered by **Django**, allowing for easy drag-and-drop interactions, interactive modals, and localized data caching.

---

## 2. Tech Stack

### Core Logic & NLP
* **Python 3.10+**: The primary programming language.
* **spaCy (`en_core_web_sm`)**: Used for robust text extraction and lemmatized skill matching.
* **rank-bm25**: Industry-standard algorithm used for text retrieval and document similarity (replaces older TF-IDF methods).
* **PyPDF2 & python-docx**: Libraries used for extracting raw text from `.pdf` and `.docx` file formats.

### Artificial Intelligence
* **Google GenAI SDK (`google-genai`)**: The official SDK used to interface with Google's Gemini models.
* **Model**: `gemini-3.1-flash-lite-preview` (Configurable via `config.json`). Used for deep semantic scoring and rationale generation.

### Web Infrastructure
* **Django 5.0+**: The core web framework handling routing, file uploads, and synchronous execution of the Python pipeline.
* **Vanilla HTML/CSS/JS**: A custom, dependency-free frontend using modern CSS variables, flexbox/grid layouts, and vanilla JavaScript for DOM manipulation.
* **SQLite3**: Django's default database (currently unused for persistent storage, as the app relies on ephemeral filesystem states).

---

## 3. How the Pipeline Works

The application follows a strictly defined, multi-stage pipeline:

1. **Ingestion & Extraction (`src/extractor.py`)**
   The user uploads a JD and a batch of resumes via the UI. Django saves these to the `input/` folder. The extractor engine determines the file type (PDF, DOCX, TXT) and pulls raw UTF-8 text from the documents, stripping out unreadable formatting.

2. **Parsing & Skill Matching (`src/jd_parser.py` & `src/resume_parser.py`)**
   The system loads a static JSON dictionary of ~500 tech industry skills. Using `spaCy`'s lemmatization, it scans the JD and resumes for these specific skills, identifying exactly what is required versus what the candidate actually possesses.

3. **Deterministic NLP Scoring (`src/nlp_engine.py`)**
   This stage assigns a baseline score (0.0 - 10.0) without calling any external APIs:
   * **Skill Recall**: Evaluates what percentage of the JD's required skills were explicitly found in the resume (`matches / required_skills`).
   * **BM25 Text Retrieval**: Treats the JD as a search query and ranks the resumes based on keyword saturation and document length normalization, scaling the batch relatively from 0 to 10.
   * *These two scores are blended 50/50 to create the final `nlp_score`.*

4. **Generative AI Semantic Scoring (`src/ai_scorer.py`)**
   The raw text of the JD and the resume are injected into a highly specific system prompt and sent to Google Gemini. The AI is instructed to return a strictly formatted JSON response containing a numerical score (0-10) and a brief rationale explaining *why* it assigned that score based on the candidate's nuanced experience. Responses are cached locally to save API quotas on subsequent runs.

5. **Final Aggregation (`src/scoring_engine.py`)**
   The deterministic NLP score and the subjective AI score are combined using a weighted formula (e.g., `0.4 * NLP + 0.6 * AI`). The final results are sorted descending by score and returned to the frontend.

---

## 4. Architecture & File Registry

### Web Application (Django)
* `manage.py`: The command-line utility for Django (used to run the server).
* `screener_web/`: The Django project core containing settings, ASGI/WSGI configs, and primary URL routing.
* `dashboard/views.py`: The primary API controller. It exposes a `POST /run-pipeline/` endpoint that accepts multipart form data, triggers the `src/` pipeline, and returns JSON results.
* `dashboard/templates/dashboard/index.html`: The single-page application layout.
* `dashboard/static/dashboard/style.css`: The styling system utilizing a custom slate-grey and primary blue theme.
* `dashboard/static/dashboard/script.js`: Handles drag-and-drop file state, POST requests via Fetch API, progress bar animations, and the pop-up results modal.

### Core Pipeline (`src/`)
* `extractor.py`: Utility functions utilizing `PyPDF2` and `docx` to yield raw text.
* `jd_parser.py`: Generates a `JDProfile` dataclass containing the raw text and extracted skill sets.
* `resume_parser.py`: Generates a `ResumeProfile` dataclass containing the raw text, candidate name, and extracted skills.
* `nlp_engine.py`: Contains the algorithms for Skill Recall and BM25 batch scoring.
* `ai_scorer.py`: Manages the API connection, handles rate-limit backoffs (HTTP 429), strictly enforces JSON schema extraction via Regex, and handles local caching.
* `scoring_engine.py`: Defines the `CandidateResult` dataclass and blends the dual-layer scoring architectures.

### System Assets
* `config.json`: The global configuration file where users can set weighting (`nlp_weight`, `ai_weight`), select the AI model, and define API delay throttles.
* `data/skills_taxonomy.json`: The static knowledge base of recognizable industry skills.

---

## 5. Security & Limitations

* **API Exposure**: The `GOOGLE_API_KEY` is loaded securely via a `.env` file and is never exposed to the frontend browser. 
* **State Management**: Uploaded resumes are stored ephemerally in `input/resumes/`. The folder is wiped at the start of every new run to prevent data bloating, meaning long-term persistence requires external storage.
* **Rate Limiting**: Because free-tier LLM APIs are highly restricted (e.g., 15 requests per minute), the system intentionally introduces an `api_delay_seconds` thread sleep between consecutive resume evaluations. Processing large batches requires patience.
