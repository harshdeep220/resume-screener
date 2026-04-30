# Project Report: AI-Powered Resume Screener

## Abstract
This project report details the development of an AI-Powered Resume Screener, an automated HR tool designed to streamline the recruitment process. The system accepts a job description (JD) and a directory of applicant resumes, extracts and normalises text across various file formats (PDF, DOCX, TXT), and scores each candidate. It employs a two-layer scoring mechanism: a fast Natural Language Processing (NLP) layer for keyword and skill overlap, followed by a semantic relevance scoring layer utilizing Google's Gemini AI. The tool features a modern Django-based web dashboard for visual analysis and a rich Command Line Interface (CLI), ultimately generating a ranked shortlist of candidates with justifiable rationales and exportable results.

## Introduction
### Context
The recruitment process often involves sifting through hundreds of resumes for a single job posting, making manual screening a time-consuming and bias-prone task. Automated applicant tracking systems exist, but many rely solely on rigid keyword matching, failing to understand the contextual relevance of a candidate's experience.

### Problem Addressed
HR professionals need a fast, cost-effective, and intelligent way to rank candidates. Pure keyword matching misses semantic context, while processing every resume through a Large Language Model (LLM) is computationally expensive and slow due to rate limits. Furthermore, candidates use varied resume formats, requiring robust text extraction.

### Objectives
- Build an automated resume screening pipeline supporting multiple document formats (PDF, DOCX, TXT).
- Implement a hybrid scoring system blending fast NLP keyword extraction (BM25/TF-IDF) with advanced LLM semantic analysis.
- Provide clear, AI-generated rationales for each candidate's score to ensure transparency.
- Offer both a rich CLI interface for offline batch processing and a Django-powered web dashboard for interactive analysis.
- Ensure cost efficiency and rate-limit compliance by implementing smart caching and using an optimal LLM model (Gemini 1.5 Flash).

## Methodology
### High-Level Approach
The system follows a sequential pipeline:
1. **Input & Extraction**: Accepts the JD and resumes, converting them into clean Unicode text.
2. **Parsing & Taxonomy Matching**: Uses a curated taxonomy of over 500 skills to deterministically extract required skills from the JD and matched skills from the resumes.
3. **NLP Scoring Layer**: Calculates a deterministic score using Jaccard similarity for skill set overlap and TF-IDF cosine similarity for overall keyword matching.
4. **AI Scoring Layer**: Sends a truncated version of the JD and each resume to the Google Gemini API to assess semantic relevance and generate a one-sentence rationale.
5. **Weighted Aggregation**: Blends the NLP score and AI score using configurable weights to produce a final ranking.

### Design Patterns & Heuristics
- **Fallback Section Detection**: If standard resume headers (e.g., "Experience", "Education") are missing, the system falls back to evaluating the entire document, ensuring no candidate is unfairly penalized for unconventional formatting.
- **Batched TF-IDF**: The TF-IDF vectorizer is fit across the entire corpus of resumes for a given run, correctly weighting terms that are rare across the specific applicant pool.
- **Smart Caching**: API responses are cached using a SHA-256 hash of the combined JD and resume text, making subsequent re-evaluations free and instantaneous.

## Implementation
### Architecture & Key Components
- **Text Extractor (`src/extractor.py`)**: Utilizes `PyMuPDF` for fast, pure-Python PDF extraction and `python-docx` for Word documents, including table parsing.
- **JD & Resume Parsers (`src/jd_parser.py`, `src/resume_parser.py`)**: Uses `spaCy` for lemmatization and tokenization to extract keywords and sections. Matches text against a curated JSON skills taxonomy.
- **NLP Engine (`src/nlp_engine.py`)**: Implements Jaccard similarity and a `scikit-learn` `TfidfVectorizer` to compute the `nlp_score` (0-10).
- **AI Scorer (`src/ai_scorer.py`)**: Integrates with `google-generativeai` (Gemini 1.5 Flash). Includes exponential backoff, rate limiting (delaying requests to respect the 15 RPM free-tier limit), and text truncation (limiting payloads to 2000 characters).
- **Web Dashboard (`dashboard/views.py` & `screener_web/`)**: A Django application providing a modern, user-friendly frontend for interacting with the screening pipeline, displaying results visually, and handling file uploads/exports.
- **Output Module (`src/output.py`)**: Uses the `Rich` library for colour-coded terminal tables and `pandas` for CSV and JSON data exports.

### Technologies
- **Core**: Python 3.10+
- **NLP & Extraction**: `spaCy`, `PyMuPDF`, `python-docx`
- **Machine Learning**: `scikit-learn`
- **AI Integration**: Google AI Studio (`google-generativeai` with Gemini 1.5 Flash)
- **Web Framework**: Django
- **Data Manipulation & CLI**: `pandas`, `Rich`

## Results (Expected/Inferred)
- **Efficiency**: The system significantly reduces manual screening time. A batch of 50 resumes can be processed and ranked in approximately 2 minutes (due to API rate limiting), with subsequent runs completing instantly due to caching.
- **Accuracy**: The two-layer approach mitigates the risk of missing highly qualified candidates who use different terminology (handled by Gemini) while ensuring baseline technical requirements are met (handled by NLP skill matching).
- **Usability**: The provision of both a visual web dashboard and a comprehensive CLI allows flexibility for different user personas (e.g., non-technical recruiters vs. technical hiring managers). Outputs include transparent rationales, preventing the "black box" problem of AI screening.

## Conclusion (Inferred)
The AI-Powered Resume Screener successfully bridges the gap between traditional applicant tracking systems and modern generative AI. By intelligently combining deterministic NLP techniques with the semantic understanding of Large Language Models, the project delivers a robust, cost-effective, and highly scalable solution for initial applicant triaging. The deliberate design choices—such as caching, fallback parsing mechanisms, and a dual-interface approach—result in a production-ready tool that respects both computational limits and real-world resume variability.

## Future Scope (Inferred)
- **Optical Character Recognition (OCR)**: Integrating libraries like `pytesseract` to support image-based or scanned PDFs, which currently fail text extraction.
- **Dynamic Skill Taxonomy**: Upgrading the static JSON taxonomy to a dynamic system or utilizing an LLM to auto-extract and categorize new, unrecognized skills dynamically.
- **Enhanced Document Context**: Transitioning to models with larger context windows or implementing Retrieval-Augmented Generation (RAG) to evaluate very long documents without truncation.
- **Bias Auditing Module**: Adding a layer to analyze the AI's rationales for potential unconscious biases regarding age, gender, or educational background.

## References
- Google AI Studio (Gemini SDK): https://aistudio.google.com/
- PyMuPDF Documentation: https://pymupdf.readthedocs.io/
- scikit-learn Documentation: https://scikit-learn.org/
- spaCy Documentation: https://spacy.io/
- Django Framework: https://www.djangoproject.com/
- Rich CLI Library: https://rich.readthedocs.io/
