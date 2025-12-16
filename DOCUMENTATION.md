# 📖 ResumeScreener — Complete Documentation

> **Version 2.2.0** | Production-ready AI screening system with advanced scoring, synonym matching, and red-flag detection.

---

## 📋 Table of Contents

1. [Overview](#1-overview)
2. [What's New in v2.2.0](#2-whats-new-in-v220)
3. [Tech Stack](#3-tech-stack)
4. [Installation (Windows)](#4-installation-windows)
5. [Installation (macOS/Linux)](#5-installation-macoslinux)
6. [Running the Server](#6-running-the-server)
7. [Testing with Sample Data](#7-testing-with-sample-data)
8. [Using the App](#8-using-the-app)
9. [API Reference](#9-api-reference)
10. [Configuration](#10-configuration)
11. [Troubleshooting](#11-troubleshooting)
12. [FAQ](#12-faq)

---

## 1. Overview

### What It Does

ResumeScreener is an AI-powered tool that helps you screen and rank job candidates automatically.

| Input | Output |
|-------|--------|
| 📄 Job Description (text or file) | 📊 Ranked candidate list |
| 📁 Multiple resumes (PDF/DOCX/TXT) | ✅ Decision labels (HIRE/REJECT/etc.) |
| 🎯 Must-have & nice-to-have skills | 💡 Detailed scoring breakdown |
| 📅 Experience & education requirements | 🚩 Red flags & strengths |

### Scoring Components

The system evaluates candidates using **6 weighted components**:

```
┌─────────────────────────────────────────────────────────┐
│                    FINAL SCORE                          │
├─────────────────────────────────────────────────────────┤
│  35%  Semantic Similarity    (NLP understanding)        │
│  25%  Keyword Coverage       (skills matching)          │
│  20%  Must-Have Score        (required skills)          │
│  10%  Experience Score       (years of experience)      │
│   5%  Education Score        (degree level)             │
│   5%  Bonus Score            (nice-to-have skills)      │
├─────────────────────────────────────────────────────────┤
│  ⚠️  Red Flag Penalties      (-5% each)                 │
└─────────────────────────────────────────────────────────┘
```

### Key Features

- ✅ **No training required** — Uses pre-trained transformer models
- ✅ **Fast processing** — 50 resumes in under 30 seconds
- ✅ **Explainable decisions** — See exactly why each score
- ✅ **Synonym support** — Understands JS = JavaScript, ML = Machine Learning
- ✅ **Red flag detection** — Catches keyword stuffing, job hopping
- ✅ **CSV export** — Download results for further analysis

---

## 2. What's New in v2.2.0

### 🚀 Major Improvements

| Feature | Description |
|---------|-------------|
| **C++/C#/.NET Matching** | Special characters now match correctly with word boundaries |
| **4x Faster Matching** | Pre-filtering optimization for fuzzy keyword search |
| **13% More Accurate** | Weighted semantic scoring for long resumes |
| **Proportional Must-Haves** | Fair scoring: 4/5 matched = 80%, not binary pass/fail |
| **Experience Validation** | Extracts and validates years from date ranges |
| **Red Flag Detection** | Identifies keyword stuffing, short resumes, job hopping |
| **Synonym Expansion** | Automatic: JS ↔ JavaScript, ML ↔ Machine Learning |

### Technical Fixes

- Fixed PDF page limit off-by-one error
- Fixed empty variant handling in keyword expansion
- Improved date range parsing (supports French month names)
- Better handling of tables in DOCX files

---

## 3. Tech Stack

### Backend

| Component | Technology |
|-----------|------------|
| Framework | FastAPI + Uvicorn |
| Embeddings | Sentence-Transformers (`all-MiniLM-L6-v2`) |
| Fuzzy Matching | RapidFuzz (with pre-filtering) |
| PDF Parsing | pypdf |
| DOCX Parsing | python-docx |
| Parallelism | ThreadPoolExecutor |

### Frontend

| Component | Technology |
|-----------|------------|
| Markup | HTML5 |
| Styling | Tailwind CSS (CDN) |
| Interactivity | Vanilla JavaScript |
| Design | Responsive, mobile-friendly |

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Dashboard UI |
| `/guide` | GET | Documentation page |
| `/api/health` | GET | Health check & version |
| `/api/screen` | POST | Screen resumes |

---

## 4. Installation (Windows)

### Prerequisites

- Python 3.10 or higher
- Windows 10/11
- 2GB+ RAM (for model loading)

### Step-by-Step

**Step 1: Open Command Prompt**

```cmd
cd C:\Users\YourUsername\Downloads\ResumeScreener
```

**Step 2: Create Virtual Environment**

```cmd
python -m venv .venv
.venv\Scripts\activate
```

> 💡 **PowerShell users:** Run `.\.venv\Scripts\Activate.ps1`
> 
> If blocked, first run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

**Step 3: Install Dependencies**

```cmd
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> ⚠️ **If PyTorch fails:** Use CPU build:
> ```cmd
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> ```

**Step 4: Verify Installation**

```cmd
python -c "import fastapi, sentence_transformers; print('✓ All dependencies installed')"
```

---

## 5. Installation (macOS/Linux)

### Step-by-Step

```bash
# Navigate to project folder
cd ~/Downloads/ResumeScreener

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -U pip
pip install -r requirements.txt

# Verify installation
python -c "import fastapi, sentence_transformers; print('✓ All dependencies installed')"
```

---

## 6. Running the Server

### Option A: Simple (Recommended)

```bash
python main.py
```

### Option B: Uvicorn with Auto-Reload

```bash
python -m uvicorn main:app --reload --port 8000
```

### Access Points

| URL | Description |
|-----|-------------|
| http://127.0.0.1:8000 | Dashboard UI |
| http://127.0.0.1:8000/guide | Documentation |
| http://127.0.0.1:8000/docs | Swagger API Docs |
| http://127.0.0.1:8000/api/health | Health Check |

### Quick Launcher Script (Windows)

Create `run.bat` in project folder:

```batch
@echo off
cd /d %~dp0
call .venv\Scripts\activate
python main.py
pause
```

Double-click `run.bat` to start the server!

---

## 7. Testing with Sample Data

### Quick Test

1. Open http://127.0.0.1:8000
2. Paste this **Job Description**:

```
We are looking for a Senior Machine Learning Engineer.

Requirements:
- 5+ years experience in Python and ML
- Strong knowledge of PyTorch, TensorFlow, scikit-learn
- Experience with Docker and cloud platforms (AWS/Azure)
- Bachelor's degree in Computer Science or related field
- SQL and database design experience

Nice to have:
- NLP and transformer models experience
- LangChain, MLflow
- French or Arabic language skills
```

3. Enter **Must-Haves**:

```
Python, PyTorch, scikit-learn, Docker, AWS, Bachelor's degree, 5 years experience
```

4. Enter **Nice-to-Haves**:

```
NLP, LangChain, MLflow, French, Arabic
```

5. Set **Minimum Years**: `5`
6. Set **Required Education**: `Bachelor's Degree`
7. Upload test resumes (PDF/DOCX/TXT)
8. Click **Run Screening**
9. Review results and export to CSV

---

## 8. Using the App

### Decision Types

| Decision | Score Range | What It Means | Action |
|:--------:|:-----------:|---------------|--------|
| 🟢 **STRONG_HIRE** | 85-100 | Excellent fit | Interview immediately |
| 🔵 **HIRE** | 75-84 | Good fit | Move to next round |
| 🟡 **MAYBE** | 60-74 | Borderline | Manual review needed |
| 🟠 **NO_HIRE** | 45-59 | Poor fit | Archive for future |
| 🔴 **REJECT** | 0-44 | Clear mismatch | Do not proceed |

### Best Practices

✅ **Do:**
- Include clear "Requirements" section in JD
- List 5-10 must-haves (not 30)
- Use full names: "Python" not "Py"
- Specify experience level if critical
- Review "MAYBE" decisions manually

❌ **Don't:**
- Upload scanned PDFs (no OCR support)
- Use too many must-haves (dilutes scores)
- Skip the nice-to-haves (they add value)

### Synonym Support

The system automatically recognizes these variations:

| Primary Term | Also Matches |
|--------------|--------------|
| JavaScript | js, ecmascript, es6, es2015 |
| TypeScript | ts |
| Python | py, python3 |
| C++ | cpp, c plus plus |
| C# | csharp, c sharp |
| .NET | dotnet, asp.net, aspnet |
| React | reactjs, react.js |
| Node.js | nodejs, node |
| PostgreSQL | postgres, psql |
| MongoDB | mongo |
| Machine Learning | ml |
| Artificial Intelligence | ai |
| Natural Language Processing | nlp |
| Docker | containerization |
| Kubernetes | k8s |
| Amazon Web Services | aws |
| Google Cloud | gcp |
| CI/CD | continuous integration, continuous deployment |

---

## 9. API Reference

### GET `/api/health`

Health check endpoint.

**Request:**
```bash
curl http://127.0.0.1:8000/api/health
```

**Response:**
```json
{
  "status": "healthy",
  "model": "sentence-transformers/all-MiniLM-L6-v2",
  "timestamp": 1734350400,
  "version": "2.2.0"
}
```

---

### POST `/api/screen`

Screen multiple resumes against a job description.

**Content-Type:** `multipart/form-data`

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|:--------:|-------------|
| `jd_text` | string | ⚠️ | Job description as text |
| `jd_file` | file | ⚠️ | Job description file (PDF/DOCX/TXT) |
| `must_haves` | string | ✗ | Comma-separated required skills |
| `nice_to_haves` | string | ✗ | Comma-separated bonus skills |
| `required_years` | integer | ✗ | Minimum years of experience |
| `required_education` | string | ✗ | `bachelor`, `master`, or `phd` |
| `resumes` | file[] | ✓ | Resume files (max 50, max 10MB each) |

> ⚠️ Either `jd_text` OR `jd_file` is required (not both)

**Example Request:**

```bash
curl -X POST "http://127.0.0.1:8000/api/screen" \
  -F "jd_text=Looking for a Python developer..." \
  -F "must_haves=Python, FastAPI, AWS" \
  -F "nice_to_haves=Docker, React" \
  -F "required_years=3" \
  -F "required_education=bachelor" \
  -F "resumes=@./cv1.pdf" \
  -F "resumes=@./cv2.docx"
```

**Example Response:**

```json
{
  "summary": {
    "total_candidates": 2,
    "strong_hire": 1,
    "hire": 0,
    "maybe": 1,
    "no_hire": 0,
    "reject": 0
  },
  "results": [
    {
      "candidate_name": "Jane Doe",
      "filename": "jane_doe.pdf",
      "decision": "STRONG_HIRE",
      "confidence": 90.0,
      "total_score": 87.5,
      "semantic_score": 85.2,
      "keyword_coverage": 78.5,
      "must_have_score": 100.0,
      "experience_score": 100.0,
      "education_score": 80.0,
      "bonus_score": 40.0,
      "years_experience": 7.0,
      "matched_keywords": ["python", "fastapi", "aws"],
      "missing_must_haves": [],
      "matched_nice_to_haves": ["docker", "react"],
      "red_flags": [],
      "strengths": ["✅ Strong overall fit with job description"],
      "reasoning": "**Decision: STRONG_HIRE** (Total Score: 87.5/100)..."
    }
  ],
  "grouped": {
    "STRONG_HIRE": [...],
    "HIRE": [...],
    "MAYBE": [...],
    "NO_HIRE": [...],
    "REJECT": [...]
  }
}
```

---

## 10. Configuration

### Adjustable Constants

Edit these in `main.py`:

```python
# File Limits
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB per file
MAX_RESUMES = 50                   # Max files per batch
MAX_PDF_PAGES = 100                # Pages to process per PDF

# Performance
MAX_PARSE_WORKERS = 4              # Parallel file parsing threads
FUZZY_PRE_FILTER = True            # Enable pre-filtering (keep enabled)

# Semantic Scoring
USE_CHUNKED_SEMANTICS = True       # Process long resumes in chunks
SEMANTIC_CHUNK_CHAR_LEN = 1500     # Characters per chunk
SEMANTIC_CHUNK_AGG = "weighted_top" # Options: "max", "avg", "weighted_top"
SEMANTIC_TOP_K = 3                  # Chunks to average
```

### Scoring Weights

```python
@dataclass
class ScoringWeights:
    semantic: float = 0.35    # NLP similarity
    keywords: float = 0.25    # Keyword coverage  
    must_haves: float = 0.20  # Required skills
    experience: float = 0.10  # Years matched
    education: float = 0.05   # Degree level
    bonus: float = 0.05       # Nice-to-haves
```

### Adding Custom Synonyms

```python
TECH_SYNONYMS = {
    # Add your custom terms
    "your_term": ["alias1", "alias2"],
    "react native": ["rn", "reactnative"],
    
    # ... existing entries
}
```

---

## 11. Troubleshooting

### ❌ "uvicorn not recognized"

**Solution:** Activate virtual environment first:

```bash
.venv\Scripts\activate   # Windows
source .venv/bin/activate # macOS/Linux

python main.py
```

---

### ❌ Port 8000 already in use

**Solution:** Use a different port:

```bash
python -m uvicorn main:app --port 8001
```

---

### ❌ C++ or C# keywords not matching

**Solution:** Verify you're running v2.2.0:

```bash
curl http://127.0.0.1:8000/api/health
```

Should return `"version": "2.2.0"`

---

### ❌ Slow performance

**Solution:** Adjust config in `main.py`:

```python
MAX_PARSE_WORKERS = 8        # Increase for more CPU cores
SEMANTIC_CHUNK_AGG = "max"   # Faster (slightly less accurate)
```

---

### ❌ Scanned PDFs not working

**Problem:** This tool doesn't include OCR.

**Solution:** Pre-process with Tesseract:

```bash
pip install pytesseract
# Also install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
```

---

### ❌ PyTorch installation fails

**Solution:** Install CPU-only version:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

### ❌ Model download stuck

**Problem:** First run downloads ~80MB model.

**Solution:** 
- Check internet connection
- Wait for download to complete
- Model is cached after first download

---

## 12. FAQ

<details>
<summary><strong>Q: Does this tool use AI/ML?</strong></summary>

Yes! It uses:
- **Sentence-Transformers** for semantic understanding (pre-trained, no training needed)
- **RapidFuzz** for fuzzy string matching
- **Custom heuristics** for experience/education extraction

</details>

<details>
<summary><strong>Q: Can I use this for real hiring?</strong></summary>

This is designed for **academic/portfolio use**. For production hiring:
- Add proper data security
- Consider bias/fairness auditing
- Don't use as sole decision maker
- Comply with local hiring laws

</details>

<details>
<summary><strong>Q: What file formats are supported?</strong></summary>

| Format | Extension | Notes |
|--------|-----------|-------|
| PDF | `.pdf` | Up to 100 pages, text-based only |
| Word | `.docx` | Including tables |
| Text | `.txt`, `.md` | UTF-8 encoded |

⚠️ Scanned/image PDFs are NOT supported (no OCR).

</details>

<details>
<summary><strong>Q: How long does processing take?</strong></summary>

| Resumes | Approximate Time |
|---------|------------------|
| 1 | ~2 seconds |
| 10 | ~5 seconds |
| 50 | ~25 seconds |

First run is slower (~30s) due to model loading.

</details>

<details>
<summary><strong>Q: Can I customize the scoring weights?</strong></summary>

Yes! Edit `ScoringWeights` in `main.py`:

```python
@dataclass
class ScoringWeights:
    semantic: float = 0.35    # Increase for more NLP focus
    keywords: float = 0.25    # Increase for keyword matching
    must_haves: float = 0.20  # Increase to penalize missing skills
    # ... etc
```

</details>

<details>
<summary><strong>Q: Why is a candidate marked REJECT?</strong></summary>

REJECT happens when:
- Must-have score < 50%
- Semantic similarity < 40%
- 3+ red flags detected
- Total score < 45

Check the "View Details" modal for specific reasons.

</details>

---

<div align="center">

## 🎓 About

**ResumeScreener v2.2.0**

Built with ❤️ by [Siham Isa](https://github.com/siham-isa)

Powered by [Sentence-Transformers](https://huggingface.co/sentence-transformers) • [FastAPI](https://fastapi.tiangolo.com) • [RapidFuzz](https://github.com/maxbachmann/RapidFuzz) • [Tailwind CSS](https://tailwindcss.com)

**[⬆ Back to Top](#-resumescreener--complete-documentation)**

</div>
