<p align="center">
  <img src="https://img.shields.io/badge/version-2.2.0-blue?style=for-the-badge" alt="Version">
  <img src="https://img.shields.io/badge/python-3.10+-green?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License">
</p>

<h1 align="center">🎯 ResumeScreener</h1>

<p align="center">
  <strong>AI-powered resume screening & ranking tool</strong><br>
  Combining semantic matching, fuzzy keyword search, and intelligent scoring — no training required.
</p>

<p align="center">
  <a href="#-features">Features</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-how-it-works">How It Works</a> •
  <a href="#-api-reference">API</a> •
  <a href="#-configuration">Config</a>
</p>

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Semantic Matching** | Uses Sentence-Transformers embeddings with chunk-aware processing for long resumes |
| **Smart Keywords** | Weighted keyword coverage with fuzzy matching and synonym expansion |
| **Must-Have Validation** | Proportional scoring (4/5 matched = 80%, not binary pass/fail) |
| **Experience Extraction** | Parses date ranges and "X years" patterns automatically |
| **Education Detection** | Maps Bachelor/Master/PhD levels to scores |
| **Red Flag Detection** | Identifies keyword stuffing, short resumes, and frequent job changes |
| **Dashboard UI** | Clean interface with filtering, details modal, and CSV export |

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/resume-screener.git
cd resume-screener

# Create virtual environment
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -U pip
pip install -r requirements.txt
```

### Run the Server

```bash
python main.py
```

Open your browser at **http://127.0.0.1:8000**

> 📘 Swagger docs available at `/docs`

## 🧠 How It Works

### Scoring Formula

```
Total Score = 35% Semantic + 25% Keywords + 20% Must-Haves 
            + 10% Experience + 5% Education + 5% Bonus
```

Red flags apply a **-5% penalty each** to the final score.

### Decision Thresholds

| Decision | Score Range | Action |
|----------|-------------|--------|
| 🟢 **STRONG_HIRE** | 85-100 | Schedule interview immediately |
| 🔵 **HIRE** | 75-84 | Move to next round |
| 🟡 **MAYBE** | 60-74 | Needs manual review |
| 🟠 **NO_HIRE** | 45-59 | Not recommended |
| 🔴 **REJECT** | 0-44 | Clear mismatch |

### Synonym Support

The system automatically recognizes common tech variations:

- `JavaScript` ↔ `JS`, `ES6`, `ECMAScript`
- `C++` ↔ `cpp`, `C plus plus`
- `C#` ↔ `csharp`, `C sharp`
- `.NET` ↔ `dotnet`, `ASP.NET`
- `Machine Learning` ↔ `ML`
- `PostgreSQL` ↔ `Postgres`, `psql`
- And 30+ more...

## 📡 API Reference

### Health Check

```bash
GET /api/health
```

```json
{
  "status": "healthy",
  "model": "sentence-transformers/all-MiniLM-L6-v2",
  "version": "2.2.0"
}
```

### Screen Resumes

```bash
POST /api/screen
Content-Type: multipart/form-data
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `jd_text` | string | No* | Job description text |
| `jd_file` | file | No* | Job description file (PDF/DOCX/TXT) |
| `must_haves` | string | No | Comma-separated required skills |
| `nice_to_haves` | string | No | Comma-separated bonus skills |
| `required_years` | int | No | Minimum years of experience |
| `required_education` | string | No | `bachelor`, `master`, or `phd` |
| `resumes` | file[] | Yes | Resume files (max 50) |

> *Either `jd_text` or `jd_file` is required

#### Example Request

```bash
curl -X POST "http://127.0.0.1:8000/api/screen" \
  -F "jd_text=Looking for a Python developer with AWS experience..." \
  -F "must_haves=Python, AWS, Docker" \
  -F "nice_to_haves=React, TypeScript" \
  -F "required_years=3" \
  -F "resumes=@./cv1.pdf" \
  -F "resumes=@./cv2.docx"
```

#### Example Response

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
      "decision": "STRONG_HIRE",
      "confidence": 90.0,
      "total_score": 87.5,
      "semantic_score": 85.2,
      "keyword_coverage": 78.5,
      "must_have_score": 100.0,
      "matched_keywords": ["python", "aws", "docker"],
      "missing_must_haves": [],
      "red_flags": [],
      "strengths": ["✅ Strong overall fit with job description"]
    }
  ]
}
```

## ⚙️ Configuration

Edit constants in `main.py` to customize behavior:

```python
# Performance
MAX_PARSE_WORKERS = 4          # Parallel file parsing threads
FUZZY_PRE_FILTER = True        # Enable pre-filtering optimization

# Limits
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB per file
MAX_RESUMES = 50                   # Files per batch
MAX_PDF_PAGES = 100                # Pages to process per PDF

# Semantic Scoring
SEMANTIC_CHUNK_AGG = "weighted_top"  # Options: "max", "avg", "weighted_top"
SEMANTIC_TOP_K = 3                    # Chunks to average (for weighted_top)
```

### Custom Scoring Weights

```python
@dataclass
class ScoringWeights:
    semantic: float = 0.35
    keywords: float = 0.25
    must_haves: float = 0.20
    experience: float = 0.10
    education: float = 0.05
    bonus: float = 0.05
```

## 📁 Project Structure

```
resume-screener/
├── main.py              # FastAPI backend + scoring engine
├── requirements.txt     # Python dependencies
├── static/
│   ├── index.html       # Dashboard UI
│   └── important.html   # Documentation page (/guide)
└── README.md
```

## 🛠️ Tech Stack

- **Backend:** FastAPI, Uvicorn
- **NLP:** Sentence-Transformers (`all-MiniLM-L6-v2`)
- **Parsing:** pypdf, python-docx
- **Matching:** RapidFuzz
- **Frontend:** HTML, Tailwind CSS, Vanilla JS

## 📋 Supported Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| PDF | `.pdf` | Up to 100 pages |
| Word | `.docx` | Including tables |
| Text | `.txt`, `.md` | UTF-8 encoded |

## 🐛 Troubleshooting

<details>
<summary><strong>Port 8000 already in use</strong></summary>

```bash
python -m uvicorn main:app --port 8001
```
</details>

<details>
<summary><strong>C++ or C# keywords not matching</strong></summary>

Verify you're running v2.2.0:
```bash
curl http://127.0.0.1:8000/api/health
```
</details>

<details>
<summary><strong>Scanned PDFs not working</strong></summary>

This tool doesn't include OCR. For scanned documents, pre-process with Tesseract:
```bash
pip install pytesseract
```
</details>

## 📄 License

MIT License — feel free to use and modify.

---

<p align="center">
  Built with ❤️ by <strong>Boubaker Khemili</strong>
</p>
