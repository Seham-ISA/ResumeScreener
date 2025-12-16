<div align="center">

<!-- Custom Banner -->
<img src="si/>

<!-- Badges Row 1 - Status -->
[![Version](https://img.shields.io/badge/version-2.2.0-blue?style=flat-square)](https://github.com/siham-isa/resume-screener/releases)
[![Python](https://img.shields.io/badge/python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.112+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen?style=flat-square)](CONTRIBUTING.md)

<!-- Badges Row 2 - Tech -->
[![Sentence Transformers](https://img.shields.io/badge/🤗_Transformers-NLP-yellow?style=flat-square)](https://huggingface.co/sentence-transformers)
[![TailwindCSS](https://img.shields.io/badge/Tailwind-CSS-06B6D4?style=flat-square&logo=tailwindcss&logoColor=white)](https://tailwindcss.com)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000?style=flat-square)](https://github.com/psf/black)

<br/>

**Production-ready resume screening system combining semantic NLP, fuzzy matching, and explainable AI scoring.**

*Zero training required • Process 50 resumes in <30 seconds • Export-ready results*

[**Live Demo**](#-quick-start) · [**Documentation**](docs/) · [**Report Bug**](../../issues) · [**Request Feature**](../../issues)

</div>

---

## 📋 Table of Contents

<details>
<summary>Click to expand</summary>

- [Why ResumeScreener?](#-why-resumescreener)
- [Features](#-features)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [API Reference](#-api-reference)
- [Configuration](#%EF%B8%8F-configuration)
- [Performance](#-performance)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

</details>

---

## 🎯 Why ResumeScreener?

<table>
<tr>
<td width="50%">

### The Problem

Hiring managers spend **23 hours** screening resumes for a single hire. Traditional ATS systems use rigid keyword matching that:

- ❌ Miss qualified candidates with different terminology
- ❌ Can't understand context or semantic meaning
- ❌ Provide no explainability for decisions
- ❌ Require expensive training data

</td>
<td width="50%">

### Our Solution

ResumeScreener uses **pretrained transformer models** to understand meaning, not just keywords:

- ✅ Semantic similarity catches "ML Engineer" ↔ "Machine Learning"
- ✅ Fuzzy matching handles typos and variations
- ✅ Full explainability: see exactly why each score
- ✅ Works out-of-the-box, zero training needed

</td>
</tr>
</table>

---

## ✨ Features

<div align="center">

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SCORING PIPELINE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐            │
│   │   JOB    │    │  RESUME  │    │ MATCHING │    │  SCORE   │            │
│   │   DESC   │───▶│  PARSER  │───▶│  ENGINE  │───▶│ & RANK   │            │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘            │
│        │                               │                                    │
│        ▼                               ▼                                    │
│   ┌──────────┐                  ┌──────────────┐                           │
│   │ KEYWORD  │                  │   SEMANTIC   │                           │
│   │EXTRACTION│                  │  SIMILARITY  │                           │
│   └──────────┘                  └──────────────┘                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

</div>

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Semantic Matching** | `all-MiniLM-L6-v2` | Understands meaning beyond keywords |
| **Fuzzy Search** | RapidFuzz | Handles typos, abbreviations, variations |
| **Synonym Engine** | Custom mapping | `JS` ↔ `JavaScript`, `C++` ↔ `cpp` |
| **Experience Parser** | Regex + NLP | Extracts years from date ranges |
| **Red Flag Detection** | Heuristics | Catches keyword stuffing, job hopping |
| **Chunk Processing** | Sliding window | Handles resumes of any length |

### Scoring Breakdown

```python
final_score = (
    0.35 × semantic_similarity  +  # How well does the resume match JD meaning?
    0.25 × keyword_coverage     +  # What % of important keywords found?
    0.20 × must_have_score      +  # Proportional: 4/5 = 80%, not binary
    0.10 × experience_score     +  # Years extracted vs required
    0.05 × education_score      +  # Degree level detection
    0.05 × bonus_score             # Nice-to-have matches
) × (1 - 0.05 × red_flag_count)    # Penalty for red flags
```

---

## 🏗 Architecture

```
resume-screener/
│
├── 📄 main.py                 # FastAPI application + scoring engine
│   ├── ScreeningEngine        # Core ML pipeline
│   ├── ScoringWeights         # Configurable weight dataclass
│   └── API Routes             # /api/screen, /api/health
│
├── 📁 static/
│   ├── index.html             # Dashboard SPA (Tailwind + Vanilla JS)
│   └── important.html         # Documentation page
│
├── 📋 requirements.txt        # Pinned dependencies
└── 📖 README.md
```

### System Design

```
                                    ┌─────────────────┐
                                    │   Web Browser   │
                                    │  (Dashboard UI) │
                                    └────────┬────────┘
                                             │ HTTP
                                             ▼
┌────────────────────────────────────────────────────────────────────────┐
│                            FastAPI Server                               │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐ │
│  │  File Parser │    │   Embedding  │    │     Scoring Engine       │ │
│  │              │    │    Model     │    │                          │ │
│  │ • PDF (pypdf)│    │              │    │ • Semantic similarity    │ │
│  │ • DOCX       │───▶│  MiniLM-L6   │───▶│ • Keyword matching       │ │
│  │ • TXT        │    │   (384-dim)  │    │ • Experience extraction  │ │
│  │              │    │              │    │ • Red flag detection     │ │
│  └──────────────┘    └──────────────┘    └──────────────────────────┘ │
│         │                                            │                 │
│         └──────────────┬─────────────────────────────┘                 │
│                        ▼                                               │
│              ┌──────────────────┐                                      │
│              │  ThreadPoolExec  │  Parallel processing                 │
│              │  (4 workers)     │  for batch uploads                   │
│              └──────────────────┘                                      │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

| Requirement | Version | Check |
|-------------|---------|-------|
| Python | 3.10+ | `python --version` |
| pip | Latest | `pip --version` |
| RAM | 2GB+ | For model loading |

### Installation

```bash
# Clone repository
git clone https://github.com/siham-isa/resume-screener.git
cd resume-screener

# Create isolated environment
python -m venv .venv

# Activate (choose your OS)
source .venv/bin/activate      # Linux/macOS
.venv\Scripts\activate         # Windows CMD
.venv\Scripts\Activate.ps1     # Windows PowerShell

# Install dependencies
pip install -U pip && pip install -r requirements.txt

# Launch server
python main.py
```

<div align="center">

🎉 **Open http://127.0.0.1:8000 in your browser**

</div>

### One-Liner (for the impatient)

```bash
git clone https://github.com/siham-isa/resume-screener.git && cd resume-screener && python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && python main.py
```

---

## 💡 Usage

### Web Interface

1. **Paste or upload** a Job Description
2. **Define requirements:**
   - Must-haves (comma-separated)
   - Nice-to-haves (bonus skills)
   - Years of experience
   - Education level
3. **Upload resumes** (drag & drop supported, max 50)
4. **Click "Run Screening"**
5. **Review results** → Filter → Export CSV

### Decision Matrix

| Decision | Score | Confidence | Recommended Action |
|:--------:|:-----:|:----------:|:-------------------|
| 🟢 **STRONG_HIRE** | 85+ | High | Interview immediately |
| 🔵 **HIRE** | 75-84 | Good | Proceed to next round |
| 🟡 **MAYBE** | 60-74 | Medium | Manual review needed |
| 🟠 **NO_HIRE** | 45-59 | Good | Archive for future |
| 🔴 **REJECT** | <45 | High | Clear mismatch |

---

## 📡 API Reference

<details>
<summary><strong>GET /api/health</strong> — Health check</summary>

```bash
curl http://127.0.0.1:8000/api/health
```

**Response:**
```json
{
  "status": "healthy",
  "model": "sentence-transformers/all-MiniLM-L6-v2",
  "version": "2.2.0",
  "timestamp": 1734350400
}
```

</details>

<details>
<summary><strong>POST /api/screen</strong> — Screen resumes</summary>

**Request:**
```bash
curl -X POST "http://127.0.0.1:8000/api/screen" \
  -F "jd_text=Senior Python Developer with AWS experience..." \
  -F "must_haves=Python, AWS, Docker" \
  -F "nice_to_haves=Kubernetes, Terraform" \
  -F "required_years=5" \
  -F "required_education=bachelor" \
  -F "resumes=@candidate1.pdf" \
  -F "resumes=@candidate2.docx"
```

**Parameters:**

| Field | Type | Required | Description |
|-------|------|:--------:|-------------|
| `jd_text` | string | ⚠️ | Job description text |
| `jd_file` | file | ⚠️ | Job description file |
| `must_haves` | string | ✗ | Required skills (comma-sep) |
| `nice_to_haves` | string | ✗ | Bonus skills (comma-sep) |
| `required_years` | int | ✗ | Minimum experience |
| `required_education` | string | ✗ | `bachelor`/`master`/`phd` |
| `resumes` | file[] | ✓ | Resume files (max 50) |

> ⚠️ Either `jd_text` or `jd_file` required

**Response:**
```json
{
  "summary": {
    "total_candidates": 10,
    "strong_hire": 2,
    "hire": 3,
    "maybe": 3,
    "no_hire": 1,
    "reject": 1
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
      "years_experience": 7.0,
      "matched_keywords": ["python", "aws", "docker"],
      "missing_must_haves": [],
      "matched_nice_to_haves": ["kubernetes"],
      "red_flags": [],
      "strengths": ["✅ Strong overall fit"],
      "reasoning": "**Decision: STRONG_HIRE** ..."
    }
  ],
  "grouped": { ... }
}
```

</details>

---

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_RESUMES` | 50 | Max files per request |
| `MAX_FILE_SIZE` | 10MB | Per-file size limit |
| `MAX_PDF_PAGES` | 100 | Pages to process |
| `MAX_PARSE_WORKERS` | 4 | Parallel threads |

### Scoring Weights

```python
# main.py - Adjust to your needs
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
# main.py - Extend TECH_SYNONYMS dict
TECH_SYNONYMS = {
    "your_term": ["alias1", "alias2"],
    "react native": ["rn", "reactnative"],
    # ...existing entries
}
```

---

## 📊 Performance

<div align="center">

| Metric | Value | Conditions |
|--------|-------|------------|
| **Cold Start** | ~3s | Model loading |
| **Per Resume** | ~150ms | After warm-up |
| **50 Resumes** | <30s | Parallel processing |
| **Memory** | ~500MB | Model in RAM |
| **Model Size** | 80MB | Downloaded once |

</div>

### Benchmarks

```
Hardware: Intel i7-10700 / 16GB RAM / SSD
Dataset: 100 resumes (mixed PDF/DOCX), avg 2 pages each

┌────────────────────┬───────────┬───────────┐
│ Operation          │ Time      │ Per Item  │
├────────────────────┼───────────┼───────────┤
│ File Parsing       │ 4.2s      │ 42ms      │
│ Embedding          │ 8.1s      │ 81ms      │
│ Scoring            │ 2.3s      │ 23ms      │
│ Total              │ 14.6s     │ 146ms     │
└────────────────────┴───────────┴───────────┘
```

---

## 🗺 Roadmap

- [x] Core screening engine
- [x] Semantic similarity with chunking
- [x] Fuzzy matching + synonyms
- [x] Red flag detection
- [x] Web dashboard
- [ ] OCR for scanned PDFs (Tesseract)
- [ ] Batch processing queue (Celery)
- [ ] Database persistence (PostgreSQL)
- [ ] User authentication (OAuth2)
- [ ] Multi-language support
- [ ] Custom model fine-tuning

See [open issues](../../issues) for feature requests.

---

## 🤝 Contributing

Contributions make the open-source community amazing. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 🙏 Acknowledgments

<table>
  <tr>
    <td align="center"><a href="https://huggingface.co/sentence-transformers"><img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" width="60"/><br/><sub><b>Sentence Transformers</b></sub></a></td>
    <td align="center"><a href="https://fastapi.tiangolo.com"><img src="https://fastapi.tiangolo.com/img/icon-white.svg" width="60"/><br/><sub><b>FastAPI</b></sub></a></td>
    <td align="center"><a href="https://github.com/maxbachmann/RapidFuzz"><img src="https://avatars.githubusercontent.com/u/42370428" width="60"/><br/><sub><b>RapidFuzz</b></sub></a></td>
    <td align="center"><a href="https://tailwindcss.com"><img src="https://tailwindcss.com/_next/static/media/tailwindcss-mark.3c5441fc7a190fb1800d4a5c7f07ba4b1345a9c8.svg" width="60"/><br/><sub><b>Tailwind CSS</b></sub></a></td>
  </tr>
</table>

---

<div align="center">

**Built with ❤️ by [Siham ISA](https://github.com/siham-isa)**

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer" width="100%"/>

</div>
