# Resume Screening Tool (Mini Project)

A modern, responsive web app that ranks resumes against a Job Description using:
- **Semantic similarity** (pretrained embeddings, no training needed)
- **Keyword coverage**
- **Must-have penalties and nice-to-have bonuses**

The stack is **FastAPI (Python)** + **Tailwind CSS** on a lightweight static frontend.

---

## ✨ Features
- Upload **1 Job Description** (PDF/DOCX/TXT or paste text)
- Upload **multiple resumes** (PDF/DOCX/TXT)
- Automatic **keyword extraction** (YAKE), with manual **must-have** / **nice-to-have** additions
- Scores combine: `0.60 * similarity + 0.30 * coverage + 0.10 * (bonuses - penalties)`
- Explainability: matched skills, missing must-haves, and snippets
- **CSV export** of results
- Mobile-friendly, modern Tailwind UI

---

## 🚀 Quickstart

> Requires Python 3.10+

```bash
cd app
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# If torch installation fails on your platform, try:
# pip install torch --index-url https://download.pytorch.org/whl/cpu

# Run the server (serves the frontend at http://127.0.0.1:8000/)
uvicorn main:app --reload --port 8000
```

Open **http://127.0.0.1:8000/** in your browser.

---

## 📂 Project Structure

```
app/
  main.py                # FastAPI app + scoring API + static file server
  requirements.txt
  static/
    index.html
    assets/
      js/app.js
      css/styles.css
```

---

## 🧠 How Scoring Works

- **Semantic Similarity**: Sentence-transformers (`all-MiniLM-L6-v2`) compares JD vs Resume as whole documents.
- **Keyword Coverage**: YAKE (or fallback) suggests top keywords from the JD; we count how many appear in the resume (with fuzzy matching).
- **Must-haves**: Any missing must-have triggers a penalty.
- **Nice-to-haves**: Each matched nice-to-have adds a small bonus.

Final score is bounded by the formula and then ranked.

> No training required. Works out-of-the-box across domains. You can later add a database (SQLite/Postgres) if you want persistence and history.

---

## 🧪 API (for reuse)

### `POST /api/score`

**multipart/form-data**

- `jd_text` *(optional)*: JD as plain text
- `jd_file` *(optional)*: JD file (PDF/DOCX/TXT)
- `resumes` *(required)*: one or more resume files
- `must_haves` *(optional)*: comma/newline-separated list
- `nice_to_haves` *(optional)*: comma/newline-separated list
- `weight_sim`, `weight_cov`, `weight_bonus` *(optional floats)*

**Response**:
```jsonc
{
  "keywords_used": ["python","nlp","..."],
  "must_haves": ["pytorch"],
  "nice_to_haves": ["aws"],
  "results": [
    {
      "filename": "Jane_Doe_CV.pdf",
      "candidate_name": "Jane Doe",
      "sim": 83.1,
      "coverage": 66.7,
      "total": 79.5,
      "matched_keywords": ["python","nlp"],
      "missing_must_haves": ["pytorch"],
      "matched_nice_to_haves": ["aws"],
      "snippets": ["..."]
    }
  ]
}
```

---

## 🔧 Notes & Tips
- If PDFs are image-scanned, you will need OCR (e.g., Tesseract). This MVP does **not** include OCR.
- For better accuracy, curate a small CSV of **skill synonyms** and normalize both JD and resumes.
- You can plug a database & auth later (SQLModel + OAuth), and store history of runs/admin pages.

---

## 📝 License
MIT — Use freely. Pull requests welcome.
