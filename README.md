# Data Quality Intelligence Platform

A multi-agent AI system that analyses CSV and Excel datasets, detects data quality issues, scores your data, and helps you clean it — all through a simple web interface.

Built as a Final Year B.Tech Project.

---

## What it does

Upload any CSV or Excel file and the system will:

- Detect missing values, duplicate rows, invalid emails, invalid phone numbers, impossible values (like age = 9999), and statistical outliers
- Give your dataset a quality score from 0 to 100 with a grade (A / B / C / D)
- Show AI-generated recommendations explaining what should be fixed and why
- Let you choose which cleaning actions to apply (Human-in-the-Loop design)
- Clean the data and show a before vs after comparison report
- Save every job to a local database so you can access past analyses anytime

---

## Tech stack

| Layer | Technology |
|---|---|
| Backend | Python, FastAPI |
| Agent pipeline | LangGraph (7-node StateGraph) |
| Data processing | Pandas, NumPy |
| AI recommendations | Ollama (local LLM) with smart fallback |
| Database | SQLite (built into Python, no setup needed) |
| Frontend | HTML, CSS, Vanilla JavaScript |
| File support | CSV, XLSX, XLS |

---

## Project structure

```
dqp/
├── main.py                    ← FastAPI server, all API routes
├── pipeline.py                ← LangGraph graph definitions
├── store.py                   ← Thread-safe in-memory job store
├── database.py                ← SQLite persistence layer
├── index.html                 ← Frontend (open in browser)
├── requirements.txt
│
├── agents/
│   ├── state.py               ← Shared LangGraph state schema
│   ├── validation_node.py     ← Node 1: file format and schema check
│   ├── profiling_node.py      ← Node 2: column statistics
│   ├── quality_node.py        ← Node 3: issue detection
│   ├── anomaly_node.py        ← Node 4: outlier detection (IQR)
│   ├── score_node.py          ← Node 5: quality score 0-100
│   ├── insight_node.py        ← Node 6: AI recommendations
│   └── cleaning_node.py       ← Node 7: data cleaning
│
├── sample_datasets/           ← Messy test datasets
│   ├── employees_messy.csv
│   ├── sales_orders_messy.csv
│   └── students_messy.csv
│
├── uploads/                   ← Auto-created, gitignored
├── outputs/                   ← Auto-created, gitignored
└── db/                        ← Auto-created, gitignored
```

---

## Agent pipeline

```
Upload CSV / Excel
        |
[1] Validation Agent      file format, encoding, schema
        |
[2] Profiling Agent       stats per column, nulls, distributions
        |
[3] Quality Agent         missing values, duplicates, invalid emails/phones, impossible values
[4] Anomaly Agent         outliers using IQR method
        |
[5] Score Agent           quality score 0-100, grade A/B/C/D
        |
[6] Insight Agent         AI recommendations
        |
    HUMAN DECISION
   skip        approve
     |              |
  Report    [7] Cleaning Agent
              executes selected operations
                    |
              Before/after report
              Download cleaned file
```

---

## Quality score formula

| Dimension | Weight | Formula |
|---|---|---|
| Completeness | 30% | 1 - null rate |
| Uniqueness | 25% | 1 - duplicate ratio |
| Validity | 25% | 1 - invalid format ratio |
| Consistency | 10% | Penalises high-null columns |
| Accuracy | 10% | 1 - outlier ratio |

---

## Issues detected

| Issue | How detected |
|---|---|
| Missing values | Null count per column |
| Duplicates | Row comparison ignoring ID columns |
| Invalid emails | Regex pattern matching |
| Invalid phones | Pattern match on phone/mobile columns |
| Impossible values | Domain rules: age (0-120), score (0-100), etc. |
| Statistical outliers | IQR method |
| Constant columns | Columns with only one unique value |

---

## How to run

**Step 1 — Install dependencies**

```bash
pip install -r requirements.txt
```

**Step 2 — Start the backend**

```bash
python -m uvicorn main:app --reload --port 8000
```

Keep this terminal open.

**Step 3 — Open the frontend**

Double-click index.html to open it in your browser.

**Step 4 — Upload a dataset**

Drag and drop any CSV or Excel file.

---

## Optional — AI with Ollama (free, local)

By default the system uses smart rule-based recommendations.
For full AI analysis install Ollama from ollama.com then run:

```bash
ollama pull llama3.2
```

Keep Ollama running in the background. The system detects it automatically.

---

## API reference

| Method | Endpoint | Description |
|---|---|---|
| POST | /api/upload | Upload a file |
| GET | /api/status/{job_id} | Poll job state |
| POST | /api/decide/{job_id} | Submit skip or approve |
| GET | /api/download/{job_id} | Download cleaned file |
| GET | /api/download-raw/{job_id} | Download original file |
| GET | /api/history | All past jobs |
| GET | /api/datasets | All dataset folders |
| GET | /api/history/{job_id} | Full report for a past job |
| GET | /api/history/{job_id}/files/{filename} | Download any file from history |

---

## Database

Every job is saved automatically:

```
db/
└── dataset_name/
    └── job_id/
        ├── raw.csv         original upload (always saved)
        ├── cleaned.csv     saved only if cleaning was approved
        └── report.json     full analysis report
```

---

## Sample datasets

Three messy datasets included under sample_datasets/ for testing:

- employees_messy.csv — 40 rows, age = 999 and -5, duplicate employees, invalid emails
- sales_orders_messy.csv — 30 rows, negative quantity, total = 9999999, duplicate orders
- students_messy.csv — 35 rows, score = -5, age = 500, invalid emails, missing values

---

## Future improvements

- PDF report export
- Support for JSON and TSV formats
- User authentication
- Cloud deployment
- More cleaning strategies

---

## Author

Rahul, Ankur, Sachin, Shivanshu, Suryaprathap 
— CDAC (BDA) Project
— Multi-Agent AI-Based Automated Data Quality Intelligence Platform
