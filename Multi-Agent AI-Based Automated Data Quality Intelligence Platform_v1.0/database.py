"""
database.py — SQLite persistence layer.

Folder structure created per job:
  db/
  └── {dataset_name}/
      └── {job_id}/
          ├── raw.{ext}        ← original upload (always saved)
          ├── cleaned.{ext}    ← only if user approved cleaning
          └── report.json      ← full analysis + cleaning report
"""
import os
import json
import sqlite3
import shutil
from datetime import datetime
from contextlib import contextmanager

BASE_DIR = os.path.join(os.path.dirname(__file__), "db")
DB_PATH  = os.path.join(BASE_DIR, "jobs.db")
os.makedirs(BASE_DIR, exist_ok=True)


# ── connection ────────────────────────────────────────────────────────────────

@contextmanager
def _conn():
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


# ── schema ────────────────────────────────────────────────────────────────────

def init_db():
    with _conn() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id          TEXT PRIMARY KEY,
                dataset_name    TEXT NOT NULL,
                folder_path     TEXT NOT NULL,
                status          TEXT NOT NULL DEFAULT 'processing',
                decision        TEXT,
                rows            INTEGER,
                columns         INTEGER,
                score_before    REAL,
                score_after     REAL,
                grade_before    TEXT,
                grade_after     TEXT,
                issue_count     INTEGER,
                has_cleaned     INTEGER DEFAULT 0,
                created_at      TEXT NOT NULL,
                completed_at    TEXT
            )
        """)
        con.execute("CREATE INDEX IF NOT EXISTS idx_dataset ON jobs(dataset_name)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_created ON jobs(created_at DESC)")


# ── folder helpers ────────────────────────────────────────────────────────────

def _safe_name(name: str) -> str:
    """Strip extension and sanitise for use as folder name."""
    base = os.path.splitext(name)[0]
    safe = "".join(c if c.isalnum() or c in "-_ " else "_" for c in base).strip()
    return safe or "dataset"


def job_folder(dataset_name: str, job_id: str) -> str:
    folder = os.path.join(BASE_DIR, _safe_name(dataset_name), job_id)
    os.makedirs(folder, exist_ok=True)
    return folder


# ── write operations ──────────────────────────────────────────────────────────

def create_job(job_id: str, filename: str, upload_path: str) -> str:
    """
    Register a new job and copy the raw file into the DB folder.
    Returns the folder path.
    """
    folder = job_folder(filename, job_id)
    ext    = os.path.splitext(filename)[1].lower()
    raw_dst = os.path.join(folder, f"raw{ext}")
    shutil.copy2(upload_path, raw_dst)

    with _conn() as con:
        con.execute("""
            INSERT OR IGNORE INTO jobs
              (job_id, dataset_name, folder_path, status, created_at)
            VALUES (?, ?, ?, 'processing', ?)
        """, (job_id, _safe_name(filename), folder, datetime.utcnow().isoformat()))

    return folder


def save_report(job_id: str, state: dict):
    """
    Called when analysis completes (awaiting_decision).
    Writes report.json and updates DB metadata.
    """
    folder = _get_folder(job_id)
    if not folder:
        return

    # Build report dict (exclude heavy objects like raw dataframes)
    report = {
        "job_id":       job_id,
        "filename":     state.get("filename"),
        "generated_at": datetime.utcnow().isoformat(),
        "validation":   state.get("validation"),
        "profile":      state.get("profile"),
        "quality":      state.get("quality"),
        "anomaly":      state.get("anomaly"),
        "score_before": state.get("score_before"),
        "insights":     state.get("insights"),
        "decision":     state.get("decision"),
        "cleaning_result": state.get("cleaning_result"),
        "score_after":  state.get("score_after"),
    }
    report_path = os.path.join(folder, "report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    profile     = state.get("profile") or {}
    quality     = state.get("quality") or {}
    score_b     = state.get("score_before") or {}

    with _conn() as con:
        con.execute("""
            UPDATE jobs SET
              status       = ?,
              rows         = ?,
              columns      = ?,
              score_before = ?,
              grade_before = ?,
              issue_count  = ?
            WHERE job_id = ?
        """, (
            state.get("status", "awaiting_decision"),
            profile.get("total_rows"),
            profile.get("total_columns"),
            score_b.get("overall"),
            score_b.get("grade"),
            quality.get("issue_count"),
            job_id,
        ))


def save_cleaned(job_id: str, src_cleaned_path: str, state: dict):
    """
    Copy cleaned file into DB folder and update DB after cleaning completes.
    """
    folder = _get_folder(job_id)
    if not folder:
        return

    ext     = os.path.splitext(src_cleaned_path)[1].lower()
    dst     = os.path.join(folder, f"cleaned{ext}")
    shutil.copy2(src_cleaned_path, dst)

    # Rewrite report with cleaning info included
    save_report(job_id, state)

    score_a = state.get("score_after") or {}

    with _conn() as con:
        con.execute("""
            UPDATE jobs SET
              status       = 'complete',
              decision     = 'approve',
              score_after  = ?,
              grade_after  = ?,
              has_cleaned  = 1,
              completed_at = ?
            WHERE job_id = ?
        """, (
            score_a.get("overall"),
            score_a.get("grade"),
            datetime.utcnow().isoformat(),
            job_id,
        ))


def mark_skipped(job_id: str, state: dict):
    """Called when user skips cleaning."""
    save_report(job_id, state)
    with _conn() as con:
        con.execute("""
            UPDATE jobs SET status='complete', decision='skip', completed_at=?
            WHERE job_id=?
        """, (datetime.utcnow().isoformat(), job_id))


def mark_failed(job_id: str, error: str):
    with _conn() as con:
        con.execute(
            "UPDATE jobs SET status='failed', completed_at=? WHERE job_id=?",
            (datetime.utcnow().isoformat(), job_id)
        )


# ── read operations ───────────────────────────────────────────────────────────

def _get_folder(job_id: str) -> str | None:
    with _conn() as con:
        row = con.execute("SELECT folder_path FROM jobs WHERE job_id=?", (job_id,)).fetchone()
    return row["folder_path"] if row else None


def list_jobs(limit: int = 100) -> list[dict]:
    """Return all jobs ordered by most recent first."""
    with _conn() as con:
        rows = con.execute("""
            SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?
        """, (limit,)).fetchall()
    return [dict(r) for r in rows]


def list_datasets() -> list[dict]:
    """Return one row per unique dataset_name with aggregated stats."""
    with _conn() as con:
        rows = con.execute("""
            SELECT
                dataset_name,
                COUNT(*)                            AS total_jobs,
                SUM(has_cleaned)                    AS cleaned_jobs,
                MAX(created_at)                     AS last_used,
                AVG(score_before)                   AS avg_score_before,
                AVG(CASE WHEN score_after IS NOT NULL THEN score_after END) AS avg_score_after
            FROM jobs
            GROUP BY dataset_name
            ORDER BY last_used DESC
        """).fetchall()
    return [dict(r) for r in rows]


def get_job(job_id: str) -> dict | None:
    with _conn() as con:
        row = con.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
    return dict(row) if row else None


def get_job_files(job_id: str) -> dict:
    """Return paths to all files for a job that actually exist on disk."""
    folder = _get_folder(job_id)
    if not folder:
        return {}
    files = {}
    for name in os.listdir(folder):
        full = os.path.join(folder, name)
        if os.path.isfile(full):
            files[name] = full
    return files


def get_report(job_id: str) -> dict | None:
    folder = _get_folder(job_id)
    if not folder:
        return None
    path = os.path.join(folder, "report.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# Initialise on import
init_db()
