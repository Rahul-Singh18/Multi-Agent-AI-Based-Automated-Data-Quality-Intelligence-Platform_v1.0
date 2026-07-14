"""FastAPI application — Data Quality Intelligence Platform v2 with SQLite DB."""
import os
import uuid
import threading
import traceback

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

import store
import database as db
from pipeline import analysis_graph, cleaning_graph

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(title="DataQuality AI", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── background runners ────────────────────────────────────────────────────────

def _run_analysis(job_id: str, filepath: str, filename: str):
    try:
        store.update(job_id, {"status": "validating"})
        result = analysis_graph.invoke({
            "job_id": job_id, "filepath": filepath,
            "filename": filename, "status": "validating",
        })
        store.update(job_id, dict(result))
        # Persist to DB
        if result.get("status") == "awaiting_decision":
            db.save_report(job_id, result)
        elif result.get("status") == "failed":
            db.mark_failed(job_id, result.get("error", ""))
    except Exception:
        err = traceback.format_exc()
        store.update(job_id, {"status": "failed", "error": err})
        db.mark_failed(job_id, err)


def _run_cleaning(job_id: str):
    try:
        store.update(job_id, {"status": "cleaning"})
        current = store.get(job_id)
        result  = cleaning_graph.invoke(dict(current))
        store.update(job_id, dict(result))

        if result.get("status") == "complete":
            # Copy cleaned file into DB folder
            cr = result.get("cleaning_result") or {}
            clean_fname = cr.get("clean_filename")
            if clean_fname:
                src = os.path.join(OUTPUT_DIR, clean_fname)
                db.save_cleaned(job_id, src, result)
            else:
                db.save_report(job_id, result)
        else:
            db.mark_failed(job_id, result.get("error", ""))
    except Exception:
        err = traceback.format_exc()
        store.update(job_id, {"status": "failed", "error": err})
        db.mark_failed(job_id, err)


# ── request models ────────────────────────────────────────────────────────────

class DecisionRequest(BaseModel):
    action: str
    approved_actions: list[str] = []


# ── routes — analysis pipeline ────────────────────────────────────────────────

@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in (".csv", ".xlsx", ".xls"):
        raise HTTPException(400, "Only CSV and Excel files are supported.")

    job_id   = str(uuid.uuid4())
    filepath = os.path.join(UPLOAD_DIR, f"{job_id}{ext}")
    content  = await file.read()
    with open(filepath, "wb") as f:
        f.write(content)

    # Create DB record and copy raw file into DB folder immediately
    db.create_job(job_id, file.filename, filepath)

    store.create(job_id, {
        "job_id": job_id, "status": "queued",
        "filepath": filepath, "filename": file.filename,
    })

    threading.Thread(target=_run_analysis, args=(job_id, filepath, file.filename), daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/status/{job_id}")
async def status(job_id: str):
    job = store.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")

    safe = {
        "job_id": job_id, "status": job.get("status"),
        "filename": job.get("filename"), "error": job.get("error"),
    }
    if job.get("status") in ("awaiting_decision", "complete", "cleaning"):
        for k in ("validation", "profile", "quality", "anomaly", "score_before", "insights"):
            safe[k] = job.get(k)
    if job.get("status") == "complete":
        safe["cleaning_result"] = job.get("cleaning_result")
        safe["score_after"]     = job.get("score_after")
        safe["profile_after"]   = job.get("profile_after")
        safe["has_cleaned_file"]= bool((job.get("cleaning_result") or {}).get("clean_filename"))
    return safe


@app.post("/api/decide/{job_id}")
async def decide(job_id: str, body: DecisionRequest):
    job = store.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    if job.get("status") != "awaiting_decision":
        raise HTTPException(400, f"Job not awaiting decision (status: {job.get('status')}).")

    if body.action == "skip":
        store.update(job_id, {"status": "complete", "decision": "skip"})
        db.mark_skipped(job_id, store.get(job_id))
        return {"status": "complete", "decision": "skip"}

    if body.action == "approve":
        if not body.approved_actions:
            raise HTTPException(400, "No cleaning actions selected.")
        store.update(job_id, {"decision": "approve", "approved_actions": body.approved_actions})
        threading.Thread(target=_run_cleaning, args=(job_id,), daemon=True).start()
        return {"status": "cleaning"}

    raise HTTPException(400, "action must be 'skip' or 'approve'.")


@app.get("/api/download/{job_id}")
async def download(job_id: str):
    job = store.get(job_id)
    if job:
        # Try in-memory job first (current session)
        fname = (job.get("cleaning_result") or {}).get("clean_filename")
        if fname:
            path = os.path.join(OUTPUT_DIR, fname)
            if os.path.exists(path):
                return FileResponse(path, media_type="application/octet-stream",
                                    filename=f"cleaned_{job.get('filename','data.csv')}")

    # Fall back to DB folder
    files = db.get_job_files(job_id)
    for name, path in files.items():
        if name.startswith("cleaned"):
            orig_name = (store.get(job_id) or {}).get("filename") or "data"
            return FileResponse(path, media_type="application/octet-stream",
                                filename=f"cleaned_{orig_name}")
    raise HTTPException(404, "No cleaned file available for this job.")


@app.get("/api/download-raw/{job_id}")
async def download_raw(job_id: str):
    files = db.get_job_files(job_id)
    for name, path in files.items():
        if name.startswith("raw"):
            rec = db.get_job(job_id)
            fname = rec["dataset_name"] if rec else "raw"
            ext = os.path.splitext(name)[1]
            return FileResponse(path, media_type="application/octet-stream",
                                filename=f"raw_{fname}{ext}")
    raise HTTPException(404, "Raw file not found.")


# ── routes — database / history ───────────────────────────────────────────────

@app.get("/api/history")
async def history():
    """All jobs, most recent first."""
    return db.list_jobs()


@app.get("/api/datasets")
async def datasets():
    """One entry per unique dataset name with aggregated stats."""
    return db.list_datasets()


@app.get("/api/history/{job_id}")
async def history_job(job_id: str):
    """Full report for a past job from DB."""
    rec = db.get_job(job_id)
    if not rec:
        raise HTTPException(404, "Job not found in database.")
    report = db.get_report(job_id)
    files  = db.get_job_files(job_id)
    return {
        "record":  rec,
        "report":  report,
        "files":   list(files.keys()),
    }


@app.get("/api/history/{job_id}/files/{filename}")
async def history_file(job_id: str, filename: str):
    """Download any file from a past job's DB folder."""
    files = db.get_job_files(job_id)
    if filename not in files:
        raise HTTPException(404, f"File '{filename}' not found for job {job_id}.")
    return FileResponse(files[filename], media_type="application/octet-stream", filename=filename)


@app.get("/")
async def root():
    return {"message": "DataQuality AI v3 — FastAPI + LangGraph + SQLite"}
