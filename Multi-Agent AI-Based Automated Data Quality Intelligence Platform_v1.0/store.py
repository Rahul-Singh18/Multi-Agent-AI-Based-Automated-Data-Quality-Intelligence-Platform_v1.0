"""Thread-safe in-memory job store."""
import threading
from typing import Optional

_lock = threading.Lock()
_jobs: dict[str, dict] = {}


def create(job_id: str, initial: dict) -> None:
    with _lock:
        _jobs[job_id] = initial


def get(job_id: str) -> Optional[dict]:
    with _lock:
        return _jobs.get(job_id)


def update(job_id: str, data: dict) -> None:
    with _lock:
        if job_id in _jobs:
            _jobs[job_id].update(data)


def all_jobs() -> dict:
    with _lock:
        return dict(_jobs)
