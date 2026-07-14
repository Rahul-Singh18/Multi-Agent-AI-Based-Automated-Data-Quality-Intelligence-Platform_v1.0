"""Shared LangGraph state — passed between every node in the pipeline."""
from typing import Any, Optional
from typing_extensions import TypedDict


class PipelineState(TypedDict, total=False):
    # ── inputs ──────────────────────────────────────────────────────
    job_id: str
    filepath: str
    filename: str

    # ── agent outputs ────────────────────────────────────────────────
    validation: dict
    profile: dict
    quality: dict
    anomaly: dict
    score_before: dict
    insights: dict

    # ── human decision ───────────────────────────────────────────────
    decision: str                   # "skip" | "approve"
    approved_actions: list[str]

    # ── cleaning outputs ─────────────────────────────────────────────
    cleaning_result: dict
    score_after: dict
    profile_after: dict

    # ── control ──────────────────────────────────────────────────────
    status: str
    error: Optional[str]
