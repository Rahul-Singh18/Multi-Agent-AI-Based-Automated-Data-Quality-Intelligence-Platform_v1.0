"""Node 5 — Quality Score Agent."""
import pandas as pd
from agents.state import PipelineState
from agents.validation_node import load_dataframe

WEIGHTS = {"completeness": 0.30, "uniqueness": 0.25, "validity": 0.25, "consistency": 0.10, "accuracy": 0.10}


def _score(df: pd.DataFrame, profile: dict, quality: dict, anomaly: dict) -> dict:
    total = len(df)
    cells = total * len(df.columns)

    completeness = 1 - profile["total_nulls"] / cells if cells else 1.0
    dup = next((i["affected_rows"] for i in quality["issues"] if i["type"] == "duplicates"), 0)
    uniqueness = 1 - dup / total if total else 1.0
    inv = sum(i["affected_rows"] for i in quality["issues"] if i["type"] in ("invalid_email", "invalid_phone", "constant_column"))
    validity = 1 - min(inv / total, 1.0) if total else 1.0
    high_null = sum(1 for cp in profile["column_profiles"].values() if cp["null_pct"] > 50)
    consistency = 1 - high_null / len(df.columns) if df.columns.size else 1.0
    accuracy = 1 - min(anomaly["total_outlier_rows"] / total, 1.0) if total else 1.0

    comps = {
        "completeness": round(completeness * 100, 1),
        "uniqueness":   round(uniqueness * 100, 1),
        "validity":     round(validity * 100, 1),
        "consistency":  round(consistency * 100, 1),
        "accuracy":     round(accuracy * 100, 1),
    }
    overall = sum(WEIGHTS[k] * comps[k] / 100 for k in WEIGHTS) * 100
    return {
        "overall": round(overall, 1),
        "components": comps,
        "weights": WEIGHTS,
        "grade": "A" if overall >= 85 else "B" if overall >= 70 else "C" if overall >= 55 else "D",
    }


def score_node(state: PipelineState) -> PipelineState:
    df = load_dataframe(state["filepath"])
    score = _score(df, state["profile"], state["quality"], state["anomaly"])
    return {**state, "status": "generating_insights", "score_before": score}


# Exported for reuse in cleaning node
compute_score = _score
