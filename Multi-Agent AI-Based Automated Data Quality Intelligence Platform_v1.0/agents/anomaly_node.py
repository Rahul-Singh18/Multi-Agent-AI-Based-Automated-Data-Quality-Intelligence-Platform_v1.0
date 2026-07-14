"""Node 4 — Anomaly Detection Agent (IQR outliers)."""
import pandas as pd
from agents.state import PipelineState
from agents.validation_node import load_dataframe


def anomaly_node(state: PipelineState) -> PipelineState:
    df = load_dataframe(state["filepath"])
    profile = state["profile"]
    total = len(df)
    outliers = []

    for col, cp in profile["column_profiles"].items():
        if "q1" not in cp or "q3" not in cp:
            continue
        q1, q3 = cp["q1"], cp["q3"]
        iqr = q3 - q1
        if iqr == 0:
            continue
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (df[col] < lo) | (df[col] > hi)
        cnt = int(mask.sum())
        if cnt:
            vals = df.loc[mask, col].dropna()
            pct = round(cnt / total * 100, 2)
            outliers.append({
                "column": col,
                "outlier_count": cnt,
                "outlier_pct": pct,
                "lower_bound": round(lo, 4),
                "upper_bound": round(hi, 4),
                "min_outlier": round(float(vals.min()), 4),
                "max_outlier": round(float(vals.max()), 4),
                "severity": "high" if pct > 5 else "medium" if pct > 1 else "low",
                "description": f"{cnt} outliers in '{col}'",
            })

    return {
        **state,
        "anomaly": {
            "outliers": outliers,
            "outlier_column_count": len(outliers),
            "total_outlier_rows": sum(o["outlier_count"] for o in outliers),
        },
    }
