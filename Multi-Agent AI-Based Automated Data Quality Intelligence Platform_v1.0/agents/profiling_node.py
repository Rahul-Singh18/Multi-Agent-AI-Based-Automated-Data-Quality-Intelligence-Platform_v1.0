"""Node 2 — Data Profiling Agent."""
import pandas as pd
import numpy as np
from agents.state import PipelineState
from agents.validation_node import load_dataframe


def profiling_node(state: PipelineState) -> PipelineState:
    df = load_dataframe(state["filepath"])
    total_rows = len(df)
    profile: dict = {}

    for col in df.columns:
        s = df[col]
        null_count = int(s.isnull().sum())
        null_pct = round(null_count / total_rows * 100, 2) if total_rows else 0
        cp: dict = {
            "dtype": str(s.dtype),
            "null_count": null_count,
            "null_pct": null_pct,
            "unique_count": int(s.nunique()),
        }
        if pd.api.types.is_numeric_dtype(s) and s.dtype != bool:
            nn = s.dropna()
            if len(nn):
                cp.update({
                    "mean":   round(float(nn.mean()), 4),
                    "median": round(float(nn.median()), 4),
                    "std":    round(float(nn.std()), 4),
                    "min":    round(float(nn.min()), 4),
                    "max":    round(float(nn.max()), 4),
                    "q1":     round(float(nn.quantile(0.25)), 4),
                    "q3":     round(float(nn.quantile(0.75)), 4),
                })
        else:
            top = s.value_counts().head(5)
            cp["top_values"] = {str(k): int(v) for k, v in top.items()}
        profile[col] = cp

    return {
        **state,
        "status": "analyzing",
        "profile": {
            "total_rows":    total_rows,
            "total_columns": len(df.columns),
            "total_cells":   total_rows * len(df.columns),
            "total_nulls":   int(df.isnull().sum().sum()),
            "column_profiles": profile,
        },
    }
