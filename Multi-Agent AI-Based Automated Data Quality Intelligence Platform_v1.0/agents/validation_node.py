"""Node 1 — Dataset Validation Agent."""
import os
import pandas as pd
from agents.state import PipelineState


def load_dataframe(filepath: str) -> pd.DataFrame:
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".csv":
        return pd.read_csv(filepath, low_memory=False)
    return pd.read_excel(filepath)


def validation_node(state: PipelineState) -> PipelineState:
    filepath = state["filepath"]
    ext = os.path.splitext(filepath)[1].lower()

    if ext not in [".csv", ".xlsx", ".xls"]:
        return {**state, "status": "failed", "error": "Unsupported file type. Upload CSV or Excel."}

    try:
        df = load_dataframe(filepath)
    except Exception as e:
        return {**state, "status": "failed", "error": f"Could not read file: {e}"}

    if df.empty or len(df.columns) == 0:
        return {**state, "status": "failed", "error": "File is empty or has no columns."}

    dup_cols = [c for c in df.columns if list(df.columns).count(c) > 1]

    return {
        **state,
        "status": "profiling",
        "validation": {
            "status": "passed",
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "duplicate_column_names": list(set(dup_cols)),
            "file_type": ext,
            "dtypes": {col: str(dt) for col, dt in df.dtypes.items()},
        },
    }
