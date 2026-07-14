"""Node 3 — Data Quality Agent. Detects all issue types robustly."""
import re
import pandas as pd
from agents.state import PipelineState
from agents.validation_node import load_dataframe

EMAIL_RE = re.compile(r'^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$')
PHONE_RE = re.compile(r'^\+?[\d\s\-().]{7,20}$')

# Columns that are likely IDs — skip for duplicate detection
ID_PATTERNS = re.compile(r'\bid\b|_id$|^id_|^index$|^row', re.IGNORECASE)


def _is_id_column(col: str, series: pd.Series) -> bool:
    """True if the column looks like a surrogate ID (unique values, numeric or ID-named)."""
    if ID_PATTERNS.search(col):
        return True
    if series.nunique() == len(series) and pd.api.types.is_numeric_dtype(series):
        # All unique integers → likely an ID
        if series.dropna().apply(lambda x: float(x) == int(x)).all():
            return True
    return False


def quality_node(state: PipelineState) -> PipelineState:
    df = load_dataframe(state["filepath"])
    profile = state["profile"]
    total = len(df)
    issues = []

    # ── 1. Missing values ─────────────────────────────────────────────────────
    for col, cp in profile["column_profiles"].items():
        if cp["null_count"] > 0:
            pct = cp["null_pct"]
            issues.append({
                "type": "missing_values",
                "column": col,
                "affected_rows": cp["null_count"],
                "affected_pct": pct,
                "severity": "high" if pct > 30 else "medium" if pct > 10 else "low",
                "description": f"'{col}' has {cp['null_count']} missing values ({pct}%)",
            })

    # ── 2. Duplicate rows — skip ID-like columns ──────────────────────────────
    id_cols = [c for c in df.columns if _is_id_column(c, df[c])]
    subset_cols = [c for c in df.columns if c not in id_cols]
    if subset_cols:
        dup_mask = df.duplicated(subset=subset_cols, keep='first')
    else:
        dup_mask = df.duplicated(keep='first')
    dup = int(dup_mask.sum())
    if dup:
        issues.append({
            "type": "duplicates",
            "column": "all",
            "affected_rows": dup,
            "affected_pct": round(dup / total * 100, 2),
            "severity": "high" if dup / total > 0.05 else "medium",
            "description": f"{dup} duplicate rows detected (ignoring ID columns: {id_cols})",
            "id_cols_ignored": id_cols,
        })

    # ── 3. Invalid emails ─────────────────────────────────────────────────────
    for col in df.columns:
        if any(k in col.lower() for k in ["email", "mail", "e_mail"]):
            nn = df[col].dropna().astype(str).str.strip()
            bad = nn[~nn.str.match(EMAIL_RE)]
            if len(bad):
                issues.append({
                    "type": "invalid_email",
                    "column": col,
                    "affected_rows": len(bad),
                    "affected_pct": round(len(bad) / total * 100, 2),
                    "severity": "medium",
                    "description": f"{len(bad)} invalid email addresses in '{col}'",
                    "samples": list(bad.head(3).values),
                })

    # ── 4. Invalid phone numbers ──────────────────────────────────────────────
    for col in df.columns:
        if any(k in col.lower() for k in ["phone", "mobile", "contact", "tel"]):
            nn = df[col].dropna().astype(str).str.strip()
            bad = nn[~nn.str.match(PHONE_RE)]
            if len(bad):
                issues.append({
                    "type": "invalid_phone",
                    "column": col,
                    "affected_rows": len(bad),
                    "affected_pct": round(len(bad) / total * 100, 2),
                    "severity": "low",
                    "description": f"{len(bad)} invalid phone numbers in '{col}'",
                    "samples": list(bad.head(3).values),
                })

    # ── 5. Out-of-range values (domain validation) ────────────────────────────
    range_rules = {
        "age":        (0, 120),
        "score":      (0, 100),
        "pct":        (0, 100),
        "percent":    (0, 100),
        "attendance": (0, 100),
        "quantity":   (0, None),
        "qty":        (0, None),
        "rating":     (0, 5),
        "grade":      (0, 100),
    }
    for col in df.columns:
        col_lower = col.lower()
        for keyword, (lo, hi) in range_rules.items():
            if keyword in col_lower and pd.api.types.is_numeric_dtype(df[col]):
                bad_mask = pd.Series([False] * len(df))
                if lo is not None:
                    bad_mask |= df[col].notna() & (df[col] < lo)
                if hi is not None:
                    bad_mask |= df[col].notna() & (df[col] > hi)
                cnt = int(bad_mask.sum())
                if cnt:
                    bad_vals = df.loc[bad_mask, col].values
                    issues.append({
                        "type": "out_of_range",
                        "column": col,
                        "affected_rows": cnt,
                        "affected_pct": round(cnt / total * 100, 2),
                        "severity": "high",
                        "description": f"{cnt} impossible values in '{col}' (expected {lo}–{hi if hi else '∞'})",
                        "samples": [float(v) for v in bad_vals[:3]],
                    })
                break

    # ── 6. Constant columns ───────────────────────────────────────────────────
    for col, cp in profile["column_profiles"].items():
        if cp["unique_count"] == 1:
            issues.append({
                "type": "constant_column",
                "column": col,
                "affected_rows": total,
                "affected_pct": 100.0,
                "severity": "low",
                "description": f"'{col}' has only one unique value — carries no information",
            })

    return {
        **state,
        "status": "scoring",
        "quality": {
            "issues": issues,
            "issue_count": len(issues),
            "id_cols_detected": id_cols,
        }
    }
