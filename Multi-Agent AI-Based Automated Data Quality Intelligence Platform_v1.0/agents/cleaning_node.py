"""
Node 7 — Data Cleaning Agent.
Robust, domain-aware cleaning. Fixed execution order.
Never touches the original file.
"""
import os
import re
import numpy as np
import pandas as pd
from agents.state import PipelineState
from agents.validation_node import load_dataframe

EMAIL_RE = re.compile(r'^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$')
PHONE_RE = re.compile(r'^\+?[\d\s\-().]{7,20}$')
ID_PATTERNS = re.compile(r'\bid\b|_id$|^id_|^index$|^row', re.IGNORECASE)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Domain range rules — same as quality_node
RANGE_RULES = {
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


def _is_id_col(col: str, series: pd.Series) -> bool:
    if ID_PATTERNS.search(col):
        return True
    if series.nunique() == len(series) and pd.api.types.is_numeric_dtype(series):
        try:
            if series.dropna().apply(lambda x: float(x) == int(x)).all():
                return True
        except Exception:
            pass
    return False


def _do_clean(df: pd.DataFrame, approved: list) -> tuple:
    cleaned = df.copy(deep=True)
    log = []
    approved_set = set(approved)

    # ── STEP 1: Remove duplicates (ignore ID columns) ─────────────────────────
    if "remove_duplicates" in approved_set:
        id_cols = [c for c in cleaned.columns if _is_id_col(c, cleaned[c])]
        subset  = [c for c in cleaned.columns if c not in id_cols]
        before  = len(cleaned)
        if subset:
            cleaned = cleaned.drop_duplicates(subset=subset, keep='first').reset_index(drop=True)
        else:
            cleaned = cleaned.drop_duplicates(keep='first').reset_index(drop=True)
        removed = before - len(cleaned)
        if removed:
            log.append({
                "action": "remove_duplicates",
                "label":  "Removed duplicate rows",
                "rows_affected": removed,
                "detail": f"{removed} duplicate rows removed; {len(cleaned)} rows remain",
            })

    # ── STEP 2: Fix out-of-range values (before outlier capping) ─────────────
    if "fix_out_of_range" in approved_set:
        total_fixed = 0
        for col in cleaned.columns:
            col_lower = col.lower()
            for keyword, (lo, hi) in RANGE_RULES.items():
                if keyword in col_lower and pd.api.types.is_numeric_dtype(cleaned[col]):
                    bad_mask = pd.Series([False] * len(cleaned))
                    if lo is not None:
                        bad_mask |= cleaned[col].notna() & (cleaned[col] < lo)
                    if hi is not None:
                        bad_mask |= cleaned[col].notna() & (cleaned[col] > hi)
                    cnt = int(bad_mask.sum())
                    if cnt:
                        cleaned.loc[bad_mask, col] = np.nan  # null them out
                        total_fixed += cnt
                    break
        if total_fixed:
            log.append({
                "action": "fix_out_of_range",
                "label":  "Nulled impossible values",
                "rows_affected": total_fixed,
                "detail": f"{total_fixed} out-of-range values set to null (e.g. age=9999, score=-5)",
            })

    # ── STEP 3: Cap statistical outliers (IQR, skip ID/range-ruled columns) ──
    if "cap_outliers" in approved_set:
        skip_cols = set()
        for col in cleaned.columns:
            col_lower = col.lower()
            for keyword in RANGE_RULES:
                if keyword in col_lower:
                    skip_cols.add(col)
                    break
            if _is_id_col(col, cleaned[col]):
                skip_cols.add(col)

        numeric_cols = [
            c for c in cleaned.select_dtypes(include=[np.number]).columns
            if c not in skip_cols
        ]
        total_capped = 0
        for col in numeric_cols:
            series = cleaned[col].dropna()
            if len(series) < 10:
                continue
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lo_b = q1 - 1.5 * iqr
            hi_b = q3 + 1.5 * iqr
            mask = cleaned[col].notna() & ((cleaned[col] < lo_b) | (cleaned[col] > hi_b))
            cnt  = int(mask.sum())
            if cnt:
                orig_dtype = df[col].dtype if col in df.columns else cleaned[col].dtype
                clipped = cleaned[col].clip(lower=lo_b, upper=hi_b)
                if pd.api.types.is_integer_dtype(orig_dtype):
                    clipped = clipped.round().astype('Int64')
                cleaned[col] = clipped
                total_capped += cnt
        if total_capped:
            log.append({
                "action": "cap_outliers",
                "label":  "Capped statistical outliers",
                "rows_affected": total_capped,
                "detail": f"{total_capped} values winsorized to IQR bounds (Q1-1.5×IQR, Q3+1.5×IQR)",
            })

    # ── STEP 4: Fill missing numeric values (median) ──────────────────────────
    if "fill_missing_numeric" in approved_set:
        total_filled = 0
        for col in cleaned.select_dtypes(include=[np.number]).columns:
            null_cnt = int(cleaned[col].isnull().sum())
            if null_cnt:
                med = float(cleaned[col].median())
                cleaned[col] = cleaned[col].fillna(med)
                total_filled += null_cnt
        if total_filled:
            log.append({
                "action": "fill_missing_numeric",
                "label":  "Filled missing numeric values",
                "rows_affected": total_filled,
                "detail": f"{total_filled} null cells filled with column median",
            })

    # ── STEP 5: Fill missing categorical values (mode) ────────────────────────
    if "fill_missing_categorical" in approved_set:
        total_filled = 0
        for col in cleaned.select_dtypes(include=["object", "category", "bool"]).columns:
            null_cnt = int(cleaned[col].isnull().sum())
            if null_cnt:
                modes = cleaned[col].mode()
                if len(modes):
                    cleaned[col] = cleaned[col].fillna(modes[0])
                    total_filled += null_cnt
        if total_filled:
            log.append({
                "action": "fill_missing_categorical",
                "label":  "Filled missing categorical values",
                "rows_affected": total_filled,
                "detail": f"{total_filled} null cells filled with column mode (most frequent value)",
            })

    # ── STEP 6: Standardise email addresses ───────────────────────────────────
    if "fix_emails" in approved_set:
        email_cols = [c for c in cleaned.columns
                      if any(k in c.lower() for k in ["email", "mail", "e_mail"])]
        total_fixed = 0
        for col in email_cols:
            mask = cleaned[col].notna()
            cleaned.loc[mask, col] = (
                cleaned.loc[mask, col].astype(str).str.lower().str.strip()
            )
            invalid = cleaned[col].notna() & ~cleaned[col].astype(str).str.fullmatch(EMAIL_RE)
            cnt = int(invalid.sum())
            if cnt:
                cleaned.loc[invalid, col] = np.nan
                total_fixed += cnt
        if total_fixed:
            log.append({
                "action": "fix_emails",
                "label":  "Standardised email addresses",
                "rows_affected": total_fixed,
                "detail": f"{total_fixed} malformed emails nulled after standardisation attempt",
            })

    # ── STEP 7: Standardise phone numbers ────────────────────────────────────
    if "fix_phones" in approved_set:
        phone_cols = [c for c in cleaned.columns
                      if any(k in c.lower() for k in ["phone", "mobile", "contact", "tel"])]
        total_fixed = 0
        for col in phone_cols:
            mask = cleaned[col].notna()
            cleaned.loc[mask, col] = cleaned.loc[mask, col].astype(str).str.strip()
            invalid = cleaned[col].notna() & ~cleaned[col].astype(str).str.match(PHONE_RE)
            cnt = int(invalid.sum())
            if cnt:
                cleaned.loc[invalid, col] = np.nan
                total_fixed += cnt
        if total_fixed:
            log.append({
                "action": "fix_phones",
                "label":  "Standardised phone numbers",
                "rows_affected": total_fixed,
                "detail": f"{total_fixed} invalid phone numbers nulled",
            })

    return cleaned, log


def _build_profile(df: pd.DataFrame) -> dict:
    total = len(df)
    cols  = {}
    for col in df.columns:
        s  = df[col]
        nc = int(s.isnull().sum())
        cp: dict = {
            "dtype":        str(s.dtype),
            "null_count":   nc,
            "null_pct":     round(nc / total * 100, 2) if total else 0,
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
        cols[col] = cp
    return {
        "total_rows":    total,
        "total_columns": len(df.columns),
        "total_cells":   total * len(df.columns),
        "total_nulls":   int(df.isnull().sum().sum()),
        "column_profiles": cols,
    }


def _build_quality(df: pd.DataFrame, profile: dict) -> dict:
    total = len(df)
    issues = []
    id_cols = [c for c in df.columns if _is_id_col(c, df[c])]
    for col, cp in profile["column_profiles"].items():
        if cp["null_count"]:
            pct = cp["null_pct"]
            issues.append({"type": "missing_values", "column": col,
                           "affected_rows": cp["null_count"], "affected_pct": pct,
                           "severity": "high" if pct > 30 else "medium" if pct > 10 else "low",
                           "description": f"'{col}' has {cp['null_count']} missing values"})
    subset = [c for c in df.columns if c not in id_cols]
    dup = int(df.duplicated(subset=subset or None, keep='first').sum())
    if dup:
        issues.append({"type": "duplicates", "column": "all", "affected_rows": dup,
                       "affected_pct": round(dup / total * 100, 2),
                       "severity": "high" if dup / total > 0.05 else "medium",
                       "description": f"{dup} duplicate rows"})
    for col in df.columns:
        if any(k in col.lower() for k in ["email", "mail"]):
            nn = df[col].dropna().astype(str)
            bad = nn[~nn.str.match(EMAIL_RE)]
            if len(bad):
                issues.append({"type": "invalid_email", "column": col,
                               "affected_rows": len(bad),
                               "affected_pct": round(len(bad) / total * 100, 2),
                               "severity": "medium",
                               "description": f"{len(bad)} invalid emails in '{col}'"})
    return {"issues": issues, "issue_count": len(issues), "id_cols_detected": id_cols}


def _build_anomaly(df: pd.DataFrame, profile: dict) -> dict:
    total = len(df)
    outliers = []
    id_cols = [c for c in df.columns if _is_id_col(c, df[c])]
    for col, cp in profile["column_profiles"].items():
        if col in id_cols or "q1" not in cp:
            continue
        iqr = cp["q3"] - cp["q1"]
        if iqr == 0:
            continue
        lo, hi = cp["q1"] - 1.5 * iqr, cp["q3"] + 1.5 * iqr
        mask = (df[col] < lo) | (df[col] > hi)
        cnt  = int(mask.sum())
        if cnt:
            pct = round(cnt / total * 100, 2)
            outliers.append({
                "column": col, "outlier_count": cnt, "outlier_pct": pct,
                "lower_bound": round(lo, 4), "upper_bound": round(hi, 4),
                "severity": "high" if pct > 5 else "medium" if pct > 1 else "low",
                "description": f"{cnt} outliers in '{col}'",
            })
    return {"outliers": outliers, "outlier_column_count": len(outliers),
            "total_outlier_rows": sum(o["outlier_count"] for o in outliers)}


def cleaning_node(state: PipelineState) -> PipelineState:
    try:
        df_orig  = load_dataframe(state["filepath"])
        approved = state.get("approved_actions", [])
        if not approved:
            return {**state, "status": "failed", "error": "No cleaning actions approved."}

        cleaned_df, log = _do_clean(df_orig, approved)

        # Save cleaned file
        ext         = os.path.splitext(state["filepath"])[1].lower()
        clean_fname = f"cleaned_{state['job_id']}{ext}"
        clean_path  = os.path.join(OUTPUT_DIR, clean_fname)
        if ext == ".csv":
            cleaned_df.to_csv(clean_path, index=False)
        else:
            cleaned_df.to_excel(clean_path, index=False)

        # Recalculate metrics
        from agents.score_node import compute_score
        new_profile = _build_profile(cleaned_df)
        new_quality = _build_quality(cleaned_df, new_profile)
        new_anomaly = _build_anomaly(cleaned_df, new_profile)
        new_score   = compute_score(cleaned_df, new_profile, new_quality, new_anomaly)

        return {
            **state,
            "status": "complete",
            "cleaning_result": {
                "log":           log,
                "rows_before":   len(df_orig),
                "rows_after":    len(cleaned_df),
                "rows_removed":  len(df_orig) - len(cleaned_df),
                "total_actions": len(log),
                "clean_filename": clean_fname,
            },
            "score_after":   new_score,
            "profile_after": new_profile,
        }
    except Exception as exc:
        import traceback
        return {**state, "status": "failed",
                "error": f"Cleaning failed: {exc}\n{traceback.format_exc()}"}
