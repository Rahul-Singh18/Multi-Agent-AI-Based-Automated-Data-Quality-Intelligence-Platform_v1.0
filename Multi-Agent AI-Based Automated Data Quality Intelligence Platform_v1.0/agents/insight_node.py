"""Node 6 — Insight Generation Agent. Ollama local AI, smart rule-based fallback."""
import json
import urllib.request
from agents.state import PipelineState

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"

# Full recommendation catalog — all possible actions
REC_CATALOG = [
    {"id": "remove_duplicates",        "label": "Remove duplicate rows",          "applies_when": {"duplicates"},         "description": "Drop all duplicate rows, keeping the first occurrence."},
    {"id": "fix_out_of_range",         "label": "Fix impossible values",          "applies_when": {"out_of_range"},       "description": "Null out impossible values (e.g. age=9999, score=-5, negative quantities)."},
    {"id": "fill_missing_numeric",     "label": "Fill missing numeric values",    "applies_when": {"missing_values"},     "description": "Replace nulls in numeric columns with the column median."},
    {"id": "fill_missing_categorical", "label": "Fill missing text values",       "applies_when": {"missing_values"},     "description": "Replace nulls in text columns with the most common value (mode)."},
    {"id": "fix_emails",               "label": "Standardise email addresses",    "applies_when": {"invalid_email"},      "description": "Lowercase, trim, and validate emails. Null unrecoverable ones."},
    {"id": "fix_phones",               "label": "Fix phone numbers",              "applies_when": {"invalid_phone"},      "description": "Remove or null invalid phone number entries."},
    {"id": "cap_outliers",             "label": "Cap statistical outliers",       "applies_when": {"outliers"},           "description": "Winsorize extreme values in numeric columns to IQR bounds."},
]

IMPACT_MAP = {
    "Remove duplicate rows":       "Ensures each record is counted once; removes inflation in metrics.",
    "Fix impossible values":       "Eliminates nonsensical values like age=9999 or score=-5 that corrupt analysis.",
    "Fill missing numeric values": "Fills gaps so aggregations, models, and charts work on complete data.",
    "Fill missing text values":    "Fills unknown categories so grouping and filtering produce complete results.",
    "Standardise email addresses": "Fixes format issues so emails are usable for contact or validation.",
    "Fix phone numbers":           "Removes garbage phone entries that break contact workflows.",
    "Cap statistical outliers":    "Reduces extreme values so averages and ML models are not skewed.",
}

GRADE_DESC = {
    "A": "excellent quality and is ready for use",
    "B": "good quality with some minor issues worth addressing",
    "C": "moderate quality with notable issues that should be fixed before analysis",
    "D": "poor quality — significant issues must be fixed before this data is usable",
}


def _rule_summary(profile, quality, anomaly, score, issue_types, recs):
    parts = []
    parts.append(
        f"The dataset has {profile['total_rows']} rows and {profile['total_columns']} columns "
        f"with a quality score of {score['overall']}/100 (Grade {score['grade']}), indicating "
        f"{GRADE_DESC.get(score['grade'], 'issues to review')}."
    )
    if "missing_values" in issue_types:
        mv = [i for i in quality["issues"] if i["type"] == "missing_values"]
        total_mv = sum(i["affected_rows"] for i in mv)
        parts.append(f"There are {total_mv} missing values across {len(mv)} column(s) that need attention.")
    if "out_of_range" in issue_types:
        oor = [i for i in quality["issues"] if i["type"] == "out_of_range"]
        parts.append(f"Found {len(oor)} column(s) with impossible values (like age > 120 or negative scores) that should be fixed.")
    if "duplicates" in issue_types:
        d = next((i for i in quality["issues"] if i["type"] == "duplicates"), None)
        if d:
            parts.append(f"There are {d['affected_rows']} duplicate rows that inflate record counts and skew analysis.")
    if "invalid_email" in issue_types:
        parts.append("Several email addresses are malformed and need standardisation.")
    if "outliers" in issue_types:
        parts.append(f"Statistical outliers detected in {anomaly['outlier_column_count']} column(s) may distort aggregations.")
    if not issue_types:
        parts.append("No significant data quality issues were detected. The dataset appears clean and ready to use.")
    for r in recs:
        r["impact"] = IMPACT_MAP.get(r["label"], r["description"])
    return " ".join(parts)


def _call_ollama(prompt: str) -> str:
    payload = json.dumps({
        "model": OLLAMA_MODEL, "prompt": prompt,
        "stream": False, "format": "json"
    }).encode()
    req = urllib.request.Request(
        OLLAMA_URL, data=payload,
        headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=45) as resp:
        return json.loads(resp.read()).get("response", "")


def insight_node(state: PipelineState) -> PipelineState:
    profile, quality, anomaly, score = (
        state["profile"], state["quality"], state["anomaly"], state["score_before"]
    )

    issue_types = {i["type"] for i in quality["issues"]}
    if anomaly["total_outlier_rows"] > 0:
        issue_types.add("outliers")

    recs = [r.copy() for r in REC_CATALOG if r["applies_when"] & issue_types]

    summary = None
    if recs:
        issue_lines = "\n".join(
            f"- {i['description']} (severity: {i['severity']})"
            for i in quality["issues"]
        )
        if anomaly["outlier_column_count"]:
            issue_lines += f"\n- Statistical outliers in {anomaly['outlier_column_count']} column(s)"

        impact_keys = ", ".join(f'"{r["label"]}": "one sentence"' for r in recs)
        prompt = (
            "You are a Data Quality analyst. Respond ONLY with valid JSON, no markdown.\n\n"
            f"Dataset: {profile['total_rows']} rows, {profile['total_columns']} columns\n"
            f"Quality Score: {score['overall']}/100 (Grade {score['grade']})\n"
            f"Issues:\n{issue_lines or 'None detected'}\n\n"
            f"Return exactly: "
            f'{{"executive_summary": "3-4 sentence analysis", '
            f'"impact_statements": {{{impact_keys}}}}}'
        )
        try:
            raw = _call_ollama(prompt).strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1].lstrip("json").strip()
            out = json.loads(raw)
            summary = out.get("executive_summary", "")
            impacts = out.get("impact_statements", {})
            for r in recs:
                r["impact"] = impacts.get(r["label"], IMPACT_MAP.get(r["label"], r["description"]))
        except Exception:
            pass

    if not summary:
        summary = _rule_summary(profile, quality, anomaly, score, issue_types, recs)

    return {
        **state,
        "status": "awaiting_decision",
        "insights": {
            "executive_summary":   summary,
            "recommendations":     recs,
            "recommendation_count": len(recs),
        },
    }
