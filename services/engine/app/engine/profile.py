"""Steps B + C — profile the data and compute the deterministic Health Score.

The Health Score is a pure function of the profile: same file in, same score
out, always. No ML and no LLM touches it. The exact deductions live in
health_score() only, and tests/test_profile.py locks them in.
"""
from __future__ import annotations

import pandas as pd

from .parse import infer_types


def profile_dataframe(df: pd.DataFrame) -> dict:
    types = infer_types(df)
    columns = []
    for col in df.columns:
        s = df[col].astype(str)
        non_empty = s[s != ""]
        entry = {
            "name": col,
            "type": types[col],
            "missing_pct": round(100.0 * (len(s) - len(non_empty)) / max(len(s), 1), 2),
            "unique": int(non_empty.nunique()),
        }
        if types[col] == "numeric":
            nums = pd.to_numeric(non_empty, errors="coerce").dropna()
            if len(nums):
                entry.update({
                    "min": float(nums.min()), "max": float(nums.max()),
                    "mean": round(float(nums.mean()), 4),
                    "median": float(nums.median()),
                    "negatives": int((nums < 0).sum()),
                })
        if types[col] in ("categorical", "boolean"):
            entry["top_values"] = {
                str(k): int(v) for k, v in non_empty.value_counts().head(5).items()
            }
        columns.append(entry)

    total_cells = max(len(df) * len(df.columns), 1)
    missing_cells = int(sum((df[c].astype(str) == "").sum() for c in df.columns))
    return {
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
        "missing_pct": round(100.0 * missing_cells / total_cells, 2),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_pct": round(100.0 * df.duplicated().mean(), 2) if len(df) else 0.0,
        "columns": columns,
    }


def health_score(profile: dict) -> tuple[int, list[str]]:
    """Start at 100, subtract for measured problems (caps per BLUEPRINT §3C)."""
    score, reasons = 100.0, []

    pen = min(profile["missing_pct"] * 1.5, 25)
    if pen > 0:
        score -= pen
        if pen >= 0.5:
            reasons.append(f"missing values: {profile['missing_pct']}% of cells (-{pen:.0f})")

    pen = min(profile["duplicate_pct"] * 2, 20)
    if pen > 0:
        score -= pen
        if pen >= 0.5:
            reasons.append(f"duplicate rows: {profile['duplicate_pct']}% (-{pen:.0f})")

    mixed = [c for c in profile["columns"] if c["type"] == "text" and c.get("unique", 0) > 0
             and c["name"].lower() in ("price", "amount", "age", "quantity", "qty")]
    # Day 2-3: mixed-type detection moves into parse.py; this is the v1 hook.
    pen = min(len(mixed) * 5, 15)
    if pen > 0:
        score -= pen
        reasons.append(f"possibly mixed-type columns: {len(mixed)} (-{pen})")

    dead = [c for c in profile["columns"] if c["type"] in ("constant", "empty")]
    pen = min(len(dead) * 4, 12)
    if pen > 0:
        score -= pen
        reasons.append(f"constant/empty columns: {', '.join(c['name'] for c in dead)} (-{pen})")

    return max(int(round(score)), 0), reasons
