"""Step F — relationships. DAY 5 in the build plan (stub on Day 1).

Planned implementation (BLUEPRINT §3F): Pearson + Spearman for numeric pairs
(report |r| >= 0.4, top 10), categorical-vs-numeric group differences
(report max/min group-mean ratio >= 1.5 with n >= 30 per group), heatmap data
for the top-8 numeric columns by variance.
"""
from __future__ import annotations

import pandas as pd


def find_relationships(df: pd.DataFrame, types: dict[str, str]) -> dict:
    # TODO Day 5: correlations + group differences (see docstring).
    return {"skipped": True, "reason": "not implemented yet (Day 5)",
            "correlations": [], "group_differences": []}
