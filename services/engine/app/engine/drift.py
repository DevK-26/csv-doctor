"""Step E — drift / change-points. DAY 5 in the build plan (stub on Day 1).

Planned implementation (BLUEPRINT §3E): if a datetime column exists, sort by
it, compute rolling mean/std, and detect change-points (ruptures PELT if
available, else a ~40-line CUSUM fallback). This is the simplified, honest
product version of the ADWIN/DDM comparison in Khushi's concept-drift paper.
No datetime column -> section is skipped, never faked.
"""
from __future__ import annotations

import pandas as pd


def find_change_points(df: pd.DataFrame, types: dict[str, str]) -> dict:
    time_cols = [c for c, t in types.items() if t == "datetime"]
    if not time_cols:
        return {"skipped": True,
                "reason": "no datetime column in this file",
                "change_points": []}
    # TODO Day 5: rolling stats + change-point detection (see docstring).
    return {"skipped": True, "reason": "not implemented yet (Day 5)",
            "change_points": []}
