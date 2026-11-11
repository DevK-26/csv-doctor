"""Day 1 golden tests for parse + profile + health score.

Run:  python -m pytest tests/ -v   (from services/engine/)
These lock in the deterministic behaviour: if a future change alters a
score, a test must fail loudly instead of users seeing a different number.
"""
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from app.engine.parse import infer_types, parse_csv_bytes
from app.engine.profile import health_score, profile_dataframe

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_and_types_on_dirty_fixture():
    df = parse_csv_bytes((FIXTURES / "dirty_sales.csv").read_bytes())
    assert len(df) == 8
    types = infer_types(df)
    assert types["date"] == "datetime"
    assert types["revenue"] == "numeric"
    assert types["region"] == "categorical"
    assert types["order_id"] == "id"          # IDs must never become numbers
    assert types["notes"] == "constant"


def test_profile_counts_duplicates_and_missing():
    df = parse_csv_bytes((FIXTURES / "dirty_sales.csv").read_bytes())
    prof = profile_dataframe(df)
    assert prof["rows"] == 8
    assert prof["duplicate_rows"] == 2        # rows 2 and 7 repeat earlier rows
    assert prof["missing_pct"] > 0            # revenue + region have blanks


def test_health_score_punishes_dirty_and_rewards_clean():
    dirty = profile_dataframe(parse_csv_bytes((FIXTURES / "dirty_sales.csv").read_bytes()))
    dirty_score, reasons = health_score(dirty)
    assert dirty_score < 90 and reasons

    clean = pd.DataFrame({"a": [1, 2, 3, 4], "b": ["x", "y", "x", "y"]}).astype(str)
    clean.loc[:, :] = clean.astype(str)
    clean_prof = profile_dataframe(clean)
    clean_score, _ = health_score(clean_prof)
    assert clean_score > dirty_score
