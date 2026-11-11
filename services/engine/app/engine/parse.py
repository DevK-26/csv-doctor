"""Step A — parse a CSV and infer each column's real type.

Why this exists: every later step (profiling, anomalies, drift) trusts these
types. A wrong type here poisons everything after it — e.g. an ID column like
00123 must stay TEXT, or it becomes the number 123 and loses its meaning.
That is why IDs are detected explicitly and excluded from numeric maths.
"""
from __future__ import annotations

import io
import re

import pandas as pd

MAX_ROWS = 50_000
MAX_BYTES = 10 * 1024 * 1024  # 10 MB

_ID_NAME = re.compile(r"(^|_)(id|code|sku|ref)($|_)", re.I)


class ParseError(ValueError):
    """Raised with a human-readable reason the API can show as-is."""


def parse_csv_bytes(raw: bytes) -> pd.DataFrame:
    if len(raw) > MAX_BYTES:
        raise ParseError("file_too_large: CSV Doctor v1 accepts files up to 10 MB.")
    for encoding in ("utf-8", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(raw), sep=None, engine="python",
                             encoding=encoding, dtype=str, keep_default_na=False)
            break
        except Exception:
            continue
    else:
        raise ParseError("not_a_csv: this file could not be parsed as CSV.")
    if df.empty or len(df.columns) == 0:
        raise ParseError("empty_file: the CSV has no rows or no columns.")
    if len(df) > MAX_ROWS:
        raise ParseError(f"too_many_rows: v1 accepts up to {MAX_ROWS:,} rows, got {len(df):,}.")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _is_datetime(series: pd.Series) -> bool:
    sample = series[series != ""].head(200)
    if len(sample) < 3:
        return False
    try:
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    except Exception:
        return False
    return parsed.notna().mean() >= 0.8


def _is_numeric(series: pd.Series) -> bool:
    sample = series[series != ""].head(500)
    if len(sample) == 0:
        return False
    return pd.to_numeric(sample, errors="coerce").notna().mean() >= 0.9


def infer_column_type(name: str, series: pd.Series) -> str:
    """One of: empty, constant, id, datetime, numeric, boolean, categorical, text."""
    values = series.astype(str)
    non_empty = values[values != ""]
    if len(non_empty) == 0:
        return "empty"
    if non_empty.nunique() == 1:
        return "constant"
    # Targeting guard: unique / ID-named columns are identifiers, not measures.
    if _ID_NAME.search(name) or (non_empty.nunique() == len(non_empty) and _is_numeric(non_empty)):
        if _ID_NAME.search(name) or non_empty.nunique() > 20:
            return "id"
    lowered = set(non_empty.str.lower().unique())
    if lowered <= {"true", "false", "yes", "no", "0", "1"} and len(lowered) <= 2:
        return "boolean"
    if _is_datetime(non_empty):
        return "datetime"
    if _is_numeric(non_empty):
        return "numeric"
    if non_empty.nunique() <= max(20, int(0.05 * len(non_empty))):
        return "categorical"
    return "text"


def infer_types(df: pd.DataFrame) -> dict[str, str]:
    return {col: infer_column_type(col, df[col]) for col in df.columns}
