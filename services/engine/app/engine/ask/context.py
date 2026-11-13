"""Dataset context for Ask-Your-Data: the typed frame and the masked schema.

typed_frame() turns the all-strings frame from parse_csv_bytes into the typed
frame that generated programs see (numbers are numbers, dates are dates,
blank cells are missing). build_schema_context() is what a generator is
allowed to see: names, types, ranges and the value lists of low-cardinality
columns — never raw rows, and never values the facts masker would rewrite.
"""
from __future__ import annotations

import pandas as pd

from .. import profile as profile_mod
from ..facts import mask

MAX_VALUES_PER_COLUMN = 200


class ContextError(ValueError):
    """The dataset cannot be exposed to a generator safely; message is shown as-is."""


def typed_frame(df: pd.DataFrame, types: dict[str, str]) -> pd.DataFrame:
    """Typed copy of a parsed frame. Unparseable numbers/dates become missing, not errors."""
    out = {}
    for col in df.columns:
        s = df[col].astype(str)
        blank = s == ""
        kind = types[col]
        if kind == "numeric":
            out[col] = pd.to_numeric(s.mask(blank), errors="coerce").astype("float64")
        elif kind == "datetime":
            try:
                out[col] = pd.to_datetime(s.mask(blank), errors="coerce", format="mixed")
            except (ValueError, TypeError):
                out[col] = s.mask(blank, None)
        else:
            out[col] = s.mask(blank, None)
    return pd.DataFrame(out, index=df.index)


def build_schema_context(df: pd.DataFrame, types: dict[str, str]) -> dict:
    """Schema + light facts pack for generators (all values masked or omitted)."""
    prof = profile_mod.profile_dataframe(df)
    score, _ = profile_mod.health_score(prof)
    typed = typed_frame(df, types)
    by_name = {c["name"]: c for c in prof["columns"]}
    columns = []
    for col in df.columns:
        base = by_name[col]
        entry = {"name": mask(col), "type": types[col],
                 "missing_pct": base["missing_pct"], "unique": base["unique"]}
        if entry["name"] != col:
            raise ContextError("unqueryable_column: a column name looks like an e-mail/phone number "
                               "and cannot be shared with the code generator.")
        if types[col] == "numeric":
            for key in ("min", "max", "mean"):
                if key in base:
                    entry[key] = base[key]
        elif types[col] == "datetime":
            valid = typed[col].dropna()
            if len(valid):
                entry["min"] = valid.min().date().isoformat()
                entry["max"] = valid.max().date().isoformat()
        elif types[col] in ("categorical", "boolean"):
            counts = typed[col].dropna().value_counts()
            ordered = sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))
            values = [str(v) for v, _ in ordered if mask(str(v)) == str(v)]
            entry["values"] = values[:MAX_VALUES_PER_COLUMN]
            entry["values_truncated"] = len(values) > MAX_VALUES_PER_COLUMN
        columns.append(entry)
    return {
        "rows": prof["rows"],
        "cols": prof["cols"],
        "columns": columns,
        "facts": {"health_score": score, "missing_pct": prof["missing_pct"],
                  "duplicate_rows": prof["duplicate_rows"]},
    }
