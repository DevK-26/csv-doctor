"""Plan -> pandas program text.

Used by the deterministic local generator. The program is ordinary, readable
pandas that the sandbox runs; it is not the verifier's code path (verify.py
recomputes the plan independently, without pandas aggregation).
"""
from __future__ import annotations

from .plan import Filter, QuerySpec


def _lit(value) -> str:
    return repr(value)


def _ts(value: str) -> str:
    return f"pd.Timestamp({value!r})"


def _condition(f: Filter, kinds: dict[str, str]) -> str:
    target = _ts(f.value) if kinds[f.column] == "datetime" else _lit(f.value)
    return f"(df[{f.column!r}] {f.op} {target})"


def _metric(spec: QuerySpec, frame_var: str) -> str:
    m = spec.metric
    if m.agg == "count" and m.column is None:
        return f"len({frame_var})"
    return f"{frame_var}[{m.column!r}].{m.agg}()"


def render_program(spec: QuerySpec, schema: dict, defensive: bool = False) -> str:
    """Return pandas source that assigns the answer to `result`.

    defensive=True is used for the one retry: it re-coerces every column the
    plan touches instead of trusting the typed frame it was handed.
    """
    kinds = {c["name"]: c["type"] for c in schema["columns"]}
    lines = ["import pandas as pd", ""]

    if defensive:
        touched = [f.column for f in spec.filters] + [spec.metric.column, spec.group_by]
        if spec.compare:
            touched.append(spec.compare.column)
        lines.append("df = df.copy()")
        for col in dict.fromkeys(c for c in touched if c):
            if kinds[col] == "numeric":
                lines.append(f"df[{col!r}] = pd.to_numeric(df[{col!r}], errors='coerce')")
            elif kinds[col] == "datetime":
                lines.append(f"df[{col!r}] = pd.to_datetime(df[{col!r}], errors='coerce')")
        lines.append("")

    if spec.filters:
        cond = " & ".join(_condition(f, kinds) for f in spec.filters)
        lines.append(f"d = df[{cond}]")
    else:
        lines.append("d = df")

    if spec.compare:
        c = spec.compare
        lines += [
            f"cut = {_ts(c.split)}",
            f"before = d[d[{c.column!r}] < cut]",
            f"after = d[d[{c.column!r}] >= cut]",
            "result = {",
            f"    'before': {_metric(spec, 'before')},",
            f"    'after': {_metric(spec, 'after')},",
            "    'n_before': len(before),",
            "    'n_after': len(after),",
            "}",
        ]
    elif spec.group_by:
        m = spec.metric
        grouped = f"d.groupby({spec.group_by!r})"
        if m.agg == "count" and m.column is None:
            lines.append(f"g = {grouped}.size()")
        else:
            lines.append(f"g = {grouped}[{m.column!r}].{m.agg}()")
        if spec.top:
            asc = not spec.top.descending
            lines.append(f"g = g.sort_values(ascending={asc}, kind='mergesort').head({spec.top.n})")
        lines.append("result = g")
    else:
        lines.append(f"result = {_metric(spec, 'd')}")
    return "\n".join(lines) + "\n"
