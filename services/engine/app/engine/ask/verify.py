"""Deterministic verification of a sandboxed program's answer.

The program's reported result is checked two ways before anyone sees it:

* Sanity bounds taken from the dataset alone (finite, right type, counts
  within [0, rows], means/medians/min/max inside the column's observed range,
  group labels that really exist...).
* An independent re-computation of the *plan* in plain Python (lists and
  math.fsum — no pandas groupby/agg, no code from the program) compared with
  the reported numbers at 1e-9 relative tolerance.

A program whose numbers pass both is "verified". Anything else is reported by
the agent as unverified and its number is never displayed as an answer.
What verification does NOT cover: whether the plan itself captures what the
user meant. That is why the plan and the program are always shown alongside
the answer.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from .plan import QuerySpec, TopN

REL_TOL = 1e-9
ABS_TOL = 1e-9
METHOD = ("independent re-computation of the plan in plain Python (no pandas aggregation), "
          "plus dataset-derived sanity bounds")


class ResultShapeError(ValueError):
    """The program's `result` does not have the shape its plan promises (retry-able)."""


@dataclass
class Check:
    name: str
    passed: bool
    detail: str

    def to_dict(self) -> dict:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


@dataclass
class Verification:
    passed: bool
    checks: list[Check]
    reported: dict = field(repr=False)   # canonical form of what the program said
    expected: dict = field(repr=False)   # trusted re-computation, same canonical form

    def to_api(self) -> dict:
        # Numbers are deliberately absent: a failed verification must not leak a wrong value.
        return {"passed": self.passed, "method": METHOD,
                "checks": [c.to_dict() for c in self.checks]}


def _missing(v: Any) -> bool:
    return v is None or v != v  # NaN and NaT are the only values unequal to themselves


# ---------------------------------------------------------------- independent recompute

def _column(frame: pd.DataFrame, name: str) -> list:
    return frame[name].tolist()


def _to_dt(value: Any) -> datetime:
    return datetime.fromisoformat(str(value))


def _passes(v: Any, op: str, target: Any, datetime_col: bool) -> bool:
    if _missing(v):
        return op == "!="
    if datetime_col:
        v = v.to_pydatetime()
        target = _to_dt(target)
    if op == "==":
        return v == target
    if op == "!=":
        return v != target
    if op == ">":
        return v > target
    if op == ">=":
        return v >= target
    if op == "<":
        return v < target
    return v <= target


def _aggregate(agg: str, values: list) -> float | int | None:
    """`values` are the non-missing cells of the metric column."""
    if agg == "nunique":
        return len(set(values))
    if agg == "sum":
        return math.fsum(values)
    if not values:
        return None
    if agg == "mean":
        return math.fsum(values) / len(values)
    if agg == "median":
        s = sorted(values)
        mid = len(s) // 2
        return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2
    return min(values) if agg == "min" else max(values)


def _metric_over(spec: QuerySpec, frame: pd.DataFrame, idx: list[int]) -> tuple[Any, int, int]:
    """(value, matching rows, rows with a usable metric value) for the given row indices."""
    m = spec.metric
    if m.column is None:
        return len(idx), len(idx), len(idx)
    col = _column(frame, m.column)
    vals = [col[i] for i in idx if not _missing(col[i])]
    value = len(vals) if m.agg == "count" else _aggregate(m.agg, vals)
    return value, len(idx), len(vals)


def _filtered_rows(spec: QuerySpec, frame: pd.DataFrame, schema: dict) -> list[int]:
    kinds = {c["name"]: c["type"] for c in schema["columns"]}
    cols = [(_column(frame, f.column), f, kinds[f.column] == "datetime") for f in spec.filters]
    return [i for i in range(len(frame))
            if all(_passes(c[i], f.op, f.value, dt) for c, f, dt in cols)]


def _order(pairs: list[tuple[str, Any]], top: TopN) -> list[tuple[str, Any]]:
    """Value order (missing last), ties by label ascending — the rule generators must follow."""
    present = sorted((p for p in pairs if not _missing(p[1])),
                     key=lambda p: (-p[1] if top.descending else p[1], p[0]))
    absent = sorted((p for p in pairs if _missing(p[1])), key=lambda p: p[0])
    return (present + absent)[:top.n]


def recompute(spec: QuerySpec, frame: pd.DataFrame, schema: dict) -> dict:
    """Trusted answer for `spec`, in the canonical shape produced by canonicalise()."""
    rows = _filtered_rows(spec, frame, schema)
    if spec.compare:
        col = _column(frame, spec.compare.column)
        cut = _to_dt(spec.compare.split)
        before = [i for i in rows if not _missing(col[i]) and col[i].to_pydatetime() < cut]
        after = [i for i in rows if not _missing(col[i]) and col[i].to_pydatetime() >= cut]
        vb, nb, ub = _metric_over(spec, frame, before)
        va, na, ua = _metric_over(spec, frame, after)
        return {"kind": "compare", "before": vb, "after": va, "n_before": nb, "n_after": na,
                "valid_before": ub, "valid_after": ua}
    if spec.group_by:
        keys = _column(frame, spec.group_by)
        buckets: dict[str, list[int]] = {}
        for i in rows:
            if not _missing(keys[i]):
                buckets.setdefault(str(keys[i]), []).append(i)
        pairs = [(k, _metric_over(spec, frame, buckets[k])[0]) for k in sorted(buckets)]
        if spec.top:
            pairs = _order(pairs, spec.top)
        return {"kind": "groups", "groups": pairs, "rows": len(rows)}
    value, n, valid = _metric_over(spec, frame, rows)
    return {"kind": "scalar", "value": value, "rows": n, "valid": valid}


# ---------------------------------------------------------------- canonical program result

def _num(sv: Any, what: str) -> float | int | None:
    """Serialised scalar -> number or None. Anything else is a shape error."""
    if not isinstance(sv, dict) or sv.get("kind") != "scalar":
        raise ResultShapeError(f"{what} must be a number, got {sv.get('kind') if isinstance(sv, dict) else type(sv).__name__}")
    v = sv["value"]
    if v is None:
        return None
    if isinstance(v, bool):
        raise ResultShapeError(f"{what} must be a number, got a boolean")
    if isinstance(v, dict):
        return {"nan": None, "inf": math.inf, "-inf": -math.inf}[v["special"]]
    return v


def canonicalise(raw: dict, spec: QuerySpec) -> dict:
    """Map the worker's serialised `result` onto the shape the plan promises, or raise ResultShapeError."""
    kind = raw.get("kind")
    if spec.compare:
        if kind != "dict":
            raise ResultShapeError("`result` must be a dict with keys 'before' and 'after' "
                                   "(optionally 'n_before', 'n_after') for a before/after comparison")
        items = raw["items"]
        for key in ("before", "after"):
            if key not in items:
                raise ResultShapeError(f"`result` is missing the key {key!r}")
        out: dict = {"kind": "compare", "before": _num(items["before"], "result['before']"),
                     "after": _num(items["after"], "result['after']")}
        for key in ("n_before", "n_after"):
            out[key] = _num(items[key], f"result[{key!r}]") if key in items else None
        return out
    if spec.group_by:
        if kind == "series":
            labels, values = raw["index"], [{"kind": "scalar", "value": v} for v in raw["values"]]
        elif kind == "frame" and len(raw["columns"]) == 1:
            labels = raw["index"]
            values = [{"kind": "scalar", "value": r[0]} for r in raw["rows"]]
        elif kind == "frame" and len(raw["columns"]) == 2:
            labels = [r[0] for r in raw["rows"]]
            values = [{"kind": "scalar", "value": r[1]} for r in raw["rows"]]
        elif kind == "dict":
            labels, values = list(raw["items"]), list(raw["items"].values())
        else:
            raise ResultShapeError("`result` must be a pandas Series indexed by group label "
                                   "(or a dict / two-column DataFrame) for a grouped question")
        if raw.get("truncated"):
            raise ResultShapeError("`result` has too many groups to verify (limit 200)")
        return {"kind": "groups",
                "groups": [(str(l), _num(v, "group value")) for l, v in zip(labels, values)]}
    if kind != "scalar":
        raise ResultShapeError(f"`result` must be a single number, got {kind}")
    return {"kind": "scalar", "value": _num(raw, "`result`")}


# ---------------------------------------------------------------- comparison + sanity

def _close(a: Any, b: Any) -> bool:
    if _missing(a) or _missing(b):
        return _missing(a) and _missing(b)
    return math.isclose(a, b, rel_tol=REL_TOL, abs_tol=ABS_TOL)


def _agree(reported: Any, expected: Any, agg: str, usable: int) -> bool:
    """Equality, except that an aggregate over nothing may be reported as 0 or as missing."""
    if usable == 0 and agg in ("sum", "count", "nunique"):
        return _missing(reported) or reported == 0
    return _close(reported, expected)


def _bounds(frame: pd.DataFrame, column: str | None) -> tuple[float, float] | None:
    if column is None:
        return None
    vals = [v for v in _column(frame, column) if not _missing(v)]
    return (min(vals), max(vals)) if vals else None


def _sanity_scalar(name: str, value: Any, spec: QuerySpec, frame: pd.DataFrame,
                   checks: list[Check]) -> None:
    agg, rows = spec.metric.agg, len(frame)
    if _missing(value):
        checks.append(Check(f"{name}:type", True, "no value (empty aggregate)"))
        return
    finite = isinstance(value, (int, float)) and math.isfinite(value)
    checks.append(Check(f"{name}:finite_number", finite, "finite number" if finite else "not a finite number"))
    if not finite:
        return
    if agg in ("count", "nunique"):
        ok = float(value).is_integer() and 0 <= value <= rows
        checks.append(Check(f"{name}:count_range", ok, f"count must be an integer in [0, {rows}]"))
        return
    b = _bounds(frame, spec.metric.column)
    if b is None:
        return
    lo, hi = b
    if agg == "sum":
        cap = rows * max(abs(lo), abs(hi))
        ok = abs(value) <= cap * (1 + REL_TOL) + ABS_TOL
        checks.append(Check(f"{name}:sum_range", ok, "|sum| cannot exceed rows x largest magnitude in the column"))
    else:
        slack = REL_TOL * max(abs(lo), abs(hi), 1.0)
        ok = lo - slack <= value <= hi + slack
        checks.append(Check(f"{name}:within_column_range", ok,
                            f"{agg} must lie inside the column's observed range"))


def verify(raw_result: dict, spec: QuerySpec, frame: pd.DataFrame, schema: dict) -> Verification:
    """Verify a serialised sandbox result against `spec`. Raises ResultShapeError on wrong shape."""
    reported = canonicalise(raw_result, spec)
    expected = recompute(spec, frame, schema)
    checks: list[Check] = []
    agg, rows = spec.metric.agg, len(frame)

    if reported["kind"] == "scalar":
        _sanity_scalar("value", reported["value"], spec, frame, checks)
        ok = _agree(reported["value"], expected["value"], agg, expected["valid"])
        checks.append(Check("recompute:value", ok,
                            "matches the independent re-computation" if ok
                            else "disagrees with the independent re-computation"))
    elif reported["kind"] == "compare":
        for side in ("before", "after"):
            _sanity_scalar(side, reported[side], spec, frame, checks)
            ok = _agree(reported[side], expected[side], agg, expected[f"valid_{side}"])
            checks.append(Check(f"recompute:{side}", ok,
                                f"{side} value " + ("matches" if ok else "disagrees with")
                                + " the independent re-computation"))
        for key in ("n_before", "n_after"):
            if reported[key] is not None:
                ok = reported[key] == expected[key]
                checks.append(Check(f"recompute:{key}", ok, f"{key} row count "
                                    + ("matches" if ok else "disagrees with") + " the re-computation"))
    else:
        known = {str(v) for v in _column(frame, spec.group_by) if not _missing(v)}
        labels = [g[0] for g in reported["groups"]]
        checks.append(Check("groups:labels_exist", all(l in known for l in labels),
                            "every group label must occur in the group column"))
        checks.append(Check("groups:unique", len(set(labels)) == len(labels), "group labels must be unique"))
        for label, value in reported["groups"]:
            _sanity_scalar(f"group[{label}]", value, spec, frame, checks)
        if agg == "count":
            total = sum(v for _, v in reported["groups"] if isinstance(v, (int, float)))
            checks.append(Check("groups:count_total", total <= rows, f"group counts cannot sum past {rows} rows"))
        exp = expected["groups"]
        if spec.top:
            same = [g[0] for g in reported["groups"]] == [g[0] for g in exp]
            checks.append(Check("recompute:ranking", same, "group ranking "
                                + ("matches" if same else "disagrees with") + " the re-computation"))
        else:
            same = sorted(labels) == sorted(g[0] for g in exp)
            checks.append(Check("recompute:groups", same, "set of groups "
                                + ("matches" if same else "disagrees with") + " the re-computation"))
        expected_by = dict(exp)
        reported_by = dict(reported["groups"])
        ok = same and all(_close(reported_by[k], expected_by[k])
                          for k in expected_by if k in reported_by)
        checks.append(Check("recompute:group_values", ok, "group values "
                            + ("match" if ok else "disagree with") + " the independent re-computation"))

    return Verification(passed=all(c.passed for c in checks), checks=checks,
                        reported=reported, expected=expected)
