"""Query plan — the closed vocabulary shared by the generator and the verifier.

A generator returns pandas code *and* a plan describing what that code is
supposed to compute. The plan is data, not code: it can be validated against
the schema and re-executed by the trusted verifier (verify.py) with a
different implementation, which is what makes the answer checkable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any

AGGS = ("count", "sum", "mean", "median", "min", "max", "nunique")
NUMERIC_AGGS = ("sum", "mean", "median", "min", "max")
OPS = ("==", "!=", ">", ">=", "<", "<=")
ORDER_OPS = (">", ">=", "<", "<=")
MAX_TOP = 100


class PlanError(ValueError):
    """The plan is malformed or refers to columns/values that do not exist."""


@dataclass(frozen=True)
class Filter:
    column: str
    op: str
    value: Any


@dataclass(frozen=True)
class Metric:
    agg: str
    column: str | None = None  # None only for count (= number of rows)


@dataclass(frozen=True)
class Compare:
    """Before/after split on a datetime column: before = < split, after = >= split."""
    column: str
    split: str  # ISO date, YYYY-MM-DD


@dataclass(frozen=True)
class TopN:
    n: int = 1
    descending: bool = True


@dataclass(frozen=True)
class QuerySpec:
    metric: Metric
    filters: tuple[Filter, ...] = field(default_factory=tuple)
    group_by: str | None = None
    compare: Compare | None = None
    top: TopN | None = None

    @property
    def kind(self) -> str:
        if self.compare:
            return "compare"
        return "groups" if self.group_by else "scalar"

    def to_dict(self) -> dict:
        return {
            "metric": {"agg": self.metric.agg, "column": self.metric.column},
            "filters": [{"column": f.column, "op": f.op, "value": f.value} for f in self.filters],
            "group_by": self.group_by,
            "compare": ({"column": self.compare.column, "split": self.compare.split}
                        if self.compare else None),
            "top": ({"n": self.top.n, "descending": self.top.descending}
                    if self.top else None),
        }

    @classmethod
    def from_dict(cls, raw: Any) -> "QuerySpec":
        """Strict parse of a JSON-style plan; raises PlanError on any deviation."""
        if not isinstance(raw, dict):
            raise PlanError("plan must be a JSON object")
        unknown = set(raw) - {"metric", "filters", "group_by", "compare", "top"}
        if unknown:
            raise PlanError(f"plan has unknown keys: {sorted(unknown)}")
        m = raw.get("metric")
        if not isinstance(m, dict) or "agg" not in m:
            raise PlanError("plan.metric must be an object with an 'agg'")
        metric = Metric(agg=str(m["agg"]), column=m.get("column"))
        filters = []
        for f in raw.get("filters") or []:
            if not isinstance(f, dict) or not {"column", "op", "value"} <= set(f):
                raise PlanError("each filter needs column, op and value")
            filters.append(Filter(str(f["column"]), str(f["op"]), f["value"]))
        compare = None
        if raw.get("compare"):
            c = raw["compare"]
            if not isinstance(c, dict) or not {"column", "split"} <= set(c):
                raise PlanError("plan.compare needs column and split")
            compare = Compare(str(c["column"]), str(c["split"]))
        top = None
        if raw.get("top"):
            t = raw["top"]
            if not isinstance(t, dict):
                raise PlanError("plan.top must be an object")
            top = TopN(n=t.get("n", 1), descending=bool(t.get("descending", True)))
        return cls(metric=metric, filters=tuple(filters),
                   group_by=raw.get("group_by"), compare=compare, top=top)


def _col(schema: dict, name: Any, what: str) -> dict:
    cols = {c["name"]: c for c in schema["columns"]}
    if not isinstance(name, str) or name not in cols:
        raise PlanError(f"{what}: column {name!r} does not exist")
    return cols[name]


def iso_date(value: Any, what: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise PlanError(f"{what}: {value!r} is not an ISO date (YYYY-MM-DD)") from exc


def validate_plan(spec: QuerySpec, schema: dict) -> None:
    """Check a plan against the dataset schema. Raises PlanError with an actionable message."""
    m = spec.metric
    if m.agg not in AGGS:
        raise PlanError(f"metric.agg {m.agg!r} is not one of {list(AGGS)}")
    if m.agg in NUMERIC_AGGS:
        col = _col(schema, m.column, "metric")
        if col["type"] != "numeric":
            raise PlanError(f"metric: {m.agg} needs a numeric column, "
                            f"but {m.column!r} is {col['type']}")
    elif m.agg == "nunique":
        _col(schema, m.column, "metric")
    elif m.column is not None:
        _col(schema, m.column, "metric")

    for f in spec.filters:
        col = _col(schema, f.column, "filter")
        if f.op not in OPS:
            raise PlanError(f"filter: operator {f.op!r} is not one of {list(OPS)}")
        if col["type"] == "numeric":
            if isinstance(f.value, bool) or not isinstance(f.value, (int, float)) \
                    or not math.isfinite(f.value):
                raise PlanError(f"filter: {f.column!r} is numeric, got value {f.value!r}")
        elif col["type"] == "datetime":
            iso_date(f.value, f"filter on {f.column!r}")
        else:
            if not isinstance(f.value, str):
                raise PlanError(f"filter: {f.column!r} is {col['type']}, value must be text")
            if f.op in ORDER_OPS:
                raise PlanError(f"filter: {f.op!r} is not valid for {col['type']} column {f.column!r}")

    if spec.group_by is not None:
        gtype = _col(schema, spec.group_by, "group_by")["type"]
        if gtype in ("numeric", "datetime"):
            raise PlanError(f"group_by: {spec.group_by!r} is {gtype}; grouping needs a "
                            "categorical/text column (time-period grouping is not supported)")
    if spec.compare is not None:
        col = _col(schema, spec.compare.column, "compare")
        if col["type"] != "datetime":
            raise PlanError(f"compare: {spec.compare.column!r} is not a datetime column")
        iso_date(spec.compare.split, "compare.split")
        if spec.group_by is not None:
            raise PlanError("compare together with group_by is not supported")
    if spec.top is not None:
        if spec.group_by is None:
            raise PlanError("top needs a group_by")
        if isinstance(spec.top.n, bool) or not isinstance(spec.top.n, int) \
                or not 1 <= spec.top.n <= MAX_TOP:
            raise PlanError(f"top.n must be an integer between 1 and {MAX_TOP}")
