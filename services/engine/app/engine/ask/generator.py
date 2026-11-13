"""Code generators for Ask-Your-Data.

A generator turns (question, schema) into a pandas program plus a *plan* that
says what the program computes. Two implementations ship:

* LocalRulesGenerator — deterministic, no network, no API key. Understands the
  common shapes: filter + group + aggregate + before/after compare.
* LLMCodeGenerator — the upgrade path. It is client-agnostic: give it any
  `complete(messages) -> str` callable that talks to your provider. It sends
  the masked schema (never rows), requires a JSON reply {plan, code}, and
  parses it strictly. Everything downstream (sandbox, verification, retry,
  honest failure) is identical for both generators.

Generators never see raw rows: only the schema context from context.py.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Callable, Protocol

from .plan import (Compare, Filter, Metric, PlanError, QuerySpec, TopN, validate_plan)
from .render import render_program


@dataclass(frozen=True)
class GenerationRequest:
    question: str
    schema: dict
    attempt: int = 1
    previous_program: str | None = None
    previous_error: str | None = None


@dataclass(frozen=True)
class GeneratedProgram:
    code: str
    plan: QuerySpec | None  # None => the answer cannot be verified and will not be shown
    assumptions: tuple[str, ...] = ()


class Unanswerable(Exception):
    """The generator understood the request and it cannot be answered from this data.

    `reason` is a stable code (unknown_column, not_numeric, unsupported_question).
    Not retried: no amount of re-prompting adds a column that is not there.
    """
    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


class GenerationError(Exception):
    """The generator produced unusable output (malformed JSON, invalid plan). Retried once."""


class CodeGenerator(Protocol):
    name: str

    def generate(self, request: GenerationRequest) -> GeneratedProgram: ...


# ------------------------------------------------------------------ LLM adapter

LLM_SYSTEM_PROMPT = """You answer questions about a CSV by WRITING a small pandas program. \
You never answer from memory of the data and you never see its rows, only the schema.

Reply with ONE JSON object and nothing else:
{"plan": {...}, "code": "...", "assumptions": ["..."]}
or, if the data cannot answer the question:
{"unanswerable": {"reason": "unknown_column|not_numeric|unsupported_question", "message": "..."}}

Program rules:
- A typed DataFrame `df` plus `pd` (pandas) and `np` (numpy) are already defined.
- Allowed imports: pandas, numpy, math, statistics, datetime, re, collections, itertools, \
functools, decimal, fractions, string. No files, network, os/sys, eval/exec/getattr, \
dunder or underscore attributes, read_*/to_* I/O, .query/.eval/.plot/.sample. Deterministic only.
- Assign the answer to `result`:
  scalar plan  -> a number
  group_by plan -> a pandas Series indexed by group label (with `top`: ordered by value, \
ties by label ascending, then head(n))
  compare plan -> {"before": number, "after": number, "n_before": rows, "n_after": rows} where \
before is `< split`, after is `>= split`, n_* count rows after filters.
- Missing cells are NaN/None; count rows with len(), non-missing with .count().

The plan is machine-checked by re-computing it independently, so it must match the code exactly:
{"metric": {"agg": "count|sum|mean|median|min|max|nunique", "column": "<col or null for row count>"},
 "filters": [{"column": "...", "op": "==|!=|>|>=|<|<=", "value": ...}],
 "group_by": "<categorical column or null>",
 "compare": {"column": "<datetime column>", "split": "YYYY-MM-DD"} or null,
 "top": {"n": 1, "descending": true} or null}
Datetime filter values are ISO dates. Only use columns that exist in the schema."""


def build_llm_messages(request: GenerationRequest) -> list[dict[str, str]]:
    user = f"Question: {request.question}\n\nSchema (JSON):\n{json.dumps(request.schema, sort_keys=True)}"
    if request.attempt > 1:
        user += (f"\n\nYour previous program failed. This is your ONE retry.\n"
                 f"Previous program:\n{request.previous_program}\n"
                 f"Error: {request.previous_error}\nReturn corrected JSON.")
    return [{"role": "system", "content": LLM_SYSTEM_PROMPT}, {"role": "user", "content": user}]


def parse_llm_response(text: str) -> GeneratedProgram:
    """Strict parse of an LLM reply. Raises Unanswerable or GenerationError."""
    body = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", body, re.S)
    if fence:
        body = fence.group(1)
    try:
        data = json.loads(body)
    except ValueError as exc:
        raise GenerationError("the reply was not valid JSON") from exc
    if not isinstance(data, dict):
        raise GenerationError("the reply must be a JSON object")
    if "unanswerable" in data:
        info = data["unanswerable"] if isinstance(data["unanswerable"], dict) else {}
        raise Unanswerable(str(info.get("reason", "unsupported_question")),
                           str(info.get("message", "the question cannot be answered from this data")))
    code = data.get("code")
    if not isinstance(code, str) or not code.strip():
        raise GenerationError("the reply has no 'code' string")
    plan = None
    if data.get("plan") is not None:
        try:
            plan = QuerySpec.from_dict(data["plan"])
        except PlanError as exc:
            raise GenerationError(f"invalid plan: {exc}") from exc
    assumptions = data.get("assumptions") or []
    if not isinstance(assumptions, list):
        raise GenerationError("'assumptions' must be a list of strings")
    return GeneratedProgram(code=code, plan=plan, assumptions=tuple(str(a) for a in assumptions))


class LLMCodeGenerator:
    """Adapter for any chat-completion provider: pass `complete(messages) -> reply text`."""

    def __init__(self, complete: Callable[[list[dict[str, str]]], str], name: str = "llm"):
        self._complete = complete
        self.name = name

    def generate(self, request: GenerationRequest) -> GeneratedProgram:
        return parse_llm_response(self._complete(build_llm_messages(request)))


# ------------------------------------------------------------------ local rules generator

MONTHS = {"january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3, "april": 4,
          "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7, "august": 8, "aug": 8,
          "september": 9, "sept": 9, "sep": 9, "october": 10, "oct": 10, "november": 11,
          "nov": 11, "december": 12, "dec": 12}
_MONTH_RE = "|".join(sorted(MONTHS, key=len, reverse=True))
TIME_UNITS = {"day", "days", "week", "weeks", "month", "months", "quarter", "quarters",
              "year", "years", "hour", "hours"}
STOP = {"the", "a", "an", "each", "every", "this", "that", "it", "them", "all", "any", "of",
        "and", "or", "to", "is", "are", "was", "were", "be", "in", "for", "from", "at", "my",
        "our", "their", "there", "only", "with", "than", "number", "rows", "row", "records"}
RATE_NAME = re.compile(r"rate|ratio|pct|percent|avg|average|mean|score|age|price|rating|"
                       r"satisfaction|temperature|seconds|duration|bounce", re.I)
CHANGE_WORDS = re.compile(r"\b(chang\w*|differ\w*|compar\w*|increas\w*|decreas\w*|drop\w*|"
                          r"ris(?:e|en|ing)|rose|grew|grow\w*|fell|fall\w*|vs\.?|versus|"
                          r"shift\w*|improv\w*|declin\w*|up or down)\b", re.I)
OP_PHRASES = [  # longest first
    (">=", r"greater than or equal to|at least|no less than|>="),
    ("<=", r"less than or equal to|at most|no more than|<="),
    ("!=", r"not equal to|!="),
    (">", r"greater than|more than|exceeding|exceeds|above|over|>"),
    ("<", r"less than|fewer than|below|under|<"),
    ("==", r"equal to|equals|exactly|==|="),
]
_OP_RE = "|".join(p for _, p in OP_PHRASES)
_NUM_RE = r"-?\d[\d,]*(?:\.\d+)?"
NEGATORS = re.compile(r"(?:excluding|except|not in|not|other than|without|outside)\s+(?:the\s+)?$", re.I)
BOUNDARY = {"after": "after", "following": "after", "since": "since", "from": "since",
            "before": "before", "prior to": "before", "until": "before", "in": "in",
            "during": "in", "on": "in"}
_PREP_RE = "|".join(sorted((p.replace(" ", r"\s+") for p in BOUNDARY), key=len, reverse=True))
COUNT_RE = re.compile(r"\b(how many|number of|count of|count)\b", re.I)
MEAN_RE = re.compile(r"\b(average|avg|mean|typical)\b", re.I)
MEDIAN_RE = re.compile(r"\bmedian\b", re.I)
SUM_RE = re.compile(r"\b(total|sum|combined|altogether)\b", re.I)
MAX_RE = re.compile(r"\b(maximum|max|highest|largest|biggest|most|greatest|top)\b", re.I)
MIN_RE = re.compile(r"\b(minimum|min|lowest|smallest|least|fewest|bottom)\b", re.I)
DISTINCT_RE = re.compile(r"\b(distinct|unique|different)\b", re.I)
AGG_NOUN_RE = re.compile(r"\b(?:average|avg|mean|median|total|sum|maximum|max|minimum|min|highest|"
                         r"lowest|largest|smallest)\s+(?:of\s+)?(?:the\s+)?([a-z][\w]*)", re.I)
GROUP_TRIGGER_RE = re.compile(r"\b(?:by|per|for each|each|across|grouped by|broken down by|split by|"
                              r"which|top\s+\d+|bottom\s+\d+)\s+(?:the\s+)?([a-z_][\w]*)", re.I)


@dataclass
class _Mention:
    start: int
    end: int
    column: str
    value: str | None = None


def _aliases(name: str) -> set[str]:
    base = re.sub(r"[_\-]+", " ", name.lower()).strip()
    out = {base}
    if len(base) > 3:
        out.add(base[:-1] if base.endswith("s") else base + "s")
    return out


def _find_columns(q: str, columns: list[dict]) -> list[_Mention]:
    cands = []
    for c in columns:
        for alias in _aliases(c["name"]):
            for m in re.finditer(rf"(?<![\w]){re.escape(alias).replace(r'\ ', r'[ _-]')}(?![\w])", q, re.I):
                cands.append((m.start(), m.end(), c["name"]))
    cands.sort(key=lambda t: (-(t[1] - t[0]), t[0]))
    taken: list[_Mention] = []
    for s, e, name in cands:
        if all(e <= t.start or s >= t.end for t in taken):
            taken.append(_Mention(s, e, name))
    return sorted(taken, key=lambda t: t.start)


def _month_period(month: int, day: int | None, year: int, ) -> tuple[date, date]:
    if day is None:
        start = date(year, month, 1)
        end = date(year + (month == 12), month % 12 + 1, 1)
    else:
        start = date(year, month, day)
        end = start + timedelta(days=1)
    return start, end


def _pick_year(month: int, lo: date, hi: date) -> int:
    years = [y for y in range(lo.year, hi.year + 1)
             if _month_period(month, None, y)[0] <= hi and _month_period(month, None, y)[1] > lo]
    return max(years) if years else hi.year


def _date_phrases(q: str, lo: date | None, hi: date | None, assumptions: list[str]):
    """Yield (kind, start, end, span): kind in after/since/before/in; period is [start, end)."""
    found = []
    for m in re.finditer(rf"\b({_PREP_RE})\s+(\d{{4}})-(\d{{2}})-(\d{{2}})\b", q, re.I):
        try:
            d = date(int(m[2]), int(m[3]), int(m[4]))
        except ValueError:
            continue
        found.append((BOUNDARY[re.sub(r"\s+", " ", m[1].lower())], d, d + timedelta(days=1), m.span()))
    month_pat = (rf"\b({_PREP_RE})\s+(?:the\s+(?:start|beginning|end)\s+of\s+)?({_MONTH_RE})\b"
                 rf"(?:\s+(\d{{1,2}})(?!\d)(?:st|nd|rd|th)?)?(?:,?\s*(\d{{4}})\b)?")
    for m in re.finditer(month_pat, q, re.I):
        month = MONTHS[m[2].lower()]
        day = int(m[3]) if m[3] else None
        year = int(m[4]) if m[4] else (_pick_year(month, lo, hi) if lo and hi else None)
        if year is None:
            raise Unanswerable("unknown_column", "The question refers to a time period but the "
                               "dataset has no usable date column.")
        try:
            start, end = _month_period(month, day, year)
        except ValueError:
            continue
        if not m[4]:
            assumptions.append(f"Read '{m[2]}' as {start.strftime('%B %Y')}"
                               f" (the data covers {lo} to {hi}).")
        found.append((BOUNDARY[re.sub(r"\s+", " ", m[1].lower())], start, end, m.span()))
    for m in re.finditer(rf"\b({_PREP_RE})\s+(\d{{4}})\b(?!-)", q, re.I):
        year = int(m[2])
        if not 1900 <= year <= 2100:
            continue
        found.append((BOUNDARY[re.sub(r"\s+", " ", m[1].lower())], date(year, 1, 1),
                      date(year + 1, 1, 1), m.span()))
    return sorted(found, key=lambda t: t[3][0])


def _iso(d: date) -> str:
    return d.isoformat()


def _column_list(columns: list[dict]) -> str:
    return ", ".join(f"{c['name']} ({c['type']})" for c in columns)


class LocalRulesGenerator:
    """Deterministic question -> plan -> pandas program. Same input, same output."""
    name = "local-rules-v1"

    def generate(self, request: GenerationRequest) -> GeneratedProgram:
        spec, assumptions = self._interpret(request.question, request.schema)
        try:
            validate_plan(spec, request.schema)
        except PlanError as exc:
            raise Unanswerable("unsupported_question", str(exc)) from exc
        # The retry gets a defensive variant that re-coerces the columns it touches.
        code = render_program(spec, request.schema, defensive=request.attempt > 1)
        return GeneratedProgram(code=code, plan=spec, assumptions=tuple(assumptions))

    # -- interpretation ---------------------------------------------------
    def _interpret(self, question: str, schema: dict) -> tuple[QuerySpec, list[str]]:
        q = " ".join(question.strip().split())
        columns = schema["columns"]
        kinds = {c["name"]: c["type"] for c in columns}
        assumptions: list[str] = []
        mentions = _find_columns(q, columns)
        spans = [(m.start, m.end) for m in mentions]
        filters: list[Filter] = []
        filter_cols: set[str] = set()

        def overlaps(s: int, e: int) -> bool:
            return any(s < b and e > a for a, b in spans)

        # numeric comparisons: "units over 50" / "more than 50 units"
        for m in mentions:
            if kinds[m.column] != "numeric":
                continue
            right = re.match(rf"\s*(?:is|was|were|are)?\s*({_OP_RE})\s*({_NUM_RE})", q[m.end:], re.I)
            left = re.search(rf"({_OP_RE})\s*({_NUM_RE})\s*$", q[:m.start], re.I)
            hit = right or left
            if hit:
                op = next(o for o, pat in OP_PHRASES if re.fullmatch(pat, hit[1], re.I))
                filters.append(Filter(m.column, op, float(hit[2].replace(",", ""))))
                span = (left.start(), left.end()) if (left and not right) else (m.end, m.end + hit.end())
                spans.append(span)
                # "at least"/"at most" must not be mistaken for superlatives later
                q = q[:span[0]] + " " * (span[1] - span[0]) + q[span[1]:]
                m.value = "filter"

        # categorical values: known values first, then explicit "col is X" / "for X col"
        for col in columns:
            if col["type"] not in ("categorical", "boolean"):
                continue
            named = any(m.column == col["name"] for m in mentions)
            if col["type"] == "boolean" and not named:
                continue
            for value in sorted(col.get("values", []), key=len, reverse=True):
                if len(value) < 2:
                    continue
                for hit in re.finditer(rf"(?<![\w]){re.escape(value)}(?![\w])", q, re.I):
                    if overlaps(hit.start(), hit.end()):
                        continue
                    neg = NEGATORS.search(q[:hit.start()])
                    filters.append(Filter(col["name"], "!=" if neg else "==", value))
                    filter_cols.add(col["name"])
                    spans.append((hit.start(), hit.end()))
        for m in mentions:
            if kinds[m.column] not in ("categorical", "boolean", "text") or m.column in filter_cols:
                continue
            known = {v.lower(): v for v in next(c for c in columns if c["name"] == m.column).get("values", [])}
            after = re.match(r"\s*(?:is|=|==|equals|equal to|:)\s*(?:\"([^\"]+)\"|'([^']+)'|([\w\-.]+))",
                             q[m.end:], re.I)
            before = re.search(r"\b(?:for|in|from|at|only|within)\s+(?:the\s+)?(?:\"([^\"]+)\"|'([^']+)'|([\w\-.]+))\s*$",
                               q[:m.start], re.I)
            hit = after or before
            if not hit:
                continue
            raw = next(g for g in hit.groups() if g)
            if raw.lower() in STOP or raw.lower() in MONTHS or raw.lower() in TIME_UNITS:
                continue
            value = known.get(raw.lower(), raw)
            neg = NEGATORS.search(q[:m.start]) if before else None
            filters.append(Filter(m.column, "!=" if neg else "==", value))
            filter_cols.add(m.column)

        # time phrases
        date_cols = [c["name"] for c in columns if c["type"] == "datetime"]
        mentioned_dates = [m.column for m in mentions if kinds[m.column] == "datetime"]
        date_col = (mentioned_dates or date_cols or [None])[0]
        lo = hi = None
        if date_col:
            meta = next(c for c in columns if c["name"] == date_col)
            if "min" in meta:
                lo, hi = date.fromisoformat(meta["min"]), date.fromisoformat(meta["max"])
        phrases = _date_phrases(q, lo, hi, assumptions)
        if phrases and date_col is None:
            raise Unanswerable("unknown_column", "The question refers to a time period but the "
                               "dataset has no date column.")
        compare = None
        change = bool(CHANGE_WORDS.search(q))
        if phrases and change and len(phrases) == 1 and phrases[0][0] in ("after", "since", "before"):
            kind, start, end, _ = phrases[0]
            split = end if kind == "after" else start
            compare = Compare(date_col, _iso(split))
            assumptions.append(f"'Before' means dates < {split}; 'after' means dates >= {split}.")
        else:
            for kind, start, end, _ in phrases:
                if kind in ("after", "since"):
                    filters.append(Filter(date_col, ">=", _iso(end if kind == "after" else start)))
                elif kind == "before":
                    filters.append(Filter(date_col, "<", _iso(start)))
                else:
                    filters += [Filter(date_col, ">=", _iso(start)), Filter(date_col, "<", _iso(end))]

        # grouping / ranking
        group_by, hinted_metric, top = None, None, None
        for trig in GROUP_TRIGGER_RE.finditer(q):
            word = trig.group(1)
            wl = word.lower()
            ment = next((m for m in mentions if m.start == trig.start(1)), None)
            word_is_trigger = trig.group(0).lower().startswith("which")
            if ment:
                kind = kinds[ment.column]
                if kind == "datetime":
                    raise Unanswerable("unsupported_question", "Grouping by a time period is not "
                                       "supported yet; ask for a before/after comparison instead.")
                if kind == "numeric":
                    hinted_metric = ment.column
                elif group_by is None:
                    group_by = ment.column
            elif wl in TIME_UNITS:
                raise Unanswerable("unsupported_question", "Grouping by a time period is not "
                                   "supported yet; ask for a before/after comparison instead.")
            elif (not word_is_trigger and wl not in STOP and wl not in MONTHS and len(wl) >= 3
                  and not any(f.value and str(f.value).lower() == wl for f in filters)
                  and not overlaps(trig.start(1), trig.end(1))):
                raise Unanswerable("unknown_column", f"No column matches {word!r}. "
                                   f"Available columns: {_column_list(columns)}.")
        topn = re.search(r"\b(top|bottom)\s+(\d+)\b", q, re.I)
        sup_max, sup_min = MAX_RE.search(q), MIN_RE.search(q)
        if group_by and (topn or re.search(r"\bwhich\b", q, re.I)) and (sup_max or sup_min or topn):
            bottom = bool(sup_min) or bool(topn and topn[1].lower() == "bottom")
            top = TopN(n=int(topn[2]) if topn else 1, descending=not bottom)

        # metric
        used = filter_cols | ({group_by} if group_by else set())  # categorical filters + group
        free = [m for m in mentions if m.column not in used and kinds[m.column] != "datetime"
                and m.value != "filter"]
        numeric_free = [m.column for m in free if kinds[m.column] == "numeric"]
        if hinted_metric:
            numeric_free.insert(0, hinted_metric)
        explicit = ("count" if COUNT_RE.search(q) else "mean" if MEAN_RE.search(q) else
                    "median" if MEDIAN_RE.search(q) else "sum" if SUM_RE.search(q) else None)
        metric = self._metric(q, explicit, numeric_free, free, kinds, columns, group_by, top, compare,
                              sup_max, sup_min, assumptions)
        if compare and group_by:
            raise Unanswerable("unsupported_question", "Combining a before/after comparison with "
                               "grouping is not supported yet; ask them separately.")
        return QuerySpec(metric=metric, filters=tuple(filters), group_by=group_by,
                         compare=compare, top=top), assumptions

    def _metric(self, q, explicit, numeric_free, free, kinds, columns, group_by, top, compare,
                sup_max, sup_min, assumptions) -> Metric:
        col = numeric_free[0] if numeric_free else None
        if explicit == "count":
            if DISTINCT_RE.search(q):
                target = next((m.column for m in free if kinds[m.column] != "numeric"), None) or col
                if target is None:
                    raise Unanswerable("unsupported_question", "Say which column to count distinct values of.")
                return Metric("nunique", target)
            if col:
                assumptions.append(f"Read the count question as the total of the {col!r} column; "
                                   "ask 'how many rows' to count rows instead.")
                return Metric("sum", col)
            return Metric("count", None)
        if explicit in ("mean", "median", "sum"):
            return Metric(explicit, self._need_numeric(q, col, free, kinds, columns))
        if compare:
            col = self._need_numeric(q, col, free, kinds, columns)
            assumptions.append(f"Compared the mean of {col!r} per row, since the two periods "
                               "can differ in length (this is a difference, not a significance test).")
            return Metric("mean", col)
        if top is not None or (group_by and (sup_max or sup_min)):
            if col is None:
                assumptions.append("No measure named, so ranked groups by row count.")
                return Metric("count", None)
            agg = "mean" if RATE_NAME.search(col) else "sum"
            assumptions.append(f"No aggregation named; used {agg} of {col!r}.")
            return Metric(agg, col)
        if sup_max or sup_min:
            return Metric("max" if sup_max else "min", self._need_numeric(q, col, free, kinds, columns))
        if col is not None:
            agg = "mean" if RATE_NAME.search(col) else "sum"
            assumptions.append(f"No aggregation named; used {agg} of {col!r}.")
            return Metric(agg, col)
        raise Unanswerable("unsupported_question",
                           "I couldn't tell what to compute. Try e.g. 'average <column> by <column>', "
                           "'how many rows where <column> is <value>', or 'did <column> change after "
                           f"<month>'. Columns: {_column_list(columns)}.")

    @staticmethod
    def _need_numeric(q, col, free, kinds, columns) -> str:
        if col:
            return col
        non_numeric = next((m.column for m in free), None)
        if non_numeric:
            raise Unanswerable("not_numeric", f"Column {non_numeric!r} is {kinds[non_numeric]}, "
                               "so it cannot be averaged, summed or compared.")
        noun = AGG_NOUN_RE.search(q)
        word = noun.group(1) if noun and noun.group(1).lower() not in STOP else None
        numeric = [c["name"] for c in columns if c["type"] == "numeric"]
        if word:
            raise Unanswerable("unknown_column", f"No column matches {word!r}. "
                               f"Numeric columns: {', '.join(numeric) or 'none'}.")
        raise Unanswerable("unknown_column", "I couldn't find a numeric column in the question. "
                           f"Numeric columns: {', '.join(numeric) or 'none'}.")
