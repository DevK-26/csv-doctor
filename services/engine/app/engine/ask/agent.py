"""Ask-Your-Data orchestration: generate -> sandbox -> verify -> answer or honest failure.

Outcomes (the `status` field of the result):

* "verified"   — the program ran in the sandbox AND its numbers matched an
                 independent re-computation of its plan. Only then is an
                 `answer` returned.
* "unverified" — the program ran but verification failed. `answer` is null and
                 no number from the program is shown. The program, plan and the
                 failed checks are returned so the user can see why.
* "failed"     — no runnable program after one retry, or the question cannot be
                 answered from this data (missing column, unsupported shape).

Retry policy: a program that errors (policy violation, exception, timeout, no
`result`, wrong result shape) or a malformed/invalid plan gets exactly ONE
retry with the error fed back. A verification failure is not retried: the
program ran and produced a wrong answer, which is the case that must be
reported, not papered over.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from .context import build_schema_context, typed_frame
from .generator import (CodeGenerator, GenerationError, GenerationRequest, LocalRulesGenerator,
                        Unanswerable)
from .plan import PlanError, QuerySpec, validate_plan
from .sandbox import Limits, SandboxBusy, SandboxError, run_program
from .verify import ResultShapeError, Verification, verify

MAX_QUESTION_CHARS = 500
MAX_ATTEMPTS = 2  # first try + exactly one retry
SANDBOX_LAYERS = ["static policy check", "isolated subprocess with rlimits and wall-clock kill",
                  "restricted builtins + audit hook (no files, no network, no process creation)"]


class AskInputError(ValueError):
    """Bad request (empty/oversized question); the API turns it into a 422."""


def _fmt(v: Any) -> str:
    if isinstance(v, int):
        return f"{v:,}"
    return f"{v:,.4f}".rstrip("0").rstrip(".")


def _fmt_change(v: float) -> str:
    return f"{v:+,.4f}".rstrip("0").rstrip(".")


def _describe_metric(spec: QuerySpec) -> str:
    m = spec.metric
    if m.agg == "count":
        return "number of rows" if m.column is None else f"count of non-missing {m.column}"
    if m.agg == "nunique":
        return f"number of distinct {m.column}"
    return f"{m.agg} of {m.column}"


def _describe_filters(spec: QuerySpec) -> str:
    return " and ".join(f"{f.column} {f.op} {f.value!r}" if isinstance(f.value, str)
                        else f"{f.column} {f.op} {f.value:g}" if isinstance(f.value, float)
                        else f"{f.column} {f.op} {f.value}" for f in spec.filters)


def _build_answer(spec: QuerySpec, v: Verification) -> dict:
    """Answer fields come only from numbers that passed verification."""
    rep, exp = v.reported, v.expected
    what = _describe_metric(spec)
    scope = f" where {_describe_filters(spec)}" if spec.filters else ""
    if rep["kind"] == "scalar":
        if (exp["rows"] == 0 and spec.metric.agg != "count") or rep["value"] is None:
            why = ("No rows matched the filters" if exp["rows"] == 0
                   else f"The {exp['rows']} matching rows have no usable values")
            return {"kind": "empty", "value": None, "matched_rows": exp["rows"],
                    "text": f"{why}{scope}, so there is no {what} to report."}
        over = "" if spec.metric.agg == "count" and spec.metric.column is None \
            else f" (over {exp['rows']:,} matching rows)"
        return {"kind": "scalar", "value": rep["value"], "matched_rows": exp["rows"],
                "text": f"The {what}{scope} is {_fmt(rep['value'])}{over}."}
    if rep["kind"] == "compare":
        c = spec.compare
        b, a = rep["before"], rep["after"]
        nb, na = exp["n_before"], exp["n_after"]
        base = {"kind": "compare", "before": b, "after": a, "n_before": nb, "n_after": na,
                "split": c.split, "change": None, "pct_change": None}
        if b is None or a is None:
            side = "before" if b is None else "on or after"
            base["text"] = (f"Cannot compare: no usable rows {side} {c.split}{scope} "
                            f"(rows before: {nb:,}, on/after: {na:,}).")
            return base
        change = a - b
        base["change"] = change
        base["pct_change"] = (change / abs(b) * 100) if b != 0 else None
        pct = f" ({change / abs(b) * 100:+.1f}%)" if b != 0 else ""
        base["text"] = (f"The {what}{scope} was {_fmt(b)} before {c.split} (n={nb:,}) and "
                        f"{_fmt(a)} on or after (n={na:,}): a change of {_fmt_change(change)}"
                        f"{pct}. This is a difference between two periods, not a significance test.")
        return base
    groups = [{"label": k, "value": val} for k, val in rep["groups"]]
    if not groups:
        return {"kind": "empty", "value": None, "groups": [], "matched_rows": exp["rows"],
                "text": f"No rows matched the filters{scope}, so there are no groups to report."}
    shown = "; ".join(f"{g['label']} = {_fmt(g['value']) if g['value'] is not None else 'n/a'}"
                      for g in groups[:10])
    more = f" (+{len(groups) - 10} more)" if len(groups) > 10 else ""
    lead = (f"Top {spec.top.n} by {what}, grouped by {spec.group_by}" if spec.top
            else f"{what[0].upper()}{what[1:]}, grouped by {spec.group_by}")
    return {"kind": "groups", "groups": groups, "matched_rows": exp["rows"],
            "text": f"{lead}{scope}: {shown}{more}."}


def _result(question: str, generator: CodeGenerator, status: str, message: str, *,
            reason: str | None = None, answer: dict | None = None, program: str | None = None,
            plan: QuerySpec | None = None, assumptions: tuple[str, ...] = (),
            attempts: list[dict], verification: Verification | None = None,
            limits: Limits) -> dict:
    return {
        "question": question, "status": status, "reason": reason, "message": message,
        "answer": answer, "program": program,
        "plan": plan.to_dict() if plan else None, "assumptions": list(assumptions),
        "attempts": attempts,
        "verification": verification.to_api() if verification else None,
        "generator": generator.name,
        "sandbox": {"layers": SANDBOX_LAYERS, "cpu_seconds": limits.cpu_seconds,
                    "wall_seconds": limits.wall_seconds, "memory_mb": limits.memory_mb},
    }


def answer_question(question: str, df: pd.DataFrame, types: dict[str, str],
                    generator: CodeGenerator | None = None,
                    limits: Limits = Limits()) -> dict:
    """Answer `question` about `df` (parse_csv_bytes output) or fail honestly.

    Returns a JSON-ready dict (see module docstring for `status`). Raises
    AskInputError for a bad question, ContextError for unshareable schemas and
    SandboxBusy when the sandbox is saturated; everything else is a result.
    """
    question = " ".join(str(question).split())
    if not question:
        raise AskInputError("empty_question: ask a question about the data.")
    if len(question) > MAX_QUESTION_CHARS:
        raise AskInputError(f"question_too_long: keep it under {MAX_QUESTION_CHARS} characters.")
    generator = generator or LocalRulesGenerator()
    schema = build_schema_context(df, types)
    frame = typed_frame(df, types)
    known = frozenset(c["name"] for c in schema["columns"])

    attempts: list[dict] = []
    previous_program = previous_error = None
    last_reason = "program_error"
    last_plan: QuerySpec | None = None
    last_assumptions: tuple[str, ...] = ()

    for n in range(1, MAX_ATTEMPTS + 1):
        entry: dict = {"attempt": n, "program": None, "error": None, "reason": None}
        attempts.append(entry)
        try:
            gen = generator.generate(GenerationRequest(
                question=question, schema=schema, attempt=n,
                previous_program=previous_program, previous_error=previous_error))
            entry["program"] = gen.code
            last_assumptions = gen.assumptions
            if gen.plan is None:
                raise GenerationError("the generator returned no plan, so the answer could not be "
                                      "checked; a plan is required")
            validate_plan(gen.plan, schema)
            last_plan = gen.plan
            raw = run_program(gen.code, frame, limits, known_strings=known)
            outcome = verify(raw, gen.plan, frame, schema)
        except Unanswerable as exc:
            entry["error"], entry["reason"] = str(exc), exc.reason
            return _result(question, generator, "failed", str(exc), reason=exc.reason,
                           attempts=attempts, limits=limits)
        except SandboxBusy:
            raise
        except SandboxError as exc:
            entry["error"], entry["reason"] = str(exc), exc.reason
        except (GenerationError, PlanError, ResultShapeError) as exc:
            reason = "invalid_plan" if isinstance(exc, PlanError) else (
                "bad_result_shape" if isinstance(exc, ResultShapeError) else "generator_error")
            entry["error"], entry["reason"] = str(exc), reason
        except Exception as exc:  # noqa: BLE001 - e.g. a provider outage inside a pluggable generator
            entry["error"] = f"the generator failed: {type(exc).__name__}"
            entry["reason"] = "generator_error"
        else:
            if outcome.passed:
                return _result(
                    question, generator, "verified",
                    "Verified: the program's result matches an independent re-computation.",
                    answer=_build_answer(gen.plan, outcome), program=gen.code, plan=gen.plan,
                    assumptions=gen.assumptions, attempts=attempts, verification=outcome,
                    limits=limits)
            return _result(
                question, generator, "unverified",
                "This answer could not be verified, so no number is shown. The program ran, but its "
                "result did not match an independent re-computation or failed a sanity check. "
                "The program and the failed checks are included so you can see why.",
                reason="verification_failed", program=gen.code, plan=gen.plan,
                assumptions=gen.assumptions, attempts=attempts, verification=outcome,
                limits=limits)
        previous_program, previous_error = entry["program"], entry["error"]
        last_reason = entry["reason"]

    return _result(
        question, generator, "failed",
        f"I could not produce a working program after {MAX_ATTEMPTS} attempts (one retry). "
        f"Last error: {previous_error}",
        reason=last_reason, program=previous_program, plan=last_plan,
        assumptions=last_assumptions, attempts=attempts, limits=limits)
