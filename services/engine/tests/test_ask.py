"""Phase B2 tests: Ask-Your-Data agent (app/engine/ask).

Covers: the local generator's question shapes, sandbox policy (static layer
and runtime layers on their own), verification, the one-retry policy, honest
failure, and the /ask endpoint. Sandbox tests spawn real worker processes.
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.engine.ask import (AskInputError, GeneratedProgram, GenerationError, LLMCodeGenerator,
                            LocalRulesGenerator, Unanswerable, answer_question)
from app.engine.ask import sandbox as sbx
from app.engine.ask.context import ContextError, build_schema_context, typed_frame
from app.engine.ask.generator import (GenerationRequest, build_llm_messages, parse_llm_response)
from app.engine.ask.plan import Compare, Filter, Metric, PlanError, QuerySpec, TopN, validate_plan
from app.engine.ask.render import render_program
from app.engine.ask.sandbox import Limits, ProgramError, SandboxTimeout, SandboxViolation
from app.engine.ask.verify import ResultShapeError, verify
from app.engine.parse import infer_types, parse_csv_bytes
from app.main import app, get_generator

SAMPLES = Path(__file__).resolve().parents[3] / "samples"
SALES = (SAMPLES / "sales.csv").read_bytes()
FAST = Limits(cpu_seconds=3, wall_seconds=20)


@pytest.fixture(scope="module")
def sales():
    df = parse_csv_bytes(SALES)
    return df, infer_types(df)


@pytest.fixture(scope="module")
def sales_schema(sales):
    return build_schema_context(*sales)


@pytest.fixture(scope="module")
def sales_frame(sales):
    return typed_frame(*sales)


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def parse(text: str):
    df = parse_csv_bytes(textwrap.dedent(text).strip().encode())
    return df, infer_types(df)


class Scripted:
    """Generator that replays prepared steps (a GeneratedProgram or an exception)."""
    name = "scripted"

    def __init__(self, *steps):
        self.steps, self.requests = list(steps), []

    def generate(self, request):
        self.requests.append(request)
        step = self.steps[min(len(self.requests), len(self.steps)) - 1]
        if isinstance(step, Exception):
            raise step
        return step

    @property
    def calls(self) -> int:
        return len(self.requests)


def prog(code: str, plan: QuerySpec | None) -> GeneratedProgram:
    return GeneratedProgram(code=textwrap.dedent(code).strip() + "\n", plan=plan)


MEAN_UNITS = QuerySpec(metric=Metric("mean", "units"))


# ============================================================ plan + context

def test_plan_round_trip_and_strict_parse():
    spec = QuerySpec(Metric("mean", "returns"), (Filter("region", "==", "North"),),
                     compare=Compare("date", "2025-04-01"))
    assert QuerySpec.from_dict(spec.to_dict()) == spec
    with pytest.raises(PlanError, match="unknown keys"):
        QuerySpec.from_dict({"metric": {"agg": "sum", "column": "x"}, "sql": "drop"})
    with pytest.raises(PlanError):
        QuerySpec.from_dict(["not", "an", "object"])


@pytest.mark.parametrize("spec,msg", [
    (QuerySpec(Metric("mean", "region")), "needs a numeric column"),
    (QuerySpec(Metric("mean", "profit")), "does not exist"),
    (QuerySpec(Metric("median", "units"), group_by="units"), "categorical"),
    (QuerySpec(Metric("count"), (Filter("region", ">", "N"),)), "not valid"),
    (QuerySpec(Metric("count"), (Filter("units", "==", "ten"),)), "numeric"),
    (QuerySpec(Metric("count"), compare=Compare("region", "2025-01-01")), "not a datetime"),
    (QuerySpec(Metric("count"), compare=Compare("date", "March")), "ISO date"),
    (QuerySpec(Metric("count"), top=TopN(1)), "needs a group_by"),
    (QuerySpec(Metric("bogus", "units")), "not one of"),
])
def test_validate_plan_rejects(spec, msg, sales_schema):
    with pytest.raises(PlanError, match=msg):
        validate_plan(spec, sales_schema)


def test_typed_frame_and_schema_context(sales, sales_schema, sales_frame):
    assert str(sales_frame["units"].dtype) == "float64"
    assert str(sales_frame["date"].dtype).startswith("datetime64")
    assert sales_frame["region"].dtype == object
    cols = {c["name"]: c for c in sales_schema["columns"]}
    assert cols["region"]["values"][:1] == ["North"]  # most frequent first
    assert cols["date"]["min"] == "2025-01-01" and cols["date"]["max"].startswith("2025-06")
    assert "values" not in cols["order_id"]  # IDs never leave as value lists
    assert "20010" not in json.dumps(sales_schema)  # no raw rows in the context


def test_schema_context_drops_maskable_values_and_blanks_become_missing():
    df, types = parse(
        """
        contact,score
        a@x.com,1
        b@x.com,2
        b@x.com,3
        plain,4
        plain,
        """)
    schema = build_schema_context(df, types)
    contact = next(c for c in schema["columns"] if c["name"] == "contact")
    assert all("@" not in v for v in contact.get("values", []))
    assert typed_frame(df, types)["score"].isna().sum() == 1
    with pytest.raises(ContextError):
        build_schema_context(*parse("a@b.com,x\n1,2\n3,4\n"))


# ============================================================ local generator

def plan_of(question: str, schema: dict) -> dict:
    gen = LocalRulesGenerator().generate(GenerationRequest(question, schema))
    return gen.plan.to_dict()


def test_headline_question_becomes_before_after_compare(sales_schema):
    gen = LocalRulesGenerator().generate(GenerationRequest(
        "did returns change after March for the North region?", sales_schema))
    assert gen.plan.to_dict() == {
        "metric": {"agg": "mean", "column": "returns"},
        "filters": [{"column": "region", "op": "==", "value": "North"}],
        "group_by": None, "compare": {"column": "date", "split": "2025-04-01"}, "top": None}
    assert any("March 2025" in a for a in gen.assumptions)
    assert "pd.Timestamp('2025-04-01')" in gen.code and "groupby" not in gen.code


@pytest.mark.parametrize("question,expected", [
    ("how many rows where region is North",
     dict(metric=("count", None), filters=[("region", "==", "North")])),
    ("how many orders have units over 50",
     dict(metric=("count", None), filters=[("units", ">", 50.0)])),
    ("how many orders with units at least 50",  # 'at least' is a filter, not a superlative
     dict(metric=("count", None), filters=[("units", ">=", 50.0)])),
    ("average revenue by product", dict(metric=("mean", "revenue"), group_by="product")),
    ("total units in north", dict(metric=("sum", "units"), filters=[("region", "==", "North")])),
    ("which region had the highest revenue?",
     dict(metric=("sum", "revenue"), group_by="region", top=(1, True))),
    ("top 2 products by units", dict(metric=("sum", "units"), group_by="product", top=(2, True))),
    ("which product has the lowest returns",
     dict(metric=("sum", "returns"), group_by="product", top=(1, False))),
    ("median revenue excluding North", dict(metric=("median", "revenue"),
                                            filters=[("region", "!=", "North")])),
    ("sum of revenue before 2025-02-01",
     dict(metric=("sum", "revenue"), filters=[("date", "<", "2025-02-01")])),
    ("total revenue in March", dict(metric=("sum", "revenue"), filters=[
        ("date", ">=", "2025-03-01"), ("date", "<", "2025-04-01")])),
    ("total revenue since April for West",
     dict(metric=("sum", "revenue"), filters=[("region", "==", "West"), ("date", ">=", "2025-04-01")])),
    ("how many distinct products", dict(metric=("nunique", "product"))),
    ("maximum units in the South region", dict(metric=("max", "units"),
                                               filters=[("region", "==", "South")])),
    ("did revenue change from March 15 for Gizmo",
     dict(metric=("mean", "revenue"), filters=[("product", "==", "Gizmo")],
          compare=("date", "2025-03-15"))),
])
def test_local_generator_question_shapes(question, expected, sales_schema):
    plan = plan_of(question, sales_schema)
    assert (plan["metric"]["agg"], plan["metric"]["column"]) == expected["metric"]
    got = sorted((f["column"], f["op"], f["value"]) for f in plan["filters"])
    assert got == sorted(expected.get("filters", []))
    assert plan["group_by"] == expected.get("group_by")
    top = (plan["top"]["n"], plan["top"]["descending"]) if plan["top"] else None
    assert top == expected.get("top")
    cmp_ = (plan["compare"]["column"], plan["compare"]["split"]) if plan["compare"] else None
    assert cmp_ == expected.get("compare")


@pytest.mark.parametrize("question,reason,needle", [
    ("what is the average profit", "unknown_column", "profit"),
    ("average units by country", "unknown_column", "country"),
    ("average region", "not_numeric", "region"),
    ("what is the average units per month", "unsupported_question", "time period"),
    ("did units change after March by product", "unsupported_question", "grouping"),
    ("hello there", "unsupported_question", "couldn't tell"),
])
def test_local_generator_unanswerable(question, reason, needle, sales_schema):
    with pytest.raises(Unanswerable) as exc:
        LocalRulesGenerator().generate(GenerationRequest(question, sales_schema))
    assert exc.value.reason == reason and needle in str(exc.value)


def test_local_generator_is_deterministic(sales_schema):
    q = "did returns change after March for the North region?"
    a = LocalRulesGenerator().generate(GenerationRequest(q, sales_schema))
    b = LocalRulesGenerator().generate(GenerationRequest(q, sales_schema))
    assert a == b


def test_retry_variant_recoerces_columns(sales_schema):
    spec = QuerySpec(Metric("mean", "returns"), (Filter("region", "==", "North"),))
    assert "to_numeric" not in render_program(spec, sales_schema)
    assert "to_numeric" in render_program(spec, sales_schema, defensive=True)


# ============================================================ sandbox: static layer

@pytest.mark.parametrize("code", [
    "result = open('/etc/passwd').read()",
    "import os\nresult = os.getcwd()",
    "from os import system\nresult = 1",
    "import socket\nresult = 1",
    "import subprocess\nresult = 1",
    "import numpy.ctypeslib\nresult = 1",
    "from numpy import ctypeslib\nresult = 1",
    "from pandas import read_csv\nresult = 1",
    "import ctypes\nresult = 1",
    "result = __import__('os')",
    "result = eval('1+1')",
    "exec('result = 1')",
    "result = pd.read_csv('/etc/passwd')",
    "result = pd.read_pickle('x')",
    "df.to_csv('/tmp/x.csv')\nresult = 1",
    "result = np.load('/tmp/x.npy')",
    "result = getattr(df, 'shape')",
    "result = (1).__class__.__base__.__subclasses__()",
    "result = df.__class__",
    "result = df._mgr",
    "result = pd.io.common.os.getcwd()",
    "result = np.core._internal.ctypes",
    "result = df.agg('__class__')",
    "result = df.query('units > 1')",
    "result = pd.eval('1 + 1')",
    "class A: pass\nresult = 1",
    "result = type(df)",
    "result = globals()",
    "result = '{0.__class__}'.format(df)",
    "result = (x for x in []).gi_frame.f_back",
    "import importlib\nresult = 1",
    "result = np.random.rand(3)",
])
def test_static_check_blocks(code):
    with pytest.raises(SandboxViolation):
        sbx.check_program(code)


def test_static_check_allows_normal_analysis_and_real_column_names():
    sbx.check_program("import pandas as pd\nimport numpy as np\nfrom datetime import datetime\n"
                      "d = df[df['units'] > 3]\nresult = d.groupby('region')['revenue'].mean()\n")
    with pytest.raises(SandboxViolation):
        sbx.check_program("result = df['type'].count()")
    sbx.check_program("result = df['type'].count()", known_strings=frozenset({"type"}))
    with pytest.raises(SandboxViolation):  # dunder strings are never exempt
        sbx.check_program("result = df['__x__']", known_strings=frozenset({"__x__"}))
    with pytest.raises(ProgramError, match="SyntaxError"):
        sbx.check_program("result = (")
    with pytest.raises(ProgramError):
        sbx.check_program("   ")
    with pytest.raises(SandboxViolation, match="longer"):
        sbx.check_program("x = 1\n" * 5000)


# ============================================================ sandbox: runtime layers on their own
# static_check=False: the static layer is bypassed on purpose so each test proves the
# process/runtime layers refuse the operation by themselves.

def run_raw(code, frame, **limits):
    return sbx.run_program(textwrap.dedent(code), frame, Limits(**{"cpu_seconds": 3, **limits}),
                           static_check=False)


def test_sandbox_runs_a_normal_program(sales_frame):
    out = sbx.run_program("result = df[df['region'] == 'North']['returns'].mean()",
                          sales_frame, FAST)
    assert out["kind"] == "scalar" and out["value"] == pytest.approx(2.5)


def test_runtime_blocks_file_read(sales_frame):
    with pytest.raises(SandboxViolation, match="open"):
        run_raw("result = len(pd.read_csv('/etc/passwd'))", sales_frame)
    with pytest.raises(SandboxViolation):
        run_raw("result = pd.io.common.os.open('/etc/passwd', 0)", sales_frame)
    with pytest.raises(ProgramError, match="NameError"):  # `open` is not even a builtin
        run_raw("result = open('/etc/passwd').read()", sales_frame)


def test_runtime_blocks_file_write_and_creates_nothing(sales_frame, tmp_path):
    target = tmp_path / "pwned.csv"
    with pytest.raises(SandboxViolation):
        run_raw(f"df.to_csv({str(target)!r})\nresult = 1", sales_frame)
    with pytest.raises(SandboxViolation):
        run_raw(f"pd.io.common.os.open({str(tmp_path / 'p2')!r}, 65)\nresult = 1", sales_frame)
    assert list(tmp_path.iterdir()) == []


def test_runtime_blocks_os_and_process_creation(sales_frame, tmp_path):
    marker = tmp_path / "ran"
    with pytest.raises(SandboxViolation):
        run_raw(f"result = pd.io.common.os.system('touch {marker}')", sales_frame)
    with pytest.raises(SandboxViolation):
        run_raw("result = pd.io.common.os.popen('id').read()", sales_frame)
    with pytest.raises(SandboxViolation):
        run_raw("result = len(pd.io.common.os.listdir('/'))", sales_frame)
    assert not marker.exists()


def test_runtime_blocks_network_and_native_code(sales_frame):
    for code in ("import socket\nresult = 1",
                 "result = pd.io.common.urlopen('http://127.0.0.1:9')",
                 "result = np.core._internal.ctypes.CDLL(None).getpid()"):
        with pytest.raises(SandboxViolation):
            run_raw(code, sales_frame)


def test_audit_hook_denies_sockets_directly():
    """The socket.* audit events are refused even when a socket module is reachable."""
    worker = sbx.WORKER
    script = textwrap.dedent(f"""
        import importlib.util, os, socket
        spec = importlib.util.spec_from_file_location("w", {str(worker)!r})
        w = importlib.util.module_from_spec(spec); spec.loader.exec_module(w)
        v = []
        w.install_audit_hook(v, os.getcwd())
        for fn in (lambda: socket.socket(),
                   lambda: socket.create_connection(("127.0.0.1", 9)),
                   lambda: socket.getaddrinfo("example.com", 80)):
            try:
                fn()
            except PermissionError:
                continue
            raise SystemExit("socket operation was NOT blocked")
        print("blocked", len(v))
    """)
    done = subprocess.run([sys.executable, "-I", "-c", script], capture_output=True, text=True,
                          timeout=60, cwd="/")
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "blocked 3"


def test_swallowing_the_error_does_not_hide_a_violation(sales_frame):
    code = """
        try:
            pd.io.common.os.open('/etc/passwd', 0)
        except Exception:
            pass
        result = 1
    """
    with pytest.raises(SandboxViolation):
        run_raw(code, sales_frame)


def test_child_has_no_parent_secrets(sales_frame, monkeypatch):
    monkeypatch.setenv("CSVDOCTOR_SECRET_FOR_TEST", "hunter2")
    out = run_raw("result = sorted(pd.io.common.os.environ.keys())", sales_frame)
    names = [v["value"] for v in out["values"]]
    assert "CSVDOCTOR_SECRET_FOR_TEST" not in names and "HOME" not in names


def test_infinite_loop_hits_cpu_limit(sales_frame):
    with pytest.raises(SandboxTimeout, match="CPU limit"):
        sbx.run_program("while True:\n    pass\n", sales_frame,
                        Limits(cpu_seconds=1, wall_seconds=30))


def test_infinite_loop_hits_wall_clock_limit(sales_frame):
    with pytest.raises(SandboxTimeout, match="time limit"):
        sbx.run_program("while True:\n    pass\n", sales_frame,
                        Limits(cpu_seconds=60, wall_seconds=2))


def test_memory_bomb_is_contained(sales_frame):
    with pytest.raises((ProgramError, SandboxTimeout)):
        sbx.run_program("result = np.ones(10**10).sum()", sales_frame,
                        Limits(cpu_seconds=3, wall_seconds=20, memory_mb=256))


def test_runtime_errors_and_missing_result_are_reported_with_context(sales_frame):
    with pytest.raises(ProgramError, match=r"KeyError: 'profit' \(line 1"):
        sbx.run_program("result = df['profit'].sum()", sales_frame, FAST)
    with pytest.raises(ProgramError, match="did not assign"):
        sbx.run_program("x = 1", sales_frame, FAST)
    with pytest.raises(ProgramError, match="unsupported result type"):
        sbx.run_program("result = {1, 2}", sales_frame, FAST)
    with pytest.raises(ProgramError, match="too much output"):
        sbx.run_program("print('x' * 100000)\nresult = 1", sales_frame, FAST)


# ============================================================ verification

def raw_scalar(v):
    return {"kind": "scalar", "value": v}


def test_verify_accepts_correct_and_rejects_tampered_scalar(sales, sales_schema, sales_frame):
    spec = QuerySpec(Metric("mean", "revenue"), (Filter("region", "==", "North"),))
    truth = float(sales_frame[sales_frame["region"] == "North"]["revenue"].mean())
    ok = verify(raw_scalar(truth), spec, sales_frame, sales_schema)
    assert ok.passed
    bad = verify(raw_scalar(truth * 1.001), spec, sales_frame, sales_schema)
    assert not bad.passed
    assert any(c.name == "recompute:value" and not c.passed for c in bad.checks)


def test_verify_sanity_checks_fire_independently_of_recompute(sales_schema, sales_frame):
    n = len(sales_frame)
    over = verify(raw_scalar(n * 2), QuerySpec(Metric("count")), sales_frame, sales_schema)
    assert any(c.name == "value:count_range" and not c.passed for c in over.checks)
    out_of_range = verify(raw_scalar(10**9), QuerySpec(Metric("mean", "units")),
                          sales_frame, sales_schema)
    assert any(c.name == "value:within_column_range" and not c.passed for c in out_of_range.checks)
    inf = verify(raw_scalar({"special": "inf"}), QuerySpec(Metric("sum", "units")),
                 sales_frame, sales_schema)
    assert not inf.passed


@pytest.mark.parametrize("raw", [
    {"kind": "text", "value": "forty-two"},
    {"kind": "scalar", "value": True},
    {"kind": "series", "index": ["a"], "values": [1], "truncated": False},
    {"kind": "dict", "items": {}},
])
def test_verify_rejects_wrong_shape_for_scalar_plan(raw, sales_schema, sales_frame):
    with pytest.raises(ResultShapeError):
        verify(raw, QuerySpec(Metric("count")), sales_frame, sales_schema)


def test_verify_groups_and_ranking(sales_schema, sales_frame):
    spec = QuerySpec(Metric("sum", "revenue"), group_by="region", top=TopN(2, True))
    truth = sales_frame.groupby("region")["revenue"].sum().sort_values(ascending=False,
                                                                       kind="mergesort").head(2)
    good = {"kind": "series", "index": list(truth.index), "values": [float(v) for v in truth.values],
            "truncated": False}
    assert verify(good, spec, sales_frame, sales_schema).passed
    swapped = dict(good, index=list(reversed(good["index"])))
    assert not verify(swapped, spec, sales_frame, sales_schema).passed
    invented = dict(good, index=["North", "Atlantis"])
    res = verify(invented, spec, sales_frame, sales_schema)
    assert any(c.name == "groups:labels_exist" and not c.passed for c in res.checks)


def test_verify_api_view_never_contains_numbers(sales_schema, sales_frame):
    res = verify(raw_scalar(123456.789), QuerySpec(Metric("sum", "units")), sales_frame, sales_schema)
    assert not res.passed
    assert "123456" not in json.dumps(res.to_api())


def test_ties_and_missing_values_agree_between_program_and_verifier():
    df, types = parse(
        """
        team,pts
        a,10
        b,10
        c,5
        c,
        d,
        a,N/A
        e,1
        e,1
        e,1
        e,1
        e,1
        e,1
        e,1
        """)  # 10 of 11 non-blank cells are numbers, so pts is still inferred numeric
    out = answer_question("which team had the highest pts", df, types, limits=FAST)
    assert out["status"] == "verified"
    assert [g["label"] for g in out["answer"]["groups"]] == ["a"]  # tie a/b -> label order
    out = answer_question("average pts by team", df, types, limits=FAST)
    assert out["status"] == "verified"
    vals = {g["label"]: g["value"] for g in out["answer"]["groups"]}
    assert vals == {"a": 10, "b": 10, "c": 5, "d": None, "e": 1}  # all-missing group -> no value


# ============================================================ agent: happy path

def test_headline_question_verified_against_independent_pandas(sales):
    df, types = sales
    out = answer_question("did returns change after March for the North region?", df, types,
                          limits=FAST)
    assert out["status"] == "verified" and out["reason"] is None
    north = df[df["region"] == "North"].copy()
    north["returns"] = pd.to_numeric(north["returns"])
    d = pd.to_datetime(north["date"])
    before = north[d < "2025-04-01"]["returns"].mean()
    after = north[d >= "2025-04-01"]["returns"].mean()
    a = out["answer"]
    assert a["kind"] == "compare"
    assert a["before"] == pytest.approx(before) and a["after"] == pytest.approx(after)
    assert a["change"] == pytest.approx(after - before)
    assert a["n_before"] + a["n_after"] == len(north)
    assert "not a significance test" in a["text"]
    assert out["program"].startswith("import pandas as pd") and "result" in out["program"]
    assert out["plan"]["compare"]["split"] == "2025-04-01"
    assert out["verification"]["passed"] and all(c["passed"] for c in out["verification"]["checks"])
    assert out["attempts"] == [{"attempt": 1, "program": out["program"], "error": None, "reason": None}]
    assert out["generator"] == "local-rules-v1"


@pytest.mark.parametrize("question,check", [
    ("how many rows where region is North", lambda a: a["value"] == 52),
    ("total units in North", lambda a: a["value"] == 11790),
    ("how many orders have units over 50", lambda a: a["value"] == 46),
    ("which region had the highest revenue?", lambda a: a["groups"][0]["label"] == "North"),
    ("average revenue by product", lambda a: len(a["groups"]) == 3),
    ("how many distinct products", lambda a: a["value"] == 3),
    ("median revenue excluding North", lambda a: a["kind"] == "scalar"),
])
def test_common_question_shapes_verify_end_to_end(question, check, sales):
    out = answer_question(question, *sales, limits=FAST)
    assert out["status"] == "verified", out["message"]
    assert check(out["answer"])


def test_results_are_identical_across_reruns(sales):
    q = "did returns change after March for the North region?"
    assert answer_question(q, *sales, limits=FAST) == answer_question(q, *sales, limits=FAST)


# ============================================================ agent: verification failure

def test_wrong_number_is_never_shown_as_an_answer(sales, sales_frame):
    wrong = float(sales_frame["units"].sum())  # the program computes a sum; the plan says mean
    gen = Scripted(prog("result = df['units'].sum()", MEAN_UNITS))
    out = answer_question("what is the average units", *sales, generator=gen, limits=FAST)
    assert out["status"] == "unverified" and out["reason"] == "verification_failed"
    assert out["answer"] is None
    assert "could not be verified" in out["message"]
    assert out["verification"]["passed"] is False
    assert out["program"] == "result = df['units'].sum()\n"  # the program is still disclosed
    dump = json.dumps(out)
    assert f"{wrong:,.0f}" not in dump and str(int(wrong)) not in dump
    assert gen.calls == 1  # a wrong answer is reported, not retried away


def test_sanity_failure_is_unverified_too(sales):
    gen = Scripted(prog("result = len(df) * 2", QuerySpec(Metric("count"))))
    out = answer_question("how many rows", *sales, generator=gen, limits=FAST)
    assert out["status"] == "unverified" and out["answer"] is None
    failed = [c["name"] for c in out["verification"]["checks"] if not c["passed"]]
    assert "value:count_range" in failed and "recompute:value" in failed


def test_wrong_grouping_and_wrong_compare_are_unverified(sales):
    grouped = Scripted(prog("result = df.groupby('region')['revenue'].mean()",
                            QuerySpec(Metric("sum", "revenue"), group_by="region")))
    out = answer_question("revenue by region", *sales, generator=grouped, limits=FAST)
    assert out["status"] == "unverified" and out["answer"] is None
    cmp_spec = QuerySpec(Metric("mean", "returns"), compare=Compare("date", "2025-04-01"))
    off_by_cut = Scripted(prog("""
        cut = pd.Timestamp('2025-03-01')
        result = {'before': df[df['date'] < cut]['returns'].mean(),
                  'after': df[df['date'] >= cut]['returns'].mean()}
        """, cmp_spec))
    out = answer_question("did returns change", *sales, generator=off_by_cut, limits=FAST)
    assert out["status"] == "unverified" and out["answer"] is None


def test_program_without_a_plan_cannot_be_verified_and_shows_nothing(sales):
    gen = Scripted(prog("result = 42", None))
    out = answer_question("anything", *sales, generator=gen, limits=FAST)
    assert out["status"] == "failed" and out["answer"] is None
    assert gen.calls == 2 and "plan" in out["attempts"][0]["error"]


# ============================================================ agent: one retry, then honest failure

def test_one_retry_feeds_the_error_back_and_can_recover(sales):
    broken = prog("result = df['profit'].mean()", MEAN_UNITS)
    fixed = prog("result = df['units'].mean()", MEAN_UNITS)
    gen = Scripted(broken, fixed)
    out = answer_question("average units", *sales, generator=gen, limits=FAST)
    assert out["status"] == "verified" and gen.calls == 2
    first, second = gen.requests
    assert first.attempt == 1 and first.previous_error is None
    assert second.attempt == 2 and "KeyError: 'profit'" in second.previous_error
    assert second.previous_program == broken.code
    assert [a["error"] is None for a in out["attempts"]] == [False, True]
    assert out["attempts"][0]["reason"] == "program_error"
    assert out["program"] == fixed.code


def test_policy_violation_is_fed_back_like_any_other_error(sales):
    gen = Scripted(prog("import os\nresult = 1", MEAN_UNITS),
                   prog("result = df['units'].mean()", MEAN_UNITS))
    out = answer_question("average units", *sales, generator=gen, limits=FAST)
    assert out["status"] == "verified"
    assert out["attempts"][0]["reason"] == "sandbox_violation"
    assert "not allowed" in gen.requests[1].previous_error


def test_two_failures_end_in_honest_failure_without_a_third_try(sales):
    gen = Scripted(prog("result = df['profit'].mean()", MEAN_UNITS),
                   prog("result = df['cost'].mean()", MEAN_UNITS),
                   prog("result = df['units'].mean()", MEAN_UNITS))  # must never be reached
    out = answer_question("average units", *sales, generator=gen, limits=FAST)
    assert gen.calls == 2 and len(out["attempts"]) == 2
    assert out["status"] == "failed" and out["answer"] is None and out["verification"] is None
    assert out["reason"] == "program_error"
    assert "after 2 attempts" in out["message"] and "KeyError: 'cost'" in out["message"]
    assert out["program"] == "result = df['cost'].mean()\n"


def test_timeout_is_retried_once_then_reported(sales):
    loop = prog("while True:\n    pass\n", MEAN_UNITS)
    gen = Scripted(loop)
    out = answer_question("average units", *sales, generator=gen,
                          limits=Limits(cpu_seconds=1, wall_seconds=10))
    assert gen.calls == 2 and out["status"] == "failed" and out["reason"] == "timeout"
    assert out["answer"] is None


def test_wrong_result_shape_is_a_retryable_program_error(sales):
    gen = Scripted(prog("result = 'about 12'", MEAN_UNITS), prog("result = df['units'].mean()", MEAN_UNITS))
    out = answer_question("average units", *sales, generator=gen, limits=FAST)
    assert out["status"] == "verified" and out["attempts"][0]["reason"] == "bad_result_shape"


def test_invalid_plan_and_generator_crash_each_get_one_retry(sales):
    bad_plan = prog("result = 1", QuerySpec(Metric("mean", "nope")))
    out = answer_question("q", *sales, generator=Scripted(bad_plan), limits=FAST)
    assert out["status"] == "failed" and out["reason"] == "invalid_plan"
    crash = Scripted(RuntimeError("provider down: secret-token-123"), GenerationError("bad json"))
    out = answer_question("q", *sales, generator=crash, limits=FAST)
    assert out["status"] == "failed" and crash.calls == 2
    assert "secret-token-123" not in json.dumps(out)  # provider details never echoed


def test_unanswerable_question_fails_immediately_without_running_anything(sales):
    out = answer_question("what is the average profit", *sales, limits=FAST)
    assert out["status"] == "failed" and out["reason"] == "unknown_column"
    assert "profit" in out["message"] and out["answer"] is None and out["program"] is None
    assert len(out["attempts"]) == 1 and out["attempts"][0]["program"] is None


# ============================================================ agent: edge cases

def test_filter_matching_nothing_is_reported_as_empty_not_zero_or_nan(sales):
    out = answer_question("median revenue for the Atlantis region", *sales, limits=FAST)
    assert out["status"] == "verified"
    assert out["answer"]["kind"] == "empty" and out["answer"]["value"] is None
    assert "No rows matched" in out["answer"]["text"] and "Atlantis" in out["answer"]["text"]
    counted = answer_question("how many rows where region is Atlantis", *sales, limits=FAST)
    assert counted["status"] == "verified"
    assert counted["answer"]["kind"] == "scalar" and counted["answer"]["value"] == 0
    grouped = answer_question("average revenue by product for the Atlantis region", *sales, limits=FAST)
    assert grouped["status"] == "verified" and grouped["answer"]["kind"] == "empty"
    summed = answer_question("total units in the Atlantis region", *sales, limits=FAST)
    assert summed["answer"]["kind"] == "empty"  # sum of nothing is not presented as a real 0


def test_compare_with_an_empty_side_is_explained(sales):
    out = answer_question("did returns change after June for the North region", *sales, limits=FAST)
    assert out["status"] == "verified"
    assert out["answer"]["after"] is None and out["answer"]["change"] is None
    assert "Cannot compare" in out["answer"]["text"]


def test_missing_column_question_is_an_honest_failure(sales):
    out = answer_question("average profit by region", *sales, limits=FAST)
    assert out["status"] == "failed" and out["reason"] == "unknown_column"
    assert "profit" in out["message"] and "units, revenue, returns" in out["message"]


def test_dataset_with_columns_named_like_python_words():
    df, types = parse(
        """
        type,id,format,score
        a,1,x,10
        b,2,y,20
        a,3,x,30
        a,4,y,40
        b,5,x,50
        """)
    out = answer_question("average score by type", df, types, limits=FAST)
    assert out["status"] == "verified"
    assert {g["label"]: g["value"] for g in out["answer"]["groups"]} == {"a": pytest.approx(80 / 3), "b": 35}


# ============================================================ LLM adapter (no network: fake `complete`)

LLM_REPLY = json.dumps({
    "plan": {"metric": {"agg": "mean", "column": "units"}},
    "code": "result = df['units'].mean()",
    "assumptions": ["plain mean over all rows"]})


def test_llm_generator_round_trip_with_fenced_json(sales):
    seen = []

    def complete(messages):
        seen.append(messages)
        return f"```json\n{LLM_REPLY}\n```"

    out = answer_question("average units", *sales, generator=LLMCodeGenerator(complete, "fake-llm"),
                          limits=FAST)
    assert out["status"] == "verified" and out["generator"] == "fake-llm"
    assert out["assumptions"] == ["plain mean over all rows"]
    text = json.dumps(seen[0])
    assert "average units" in text and "North" in text  # question + categorical values
    assert "20010" not in text and "2025-01-11" not in text  # never raw rows


def test_llm_bad_json_is_retried_with_the_error_and_then_fails_honestly(sales):
    replies = iter(["sorry, here is some prose", "{still not json"])
    prompts = []

    def complete(messages):
        prompts.append(messages[-1]["content"])
        return next(replies)

    out = answer_question("average units", *sales, generator=LLMCodeGenerator(complete), limits=FAST)
    assert out["status"] == "failed" and len(prompts) == 2
    assert "ONE retry" in prompts[1] and "not valid JSON" in prompts[1]


def test_llm_unanswerable_and_unverifiable_replies():
    with pytest.raises(Unanswerable) as exc:
        parse_llm_response('{"unanswerable": {"reason": "unknown_column", "message": "no profit"}}')
    assert exc.value.reason == "unknown_column"
    with pytest.raises(GenerationError, match="invalid plan"):
        parse_llm_response('{"plan": {"metric": {}}, "code": "result = 1"}')
    with pytest.raises(GenerationError, match="no 'code'"):
        parse_llm_response('{"plan": null}')
    assert parse_llm_response('{"code": "result = 1"}').plan is None  # -> agent refuses to show it
    msgs = build_llm_messages(GenerationRequest("q", {"columns": []}))
    assert msgs[0]["role"] == "system" and "never see its rows" in msgs[0]["content"]


def test_llm_that_hallucinates_a_number_is_caught_by_verification(sales):
    reply = json.dumps({"plan": {"metric": {"agg": "mean", "column": "units"}}, "code": "result = 42.0"})
    out = answer_question("average units", *sales,
                          generator=LLMCodeGenerator(lambda m: reply), limits=FAST)
    assert out["status"] == "unverified" and out["answer"] is None


# ============================================================ HTTP endpoint

def ask(client, question, content=SALES, filename="sales.csv"):
    return client.post("/ask", files={"file": (filename, content, "text/csv")},
                       data={"question": question})


def test_ask_endpoint_happy_path(client):
    r = ask(client, "did returns change after March for the North region?")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "verified" and body["answer"]["kind"] == "compare"
    assert body["program"] and body["plan"] and body["verification"]["passed"]


def test_ask_endpoint_unverified_is_clearly_labelled_and_has_no_answer(client):
    app.dependency_overrides[get_generator] = lambda: Scripted(
        prog("result = df['units'].sum()", MEAN_UNITS))
    r = ask(client, "average units")
    body = r.json()
    assert r.status_code == 200 and body["status"] == "unverified"
    assert body["answer"] is None and "could not be verified" in body["message"]
    assert body["program"] is not None


def test_ask_endpoint_honest_failures(client):
    body = ask(client, "what is the average profit").json()
    assert body["status"] == "failed" and body["reason"] == "unknown_column"
    app.dependency_overrides[get_generator] = lambda: Scripted(prog("result = df['x'].mean()", MEAN_UNITS))
    body = ask(client, "average units").json()
    assert body["status"] == "failed" and len(body["attempts"]) == 2


@pytest.mark.parametrize("question", ["", "   ", "x" * 501])
def test_ask_endpoint_rejects_bad_questions(client, question):
    assert ask(client, question).status_code == 422


def test_ask_endpoint_rejects_bad_files_and_missing_fields(client):
    assert ask(client, "how many rows", content=b"", filename="e.csv").status_code == 422
    assert client.post("/ask", data={"question": "how many rows"}).status_code == 422
    assert client.post("/ask", files={"file": ("a.csv", SALES, "text/csv")}).status_code == 422


def test_ask_endpoint_reports_a_saturated_sandbox_as_503(client, monkeypatch):
    def busy(*a, **k):
        raise sbx.SandboxBusy("too many analyses are running; try again shortly")
    monkeypatch.setattr("app.engine.ask.agent.run_program", busy)
    r = ask(client, "how many rows")
    assert r.status_code == 503 and "try again" in r.json()["detail"]


def test_analyse_and_health_are_unchanged(client):
    assert client.get("/health").json()["ok"] is True
    r = client.post("/analyse", files={"file": ("sales.csv", SALES, "text/csv")})
    assert r.status_code == 200
    assert {"health_score", "profile", "anomalies", "drift", "relations", "facts_pack"} <= set(r.json())
