"""Phase B1 tests: fuzzy duplicate detection (app/engine/relations.py).

Ground truth for the dirty-names fixture lives in
tests/fixtures/dirty_names_truth.json (0-based data-row indices).
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import pandas as pd
import pytest

from app.engine.parse import infer_types, parse_csv_bytes
from app.engine.relations import (
    MATCH_THRESHOLD, blocking_keys, candidate_pair_set, find_duplicates,
    normalise, soundex, value_similarity)

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLES = Path(__file__).resolve().parents[3] / "samples"


@pytest.fixture(scope="module")
def dirty():
    df = parse_csv_bytes((FIXTURES / "dirty_names.csv").read_bytes())
    truth = json.loads((FIXTURES / "dirty_names_truth.json").read_text(encoding="utf-8"))
    return df, truth


@pytest.fixture(scope="module")
def dirty_result(dirty):
    df, _ = dirty
    return find_duplicates(df, infer_types(df))


def _frame(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["name", "address", "city"])


def _group_sets(result: dict) -> list[frozenset[int]]:
    return [frozenset(g["rows"]) for g in result["groups"]]


def _pairs(groups) -> set[tuple[int, int]]:
    return {p for g in groups for p in combinations(sorted(g), 2)}


def _padded(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    """Rows plus enough distinct filler people that name/address count as identity columns."""
    filler = [(f"{a} {b}", f"{i + 1} {s}", "Delhi") for i, (a, b, s) in enumerate([
        ("Meera", "Iyer", "Temple Street"), ("Arjun", "Reddy", "Banjara Hills"),
        ("Neha", "Gupta", "Sector Fifteen"), ("Karan", "Malhotra", "Defence Colony"),
        ("Fatima", "Sheikh", "Camp Road"), ("Sanjay", "Patel", "Ashram Road")])]
    return _frame(rows + filler)


def _run(df: pd.DataFrame) -> dict:
    return find_duplicates(df, infer_types(df))


# ---- fixture ground truth ------------------------------------------------------

def test_fixture_matches_its_truth_file(dirty):
    df, truth = dirty
    assert len(df) == len(truth["entities"])
    for group in truth["duplicate_groups"]:
        assert len({truth["entities"][i] for i in group}) == 1
    # Every planted group is made of rows that really look alike after cleanup.
    names = df["name"].map(normalise)
    for group in truth["duplicate_groups"]:
        assert len({n[:2] for n in names[group]}) <= 2


def test_fixture_precision_and_recall(dirty, dirty_result):
    _, truth = dirty
    predicted = _pairs(_group_sets(dirty_result))
    true_pairs = {tuple(p) for p in truth["duplicate_pairs"]}
    assert predicted == true_pairs, (
        f"missed={sorted(true_pairs - predicted)} extra={sorted(predicted - true_pairs)}")


def test_typo_variants_grouped(dirty, dirty_result):
    df, truth = dirty
    groups = _group_sets(dirty_result)
    for planted in truth["duplicate_groups"]:
        assert frozenset(planted) in groups, [df.name[i] for i in planted]


def test_jon_smith_and_john_smith_group_but_other_john_smith_does_not(dirty, dirty_result):
    df, _ = dirty
    jons = [i for i, n in enumerate(df["name"]) if n.strip() in ("Jon Smith", "John Smith")]
    assert len(jons) == 3
    in_group = [g for g in _group_sets(dirty_result) if g & set(jons)]
    assert len(in_group) == 1 and len(in_group[0]) == 2
    other = [i for i in jons if df.address[i] == "88 Ring Road"][0]
    assert other not in in_group[0]


def test_distinct_people_not_grouped(dirty, dirty_result):
    _, truth = dirty
    grouped_rows = set().union(*_group_sets(dirty_result))
    singles = [i for i, e in enumerate(truth["entities"]) if e.startswith("single_")]
    # hard negatives (same name / same address / close first name) included
    assert not grouped_rows & set(singles)
    for a, b in truth["hard_negatives"]:
        assert not any({a, b} <= g for g in _group_sets(dirty_result))


def test_every_group_is_reviewable(dirty, dirty_result):
    df, _ = dirty
    for g in dirty_result["groups"]:
        assert g["rows"] == sorted(g["rows"]) and g["size"] == len(g["rows"]) >= 2
        assert 0.0 <= g["confidence"] <= 1.0
        assert g["confidence"] >= MATCH_THRESHOLD
        assert g["reasons"], "bare group without a reason"
        assert {f["column"] for f in g["matched_fields"]} >= {"name", "address"}
        assert all(0 <= i < len(df) for i in g["rows"])
    assert dirty_result["skipped"] is False
    assert dirty_result["method"]["identity_columns"] == ["name", "address"]
    assert dirty_result["method"]["context_columns"] == ["city"]


def test_nothing_is_modified(dirty):
    df, _ = dirty
    before = df.copy(deep=True)
    find_duplicates(df, infer_types(df))
    pd.testing.assert_frame_equal(df, before)


# ---- exact duplicates ----------------------------------------------------------

def test_exact_duplicates_grouped():
    df = _padded([("Priya Sharma", "45 Park Street", "Mumbai")] * 3)
    result = _run(df)
    assert [g["rows"] for g in result["groups"]] == [[0, 1, 2]]
    group = result["groups"][0]
    assert group["kind"] == "exact" and group["confidence"] == 1.0
    assert "exact duplicate" in group["reasons"][0]
    assert result["duplicate_rows"] == 2


def test_exact_duplicates_found_even_without_name_columns():
    df = pd.DataFrame({"order_id": ["1", "2", "2", "3"], "units": ["5", "7", "7", "9"]})
    result = _run(df)
    assert result["skipped"] is False
    assert [g["rows"] for g in result["groups"]] == [[1, 2]]
    assert "name/address" in result["reason"]


def test_case_and_whitespace_variants_grouped():
    df = _padded([("Priya Sharma", "45 Park Street", "Delhi"),
                  ("  priya   SHARMA ", "45 park street", "delhi ")])
    result = _run(df)
    assert [g["rows"] for g in result["groups"]] == [[0, 1]]
    group = result["groups"][0]
    assert group["kind"] == "near" and group["confidence"] == 1.0
    assert any("same after ignoring case" in r for r in group["reasons"])


# ---- confidence ----------------------------------------------------------------

def test_confidence_higher_for_closer_matches():
    df = _padded([
        ("Vikram Singh", "18 Lake View Apartments", "Jaipur"),
        ("Vikram Singh", "18 Lake View Apartments", "Jaipur"),      # exact: pair A
        ("Rajesh Kumar", "7 Nehru Lane", "Pune"),
        ("Rajseh Kumar", "7 Nehru Lane", "Pune"),                    # one typo: pair B
        ("Mohammed Aslam", "90 Residency Road", "Bengaluru"),
        ("Muhammad Aslam", "90 Residency Road", "Bangalore"),        # more drift: pair C
    ])
    by_rows = {tuple(g["rows"]): g["confidence"] for g in _run(df)["groups"]}
    assert set(by_rows) == {(0, 1), (2, 3), (4, 5)}
    assert by_rows[(0, 1)] > by_rows[(2, 3)] > by_rows[(4, 5)]


def test_chain_lowers_group_confidence():
    # A~B and B~C link, but A vs C is the weakest pair: confidence must show it.
    df = _padded([("Anitha Raman", "5 Garden Lane", "Pune"),
                  ("Anita Raman", "5 Garden Lane", "Pune"),
                  ("Anitaa Ramana", "5 Garden Lane", "Pune")])
    groups = _run(df)["groups"]
    assert groups and groups[0]["rows"] == [0, 1, 2]
    assert groups[0]["weakest_pair"] <= groups[0]["confidence"] < 1.0


def test_value_similarity_orders_by_closeness():
    close = value_similarity(normalise("John Smith"), normalise("Jon Smith"))
    far = value_similarity(normalise("John Smith"), normalise("Jane Smythe"))
    different = value_similarity(normalise("John Smith"), normalise("Priya Sharma"))
    assert 1.0 > close > far > different
    assert value_similarity("a b", "a b") == 1.0


def test_house_number_difference_is_penalised():
    same = value_similarity(normalise("12 Park Street"), normalise("12 Park Street"))
    other = value_similarity(normalise("12 Park Street"), normalise("21 Park Street"))
    assert same == 1.0 and other < MATCH_THRESHOLD


# ---- blocking ------------------------------------------------------------------

def test_blocking_does_not_miss_planted_pairs(dirty):
    df, truth = dirty
    candidates = candidate_pair_set(df, infer_types(df))
    missed = [tuple(p) for p in truth["duplicate_pairs"] if tuple(p) not in candidates]
    assert not missed, f"blocking dropped planted pairs {missed}"


def test_blocking_prunes_most_pairs(dirty, dirty_result):
    df, _ = dirty
    stats = dirty_result["stats"]
    assert stats["naive_pairs"] == len(df) * (len(df) - 1) // 2
    assert stats["candidate_pairs"] < 0.1 * stats["naive_pairs"]
    assert stats["blocking_reduction"] > 0.9


def test_blocking_keys_survive_typical_variants():
    def shared(a: str, b: str) -> bool:
        return bool(set(blocking_keys(normalise(a))) & set(blocking_keys(normalise(b))))
    assert shared("Jon Smith", "John Smith")
    assert shared("Smith John", "John Smith")             # word order
    assert shared("Rajesh Kumar", "Rajseh Kumar")         # soundex differs, spelling key holds
    assert shared("Café Noir", "cafe noir ")
    assert not shared("Priya Sharma", "Meera Iyer")


def test_soundex_known_values():
    assert soundex("Robert") == soundex("Rupert") == "R163"
    assert soundex("Smith") == soundex("Smyth") == "S530"
    assert soundex("Tymczak") == "T522"
    assert soundex("1234") == ""


def test_oversize_block_falls_back_to_window():
    # 300 people sharing one surname and first initial: one giant bucket.
    rows = [(f"Sam{chr(97 + i % 26)}{chr(97 + i // 26)} Patel", f"{i} Main Road", "Pune")
            for i in range(300)]
    rows += [("Sam Patel", "400 Main Road", "Pune"), ("Sam Patel", "400 Main Road", "Pune")]
    result = _run(_frame(rows))
    assert result["stats"]["oversize_blocks"] >= 1
    assert result["stats"]["candidate_pairs"] < 300 * 299 // 2
    assert any(g["rows"] == [300, 301] for g in result["groups"])


# ---- edge cases ----------------------------------------------------------------

def test_empty_frame():
    df = pd.DataFrame({"name": [], "address": []}, dtype=str)
    result = find_duplicates(df, infer_types(df))
    assert result["skipped"] is True and result["groups"] == []
    assert result["duplicate_groups"] == 0


def test_single_row():
    result = _run(_frame([("Priya Sharma", "45 Park Street", "Delhi")]))
    assert result["skipped"] is True and result["groups"] == []


def test_all_unique_frame_has_no_groups():
    df = _padded([("Priya Sharma", "45 Park Street", "Mumbai"),
                  ("Rahul Verma", "9 Hill Road", "Pune")])
    result = _run(df)
    assert result["skipped"] is False and result["groups"] == []
    assert result["duplicate_groups"] == 0 and result["duplicate_rows"] == 0


def test_missing_values_do_not_create_matches():
    df = _padded([("Priya Sharma", "", "Delhi"), ("Priya Sharma", "", "Delhi"),
                  ("", "", ""), ("", "", "")])
    groups = _group_sets(_run(df))
    assert frozenset({0, 1}) in groups and frozenset({2, 3}) in groups
    assert not any({0, 2} <= g for g in groups)


def test_low_cardinality_columns_alone_never_match_rows():
    # sales-style file: region/product repeat by design, so rows must not merge.
    df = parse_csv_bytes((SAMPLES / "sales.csv").read_bytes())
    result = find_duplicates(df, infer_types(df))
    assert result["duplicate_rows"] < 0.02 * len(df)


# ---- determinism ---------------------------------------------------------------

def test_deterministic(dirty):
    df, _ = dirty
    types = infer_types(df)
    first = json.dumps(find_duplicates(df, types), sort_keys=False)
    second = json.dumps(find_duplicates(df.copy(deep=True), types), sort_keys=False)
    assert first == second


def test_benchmark_generator_is_seeded():
    from benchmarks.dedupe_benchmark import generate_dirty_set
    a, ea = generate_dirty_set(5, 60)
    b, eb = generate_dirty_set(5, 60)
    pd.testing.assert_frame_equal(a, b)
    assert ea == eb


# ---- API wiring ----------------------------------------------------------------

def test_analyse_endpoint_exposes_dedupe_groups():
    from fastapi.testclient import TestClient

    from app.main import ENGINE_VERSION, app

    raw = (FIXTURES / "dirty_names.csv").read_bytes()
    resp = TestClient(app).post("/analyse", files={"file": ("n.csv", raw, "text/csv")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["engine_version"] == ENGINE_VERSION and "b1" in ENGINE_VERSION
    rel = body["relations"]
    assert rel["skipped"] is False and rel["duplicate_groups"] == 8
    assert all(g["reasons"] for g in rel["groups"])
    # existing response shape still intact
    for key in ("health_score", "profile", "anomalies", "drift", "facts_pack"):
        assert key in body


def test_facts_pack_has_dedupe_counts_but_no_raw_values():
    from fastapi.testclient import TestClient

    from app.main import app

    raw = (FIXTURES / "dirty_names.csv").read_bytes()
    body = TestClient(app).post("/analyse", files={"file": ("n.csv", raw, "text/csv")}).json()
    rel = body["facts_pack"]["relations"]
    assert rel["duplicate_groups"] == 8
    assert rel["duplicate_rows"] == body["relations"]["duplicate_rows"]
    assert "Priya" not in json.dumps(rel) and "groups" not in rel


def test_analyse_endpoint_still_works_on_files_without_names():
    from fastapi.testclient import TestClient

    from app.main import app

    raw = (SAMPLES / "survey.csv").read_bytes()
    resp = TestClient(app).post("/analyse", files={"file": ("s.csv", raw, "text/csv")})
    assert resp.status_code == 200
    assert resp.json()["relations"]["duplicate_rows"] == 0
