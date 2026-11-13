"""Phase C1 tests: chunked + DuckDB profiling (app/engine/scale.py).

The contract: the scale path reports the same numbers as the in-memory path
(profile.profile_dataframe + profile.health_score) on the same data, whatever
the chunk size, and its DuckDB aggregates match plain-pandas ground truth.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from app.engine import profile as profile_mod
from app.engine import scale
from app.engine.parse import ParseError, parse_csv_bytes

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = sorted((ROOT.parents[1] / "samples").glob("*.csv"))
FIXTURES = sorted((Path(__file__).parent / "fixtures").glob("*.csv"))


def make_csv(path: Path, n: int = 4_000, seed: int = 7) -> Path:
    """A seeded CSV with every column kind and planted dirt (blanks, duplicates)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "order_id": [f"{i:06d}" for i in range(n)],
        "region": rng.choice(["north", "south", "east", "west", ""], n, p=[.3, .3, .2, .15, .05]),
        "price": np.round(rng.normal(50, 20, n), 2).astype(str),
        "qty": rng.integers(-3, 40, n).astype(str),
        "active": rng.choice(["true", "false", "TRUE"], n),
        "placed": pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D"),
        "note": [f"note {v}" for v in rng.integers(0, 900, n)],
    })
    df["placed"] = df["placed"].dt.strftime("%Y-%m-%d")
    df.loc[rng.choice(n, 120, replace=False), "price"] = ""
    df.loc[rng.choice(n, 60, replace=False), "note"] = ""
    df = pd.concat([df, df.iloc[:150]], ignore_index=True)   # 150 exact duplicate rows
    df.to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def sample_csv(tmp_path_factory) -> Path:
    return make_csv(tmp_path_factory.mktemp("scale") / "sample.csv")


def in_memory(path: Path) -> dict:
    prof = profile_mod.profile_dataframe(parse_csv_bytes(Path(path).read_bytes()))
    score, reasons = profile_mod.health_score(prof)
    return {"profile": prof, "health_score": score, "score_reasons": reasons}


# ---------------------------------------------------- chunked == whole frame

def test_chunked_profile_equals_whole_frame_profile(sample_csv):
    expected = in_memory(sample_csv)
    got = scale.profile_file(sample_csv, chunk_rows=500)
    assert got["profile"]["rows"] == expected["profile"]["rows"] == 4_150
    assert got["scale"]["chunks"] == 9
    # Row count, per-column nulls, exact distincts, numeric min/max/mean/median.
    assert got["profile"] == expected["profile"]
    assert got["health_score"] == expected["health_score"]
    assert got["score_reasons"] == expected["score_reasons"]


def test_sample_statistics_are_what_we_planted(sample_csv):
    cols = {c["name"]: c for c in scale.profile_file(sample_csv)["profile"]["columns"]}
    assert cols["order_id"]["type"] == "id"
    assert cols["price"]["type"] == "numeric" and cols["qty"]["negatives"] > 0
    assert cols["active"]["type"] == "boolean"
    assert cols["placed"]["type"] == "datetime"
    assert cols["region"]["missing_pct"] > 0
    assert scale.profile_file(sample_csv)["profile"]["duplicate_rows"] >= 150


@pytest.mark.parametrize("path", SAMPLES + FIXTURES, ids=lambda p: p.name)
def test_repo_sample_files_match_in_memory_profile(path):
    got = scale.profile_file(path, chunk_rows=7)   # tiny chunks: many merges
    expected = in_memory(path)
    assert got["profile"] == expected["profile"]
    assert (got["health_score"], got["score_reasons"]) == (expected["health_score"],
                                                           expected["score_reasons"])


def test_result_does_not_depend_on_chunk_size_and_is_deterministic(sample_csv):
    runs = [scale.profile_file(sample_csv, chunk_rows=n) for n in (1_000, 333, 100_000, 1_000)]
    for run in runs[1:]:
        assert run["profile"] == runs[0]["profile"]
        assert run["health_score"] == runs[0]["health_score"]
    assert runs[0] == runs[3] and runs[0]["scale"]["chunk_rows"] == 1_000


def test_health_score_reuses_the_in_memory_scoring_function(sample_csv, monkeypatch):
    seen = []
    real = profile_mod.health_score
    monkeypatch.setattr(profile_mod, "health_score", lambda p: seen.append(p) or real(p))
    result = scale.profile_file(sample_csv)
    assert seen == [result["profile"]]


# ------------------------------------------------- DuckDB vs pandas truth

def test_duckdb_aggregates_match_pandas_ground_truth(sample_csv, tmp_path):
    df = pd.read_csv(sample_csv, dtype=str, keep_default_na=False)
    n_cols = len(df.columns)
    numeric = [df.columns.get_loc("price"), df.columns.get_loc("qty")]
    con = scale._open_duckdb(sample_csv, n_cols, ",", "512MB", 2, tmp_path)
    try:
        agg = scale._aggregate_scan(con, n_cols, numeric)
        assert agg["rows"] == len(df)
        for i, name in enumerate(df.columns):
            non_empty = df[name][df[name] != ""]
            assert agg["cols"][i]["distinct"] == non_empty.nunique(), name
        for i in numeric:
            nums = pd.to_numeric(df.iloc[:, i][df.iloc[:, i] != ""], errors="coerce").dropna()
            got = agg["numeric"][i]
            assert got["count"] == len(nums)
            assert got["sum"] == pytest.approx(nums.sum(), rel=1e-12)
            assert got["sum"] / got["count"] == pytest.approx(nums.mean(), rel=1e-12)
            assert got["median"] == pytest.approx(nums.median())
            assert got["negatives"] == int((nums < 0).sum())
        assert scale._duplicate_rows(con, len(df)) == int(df.duplicated().sum())
        region = df.columns.get_loc("region")
        top = scale._top_values(con, [region])[region]
        counts = df["region"][df["region"] != ""].value_counts()
        assert top == {k: int(v) for k, v in counts.head(5).items()}
    finally:
        con.close()


def test_duckdb_number_grammar_matches_pandas_to_numeric(tmp_path):
    tokens = ["1", "2.5", " 3", "1e3", "+4", ".5", "5.", "inf", "-Infinity", "Inf ", "nan",
              "1_0", "0x10", "1e400", "abc", "1,5", "--1", "-0.25", "9"]
    path = tmp_path / "nums.csv"
    path.write_text("v,k\n" + "\n".join(f'"{t}",{i}' for i, t in enumerate(tokens)) + "\n")
    expect = pd.to_numeric(pd.Series(tokens), errors="coerce").dropna()
    con = scale._open_duckdb(path, 2, ",", "256MB", 1, tmp_path)
    try:
        got = con.execute(f"SELECT list({scale._numeric_sql('c0')}) FROM t").fetchone()[0]
    finally:
        con.close()
    got = pd.Series([g for g in got if g is not None])
    assert sorted(got) == sorted(expect)


def test_top_values_ties_are_broken_by_value(tmp_path):
    path = tmp_path / "ties.csv"
    path.write_text("c\n" + "\n".join(["b", "a", "d", "c", "e", "f", "g"] * 2) + "\n")
    cols = {c["name"]: c for c in scale.profile_file(path)["profile"]["columns"]}
    assert list(cols["c"]["top_values"]) == ["a", "b", "c", "d", "e"]


# ---------------------------------------------------------- awkward files

def test_quoted_newlines_blank_lines_and_semicolons(tmp_path):
    quoted = tmp_path / "q.csv"
    quoted.write_bytes(b'id,note,amt,flag\n1,"he said ""hi""\nthere",1.5,Yes\n2,plain,2,no\n'
                       b'3,"a,b",,YES\n\n4,,3e2,No\n2,plain,2,no\n')
    assert scale.profile_file(quoted, chunk_rows=2)["profile"] == in_memory(quoted)["profile"]
    semi = tmp_path / "s.csv"
    semi.write_text("a;b\nx;1\ny;2\nx;1\n")
    got = scale.profile_file(semi)
    assert got["scale"]["delimiter"] == ";" and got["profile"] == in_memory(semi)["profile"]


def test_latin1_file_is_decoded_like_the_in_memory_path(tmp_path):
    path = tmp_path / "latin.csv"
    path.write_bytes("name;city;n\ncafé;Zürich;1\nnaïve;Zürich;2\ncafé;Bern;3\n".encode("latin-1"))
    got = scale.profile_file(path)
    assert got["scale"]["encoding"] == "latin-1"
    assert got["profile"] == in_memory(path)["profile"]
    assert "café" in next(c for c in got["profile"]["columns"] if c["name"] == "name")["top_values"]


def test_sparse_numeric_column_is_still_typed_and_summarised(tmp_path):
    # Fewer than 500 non-empty values: the numeric decision happens at end of file.
    path = tmp_path / "sparse.csv"
    path.write_text("k,v\n" + "\n".join(f"{i}," + ("" if i % 4 else str(i * 1.5)) for i in range(400)) + "\n")
    assert scale.profile_file(path, chunk_rows=50)["profile"] == in_memory(path)["profile"]


def test_progress_is_monotonic_named_and_ends_at_100(sample_csv):
    events = []
    scale.profile_file(sample_csv, chunk_rows=500, progress=lambda p, s: events.append((p, s)))
    pcts = [p for p, _ in events]
    assert pcts == sorted(pcts) and pcts[-1] == 100.0
    assert sum(1 for _, s in events if s == "scanning chunks") == 9
    assert {"inspecting file", "exact aggregates (DuckDB)", "health score"} <= {s for _, s in events}


@pytest.mark.parametrize("content", [b"", b"a,b\n", b"\n\n"], ids=["empty", "header_only", "blank"])
def test_empty_files_raise_a_parse_error(tmp_path, content):
    path = tmp_path / "e.csv"
    path.write_bytes(content)
    with pytest.raises(ParseError, match="empty_file"):
        scale.profile_file(path)


def test_missing_file_and_bad_chunk_size(tmp_path):
    with pytest.raises(scale.ScaleError, match="not_a_csv"):
        scale.profile_file(tmp_path / "nope.csv")
    with pytest.raises(ValueError):
        scale.profile_file(tmp_path / "x.csv", chunk_rows=0)


def test_scale_works_past_the_in_memory_row_cap(tmp_path):
    # parse.py refuses > 50,000 rows; the scale path must not.
    n = 60_000
    path = tmp_path / "big.csv"
    pd.DataFrame({"a": np.arange(n) % 7, "b": [f"x{i % 3}" for i in range(n)]}).to_csv(path, index=False)
    with pytest.raises(ParseError, match="too_many_rows"):
        parse_csv_bytes(path.read_bytes())
    got = scale.profile_file(path, chunk_rows=20_000)["profile"]
    assert got["rows"] == n and got["duplicate_rows"] == n - 21
