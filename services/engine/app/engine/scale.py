"""Phase C1 — scale path: profile a CSV of ~1M rows without loading it.

`profile_file(path)` returns the same profile dict as profile.profile_dataframe
(rows, cols, missing_pct, duplicate_rows, duplicate_pct and one entry per
column) plus the deterministic Health Score, computed by the very same
profile.health_score() the in-memory path uses. Peak memory is bounded by the
chunk size and the DuckDB memory limit, not by the file size.

How the work is split (and why each statistic is exact):

1. Chunk pass (pandas `chunksize`, constant memory): row count, per-column
   null counts, numeric count/min/max, and the first 500 non-empty values of
   every column (the type sniffers only ever look at that head). All of these
   combine exactly across chunks.
2. DuckDB over the CSV file (in-process, the file is only ever read): the
   statistics that need the whole column at once and are expensive to merge
   chunk by chunk — exact distinct counts, numeric sum (so the mean) and
   median, negatives count, the boolean-type check, exact whole-row duplicate
   count, and exact top values. Nothing here is a sketch or an estimate:
   no HyperLogLog, no approximate quantiles. DuckDB is given a memory limit and
   a spill directory, so high-cardinality columns degrade to disk, not to OOM.
3. Column types come from parse.classify_column — the same rules infer_types
   applies in memory — fed with the exact counts from steps 1 and 2.

Known, deliberate differences from the in-memory path:
- The delimiter is sniffed from the header line among , ; tab | (default ",")
  rather than by pandas' Python-engine sniffer.
- top_values ties at the 5th place are broken by value, so the choice is
  deterministic (pandas breaks them by internal order).
- A file that is not valid UTF-8 is decoded as latin-1 (as parse.py does) via
  a streamed UTF-8 temp copy, because DuckDB reads UTF-8 only.
- A UTF-8 byte-order mark is dropped from the first column name (pandas' C
  parser does that; the in-memory path keeps it as part of the name).
- Numeric values are those pandas.to_numeric accepts; DuckDB is restricted to
  the same grammar, and a disagreement in the count raises instead of
  returning a quietly different mean.
"""
from __future__ import annotations

import codecs
import csv
import os
import shutil
import tempfile
from pathlib import Path
from typing import Callable

import duckdb
import pandas as pd

from . import profile as profile_mod
from .parse import ParseError, _is_numeric, classify_column

DEFAULT_CHUNK_ROWS = 100_000
HEAD_SAMPLE = 500          # parse._is_numeric looks at the first 500 non-empty values
TOP_VALUES = 5
DELIMITERS = ",;\t|"
_IO_BLOCK = 1 << 20

# fn(percent 0-100, stage name); called after every chunk and every DuckDB scan.
ProgressFn = Callable[[float, str], None]

# Strings DuckDB may treat as numbers: the grammar pandas.to_numeric accepts
# (no "nan", no digit separators, no hex), with overflow like 1e400 rejected
# the way pandas rejects it.
_NUM_RE = r"^\s*[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\s*$"
_INF_RE = r"^\s*[+-]?[iI][nN][fF](?:[iI][nN][iI][tT][yY])?$"
_BOOL_WORDS = "('true','false','yes','no','0','1')"


class ScaleError(ParseError):
    """A file the scale path cannot profile; the message is safe to show a user."""


def _noop(_pct: float, _stage: str) -> None:
    return None


def _is_valid_utf8(path: Path) -> bool:
    decoder = codecs.getincrementaldecoder("utf-8")()
    with open(path, "rb") as fh:
        while block := fh.read(_IO_BLOCK):
            try:
                decoder.decode(block)
            except UnicodeDecodeError:
                return False
    try:
        decoder.decode(b"", final=True)
    except UnicodeDecodeError:
        return False
    return True


def _transcode_latin1_to_utf8(src: Path, dst: Path) -> None:
    with open(src, "rb") as fin, open(dst, "w", encoding="utf-8", newline="") as fout:
        while block := fin.read(_IO_BLOCK):
            fout.write(block.decode("latin-1"))


def _sniff_delimiter(path: Path) -> str:
    with open(path, "r", encoding="utf-8", newline="") as fh:
        header = ""
        for line in fh:
            if line.strip():
                header = line
                break
    try:
        return csv.Sniffer().sniff(header, delimiters=DELIMITERS).delimiter
    except csv.Error:
        return ","


class _Column:
    """Mergeable per-column state for the chunk pass."""

    __slots__ = ("nulls", "head", "numeric_candidate", "num_count", "num_min", "num_max")

    def __init__(self) -> None:
        self.nulls = 0
        self.head: list[str] = []
        # None = undecided (fewer than HEAD_SAMPLE non-empty values seen so far).
        self.numeric_candidate: bool | None = None
        self.num_count = 0
        self.num_min = float("inf")
        self.num_max = float("-inf")

    def update(self, values: pd.Series) -> None:
        empty = values.eq("")
        self.nulls += int(empty.sum())
        if len(self.head) >= HEAD_SAMPLE and self.numeric_candidate is False:
            return   # nothing else to learn from a settled non-numeric column
        non_empty = values[~empty]
        if len(self.head) < HEAD_SAMPLE:
            self.head.extend(non_empty.head(HEAD_SAMPLE - len(self.head)).tolist())
            if len(self.head) == HEAD_SAMPLE:
                self.numeric_candidate = _is_numeric(pd.Series(self.head, dtype=object))
        if self.numeric_candidate is False or non_empty.empty:
            return
        # Same parser as profile.py, so min/max see exactly the values it sees.
        nums = pd.to_numeric(non_empty, errors="coerce").dropna()
        if len(nums):
            self.num_count += len(nums)
            self.num_min = min(self.num_min, float(nums.min()))
            self.num_max = max(self.num_max, float(nums.max()))

    def finish(self) -> None:
        if self.numeric_candidate is None:
            self.numeric_candidate = _is_numeric(pd.Series(self.head, dtype=object))


def _chunk_pass(path: Path, delimiter: str, chunk_rows: int,
                progress: ProgressFn) -> tuple[list[str], int, int, list[_Column]]:
    total = max(os.path.getsize(path), 1)
    names: list[str] = []
    columns: list[_Column] = []
    rows = chunks = 0
    try:
        with open(path, "rb") as fh:
            reader = pd.read_csv(fh, sep=delimiter, dtype=str, keep_default_na=False,
                                 na_filter=False, index_col=False, engine="c",
                                 chunksize=chunk_rows, encoding="utf-8")
            for chunk in reader:
                if not columns:
                    names = [str(c).strip() for c in chunk.columns]
                    columns = [_Column() for _ in names]
                for i, name in enumerate(chunk.columns):
                    columns[i].update(chunk[name])
                rows += len(chunk)
                chunks += 1
                progress(min(5.0 + 55.0 * fh.tell() / total, 60.0), "scanning chunks")
    except pd.errors.EmptyDataError as exc:
        raise ScaleError("empty_file: the CSV has no rows or no columns.") from exc
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ScaleError(f"not_a_csv: this file could not be parsed as CSV ({exc}).") from exc
    if rows == 0 or not columns:
        raise ScaleError("empty_file: the CSV has no rows or no columns.")
    for col in columns:
        col.finish()
    return names, rows, chunks, columns


def _numeric_sql(col: str) -> str:
    """SQL for 'the number pandas.to_numeric would read from this text, else NULL'."""
    finite = f"CASE WHEN regexp_full_match({col}, '{_NUM_RE}') THEN TRY_CAST({col} AS DOUBLE) END"
    return (f"CASE WHEN isfinite({finite}) THEN {finite} "
            f"WHEN regexp_full_match({col}, '{_INF_RE}') THEN "
            f"CASE WHEN ltrim({col}) LIKE '-%' THEN '-inf'::DOUBLE ELSE 'inf'::DOUBLE END END")


def _open_duckdb(path: Path, n_cols: int, delimiter: str, memory_limit: str,
                 threads: int, spill_dir: Path, parallel: bool = True) -> duckdb.DuckDBPyConnection:
    """In-process DuckDB with a view `t` over the CSV; the file is only read.

    Columns are renamed c0..cN (names come from the pandas header, so they can
    hold any characters) and NULLs become '' to match the pandas path, where
    an empty cell is the empty string.
    """
    con = duckdb.connect(":memory:", config={
        "memory_limit": memory_limit,
        "threads": threads,
        "temp_directory": str(spill_dir),
        "preserve_insertion_order": False,   # every aggregate here is order-free
        "autoinstall_known_extensions": False,
        "autoload_known_extensions": False,
    })
    con.execute("SET enable_progress_bar = false")   # else it prints a bar to the worker's stderr
    names = [f"c{i}" for i in range(n_cols)]
    try:
        con.read_csv(str(path), header=True, all_varchar=True, names=names,
                     delimiter=delimiter, quotechar='"', escapechar='"',
                     null_padding=True, parallel=parallel).create_view("raw")
    except duckdb.Error as exc:
        con.close()
        raise ScaleError(f"not_a_csv: DuckDB could not read this file ({exc}).") from exc
    select = ", ".join(f"COALESCE({n}, '') AS {n}" for n in names)
    # pandas skips blank lines; DuckDB keeps them as a single empty value
    # when there is only one column, so drop them there to keep row counts equal.
    where = " WHERE c0 <> ''" if n_cols == 1 else ""
    con.execute(f"CREATE VIEW t AS SELECT {select} FROM raw{where}")
    return con


def _aggregate_scan(con: duckdb.DuckDBPyConnection, n_cols: int,
                    numeric_cols: list[int]) -> dict:
    """Exact distinct counts, boolean-check inputs and numeric aggregates.

    Distinct counts run one column per query (a scan costs ~0.4 s per million
    rows): ten COUNT(DISTINCT) hash sets alive at once is what runs out of
    memory first, one at a time is bounded by the widest single column.
    """
    parts = ["count(*)"]
    for i in range(n_cols):
        c = f"c{i}"
        parts += [
            # Cheap boolean check: all non-empty values are boolean words, and
            # how many distinct ones there are (the hash only ever holds <= 6).
            f"bool_and({c} = '' OR lower({c}) IN {_BOOL_WORDS})",
            f"count(DISTINCT lower({c})) FILTER (WHERE {c} <> '' AND lower({c}) IN {_BOOL_WORDS})",
            f"min(lower({c})) FILTER (WHERE {c} <> '')",
            f"max(lower({c})) FILTER (WHERE {c} <> '')",
        ]
    source = "t"
    if numeric_cols:
        extra = ", ".join(f"{_numeric_sql(f'c{i}')} AS n{i}" for i in numeric_cols)
        source = f"(SELECT *, {extra} FROM t)"
        for i in numeric_cols:
            parts += [f"count(n{i})", f"sum(n{i})", f"median(n{i})",
                      f"count(*) FILTER (WHERE n{i} < 0)"]
    row = con.execute(f"SELECT {', '.join(parts)} FROM {source}").fetchone()
    out: dict = {"rows": row[0], "cols": [], "numeric": {}}
    pos = 1
    for i in range(n_cols):
        distinct = con.execute(f"SELECT count(DISTINCT NULLIF(c{i}, '')) FROM t").fetchone()[0]
        out["cols"].append({"distinct": distinct, "all_bool": bool(row[pos]),
                            "bool_distinct": row[pos + 1], "lo": row[pos + 2], "hi": row[pos + 3]})
        pos += 4
    for i in numeric_cols:
        out["numeric"][i] = {"count": row[pos], "sum": row[pos + 1],
                             "median": row[pos + 2], "negatives": row[pos + 3]}
        pos += 4
    return out


def _duplicate_rows(con: duckdb.DuckDBPyConnection, rows: int) -> int:
    distinct = con.execute("SELECT count(*) FROM (SELECT DISTINCT * FROM t)").fetchone()[0]
    return rows - distinct


def _top_values(con: duckdb.DuckDBPyConnection, cols: list[int]) -> dict[int, dict[str, int]]:
    """Exact top-5 value counts for several columns from a single scan."""
    if not cols:
        return {}
    sets = ", ".join(f"(c{i})" for i in cols)
    sel = ", ".join(f"GROUPING(c{i}), c{i}" for i in cols)
    counts: dict[int, list[tuple[str, int]]] = {i: [] for i in cols}
    for row in con.execute(f"SELECT {sel}, count(*) FROM t GROUP BY GROUPING SETS ({sets})").fetchall():
        for k, i in enumerate(cols):
            if row[2 * k] == 0 and row[2 * k + 1] != "":
                counts[i].append((row[2 * k + 1], row[-1]))
    return {i: {v: n for v, n in sorted(pairs, key=lambda p: (-p[1], p[0]))[:TOP_VALUES]}
            for i, pairs in counts.items()}


def profile_file(path: str | os.PathLike, *, chunk_rows: int = DEFAULT_CHUNK_ROWS,
                 progress: ProgressFn | None = None, memory_limit: str = "1GB",
                 threads: int | None = None) -> dict:
    """Profile a CSV on disk in bounded memory.

    Inputs: a file path; chunk_rows (rows per pandas chunk); an optional
    progress callback fn(percent, stage); DuckDB memory_limit and threads.
    Output (JSON-serialisable, identical on re-runs):
      {"profile": <same shape as profile_dataframe>, "health_score": int,
       "score_reasons": [str], "scale": {chunk_rows, chunks, encoding, delimiter}}
    Raises ScaleError (a ParseError, so the API answers 422) for empty or
    unparseable files. See the module docstring for what is exact and where
    this path deliberately differs from the in-memory one.
    """
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be >= 1")
    report = progress or _noop
    src = Path(path)
    if not src.is_file():
        raise ScaleError(f"not_a_csv: no such file {src.name!r}.")
    threads = threads or min(4, os.cpu_count() or 1)

    work = Path(tempfile.mkdtemp(prefix="csvdoctor-scale-"))
    con = None
    try:
        report(1.0, "inspecting file")
        encoding = "utf-8"
        if not _is_valid_utf8(src):
            encoding = "latin-1"
            utf8_copy = work / "utf8.csv"
            _transcode_latin1_to_utf8(src, utf8_copy)
            src = utf8_copy
        delimiter = _sniff_delimiter(src)

        names, rows, chunks, cols = _chunk_pass(src, delimiter, chunk_rows, report)
        n_cols = len(names)

        # Undecided/candidate numeric columns only: the rest can never be numeric.
        numeric_cols = [i for i, c in enumerate(cols) if c.numeric_candidate]
        report(62.0, "exact aggregates (DuckDB)")
        con = _open_duckdb(src, n_cols, delimiter, memory_limit, threads, work)
        try:
            agg = _aggregate_scan(con, n_cols, numeric_cols)
        except duckdb.Error as exc:
            if "parallel" not in str(exc).lower():
                raise
            # DuckDB's parallel reader cannot pad short rows when a quoted field
            # contains a newline; the serial reader can (slower, same results).
            con.close()
            con = _open_duckdb(src, n_cols, delimiter, memory_limit, threads, work, parallel=False)
            agg = _aggregate_scan(con, n_cols, numeric_cols)
        if agg["rows"] != rows:
            raise ScaleError(f"internal_mismatch: chunk pass saw {rows} rows, DuckDB saw "
                             f"{agg['rows']}; the file's quoting is ambiguous.")
        types: list[str] = []
        for i, name in enumerate(names):
            a = agg["cols"][i]
            lowered = {a["lo"], a["hi"]} if a["all_bool"] and a["bool_distinct"] <= 2 else None
            types.append(classify_column(name, rows - cols[i].nulls, a["distinct"], lowered,
                                         pd.Series(cols[i].head, dtype=object)))

        for i, t in enumerate(types):
            if t == "numeric" and agg["numeric"][i]["count"] != cols[i].num_count:
                raise ScaleError(f"internal_mismatch: column {names[i]!r} parses as a different "
                                 "number of numeric values in pandas and DuckDB.")

        report(80.0, "duplicate rows (DuckDB)")
        duplicate_rows = _duplicate_rows(con, rows)
        report(90.0, "top values (DuckDB)")
        top = _top_values(con, [i for i, t in enumerate(types) if t in ("categorical", "boolean")])

        report(96.0, "health score")
        columns = []
        for i, name in enumerate(names):
            entry = {
                "name": name,
                "type": types[i],
                "missing_pct": round(100.0 * cols[i].nulls / max(rows, 1), 2),
                "unique": int(agg["cols"][i]["distinct"]),
            }
            if types[i] == "numeric" and cols[i].num_count:
                n = agg["numeric"][i]
                entry.update({
                    "min": cols[i].num_min, "max": cols[i].num_max,
                    "mean": round(float(n["sum"]) / n["count"], 4),
                    "median": float(n["median"]),
                    "negatives": int(n["negatives"]),
                })
            if types[i] in ("categorical", "boolean"):
                entry["top_values"] = top[i]
            columns.append(entry)

        missing_cells = sum(c.nulls for c in cols)
        profile = {
            "rows": rows,
            "cols": n_cols,
            "missing_pct": round(100.0 * missing_cells / max(rows * n_cols, 1), 2),
            "duplicate_rows": int(duplicate_rows),
            "duplicate_pct": round(100.0 * duplicate_rows / rows, 2),
            "columns": columns,
        }
        score, reasons = profile_mod.health_score(profile)
        report(100.0, "complete")
        return {
            "profile": profile,
            "health_score": score,
            "score_reasons": reasons,
            "scale": {"chunk_rows": chunk_rows, "chunks": chunks,
                      "encoding": encoding, "delimiter": delimiter},
        }
    except duckdb.Error as exc:
        raise ScaleError(f"not_a_csv: DuckDB could not aggregate this file ({exc}).") from exc
    finally:
        if con is not None:
            con.close()
        shutil.rmtree(work, ignore_errors=True)
