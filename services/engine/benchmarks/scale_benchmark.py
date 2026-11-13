"""Phase C1 — benchmark for the scale path (BUILD-SPRINT C1, BLUEPRINT §12 T5).

What it does: generates a deterministic (fixed-seed) CSV of ~1M rows into a
temporary directory, profiles it through `app.engine.scale.profile_file`
(chunked pandas + DuckDB), and records wall time and peak memory in
`benchmarks/results/scale_benchmark.md`. The generated file is deleted when
the run ends; nothing is committed (CLAUDE.md rule 9).

How the numbers are taken:
- The profile runs in a FRESH child process, so peak memory is not inflated
  by the generator. Peak memory is the child's `resource.getrusage` maxrss
  (resident set high-water mark, which includes DuckDB's native allocations).
  The same child also reports its maxrss right after imports, so the table
  shows both the interpreter + libraries baseline and the peak while profiling.
- Wall time is `time.perf_counter()` around `profile_file` only (imports and
  generation excluded).
- A second, smaller run (--scaling-rows) goes through the same code; if peak
  memory is "constant", it barely moves between the two sizes.
- Correctness checks use planted truth, so a fast wrong answer cannot pass:
  the row count equals the rows written, and `duplicate_rows` equals the
  number of exact duplicate rows appended by the generator.

Run it from services/engine/:

    python -m benchmarks.scale_benchmark                  # ~1M rows, seed 42
    python -m benchmarks.scale_benchmark --rows 200000    # quicker local check

Determinism: the generated data and the profile are identical on every run
(one seed); wall time, memory and date are measurements and will vary.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ENGINE_ROOT = Path(__file__).resolve().parents[1]
if str(ENGINE_ROOT) not in sys.path:  # lets `python benchmarks/scale_benchmark.py` find `app`
    sys.path.insert(0, str(ENGINE_ROOT))

DEFAULT_SEED = 42
DEFAULT_ROWS = 1_000_000
DEFAULT_SCALING_ROWS = 250_000
DEFAULT_CHUNK_ROWS = 100_000
DUPLICATE_FRACTION = 0.01      # exact duplicate rows appended at the end
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "scale_benchmark.md"
_BLOCK = 100_000

REGIONS = np.array(["north", "south", "east", "west", "central"])
STATUSES = np.array(["paid", "pending", "refunded", "cancelled"])


def _block(rng: np.random.Generator, start: int, size: int) -> pd.DataFrame:
    """`size` rows with unique order ids starting at `start`, with planted blanks."""
    df = pd.DataFrame({
        "order_id": [f"ORD{i:09d}" for i in range(start, start + size)],
        "customer": [f"cust_{v}" for v in rng.integers(0, 200_000, size)],
        "region": rng.choice(REGIONS, size),
        "status": rng.choice(STATUSES, size, p=[.7, .15, .1, .05]),
        "amount": np.round(rng.gamma(2.0, 40.0, size), 2).astype(str),
        "quantity": rng.integers(-2, 60, size).astype(str),
        "discount": np.round(rng.uniform(0, 0.3, size), 3).astype(str),
        "is_member": rng.choice(["true", "false"], size),
        "placed_on": (pd.Timestamp("2023-01-01")
                      + pd.to_timedelta(rng.integers(0, 730, size), unit="D")).strftime("%Y-%m-%d"),
        "note": [f"note {v}" for v in rng.integers(0, 50_000, size)],
    })
    df.loc[rng.random(size) < 0.10, "discount"] = ""
    df.loc[rng.random(size) < 0.05, "note"] = ""
    return df


def generate_csv(path: Path, rows: int, seed: int = DEFAULT_SEED) -> dict:
    """Write `rows` rows (incl. planted exact duplicates) block by block; returns the truth.

    Only one block is in memory at a time. The duplicates are copies of the
    first rows, appended last, so duplicate_rows is known exactly.
    """
    dups = int(rows * DUPLICATE_FRACTION)
    unique = rows - dups
    rng = np.random.default_rng(seed)
    head = None
    with open(path, "w", encoding="utf-8", newline="\n") as out:
        for start in range(0, unique, _BLOCK):
            df = _block(rng, start, min(_BLOCK, unique - start))
            if head is None:
                head = df.head(dups).copy()
            df.to_csv(out, index=False, header=start == 0, lineterminator="\n")
        head.to_csv(out, index=False, header=False, lineterminator="\n")
    return {"rows": rows, "duplicate_rows": dups, "cols": 10}


def _maxrss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024   # bytes vs kB


def _child(path: str, chunk_rows: int) -> None:
    """Runs in a fresh process: profile the file, print the measurements as JSON."""
    from app.engine import scale
    baseline = _maxrss_mb()
    start = time.perf_counter()
    result = scale.profile_file(path, chunk_rows=chunk_rows)
    wall = time.perf_counter() - start
    prof = result["profile"]
    print(json.dumps({
        "wall_seconds": wall, "peak_mb": _maxrss_mb(), "baseline_mb": baseline,
        "rows": prof["rows"], "cols": prof["cols"], "duplicate_rows": prof["duplicate_rows"],
        "health_score": result["health_score"], "chunks": result["scale"]["chunks"],
        "types": {c["name"]: c["type"] for c in prof["columns"]},
    }))


def measure(path: Path, chunk_rows: int) -> dict:
    proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--child", str(path),
                           str(chunk_rows)], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"profiling child failed:\n{proc.stderr[-2000:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def run_once(rows: int, chunk_rows: int, seed: int, workdir: Path) -> dict:
    path = workdir / f"bench_{rows}.csv"
    truth = generate_csv(path, rows, seed)
    size_mb = path.stat().st_size / (1024 * 1024)
    out = measure(path, chunk_rows)
    path.unlink()
    out["file_mb"] = size_mb
    out["checks"] = {
        "row count equals rows written": out["rows"] == truth["rows"],
        "duplicate_rows equals planted duplicates": out["duplicate_rows"] == truth["duplicate_rows"],
        "column count": out["cols"] == truth["cols"],
    }
    return out


def _total_ram_gb() -> str:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                return f"{int(line.split()[1]) / 1024 / 1024:.1f} GB RAM"
    except OSError:
        pass
    return "RAM unknown"


def environment_note() -> str:
    import duckdb
    return (f"{platform.platform()}, Python {platform.python_version()}, "
            f"{os.cpu_count()} logical CPUs, {_total_ram_gb()}; "
            f"pandas {pd.__version__}, numpy {np.__version__}, duckdb {duckdb.__version__}")


def render_markdown(headline: dict, scaling: dict | None, *, rows: int, chunk_rows: int,
                    seed: int, date: str) -> str:
    def row(label: str, r: dict) -> str:
        return (f"| {label} | {r['rows']:,} | {r['cols']} | {r['file_mb']:.1f} | {r['chunks']} | "
                f"{r['wall_seconds']:.2f} | {r['rows'] / r['wall_seconds']:,.0f} | "
                f"{r['peak_mb']:.0f} | {r['baseline_mb']:.0f} |")

    lines = [
        "# Scale benchmark (chunked profiling + DuckDB)",
        "",
        "<!-- Generated by benchmarks/scale_benchmark.py - do not edit by hand; re-run the script. -->",
        "",
        "Every number on this page was measured by "
        "`services/engine/benchmarks/scale_benchmark.py` in the run described below.",
        "Nothing here is typed in by hand. Wall time, memory and date are measurements "
        "and change from run to run; the generated data and the profile do not.",
        "",
        "## Headline",
        "",
        f"- **Rows:** {headline['rows']:,} x {headline['cols']} columns "
        f"({headline['file_mb']:.1f} MB of CSV)",
        f"- **Wall time:** {headline['wall_seconds']:.2f} s "
        f"({headline['rows'] / headline['wall_seconds']:,.0f} rows/s)",
        f"- **Peak memory:** {headline['peak_mb']:.0f} MB resident (maxrss), of which "
        f"{headline['baseline_mb']:.0f} MB is the interpreter + libraries before profiling starts",
        f"- **Chunk size:** {chunk_rows:,} rows ({headline['chunks']} chunk{'s' if headline['chunks'] != 1 else ''})",
        f"- **Health Score of the generated file:** {headline['health_score']}",
        "",
        "## Runs",
        "",
        "| Run | Rows | Columns | CSV size (MB) | Chunks | Wall time (s) | Rows/s | "
        "Peak memory (MB) | Baseline before profiling (MB) |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|--:|",
        row("headline", headline),
    ]
    if scaling:
        lines.append(row("smaller file, same code", scaling))
    lines += [
        "",
        "Peak memory is the profiling process's `resource.getrusage` maxrss high-water mark, "
        "measured in a fresh child process so the data generator cannot inflate it; it "
        "includes DuckDB's native memory. Wall time covers `profile_file` only (no imports, "
        "no generation).",
    ]
    if scaling:
        growth = headline["peak_mb"] - scaling["peak_mb"]
        lines += [
            "",
            f"Going from {scaling['rows']:,} to {headline['rows']:,} rows "
            f"({headline['rows'] / scaling['rows']:.1f}x the data) changed peak memory by "
            f"{growth:+.0f} MB ({headline['peak_mb'] / scaling['peak_mb']:.2f}x). "
            "Memory is therefore bounded but not flat: the pandas pass holds one chunk "
            "whatever the file size, while DuckDB's exact distinct and duplicate-row hash "
            "tables grow with column cardinality (`order_id` is unique in every row). "
            "DuckDB is given a memory limit (1 GB by default) and a spill directory.",
        ]
    lines += [
        "",
        "## Setup",
        "",
        f"- **Date (UTC):** {date}",
        f"- **Platform:** {environment_note()}",
        f"- **Seed:** {seed} (`numpy.random.default_rng`), so the generated CSV is identical on every run.",
        f"- **Generated data:** {DUPLICATE_FRACTION:.0%} of rows are exact duplicates of the first "
        "rows (appended last), ~10% of `discount` and ~5% of `note` are blank, `customer` has "
        "~200k distinct values, `order_id` is unique. Written to a temp directory and deleted "
        "after the run; nothing is committed.",
        f"- **Path under test:** `app.engine.scale.profile_file` with chunk size {chunk_rows:,}: "
        "a pandas `chunksize` pass (row count, nulls, numeric min/max) then DuckDB over the "
        "CSV for exact distinct counts, sums, medians, duplicate rows and top values, "
        "then the same `profile.health_score` the in-memory path uses.",
        "- **Not measured here:** the in-memory path (`profile_dataframe`), which refuses "
        "files over 50,000 rows / 10 MB.",
        "",
        "## Correctness checks (against planted truth)",
        "",
        "| Check | Headline run | Smaller run |",
        "|:--|:--|:--|",
    ]
    for name, ok in headline["checks"].items():
        other = "n/a" if not scaling else ("pass" if scaling["checks"][name] else "FAIL")
        lines.append(f"| {name} | {'pass' if ok else 'FAIL'} | {other} |")
    lines += [
        "",
        "Column types inferred for the headline file: "
        + ", ".join(f"`{k}` {v}" for k, v in headline["types"].items()) + ".",
        "",
    ]
    return "\n".join(lines)


def write_report(text: str, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> int:
    """Returns 0 if the report was written and every check passed, 1 otherwise."""
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "--child":
        _child(argv[1], int(argv[2]))
        return 0

    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.scale_benchmark",
        description="Profile a generated ~1M-row CSV through the chunked scale path.")
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--scaling-rows", type=int, default=DEFAULT_SCALING_ROWS,
                        help="smaller second run to show memory vs size; 0 to skip")
    parser.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    with tempfile.TemporaryDirectory(prefix="csvdoctor-scale-bench-") as tmp:
        scaling = (run_once(args.scaling_rows, args.chunk_rows, args.seed, Path(tmp))
                   if 0 < args.scaling_rows < args.rows else None)
        headline = run_once(args.rows, args.chunk_rows, args.seed, Path(tmp))
    report = render_markdown(headline, scaling, rows=args.rows, chunk_rows=args.chunk_rows,
                             seed=args.seed, date=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    write_report(report, args.output)
    print(report)
    print(f"Report written to {args.output}")
    failed = [n for r in (headline, scaling) if r for n, ok in r["checks"].items() if not ok]
    if failed:
        print(f"Correctness check FAILED: {failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
