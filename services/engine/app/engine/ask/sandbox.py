"""Sandbox for generated pandas programs (parent side).

Three independent layers, because in-process Python sandboxes are easy to get
wrong and no single layer should be trusted alone:

1. Static policy (check_program): the program is parsed and rejected before
   anything runs if it imports outside the allow-list, touches dunder/private
   attributes, uses file/network/introspection builtins, or calls pandas/numpy
   I/O entry points.
2. Process isolation (run_program): the program runs in a fresh child
   interpreter with an empty environment, an empty temporary working
   directory, rlimits for CPU time / address space / file size / open files,
   a wall-clock kill of the whole process group, and a bounded result channel.
3. Runtime policy inside the child (_worker.py): restricted builtins with a
   guarded __import__, plus a CPython audit hook that refuses socket, ctypes,
   subprocess, os.* mutation, any file open outside the Python library
   directories, and any write at all. The hook cannot be removed once armed.

What this is NOT: a kernel jail. A determined attacker with a CPython/pandas
exploit is out of scope; production should additionally run the engine
container with no network. Its guarantee is stated in agent.py: whatever the
program reports is re-computed independently before it is shown, and the
child holds no secrets (empty environment, no readable files besides the
public Python libraries, only the user's own rows).
"""
from __future__ import annotations

import ast
import json
import os
import pickle
import signal
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

WORKER = Path(__file__).with_name("_worker.py")

# Plain module names only: dotted imports and `from pandas import x` are rejected,
# because they can pull in submodules (pandas.io.*, numpy.ctypeslib) by the back door.
ALLOWED_MODULES = ("pandas", "numpy", "math", "statistics", "datetime", "re",
                   "collections", "itertools", "functools", "decimal", "fractions", "string")
FROM_IMPORT_MODULES = {"datetime", "collections", "itertools", "functools", "math",
                       "statistics", "decimal", "fractions", "string", "re"}

MAX_PROGRAM_CHARS = 8000
MAX_CONCURRENT = 4

DENY_NAMES = {
    "open", "exec", "eval", "compile", "getattr", "setattr", "delattr", "globals",
    "locals", "vars", "dir", "input", "breakpoint", "help", "exit", "quit", "memoryview",
    "type", "object", "super", "classmethod", "staticmethod", "property", "bytes",
    "bytearray", "id", "hasattr", "callable",
    "os", "sys", "subprocess", "socket", "builtins", "importlib", "shutil", "pathlib",
    "tempfile", "urllib", "http", "ctypes", "pickle", "marshal", "sqlite3", "threading",
    "multiprocessing", "signal", "resource", "gc", "inspect", "types", "operator",
}
# Module-walking and I/O attribute names. The audit hook is the real barrier;
# rejecting these early gives the generator a precise error to retry from.
DENY_ATTRS = {
    "io", "core", "compat", "util", "testing", "lib", "ctypeslib", "f2py", "distutils",
    "random", "sample", "eval", "query", "plot", "hist", "boxplot", "style", "format",
    "format_map", "mro", "load", "loadtxt", "genfromtxt", "fromfile", "save", "savez",
    "savez_compressed", "savetxt", "tofile", "memmap", "DataSource", "popen", "system",
    "environ", "getenv", "attrgetter", "methodcaller", "to_csv", "to_excel", "to_json",
    "to_pickle", "to_parquet", "to_feather", "to_hdf", "to_sql", "to_stata", "to_clipboard",
    "to_latex", "to_html", "to_markdown", "to_xml", "to_orc", "to_gbq", "to_xarray",
    "gi_frame", "gi_code", "gi_yieldfrom", "cr_frame", "cr_code", "ag_frame", "ag_code",
    "f_globals", "f_locals", "f_builtins", "f_back", "f_code", "tb_frame", "tb_next",
    "func_globals",
}


class SandboxError(Exception):
    """Base class: `reason` is a stable machine-readable code, str(exc) the human message."""
    reason = "program_error"


class SandboxViolation(SandboxError):
    reason = "sandbox_violation"


class SandboxTimeout(SandboxError):
    reason = "timeout"


class SandboxBusy(SandboxError):
    reason = "busy"


class ProgramError(SandboxError):
    reason = "program_error"


@dataclass(frozen=True)
class Limits:
    cpu_seconds: int = 5
    wall_seconds: float = 20.0  # also covers interpreter + pandas start-up
    memory_mb: int = 1024       # extra address space on top of what the worker has loaded


_slots = threading.BoundedSemaphore(MAX_CONCURRENT)


def check_program(program: str, known_strings: frozenset[str] = frozenset()) -> None:
    """Static policy check. Raises SandboxViolation (or ProgramError for bad syntax).

    `known_strings` (real column names) are exempt from the denied-word string
    rule so a column called "type" or "format" stays queryable; any string
    containing "__" is rejected regardless.
    """
    if not isinstance(program, str) or not program.strip():
        raise ProgramError("the program is empty")
    if len(program) > MAX_PROGRAM_CHARS:
        raise SandboxViolation(f"program is longer than {MAX_PROGRAM_CHARS} characters")
    try:
        tree = ast.parse(program, filename="<program>")
    except SyntaxError as exc:
        raise ProgramError(f"SyntaxError: {exc.msg} (line {exc.lineno})") from exc
    except (RecursionError, MemoryError, ValueError) as exc:
        raise SandboxViolation("program is too deeply nested to analyse") from exc

    def bad(node: ast.AST, why: str) -> SandboxViolation:
        return SandboxViolation(f"{why} (line {getattr(node, 'lineno', '?')})")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in ALLOWED_MODULES:
                    raise bad(node, f"import of {alias.name!r} is not allowed; "
                                    f"allowed modules: {list(ALLOWED_MODULES)}")
        elif isinstance(node, ast.ImportFrom):
            if node.level or node.module not in FROM_IMPORT_MODULES:
                raise bad(node, f"'from {node.module} import ...' is not allowed; "
                                "use `import pandas as pd` / `import numpy as np`")
            if any(a.name == "*" or a.name.startswith("_") for a in node.names):
                raise bad(node, "star/private imports are not allowed")
        elif isinstance(node, (ast.ClassDef, ast.AsyncFunctionDef, ast.AsyncFor,
                               ast.AsyncWith, ast.Await)):
            raise bad(node, f"{type(node).__name__} is not allowed")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr in DENY_ATTRS or node.attr.startswith("read_"):
                raise bad(node, f"access to attribute {node.attr!r} is not allowed")
        elif isinstance(node, ast.Name):
            if node.id.startswith("__") or node.id in DENY_NAMES:
                raise bad(node, f"use of name {node.id!r} is not allowed")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            v = node.value.strip()
            if "__" in v or ((v in DENY_ATTRS or v in DENY_NAMES) and v not in known_strings):
                raise bad(node, f"string {v[:40]!r} is not allowed (could be used for dynamic lookup)")


def _decode(stdout: bytes) -> dict | None:
    for line in reversed(stdout.decode("utf-8", "replace").splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                parsed = json.loads(line)
            except ValueError:
                return None
            return parsed if isinstance(parsed, dict) else None
    return None


def run_program(program: str, frame: pd.DataFrame, limits: Limits = Limits(),
                static_check: bool = True,
                known_strings: frozenset[str] = frozenset()) -> dict:
    """Run `program` against `frame` in the sandbox; return the serialised `result`.

    Raises SandboxViolation / SandboxTimeout / ProgramError / SandboxBusy; the
    message is written to be fed straight back to the generator.
    `static_check=False` exists so tests can prove the runtime layers hold on
    their own; the agent always leaves it on.
    """
    if static_check:
        check_program(program, known_strings)
    payload = pickle.dumps({
        "program": program, "frame": frame, "allowed_modules": list(ALLOWED_MODULES),
        "cpu_seconds": limits.cpu_seconds, "memory_mb": limits.memory_mb,
    }, protocol=4)
    if not _slots.acquire(timeout=limits.wall_seconds):
        raise SandboxBusy("too many analyses are running; try again shortly")
    try:
        with tempfile.TemporaryDirectory(prefix="csvdoctor-sbx-") as workdir:
            env = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
                   "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
                   "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1"}
            proc = subprocess.Popen(
                [sys.executable, "-s", "-P", "-B", str(WORKER)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                cwd=workdir, env=env, close_fds=True, start_new_session=True)
            try:
                stdout, stderr = proc.communicate(payload, timeout=limits.wall_seconds)
            except subprocess.TimeoutExpired as exc:
                _kill_group(proc)
                raise SandboxTimeout(
                    f"the program exceeded the {limits.wall_seconds:g}s time limit") from exc
            finally:
                if proc.poll() is None:  # never signal a reaped (possibly recycled) pid
                    _kill_group(proc)
    finally:
        _slots.release()

    out = _decode(stdout)
    if out is None:
        code = proc.returncode
        if code in (-signal.SIGKILL, -signal.SIGXCPU):
            raise SandboxTimeout(f"the program exceeded the {limits.cpu_seconds}s CPU limit")
        detail = stderr.decode("utf-8", "replace").strip().splitlines()[-1:] or [""]
        raise ProgramError(f"the sandbox worker crashed (exit {code}); "
                           f"likely the memory limit was exceeded. {detail[0][:200]}")
    if out.get("violations"):
        raise SandboxViolation("sandbox policy violation: " + "; ".join(out["violations"][:3]))
    if not out.get("ok"):
        raise ProgramError(out.get("error", "the program failed"))
    return out["result"]


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass
