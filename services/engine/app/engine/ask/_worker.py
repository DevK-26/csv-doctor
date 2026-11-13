"""Sandbox worker: runs exactly ONE generated program, then exits.

Executed by sandbox.py as `python -s -P -B _worker.py` with an explicit
minimal environment and an empty temporary working directory. It must stay a
standalone script (no imports from `app`) so nothing of the parent is
reachable from it. Protocol: a pickle on stdin (produced by the trusted
parent: program, typed DataFrame, limits, allowed modules); one JSON line on
stdout.

Enforcement order matters: everything the program legitimately needs is
imported and warmed up FIRST, then resource limits are set, then the audit
hook is armed, and only then is untrusted code executed.
"""
import builtins
import importlib
import json
import math
import os
import pickle
import resource
import sys
import sysconfig

MAX_TABLE_ROWS = 200
MAX_TEXT = 2000
MAX_CAPTURE = 20_000

SAFE_BUILTIN_NAMES = (
    "abs all any bool dict divmod enumerate filter float frozenset int isinstance "
    "iter len list map max min next pow print range repr reversed round set slice "
    "sorted str sum tuple zip True False None "
    "Exception ArithmeticError ValueError TypeError KeyError IndexError "
    "ZeroDivisionError StopIteration OverflowError AttributeError NotImplementedError"
).split()

# Audit events that are always refused. Network, native code, process creation,
# filesystem mutation and anything that could undo these restrictions.
DENY_PREFIXES = (
    "socket.", "ctypes.", "subprocess.", "shutil.", "glob.", "pty.", "urllib.",
    "http.", "ftplib.", "smtplib.", "poplib.", "imaplib.", "nntplib.", "telnetlib.",
    "ssl.", "resource.", "mmap.", "sqlite3.", "multiprocessing.", "signal.", "gc.",
    "webbrowser.", "pdb.", "code.", "cpython.", "tempfile.", "syslog.", "fcntl.",
)
DENY_EVENTS = {
    "pickle.find_class", "sys.addaudithook", "sys.settrace", "sys.setprofile",
    "builtins.input", "builtins.breakpoint",
}
# os.* is denied wholesale except directory reads that are path-gated below.
OS_GATED = {"os.listdir", "os.scandir", "os.walk"}
OS_HARMLESS = {"os.readlink"}
DENY_IMPORTS = {
    "socket", "_socket", "ssl", "_ssl", "subprocess", "_posixsubprocess", "ctypes",
    "_ctypes", "multiprocessing", "sqlite3", "_sqlite3", "urllib", "http", "ftplib",
    "smtplib", "poplib", "imaplib", "telnetlib", "pty", "shutil", "glob", "tempfile",
    "webbrowser", "asyncio", "select", "selectors", "mmap", "pdb", "code", "gc",
}
WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND


def _read_roots():
    roots = set()
    for key in ("stdlib", "platstdlib", "purelib", "platlib"):
        p = sysconfig.get_paths().get(key)
        if p:
            roots.add(os.path.normpath(p))
    roots.add("/usr/share/zoneinfo")
    return tuple(roots)


def _under(path, roots):
    return any(path == r or path.startswith(r + os.sep) for r in roots)


def install_audit_hook(violations, start_cwd):
    roots = _read_roots()

    def deny(event, detail=""):
        violations.append(f"{event} {detail}".strip()[:200])
        raise PermissionError(f"sandbox: {event} is not allowed")

    def readable_path(raw):
        if not isinstance(raw, (str, bytes, os.PathLike)):
            return None
        p = os.fsdecode(raw)
        if not os.path.isabs(p):
            p = os.path.join(start_cwd, p)
        return os.path.normpath(p)

    def hook(event, args):
        if event == "open":
            path, mode, flags = args[0], args[1], args[2]
            writes = (isinstance(mode, str) and any(c in mode for c in "wax+")) or \
                     (isinstance(flags, int) and flags & WRITE_FLAGS)
            p = readable_path(path)
            # Library source is public and pandas/numpy lazily read it; nothing else is readable.
            if writes or p is None or not _under(p, roots):
                deny("open", str(path))
            return
        if event in OS_GATED:
            p = readable_path(args[0]) if args and args[0] is not None else None
            if p is None or not _under(p, roots):
                deny(event, str(args[0] if args else ""))
            return
        if event == "import":
            top = str(args[0]).split(".")[0]
            if top in DENY_IMPORTS:
                deny("import", top)
            return
        if event in OS_HARMLESS:
            return
        if event.startswith("os.") or event in DENY_EVENTS or event.startswith(DENY_PREFIXES):
            deny(event)

    sys.addaudithook(hook)


def _vm_bytes():
    with open("/proc/self/statm") as fh:
        return int(fh.read().split()[0]) * resource.getpagesize()


def set_limits(cpu_seconds, memory_mb):
    ru = resource.getrusage(resource.RUSAGE_SELF)
    soft = int(math.ceil(ru.ru_utime + ru.ru_stime)) + max(1, int(math.ceil(cpu_seconds)))
    resource.setrlimit(resource.RLIMIT_CPU, (soft, soft + 1))
    cap = _vm_bytes() + memory_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    for res, val in ((resource.RLIMIT_FSIZE, 0), (resource.RLIMIT_CORE, 0),
                     (resource.RLIMIT_NOFILE, 32)):
        resource.setrlimit(res, (val, val))


class _Capped:
    """Swallows print() output (bounded) so it can never reach the result channel."""
    def __init__(self):
        self.size = 0

    def write(self, text):
        self.size += len(text)
        if self.size > MAX_CAPTURE:
            raise OverflowError("program printed too much output")
        return len(text)

    def flush(self):
        pass


def _scalar(v, pd, np):
    if v is None or v is pd.NaT:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        return f if math.isfinite(f) else {"special": "nan" if math.isnan(f) else
                                           ("inf" if f > 0 else "-inf")}
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    if isinstance(v, str):
        return v[:MAX_TEXT]
    raise TypeError(f"unsupported value type {type(v).__name__}")


def serialise(obj, pd, np):
    if obj is None or isinstance(obj, (bool, int, float, str, np.generic, pd.Timestamp)) \
            or obj is pd.NaT:
        v = _scalar(obj, pd, np)
        if isinstance(v, str):
            return {"kind": "text", "value": v}
        return {"kind": "scalar", "value": v}
    if isinstance(obj, pd.Series):
        part = obj.head(MAX_TABLE_ROWS)
        return {"kind": "series", "truncated": len(obj) > MAX_TABLE_ROWS,
                "index": [_scalar(i, pd, np) if not isinstance(i, tuple) else str(i)
                          for i in part.index],
                "values": [_scalar(v, pd, np) for v in part.tolist()]}
    if isinstance(obj, pd.DataFrame):
        part = obj.head(MAX_TABLE_ROWS)
        return {"kind": "frame", "truncated": len(obj) > MAX_TABLE_ROWS,
                "index": [_scalar(i, pd, np) if not isinstance(i, tuple) else str(i)
                          for i in part.index],
                "columns": [str(c) for c in part.columns],
                "rows": [[_scalar(v, pd, np) for v in row] for row in part.values.tolist()]}
    if isinstance(obj, dict):
        if len(obj) > 100:
            raise TypeError("dict result too large")
        return {"kind": "dict", "items": {str(k): serialise(v, pd, np) for k, v in obj.items()}}
    if isinstance(obj, (list, tuple, np.ndarray)):
        vals = list(obj)[:MAX_TABLE_ROWS]
        return {"kind": "list", "values": [serialise(v, pd, np) for v in vals]}
    raise TypeError(f"unsupported result type {type(obj).__name__}")


def _error_text(exc, program):
    line = None
    tb = exc.__traceback__
    while tb is not None:
        if tb.tb_frame.f_code.co_filename == "<program>":
            line = tb.tb_lineno
        tb = tb.tb_next
    text = f"{type(exc).__name__}: {exc}"
    if line is not None:
        lines = program.splitlines()
        src = lines[line - 1].strip() if 0 < line <= len(lines) else ""
        text += f" (line {line}: {src})"
    return text[:500]


def _warm_up(frame, pd):
    """Trigger pandas' lazy imports while file access is still unrestricted."""
    try:
        small = frame.head(20).copy()
        for c in small.columns:
            s = small[c]
            s.dropna().nunique()
            s.value_counts()
            if str(s.dtype).startswith("float"):
                s.mean(); s.median(); s.sum(); s.min(); s.max()
            elif str(s.dtype).startswith("datetime"):
                s.dt.month; s.min(); (s >= pd.Timestamp("2000-01-01")).sum()
            else:
                (s == "x").sum(); s.dropna().astype(str).str.lower()
        first = small.columns[0]
        small.groupby(first).size().sort_values(ascending=False, kind="mergesort")
        small.sort_values(first)
    except Exception:
        pass


def main():
    payload = pickle.load(sys.stdin.buffer)
    real_stdout = sys.stdout
    start_cwd = os.getcwd()
    allowed = list(payload["allowed_modules"])
    for name in allowed:
        importlib.import_module(name)
    pd, np = sys.modules["pandas"], sys.modules["numpy"]
    frame, program = payload["frame"], payload["program"]
    _warm_up(frame, pd)

    violations = []
    out = {"ok": False, "violations": violations}
    set_limits(payload["cpu_seconds"], payload["memory_mb"])
    install_audit_hook(violations, start_cwd)

    safe = {n: getattr(builtins, n) for n in SAFE_BUILTIN_NAMES if hasattr(builtins, n)}

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level != 0 or name not in allowed:
            violations.append(f"import {name}")
            raise ImportError(f"import of {name!r} is not allowed in the sandbox; "
                              f"allowed modules: {sorted(allowed)}")
        return sys.modules[name]

    safe["__import__"] = guarded_import
    namespace = {"__builtins__": safe, "__name__": "program", "pd": pd, "np": np, "df": frame}

    sys.stdout = sys.stderr = _Capped()
    try:
        code = compile(program, "<program>", "exec")
        exec(code, namespace)
        if "result" not in namespace:
            out["error"] = "the program did not assign a variable named `result`"
        else:
            out["result"] = serialise(namespace["result"], pd, np)
            out["ok"] = True
    except BaseException as exc:  # noqa: BLE001 - every failure is reported, none escapes
        out["error"] = _error_text(exc, program)
    finally:
        sys.stdout = real_stdout
    if violations:
        out["ok"] = False
        out.setdefault("error", "sandbox policy violation: " + "; ".join(violations[:3]))
    try:
        line = json.dumps(out, allow_nan=False)
    except (TypeError, ValueError) as exc:
        line = json.dumps({"ok": False, "violations": violations,
                           "error": f"result could not be serialised: {exc}"})
    real_stdout.write(line + "\n")
    real_stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
