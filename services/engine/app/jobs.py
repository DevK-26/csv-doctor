"""Phase C1 — async analysis jobs: Celery task + progress store.

Flow: the API saves an upload to disk and calls submit_job(); a Celery worker
(or, in eager mode, the same process) runs scale.profile_file on it, and every
progress report is appended to a per-job event log in a JobStore. The API reads
that log for GET /jobs/{id} and replays it as Server-Sent Events, so a client
that connects late (or reconnects with Last-Event-ID) still sees every stage.

Modes (env CSV_DOCTOR_JOBS_MODE, default "auto"):
- "redis": Celery over a Redis broker; progress and results live in Redis, so
  the API and any number of workers share them. The API container and the
  worker must see the same upload directory (CSV_DOCTOR_UPLOAD_DIR).
- "eager": no broker. Celery's task_always_eager runs the task inline during
  submit, and progress lives in an in-process dict. This is what the tests use
  and what a dev box without Redis gets.
- "auto": "redis" if the broker answers a PING within half a second, otherwise
  "eager" with a logged warning.

URLs: CELERY_BROKER_URL, else REDIS_URL, else redis://localhost:6379/0; result
backend CELERY_RESULT_BACKEND, else REDIS_URL, else the same default. The task
keeps its own results in the JobStore, so Celery's result backend is unused
(task_ignore_result) but still configured for tooling such as Flower.
"""
from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Protocol

import redis as redis_lib
from celery import Celery

from .engine import scale
from .engine.parse import ParseError

log = logging.getLogger("csv_doctor.jobs")

DEFAULT_REDIS_URL = "redis://localhost:6379/0"
JOB_TTL_SECONDS = 24 * 3600
MAX_MEMORY_JOBS = 256
JOB_ID_RE = re.compile(r"^[0-9a-f]{32}$")
TERMINAL = ("done", "failed")


class JobQueueUnavailable(RuntimeError):
    """The broker could not accept the job; the API answers 503."""


def _redis_url() -> str:
    return os.environ.get("REDIS_URL") or DEFAULT_REDIS_URL


def build_celery() -> Celery:
    """The Celery app, with broker/backend URLs read from the environment."""
    app = Celery(
        "csv_doctor",
        broker=os.environ.get("CELERY_BROKER_URL") or _redis_url(),
        backend=os.environ.get("CELERY_RESULT_BACKEND") or _redis_url(),
    )
    app.conf.update(
        task_serializer="json", result_serializer="json", accept_content=["json"],
        task_ignore_result=True, task_eager_propagates=False,
        broker_connection_timeout=2, broker_connection_retry_on_startup=True,
    )
    return app


celery_app = build_celery()


# --------------------------------------------------------------------- stores

class JobStore(Protocol):
    def create(self, job_id: str) -> None: ...
    def publish(self, job_id: str, status: str, percent: float, stage: str, *,
                result: dict | None = None, error: str | None = None) -> None: ...
    def get(self, job_id: str) -> dict | None: ...
    def events_after(self, job_id: str, seq: int) -> list[dict]: ...


def summarise(result: dict) -> dict:
    """The small payload of the final SSE event; the full result is at GET /jobs/{id}."""
    prof = result["profile"]
    return {
        "health_score": result["health_score"], "score_reasons": result["score_reasons"],
        "rows": prof["rows"], "cols": prof["cols"], "missing_pct": prof["missing_pct"],
        "duplicate_rows": prof["duplicate_rows"], "duplicate_pct": prof["duplicate_pct"],
    }


def _make_event(status: str, percent: float, stage: str, result: dict | None,
                error: str | None) -> dict:
    event = {"status": status, "percent": round(percent, 1), "stage": stage}
    if result is not None:
        event["summary"] = summarise(result)
    if error is not None:
        event["error"] = error
    return event


def _make_record(job_id: str, status: str, percent: float, stage: str,
                 result: dict | None, error: str | None) -> dict:
    return {"job_id": job_id, "status": status, "percent": round(percent, 1), "stage": stage,
            "result": result, "error": error}


class MemoryJobStore:
    """Process-local store: eager mode and tests. Bounded, thread-safe."""

    def __init__(self, max_jobs: int = MAX_MEMORY_JOBS) -> None:
        self._jobs: OrderedDict[str, dict] = OrderedDict()
        self._lock = threading.Lock()
        self._max = max_jobs

    def create(self, job_id: str) -> None:
        with self._lock:
            self._jobs[job_id] = {"record": _make_record(job_id, "queued", 0.0, "queued", None, None),
                                  "events": [_make_event("queued", 0.0, "queued", None, None)]}
            while len(self._jobs) > self._max:
                self._jobs.popitem(last=False)

    def publish(self, job_id: str, status: str, percent: float, stage: str, *,
                result: dict | None = None, error: str | None = None) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job["record"] = _make_record(job_id, status, percent, stage, result, error)
            job["events"].append(_make_event(status, percent, stage, result, error))

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job["record"]) if job else None

    def events_after(self, job_id: str, seq: int) -> list[dict]:
        with self._lock:
            job = self._jobs.get(job_id)
            events = job["events"] if job else []
            return [{"seq": i + 1, **e} for i, e in enumerate(events) if i + 1 > seq]


class RedisJobStore:
    """Shared store for API + workers: a JSON record and an append-only event list per job."""

    def __init__(self, client: redis_lib.Redis) -> None:
        self._r = client

    @staticmethod
    def _keys(job_id: str) -> tuple[str, str]:
        return f"csvdoctor:job:{job_id}", f"csvdoctor:job:{job_id}:events"

    def create(self, job_id: str) -> None:
        rec_key, ev_key = self._keys(job_id)
        pipe = self._r.pipeline()
        pipe.set(rec_key, json.dumps(_make_record(job_id, "queued", 0.0, "queued", None, None)),
                 ex=JOB_TTL_SECONDS)
        pipe.delete(ev_key)
        pipe.rpush(ev_key, json.dumps(_make_event("queued", 0.0, "queued", None, None)))
        pipe.expire(ev_key, JOB_TTL_SECONDS)
        pipe.execute()

    def publish(self, job_id: str, status: str, percent: float, stage: str, *,
                result: dict | None = None, error: str | None = None) -> None:
        rec_key, ev_key = self._keys(job_id)
        pipe = self._r.pipeline()
        pipe.set(rec_key, json.dumps(_make_record(job_id, status, percent, stage, result, error)),
                 ex=JOB_TTL_SECONDS)
        pipe.rpush(ev_key, json.dumps(_make_event(status, percent, stage, result, error)))
        pipe.expire(ev_key, JOB_TTL_SECONDS)
        pipe.execute()

    def get(self, job_id: str) -> dict | None:
        raw = self._r.get(self._keys(job_id)[0])
        return json.loads(raw) if raw else None

    def events_after(self, job_id: str, seq: int) -> list[dict]:
        raw = self._r.lrange(self._keys(job_id)[1], seq, -1)
        return [{"seq": seq + i + 1, **json.loads(e)} for i, e in enumerate(raw)]


# ------------------------------------------------------------ mode / settings

_state: dict = {"mode": None, "store": None}
_state_lock = threading.Lock()


def configure(mode: str | None = None, store: JobStore | None = None) -> str:
    """Pin the job mode ("eager" or "redis") and optionally the store; returns the mode.

    Tests call configure("eager", MemoryJobStore()) for isolation; production
    leaves it alone and lets the first use resolve the mode from the environment.
    """
    with _state_lock:
        mode = mode or _resolve_mode()
        if mode not in ("eager", "redis"):
            raise ValueError(f"unknown jobs mode {mode!r}")
        celery_app.conf.task_always_eager = mode == "eager"
        _state["mode"] = mode
        _state["store"] = store or (
            MemoryJobStore() if mode == "eager"
            else RedisJobStore(redis_lib.Redis.from_url(_redis_url(), decode_responses=True)))
        return mode


def _resolve_mode() -> str:
    requested = os.environ.get("CSV_DOCTOR_JOBS_MODE", "auto").lower()
    if requested in ("eager", "redis"):
        return requested
    if requested != "auto":
        raise ValueError(f"CSV_DOCTOR_JOBS_MODE must be auto, eager or redis, not {requested!r}")
    try:
        redis_lib.Redis.from_url(_redis_url(), socket_connect_timeout=0.5,
                                 socket_timeout=0.5).ping()
        return "redis"
    except redis_lib.RedisError:
        log.warning("Redis unreachable at %s; running jobs eagerly in-process "
                    "(set CSV_DOCTOR_JOBS_MODE=redis to require the broker).", _redis_url())
        return "eager"


def get_store() -> JobStore:
    if _state["store"] is None:
        configure()
    return _state["store"]


def get_mode() -> str:
    get_store()
    return _state["mode"]


def upload_dir() -> Path:
    """Where uploads wait for the worker. Private to the service user."""
    path = Path(os.environ.get("CSV_DOCTOR_UPLOAD_DIR")
                or Path(tempfile.gettempdir()) / "csv-doctor-uploads")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return path


# ------------------------------------------------------------------ the task

@celery_app.task(name="csv_doctor.profile_file")
def profile_file_task(job_id: str, path: str, file_hash: str, engine_version: str,
                      chunk_rows: int = scale.DEFAULT_CHUNK_ROWS) -> None:
    """Run the scale profile on an uploaded file, recording progress; deletes the file after.

    Failures are recorded on the job (status "failed" + a message safe to
    show), never raised, so a bad upload does not look like a crashed worker.
    """
    store = get_store()

    def on_progress(percent: float, stage: str) -> None:
        if percent < 100.0:   # 100% is the terminal "done" event, not a progress tick
            store.publish(job_id, "running", percent, stage)

    try:
        store.publish(job_id, "running", 0.0, "starting")
        result = scale.profile_file(path, chunk_rows=chunk_rows, progress=on_progress)
        result.update({"file_hash": file_hash, "engine_version": engine_version})
        store.publish(job_id, "done", 100.0, "complete", result=result)
    except ParseError as exc:
        store.publish(job_id, "failed", 100.0, "failed", error=str(exc))
    except Exception:
        log.exception("job %s crashed", job_id)
        store.publish(job_id, "failed", 100.0, "failed",
                      error="internal_error: the analysis crashed; the file may be malformed.")
    finally:
        Path(path).unlink(missing_ok=True)


def submit_job(job_id: str, path: Path, file_hash: str, engine_version: str,
               chunk_rows: int = scale.DEFAULT_CHUNK_ROWS) -> None:
    """Register the job and enqueue it (eager mode: runs to completion before returning)."""
    if not JOB_ID_RE.match(job_id):
        raise ValueError("job_id must be 32 lowercase hex characters")
    store = get_store()
    try:
        store.create(job_id)
        profile_file_task.apply_async(
            args=(job_id, str(path), file_hash, engine_version, chunk_rows),
            task_id=job_id, retry=False)
    except Exception as exc:  # kombu/redis connection errors surface as several types
        try:
            store.publish(job_id, "failed", 100.0, "failed", error="queue_unavailable: job not accepted.")
        except Exception:
            pass   # the store is the thing that is down
        Path(path).unlink(missing_ok=True)
        raise JobQueueUnavailable("The job queue is unavailable; try again shortly.") from exc
