"""CSV Doctor engine API (FastAPI).

GET /health and POST /analyse: parse + profile + health score + the anomaly
ensemble (Phase A1: per-row results with reasons under "anomalies") + fuzzy
duplicate groups for review (Phase B1, under "relations").
POST /ask (Phase B2): Ask-Your-Data — a question about an uploaded CSV is
answered by a generated pandas program run in a sandbox and verified by an
independent re-computation (see app/engine/ask). The default generator is
deterministic and local; swapping in an LLM is a one-function change
(get_generator). The LLM summary is a separate web-app route (Day 10) —
this service never calls a model unless a generator that does is plugged in.
Phase C1 scale path: POST /jobs queues a chunked + DuckDB profile of a large
CSV (no 50k-row cap), GET /jobs/{id} returns its status/result, and
GET /jobs/{id}/events streams progress as Server-Sent Events (app/jobs.py).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, StreamingResponse
from redis.exceptions import RedisError

from . import jobs

from .engine import anomalies, drift, facts, profile as profile_mod, relations
from .engine.ask import (AskInputError, CodeGenerator, ContextError, LocalRulesGenerator,
                         SandboxBusy, answer_question)
from .engine.parse import ParseError, infer_types, parse_csv_bytes

# Bump on any change to analysis output: reports are cached by file hash + version.
ENGINE_VERSION = "0.3.0-b1"

MAX_JOB_BYTES = 512 * 1024 * 1024   # the scale path streams, so this is a disk limit, not RAM
SSE_POLL_SECONDS = 0.2
SSE_KEEPALIVE_SECONDS = 15.0
SSE_MAX_SECONDS = 3600.0

app = FastAPI(title="CSV Doctor Engine", version=ENGINE_VERSION)


@app.exception_handler(RedisError)
async def redis_unavailable(_request: Request, _exc: RedisError):
    return JSONResponse(status_code=503, content={
        "detail": "The job store is unavailable; try again shortly."})


@app.get("/health")
def health():
    return {"ok": True, "engine_version": ENGINE_VERSION}


@app.post("/analyse")
async def analyse(file: UploadFile = File(...)):
    raw = await file.read()
    try:
        df = parse_csv_bytes(raw)
    except ParseError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    types = infer_types(df)
    prof = profile_mod.profile_dataframe(df)
    score, reasons = profile_mod.health_score(prof)
    anom = anomalies.find_anomalies(df, types)
    dr = drift.find_change_points(df, types)
    rel = relations.find_duplicates(df, types)
    pack = facts.build_facts_pack(prof, score, reasons, anom, dr, rel)

    return {
        "file_hash": hashlib.sha256(raw).hexdigest(),
        "engine_version": ENGINE_VERSION,
        "health_score": score,
        "score_reasons": reasons,
        "profile": prof,
        "anomalies": anom,
        "drift": dr,
        "relations": rel,
        "facts_pack": pack,
    }


def get_generator() -> CodeGenerator:
    """Generator used by /ask. Override (e.g. with LLMCodeGenerator) to upgrade."""
    return LocalRulesGenerator()


@app.post("/ask")
def ask(file: UploadFile = File(...), question: str = Form(...),
        generator: CodeGenerator = Depends(get_generator)):
    """Verified answer to a natural-language question about the uploaded CSV.

    Always 200 with a `status` of verified / unverified / failed for anything
    the agent itself decides; 422 for unreadable files or bad questions; 503
    when the sandbox is saturated.
    """
    raw = file.file.read()
    try:
        df = parse_csv_bytes(raw)
        return answer_question(question, df, infer_types(df), generator)
    except (ParseError, AskInputError, ContextError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except SandboxBusy as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


async def _save_upload(file: UploadFile, dest) -> tuple[int, str]:
    """Stream the upload to `dest` in 1 MiB blocks; returns (bytes, sha256 hex)."""
    digest, size = hashlib.sha256(), 0
    try:
        with open(dest, "wb") as out:
            while block := await file.read(1 << 20):
                size += len(block)
                if size > MAX_JOB_BYTES:
                    raise HTTPException(status_code=413, detail=(
                        f"file_too_large: async jobs accept files up to {MAX_JOB_BYTES >> 20} MB."))
                digest.update(block)
                out.write(block)
        if size == 0:
            raise HTTPException(status_code=422, detail="empty_file: the upload has no content.")
    except BaseException:
        dest.unlink(missing_ok=True)
        raise
    return size, digest.hexdigest()


def _job_or_404(job_id: str) -> dict:
    record = jobs.get_store().get(job_id) if jobs.JOB_ID_RE.match(job_id) else None
    if record is None:
        raise HTTPException(status_code=404, detail="unknown_job: no job with this id.")
    return record


@app.post("/jobs", status_code=202)
async def submit_job(file: UploadFile = File(...)):
    """Queue a chunked profile of the uploaded CSV; returns the job id to follow."""
    job_id = uuid.uuid4().hex
    path = jobs.upload_dir() / f"{job_id}.csv"
    _, file_hash = await _save_upload(file, path)
    try:
        # Threadpool: in eager mode the whole job runs inside this call.
        await run_in_threadpool(jobs.submit_job, job_id, path, file_hash, ENGINE_VERSION)
    except jobs.JobQueueUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    record = jobs.get_store().get(job_id)
    return {"job_id": job_id, "status": record["status"],
            "status_url": f"/jobs/{job_id}", "events_url": f"/jobs/{job_id}/events"}


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    """Status, progress percent + stage, and (when done) the full result or the error."""
    return _job_or_404(job_id)


def _sse(event: dict) -> str:
    name = {"done": "complete", "failed": "failed"}.get(event["status"], "progress")
    return f"id: {event['seq']}\nevent: {name}\ndata: {json.dumps(event)}\n\n"


@app.get("/jobs/{job_id}/events")
async def job_events(job_id: str, request: Request):
    """Server-Sent Events: `progress` events, then one `complete` (with the result
    summary) or `failed` event. Replays from the start, or after Last-Event-ID."""
    _job_or_404(job_id)
    try:
        seq = max(int(request.headers.get("last-event-id", "0")), 0)
    except ValueError:
        seq = 0
    store = jobs.get_store()

    async def stream():
        started = last_sent = time.monotonic()
        nonlocal seq
        while True:
            for event in store.events_after(job_id, seq):
                seq = event["seq"]
                last_sent = time.monotonic()
                yield _sse(event)
                if event["status"] in jobs.TERMINAL:
                    return
            now = time.monotonic()
            if now - started > SSE_MAX_SECONDS or await request.is_disconnected():
                return
            if now - last_sent > SSE_KEEPALIVE_SECONDS:
                last_sent = now
                yield ": keep-alive\n\n"
            await asyncio.sleep(SSE_POLL_SECONDS)

    return StreamingResponse(stream(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
