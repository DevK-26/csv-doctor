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
"""
from __future__ import annotations

import hashlib

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile

from .engine import anomalies, drift, facts, profile as profile_mod, relations
from .engine.ask import (AskInputError, CodeGenerator, ContextError, LocalRulesGenerator,
                         SandboxBusy, answer_question)
from .engine.parse import ParseError, infer_types, parse_csv_bytes

# Bump on any change to analysis output: reports are cached by file hash + version.
ENGINE_VERSION = "0.3.0-b1"

app = FastAPI(title="CSV Doctor Engine", version=ENGINE_VERSION)


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
