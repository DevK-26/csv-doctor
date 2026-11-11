"""CSV Doctor engine API (FastAPI).

GET /health and POST /analyse: parse + profile + health score + the anomaly
ensemble (Phase A1: per-row results with reasons under "anomalies").
Drift/relations still return their honest 'skipped / not implemented yet'
stubs. The LLM summary is a separate web-app route (Day 10) —
this service never calls a model.
"""
from __future__ import annotations

import hashlib

from fastapi import FastAPI, File, HTTPException, UploadFile

from .engine import anomalies, drift, facts, profile as profile_mod, relations
from .engine.parse import ParseError, infer_types, parse_csv_bytes

# Bump on any change to analysis output: reports are cached by file hash + version.
ENGINE_VERSION = "0.2.0-a1"

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
    rel = relations.find_relationships(df, types)
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
