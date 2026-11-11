# CSV Doctor

Drop any CSV. In 30 seconds, know what's wrong, what's weird, and what's
interesting in your data — in plain English.

Real ML runs on the full file (profiling, anomaly detection, drift /
change-point detection). A free LLM only *explains* the computed facts —
it never sees the raw file and never invents a number.

**Independent product by Khushi (DevK-26). Built and run entirely on free tiers.**

## Status
Day 1 — repo + engine skeleton. See `BLUEPRINT.md` for the full product spec
and the 14-day build plan, and `DAY1.md` for what this skeleton contains.

## Layout
```
apps/web/          Next.js frontend (Day 7-9)
services/engine/   FastAPI + pandas/scikit-learn ML engine (Day 2-6)
samples/           Sample datasets anyone can try without uploading a file
lanes.yaml         Free-model router config (Groq -> OpenRouter :free)
```

## Run the engine tests (Day 1 check)
```bash
cd services/engine
pip install -r requirements.txt
python -m pytest tests/ -v
```

## Principles
1. Deterministic first: every number on a report is computed, reproducible,
   and unit-tested. Same file = same score, always.
2. The LLM is garnish: if its summary disagrees with a computed number,
   the number wins and the summary is dropped.
3. Privacy by architecture: raw files are never stored; auto-deleted
   within 1 hour. The LLM receives only a masked facts pack.
4. ₹0 model cost: free tiers only, every call logged and cached by file hash.
