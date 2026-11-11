# CSV Doctor — project instructions for Claude Code

## What this is

CSV Doctor is an independent product by Khushi Yadav (GitHub: DevK-26).
A user uploads a CSV and gets a diagnosis: profiling, a deterministic
Health Score, anomaly detection with explanations, drift detection,
duplicate detection, and a plain-English report. It is a resume-flagship
project: every claim in the README must be backed by code that runs and
numbers that were actually measured.

Full product spec: `BLUEPRINT.md` (esp. §12 Technical Depth v2).
Build order and acceptance criteria: `BUILD-SPRINT.md`.

## Stack

- Engine: Python 3, FastAPI (`services/engine/`), pandas/numpy,
  scikit-learn, PyTorch (autoencoder, Phase B), DuckDB (Phase C)
- Web: Next.js (`apps/web/`)
- Infra: Docker Compose + Caddy (`docker-compose.yml`, `deploy/`),
  Redis + Celery (Phase C), GitHub Actions (`.github/workflows/deploy.yml`)

## Engine layout (do not restructure)

- `services/engine/app/engine/parse.py` — CSV parsing, type inference
  (IDs must stay IDs, never numbers). WORKING — extend carefully.
- `services/engine/app/engine/profile.py` — per-column profile +
  deterministic Health Score. WORKING.
- `services/engine/app/engine/facts.py` — masked Facts Pack for reports.
  WORKING.
- `services/engine/app/engine/anomalies.py` — STUB → Phase A target.
- `services/engine/app/engine/drift.py` — STUB → Phase A target.
- `services/engine/app/engine/relations.py` — STUB → Phase B target.
- `services/engine/app/main.py` — FastAPI app (`/health`, `/analyse`).
- `services/engine/tests/` — pytest; `samples/` — 3 CSVs with planted dirt.

## Hard rules

1. **Tests are the contract.** Every new module ships with pytest tests.
   All existing tests must keep passing (`python -m pytest tests/ -v`
   from `services/engine/`). Never weaken a test to make it pass.
2. **Deterministic core.** Profiling, Health Score, drift statistics and
   benchmarks run with fixed seeds and produce identical results on
   re-runs. No randomness without an explicit seed parameter.
3. **Measured numbers only.** Benchmarks (anomaly F1/AUROC, drift
   detection delay, rows/sec) must come from scripts in
   `services/engine/benchmarks/` that anyone can re-run. Never hardcode
   a metric into the README or UI.
4. **From scratch means from scratch.** The drift detectors (PSI, KS,
   Jensen–Shannon, ADWIN, Page-Hinkley) are implemented by hand in
   `drift.py` — no library drift package. The point is the author can
   explain every line.
5. **Explainability is a feature.** Every anomaly returned by the API
   carries a human-readable reason (which features drove the score).
   No bare scores.
6. **No placeholder UI.** The web app calls the real API. If an endpoint
   doesn't exist yet, build the endpoint, not a mock.
7. **Small, reviewable steps.** One module per working session, committed
   in logical chunks with clear messages. The author reviews and must be
   able to explain each module before the next one starts.
8. **Dependencies** go in `services/engine/requirements.txt`, pinned to
   compatible versions; note in the commit why each was added.
9. Never commit secrets, `.env` files, or generated benchmark data files
   larger than 5 MB. Datasets for benchmarks are downloaded by script,
   not vendored.

## Style

- Type hints on public functions; short docstring stating inputs/outputs.
- Comments explain *why*, not *what* — keep them sparse.
- Errors: raise specific exceptions with actionable messages; the API
  layer converts them to clean 4xx responses, never stack traces.
