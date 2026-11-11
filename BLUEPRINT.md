# CSV Doctor — Full Blueprint (Independent Product)
**Builder:** Khushi (DevK-26) — independent brand, NOT Code.SCRIET, NOT IITM, no institution attached
**Chosen:** 2026-11-01 | **Working name:** CSV Doctor (final name/domain before launch)
**One-line pitch:** Drop any CSV. In 30 seconds, know what's wrong, what's weird, and what's interesting in your data — in plain English.

---

## 1. What it does — in detail

### The problem (intuition)
Almost everyone who works with data has the same first 30 minutes of pain: they get a CSV
(sales export, survey, bank/analytics export, a Kaggle dataset, sensor logs, a research dataset)
and they do NOT know: Is it clean? Are rows duplicated? Which columns are broken? Is anything
strange hiding in it? Did the pattern change halfway? Tools that exist (Excel, ChatGPT upload,
Pandas Profiling) fail normal people in different ways:
- Excel: shows the data, tells you nothing about it.
- ChatGPT/LLM upload: gives a confident summary, but it *guesses* — it does not actually run
  statistics on all 50,000 rows, it hallucinates outliers, and your file goes to a model.
- ydata-profiling / pandas-profiling: correct statistics, but the output is a 40-section
  technical report no non-data person can read or act on.

CSV Doctor sits in the gap: **real ML/statistics run on the full file (deterministic, verifiable),
explained in plain English by a free LLM that is only allowed to explain the computed facts —
never to invent them.** That separation is the product.

### User journey (v1, no login needed)
1. **Landing page** — one sentence, one drop zone, 3 sample datasets ("Try with sample sales data")
   so a visitor gets a full report in 1 click without trusting you with their file yet.
2. **Drop a CSV** (drag/drop or browse). Limits in v1: 10 MB, 50,000 rows, .csv only.
   Browser shows: filename, rows, columns detected locally before upload.
3. **Processing (15-40 sec, progress steps shown honestly):**
   Parsing → Profiling → Outlier detection → Drift/change detection → Correlations →
   Plain-English explanation. Progress is real stage names, not a fake spinner.
4. **Report page — 5 sections, in this order:**
   - **Health Score (0-100) + verdict in one line.** e.g. "72/100 — Usable, but fix 3 things first."
     Score is computed deterministically from: % missing cells, % duplicate rows, broken columns,
     outlier rate, mixed-type columns. Same file = same score, always. (LLM has zero say in the score.)
   - **Fix First (data quality):** per-column cards — type detected (number/date/text/category),
     missing %, unique count, min/max/mean where relevant; duplicate rows count + download-the-
     duplicate-rows CSV; constant columns; mixed types ("this 'price' column has 212 text values");
     impossible values (negative ages, dates in the future) from range rules.
   - **What's Weird (anomalies):** Isolation Forest + IQR/z-score ensemble flags rows that do not
     fit the rest. Report shows top 20 anomalous rows in a table, *why* each was flagged (which
     columns drove it), a scatter/line highlight, and a **Download anomalies.csv** button.
   - **What Changed (drift — your research, as a feature):** If a date/time column exists, the data
     is treated as a stream in time order: rolling-mean shift + a change-point detector finds where
     the behaviour changed ("Revenue pattern changed around 14 Mar — mean jumped 31%, variance
     doubled"). If no date column, this section is honestly skipped, not faked.
   - **What's Connected (relationships):** top correlations (numeric-numeric, with a heatmap for the
     top 8 numeric columns), category-vs-number group differences ("Region=North has 2.1x returns
     of other regions"), and 3 auto-suggested charts.
   - **Plain-English Summary (the only LLM part):** 5-7 bullet insights + 3 "questions to ask next",
     written from a *facts pack* only (see §4). Plus **Ask your data** in v1.5: chat questions are
     answered by first computing the answer in pandas, then letting the LLM phrase it.
5. **Share / export:** Download full report as PDF, download cleaned CSV (duplicates removed,
   types fixed — original never modified), share a public report link (aggregates only, no raw
   rows unless the user explicitly enables it) + a share-card image for LinkedIn/X/WhatsApp.

### Worked example (use this in demos and launch posts)
Sample file: `sales_2025.csv` — 12,483 rows, columns: date, region, product, units, revenue, returns.
Report says: Health 68/100. Fix: 341 exact duplicate rows (2.7%); `revenue` has 96 blanks, all in
Region=West in January (a pattern, not random missingness — flagged as such). Weird: 57 anomalous
rows, top one is 9,800 units on one day vs a daily max of ~400 — likely an extra zero typed.
Changed: change-point on 2025-03-14, returns rate doubled after it and never recovered.
Connected: returns strongly correlate with discount (r=0.71) only in one product category.
Summary explains all of the above in 6 bullets a shop owner understands.

### Who it is for (launch order)
1. Students/researchers with a dataset for a project/paper (they must describe their data anyway).
2. Analysts / data interns who get handed "dirty" CSVs at work.
3. Small business owners exporting sales/expenses from Tally/Shopify/Amazon/Excel.
Explicitly NOT for: live databases, streaming production monitoring, files >10 MB in v1.

---

## 2. MVP scope — freeze this

### In v1 (launch)
- [ ] Landing + drop zone + 3 sample datasets (sales, survey, website analytics — ship with repo)
- [ ] CSV parse with encoding/delimiter auto-detection, row/col limits enforced with a clear error
- [ ] Profiling engine (§3, step A+B) + Health Score (deterministic, unit-tested)
- [ ] Anomaly engine (Isolation Forest ensemble) + top-20 table + anomalies.csv download
- [ ] Drift/change-point section (only when a datetime column is detected)
- [ ] Correlation heatmap + top relationships
- [ ] LLM summary from facts pack (Groq primary, OpenRouter :free fallback, result cached)
- [ ] Report page (5 sections) + PDF export (browser print stylesheet is enough for v1)
- [ ] Cleaned CSV download (dedupe + type coercion, with a changelog of what was changed)
- [ ] Privacy: file processed in memory / temp storage, **auto-deleted within 1 hour**, page states it
- [ ] Analytics (PostHog free / Plausible): files analysed, sample-vs-own, shares, downloads — no file contents in analytics
- [ ] Rate limit: 5 analyses/IP/day logged-out (protects free-model quota)

### NOT in v1 (backlog only)
Login/accounts, saved history, Excel (.xlsx) support, >10 MB files, Ask-your-data chat (v1.5),
scheduled re-analysis / live drift monitoring (v2 — this is where the research paper's ADWIN work lands),
payments, team workspaces, API access.

---

## 3. How the ML actually works (the pipeline)

```
CSV bytes
  → A. Parse & type inference
  → B. Profile (per-column stats)
  → C. Quality rules + Health Score          [deterministic, no ML, no LLM]
  → D. Anomalies (IsolationForest + IQR)      [sklearn, runs on CPU, ₹0]
  → E. Drift / change-points (if time col)    [rolling stats + ruptures/PELT or CUSUM fallback]
  → F. Relationships (correlation, group diffs)
  → Facts Pack (JSON, ~2-4 KB — numbers only, PII-masked)
  → G. LLM explains Facts Pack                [Groq free → OpenRouter :free fallback, cached]
  → Report (stored aggregates + explanation; raw file deleted)
```

- **A. Parse:** pandas `read_csv` with `sep=None` (sniffer) + encoding fallback (utf-8 → latin-1).
  Type inference per column: datetime (try parse ≥80% values), numeric (≥90% coercible),
  categorical (text, low cardinality), text (high cardinality), boolean, empty, constant.
  **Targeting check:** never trust the first guess — a column of IDs like `00123` must stay text,
  not become numbers that lose leading zeros. ID-like columns (name contains id/code, or all-unique
  integers) are excluded from anomaly/correlation maths.
- **B. Profile:** rows, cols, missing % overall + per column, duplicate full rows, per-numeric:
  min/max/mean/median/std/IQR/zeros/negatives; per-category: top-5 values + counts; per-date: range + gaps.
- **C. Health Score (0-100, start at 100, subtract, floor 0):**
  missing cells: −(missing% × 1.5, cap −25); duplicate rows: −(dup% × 2, cap −20);
  mixed-type column: −5 each (cap −15); constant/empty column: −4 each (cap −12);
  impossible values: −1 each type (cap −10); outlier rate >5%: −8, >10%: −15.
  Formula lives in ONE function with unit tests — same file must always score the same.
- **D. Anomalies:** numeric columns only, median-imputed for the model only (original untouched).
  IsolationForest (n_estimators=200, contamination = min(0.05, max(0.01, heuristic)), random_state fixed)
  + IQR rule per column. A row is reported if IF flags it; "why" = the columns where its IQR z-score
  is extreme. Cap output at top 20 by anomaly score. If rows < 50 or numeric cols < 2, say so and skip —
  never run a model that cannot mean anything on that data.
- **E. Drift:** sort by datetime col; resample numeric targets to daily/weekly if needed;
  rolling mean/std (window = max(7, n/20)); change-points via `ruptures` Pelt (rbf, pen=10) if installed,
  else a CUSUM/rolling-shift fallback you write yourself (~40 lines — and you can explain it in a viva).
  Report each change-point: date, metric, before/after mean, % change. This is the simplified,
  honest version of your paper's ADWIN/DDM comparison — say exactly that in the README.
- **F. Relationships:** Pearson + Spearman for numeric pairs (|r| ≥ 0.4 reported, top 10);
  categorical vs numeric: group means, report if max/min ratio ≥ 1.5 with n ≥ 30 per group;
  heatmap data for top-8 numeric columns by variance.
- **Determinism rule:** every number on the report except the LLM paragraph comes from A–F and is
  reproducible. Tests: golden files — 3 fixture CSVs with known duplicates/outliers/drift, assert
  the engine finds them.

## 4. How the AI part works (and cannot hallucinate the numbers)

- The LLM **never receives the raw CSV.** It receives the Facts Pack JSON: column summaries,
  health score + reasons, top anomalies (row numbers + driving columns + values, with emails/
  phones/long IDs masked), change-points, top correlations, group differences.
- Prompt contract (system): "You are explaining computed statistics. Use ONLY numbers in the
  facts pack. Never invent a value, a cause, or a row. If a cause is a guess, label it 'Possible
  explanation — verify'. Output JSON: {bullets: [..≤7], warnings: [..], next_questions: [3]}."
- Validate the returned JSON (schema check); on failure, retry once on the fallback lane, then
  show the report *without* the summary section and a note — the report must never fail because an LLM did.
- **Model router (`lanes.yaml` — steal Aarya's idea, simplify it):**
  ```yaml
  lanes:
    - {id: groq-8b,   provider: groq,       model: llama-3.1-8b-instant,    role: quick-summary, daily_cap: 14400}
    - {id: groq-70b,  provider: groq,       model: llama-3.3-70b-versatile, role: deep-summary,  daily_cap: 1000}
    - {id: or-free,   provider: openrouter, model: openrouter/free,         role: fallback}
  rules: [cache-by-file-hash, 8b-first, 70b-only-if-rows>5000-or-user-clicks-deep, log-every-call, stop-at-cap]
  ```
  v1 default: 8B does the summary (plenty for explaining stats, and it has the big free quota);
  70B is the "Deep dive" button. Cache key = SHA-256 of file bytes + engine version — same file
  re-uploaded = 0 new calls.
- Quota guard: a `model_calls` table counts calls/day/lane; router refuses past cap and degrades
  gracefully. This table is also your resume number: "served X reports at ₹0 model cost."

## 5. Tech stack (all free tiers)

| Layer | Choice | Why / free limit |
|---|---|---|
| Frontend | Next.js (React + TypeScript) + Tailwind, Recharts for charts | Your stack; Vercel Hobby or Cloudflare Pages free |
| ML API | Python FastAPI + pandas + scikit-learn (+ ruptures optional) | ML in Python where it belongs; separate service = clean interview story |
| ML hosting | Hugging Face Spaces (Docker, free CPU) or Render free web service | HF Spaces does not expire the demo like Render's free Postgres does; expect cold starts, show honest "waking the analyser" state |
| DB | Supabase Postgres (free: 500 MB DB, 50k MAU) | Stores reports (aggregates only), shares, model-call counts — NOT raw files |
| Cache / queue | Upstash Redis (free) | Report cache by file hash, rate limiting, job status if analysis goes async |
| Files | In-memory / temp disk on the ML service, deleted ≤1 hr | Privacy by architecture, not by promise |
| LLM | Groq free (8B: 14,400 req/day; 70B: 1,000 req/day) → OpenRouter :free fallback (50 req/day) | Numbers checked Oct-Nov 2026; re-check console before launch, they change |
| Analytics | PostHog free tier | Funnels: visit → sample report → own file → share |
| Repo | GitHub, public, MIT | Build-in-public is the marketing |

Architecture: Browser → Next.js (Vercel) → FastAPI `/analyse` (HF Spaces) → engine (pandas/sklearn)
→ Facts Pack → Next.js server route calls Groq (key stays server-side) → report saved to Supabase →
report page renders from saved aggregates. If analysis >20 sec, return a job id and poll status
stored in Redis — build the sync version first, add polling only if real files time out.

## 6. Repo structure

```
csv-doctor/
  apps/web/                 # Next.js
    app/page.tsx            # landing + drop zone
    app/report/[id]/page.tsx
    app/api/summary/route.ts  # server-only LLM call (keys never in browser)
    components/ (DropZone, HealthScore, ColumnCard, AnomalyTable, Heatmap, Summary)
    lib/ (api client, share-card canvas)
  services/engine/          # FastAPI
    app/main.py             # POST /analyse, GET /health
    app/engine/parse.py     # A: parse + type inference
    app/engine/profile.py   # B + C: profile + health score
    app/engine/anomalies.py # D: IsolationForest + IQR
    app/engine/drift.py     # E: change-points
    app/engine/relations.py # F: correlations + group diffs
    app/engine/facts.py     # Facts Pack builder + PII masking
    tests/ (test_profile.py, test_anomalies.py, test_drift.py, fixtures/*.csv)
    requirements.txt, Dockerfile
  samples/ (sales.csv, survey.csv, analytics.csv + expected outputs)
  lanes.yaml                # model router config (§4)
  README.md                 # pitch, screenshots, how ML works, ₹0-cost claim + call log proof
```

## 7. Data model (Supabase)

```
reports:      id (uuid), file_hash, file_name, rows, cols, health_score,
              profile (jsonb), anomalies (jsonb), drift (jsonb), relations (jsonb),
              summary (jsonb, nullable), engine_version, is_sample, is_public, created_at
model_calls:  id, lane, model, report_id, tokens_in, tokens_out, latency_ms, ok, created_at
shares:       id, report_id, created_at   -- a share is just reports.is_public = true + this row for counting
```
No users table in v1. No raw CSV anywhere in the DB. Ever.

## 8. API contract

- `POST /analyse` (FastAPI, multipart file) → `{report_id, health_score, profile, anomalies, drift, relations, facts_pack}` or `422 {error: "file_too_large" | "not_a_csv" | "no_numeric_columns"}` with a human sentence for each.
- `POST /api/summary` (Next.js server route, body: report_id) → returns cached summary if present, else calls router (§4), saves, returns `{bullets, warnings, next_questions}`.
- `GET /report/[id]` — public only if `is_public`, else requires the unguessable uuid (v1 simplicity, stated honestly).

## 9. Build plan — 14 working days (~2-3 hrs/day)

Start after your Nov project deadlines (M5 Nov 18) — build in Dec, launch for winter-internship season.
- **D1:** Repo, monorepo folders, Next.js + FastAPI skeletons, `/health` on both, lanes.yaml, README pitch.
- **D2-3:** Engine A+B+C (parse, types, profile, health score) + unit tests on 3 fixture CSVs.
- **D4:** Engine D (anomalies) + golden test: fixture with 5 planted outliers must find ≥4.
- **D5:** Engine E+F (drift + relations) + fixture with a planted change-point must be detected ±1 day.
- **D6:** Facts Pack + masking + `/analyse` end-to-end via curl. **Milestone: engine works without any UI.**
- **D7-8:** Web: landing, drop zone, report page rendering a saved sample report (hardcode first, wire API second).
- **D9:** Wire upload → API → report; loading states per real stage; error states for bad files.
- **D10:** LLM summary route + router + cache + model_calls logging; schema validation + no-summary fallback.
- **D11:** Downloads (anomalies.csv, cleaned.csv + changelog), PDF print view, share link + share card.
- **D12:** Deploy: engine → HF Spaces (Docker), web → Vercel, DB → Supabase, Redis → Upstash. Test on phone + a real 40k-row file.
- **D13:** Polish, privacy page ("what happens to your file", step-by-step), sample datasets verified, analytics events.
- **D14:** Launch (see §10). Fix only launch-blocking bugs after this; log everything else as issues.

Definition of done for YOU (authorship rule, same as your term projects): you run every engine
module yourself on the fixtures and can explain parse → profile → IF → drift → facts pack →
LLM contract without notes. If you can't explain it, it doesn't ship.

## 10. Launch plan (public internet only, your brand)

1. **Soft:** Post the worked example (§1) as a LinkedIn build-in-public post: screenshot of the
   sales report + "I built a tool that actually runs ML on your CSV instead of asking an LLM to guess.
   It found a fake 9,800-unit sale in my sample data." Link in first comment.
2. **Reddit (value first, not spam):** r/datascience, r/dataanalysis, r/excel, r/SideProject —
   title pattern: "I built a free tool that profiles a CSV and explains anomalies in plain English —
   no signup, files auto-deleted. Roast my approach?" Offer to run reports for commenters' sample data.
3. **SEO tail:** public sample reports are indexable pages ("example: sales data quality report") —
   small now, compounds later.
4. **Numbers to chase in month 1:** 100 own-file analyses, 20 shares, 5 people who return with a
   second file. Those numbers, not vanity visits, go on the resume.

Resume bullet (fill after launch): "Built & launched CSV Doctor, an independent data-profiling
product — pandas/scikit-learn engine (Isolation Forest anomaly detection, change-point/drift
detection from my concept-drift research) with a free-tier LLM explanation layer behind a
multi-provider router; [X] files analysed at ₹0 model cost; Next.js, FastAPI, PostgreSQL."

## 10A. Traffic Engine (added 2026-11-01 — Khushi's final call)

Decision note: Khushi reaffirmed CSV Doctor *knowing* its traffic is slower than a viral
Wrapped/resume tool, because the resume story, explainability, and use of all her skills
(web + ML + AI) matter more to her than maximum raw traffic. Traffic is therefore grown
*around* the product, not by changing it:

1. **Micro-tool SEO pages (same engine, 5 doors):** `/tools/find-duplicates-in-csv`,
   `/tools/outlier-detector-online`, `/tools/csv-data-quality-checker`,
   `/tools/missing-values-checker`, `/tools/what-changed-in-my-data` — each is the main
   tool pre-set to that single job, targeting one exact Google search per page.
2. **Public Dataset Library (the shareable content):** one famous public dataset per week
   (IPL matches, Netflix titles, Titanic, Delhi weather, Bengaluru house prices) analysed
   and published as a public, indexable report + a LinkedIn/Reddit post built on its most
   surprising finding. Users' private reports never do marketing; public reports do.
3. **Answer-with-a-report:** reply to real Reddit/Stack Overflow data-cleaning questions
   with a report link on public/sample data, not an ad.
Target after this engine: Month-1 goal stays §10's (100 own-file analyses, 20 shares,
5 returning users) — quality traffic (data people, hiring managers) over raw visits.

## 11. Risks & honest limits

- **Wrong anomaly ≠ error:** an outlier can be your best customer, not a mistake. UI language is
  always "unusual — check this", never "wrong". (Your targeting-error rule, applied to the product.)
- **Free tiers move:** Groq/OpenRouter/Gemini caps changed repeatedly in 2026. The router + cache +
  graceful degradation exist because of this; re-check caps monthly.
- **Privacy is the product:** if a user can't verify deletion, they won't upload work data.
  Keep the promise architectural (no raw storage) and say exactly what the LLM receives.
- **Big files:** v1 caps are a feature. Trying to support 1 GB files on free CPU will kill the launch.
- **LLM summary is garnish:** if it ever disagrees with a computed number, the number wins and the
  summary is regenerated or dropped. State this in the README — it's your differentiator.

## 12. Technical Depth Upgrade v2 (added 2026-11-01 — Khushi's challenge, accepted)

Khushi's verdict on v1: "this is a normal project, not that technical."
She is RIGHT about the v1 core. Profiling + a basic Isolation Forest + an LLM
summary is what a dozen student projects and free web tools already do
(ydata-profiling exists). v1 stays as Layer 1 (foundation — it is also what
makes the product usable by normal people), but the following five systems
become CORE, not backlog. They are what an interviewer will actually probe,
and each one maps to something Khushi has formally studied.

### T1. Research-grade drift engine (her paper, implemented, not mentioned)
- Per-column drift, not just a global change-point: Population Stability Index
  (PSI), two-sample Kolmogorov–Smirnov test with p-values for numeric columns,
  Jensen–Shannon divergence for categorical distributions — first half vs
  second half, and in sliding windows over the time column.
- Streaming detectors implemented FROM SCRATCH (no library call): ADWIN-style
  adaptive windowing and Page–Hinkley, ~80 lines each, unit-tested on streams
  with planted abrupt and gradual drift. Detection delay is measured, not
  asserted: "detected 14 rows after the planted change."
- Output per drift event: column, detector, statistic, p-value / threshold,
  before/after distribution, detection delay. That table IS the viva defence.

### T2. Explainable anomaly ensemble + a real evaluation
- Three detectors, three families: Isolation Forest (tree ensemble), Local
  Outlier Factor (density), and a small PyTorch autoencoder (her DL course)
  for multivariate structure the other two miss. A row's final verdict is a
  weighted vote; disagreement between detectors is reported, not hidden.
- Every flagged row gets feature contributions: which columns pushed its
  score, and by how much (per-column reconstruction error for the autoencoder;
  perturbation-based contribution for IF/LOF).
- EVALUATION, the part normal projects skip: benchmark the ensemble on public
  labelled anomaly data (e.g. Kaggle credit-card fraud sample, SMD/NSL-KDD
  subsets) and publish precision / recall / F1 / AUROC per detector and for
  the ensemble in the README. The product's claims become measured numbers.

### T3. Fuzzy duplicate detection (record linkage, not df.duplicated())
- Exact duplicates are trivial (v1). v2 finds NEAR duplicates: rows that are
  the same entity with typos/variants ("Delhi" vs "delhi ", "Jon Smith" vs
  "John Smith"). Pipeline: blocking on cheap keys -> pairwise similarity
  (token/Jaro–Winkler via rapidfuzz) -> clustering into duplicate groups with
  a confidence score. Users review groups and merge — nothing auto-deleted.

### T4. Ask-Your-Data agent (text-to-analysis with deterministic execution)
- User asks: "did returns change after March for the North region?"
- The LLM does NOT answer from memory of the file. It writes a small pandas
  program from the schema + facts pack; the program runs in a sandboxed
  worker (no network, CPU/time limits, allow-listed imports); the numeric
  result is verified (types, non-empty, sanity ranges); only then does the
  LLM phrase the answer, quoting the computed result. Wrong code -> error
  fed back -> one retry -> honest failure. This is the Aarya-engine pattern
  (generate -> deterministic gate -> explain) applied to data questions,
  and it is the hardest, most defensible AI piece in the product.

### T5. Scale + backend engineering (her placement target)
- 50k-row cap removed: chunked/streaming profiling (constant-memory column
  accumulators) + DuckDB for group-by/correlation on files up to ~1M rows.
- Async jobs: FastAPI enqueues to Redis + a Celery-style worker, progress is
  streamed to the report page over SSE/WebSocket as stages complete
  (parse -> profile -> anomalies -> drift -> summary). Same Redis/Celery
  story as MAD II, now in her own product.
- Engineering hygiene that gets asked about: Pydantic/OpenAPI contract
  shared between web and engine, Docker Compose for the whole stack,
  GitHub Actions CI (engine tests + golden-file tests on every push),
  structured logging with a per-report trace id.

### Revised build shape
- Days 1-6: v1 engine as planned (foundation, already started Day 1).
- Days 7-10: T1 (drift engine from scratch + tests) and T2 (ensemble +
  benchmark). These two are the technical centre of gravity — schedule them
  BEFORE the pretty UI, not after.
- Days 11-13: T3 (fuzzy dedupe) + T5 (async jobs, chunking, CI/Docker).
- Days 14-17: Web UI (now including Ask-Your-Data), T4 (sandboxed agent).
- Days 18-21: Polish, deploy, benchmark page, launch (incl. Traffic Engine §10A).
Total: ~21 working days (was 14). The extra week is the difference between
"a normal project" and a flagship.
