# Day 1 — Repo + Engine Skeleton (step-by-step)

Goal by end of Day 1: a public GitHub repo `csv-doctor` on YOUR account, with the
skeleton below committed by YOU, and the engine tests passing on your machine.
Nothing is deployed yet. That is Day 12.

## What you have in this folder (read this first)

| Path | What it is, in plain words |
|---|---|
| `README.md` | The repo's front page: pitch + the 4 product principles |
| `BLUEPRINT.md` | The full product spec (what we're building over 14 days) |
| `lanes.yaml` | The free-model router config — which AI model answers, in what order, with daily caps |
| `samples/` | 3 ready datasets (sales has deliberately planted dirt: 2 duplicate rows, missing cells, 1 huge outlier row of 9,800 units, and a returns jump in the second half — later days must detect all of it) |
| `services/engine/app/engine/parse.py` | **Working.** Reads CSV bytes, enforces 10 MB / 50k rows, guesses each column's type. IDs stay IDs (never become numbers) |
| `services/engine/app/engine/profile.py` | **Working.** Per-column stats + the deterministic Health Score (same file = same score) |
| `services/engine/app/engine/anomalies.py` | Stub with the Day-4 plan written inside (Isolation Forest) |
| `services/engine/app/engine/drift.py` | Stub with the Day-5 plan (change-points — your concept-drift feature) |
| `services/engine/app/engine/relations.py` | Stub with the Day-5 plan (correlations) |
| `services/engine/app/engine/facts.py` | **Working.** Builds the masked Facts Pack — the only thing the AI will ever see |
| `services/engine/app/main.py` | FastAPI app: `GET /health`, `POST /analyse` (wired to parse+profile today) |
| `services/engine/tests/` | 3 golden tests locking parse/profile/score behaviour |
| `apps/web/` | Next.js landing-page skeleton (upload wiring is Day 7-9) |

## Step 1 — Create the GitHub repo (2 minutes, do it yourself)

1. Go to github.com → `+` (top right) → **New repository**
2. Name: `csv-doctor` · Description: `Drop any CSV — ML finds what's wrong, weird and interesting, AI explains it in plain English.`
3. **Public** · Do NOT tick "Add a README" / gitignore / licence (they are already in the skeleton — ticking them causes a merge conflict on first push)
4. **Create repository** · Leave the page open, you'll need its URL in Step 3

## Step 2 — Put the skeleton on your laptop

1. Download `csv-doctor-day1.zip` (sent with this guide) and unzip it on your **Desktop**
2. You should now have a folder `Desktop/csv-doctor/` containing README.md, BLUEPRINT.md, apps/, services/, samples/, lanes.yaml
   - ✅ Done-check: if you see a *second* csv-doctor folder inside the first one, move the inner one up — the README must be at `Desktop/csv-doctor/README.md`

## Step 3 — First commit + push (terminal, inside Desktop/csv-doctor)

```bash
cd ~/Desktop/csv-doctor
git init
git add .
git status            # read it: ~25 files, no .env, no zip files
git -c user.name="DevK-26" -c user.email="YOUR-GITHUB-EMAIL" commit -m "Day 1: repo + engine skeleton (parse, profile, health score, tests)"
git branch -M main
git remote add origin https://github.com/DevK-26/csv-doctor.git
git push -u origin main
```
- ✅ Done-check: refresh the GitHub repo page — 1 commit, README renders with the pitch.
- ⚠ Author check (you hit this on your DL repo): the commit author must show **DevK-26**.
  If it shows a different name, tell me before pushing anything else.

## Step 4 — Run the engine check (proves the skeleton works)

```bash
cd ~/Desktop/csv-doctor/services/engine
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/ -v
```
Expected result: **3 passed**
- `test_parse_and_types_on_dirty_fixture` — types guessed right, order_id stays an ID
- `test_profile_counts_duplicates_and_missing` — finds exactly 2 duplicate rows in the dirty fixture
- `test_health_score_punishes_dirty_and_rewards_clean` — dirty file scores lower than clean data

Then see it work on a real sample (still inside services/engine, venv active):
```bash
python -c "import sys; sys.path.insert(0,'.'); from app.engine.parse import parse_csv_bytes, infer_types; from app.engine.profile import profile_dataframe, health_score; raw=open('../../samples/sales.csv','rb').read(); df=parse_csv_bytes(raw); print(infer_types(df)); print(health_score(profile_dataframe(df)))"
```
Expected: types show `order_id: id`, `revenue: numeric`, and a health score prints.

## Step 5 — Stop here for Day 1

- ✅ Repo public, 1 commit, authored by you
- ✅ 3 engine tests passed on your machine
- ✅ You can say, in one line each, what parse.py, profile.py and facts.py do
  (if you can't, re-read their top docstrings — they're written for exactly this)

**Day 2 preview:** engine A+B+C properly — mixed-type detection, richer column
profiles, and health-score tests for every deduction. No UI yet, on purpose:
engine first, proven by tests, then the pretty part.
