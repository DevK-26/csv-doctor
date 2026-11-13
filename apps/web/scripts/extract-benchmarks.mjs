// Reads the four generated benchmark reports in services/engine/benchmarks/results/
// and writes the numbers the landing page shows to apps/web/lib/benchmarks.json.
//
//   node scripts/extract-benchmarks.mjs            write lib/benchmarks.json
//   node scripts/extract-benchmarks.mjs --check    exit 1 if the committed JSON is stale
//
// The .md files are the source of truth (CLAUDE.md rule 3): nothing is typed in
// here. If a report changes shape the parser throws instead of guessing.
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const RESULTS_DIR = resolve(here, "../../../services/engine/benchmarks/results");
const OUT_FILE = resolve(here, "../lib/benchmarks.json");
const SOURCE_PREFIX = "services/engine/benchmarks/results";

const read = (name) => readFileSync(resolve(RESULTS_DIR, name), "utf8");

function fail(file, what) {
  throw new Error(`extract-benchmarks: ${file}: could not find ${what}. Did the report format change?`);
}

/** First capture group of `re` in `text`, or a loud failure. */
function grab(file, text, re, what) {
  const m = text.match(re);
  if (!m) fail(file, what);
  return m[1];
}

const num = (s) => Number(String(s).replace(/[,+\s]/g, ""));
const stripMd = (s) => s.replace(/\*\*/g, "").replace(/`/g, "").trim();

/** Rows of the markdown table whose header row starts with `firstHeader`, as {header: cell} objects. */
function table(file, text, firstHeader) {
  const lines = text.split("\n");
  const start = lines.findIndex((l) => l.startsWith("|") && stripMd(l.split("|")[1]) === firstHeader);
  if (start < 0) fail(file, `a table whose first column is "${firstHeader}"`);
  const cells = (l) => l.trim().replace(/^\||\|$/g, "").split("|").map(stripMd);
  const headers = cells(lines[start]);
  const rows = [];
  for (let i = start + 2; i < lines.length && lines[i].startsWith("|"); i++) {
    const row = cells(lines[i]);
    rows.push(Object.fromEntries(headers.map((h, j) => [h, row[j]])));
  }
  return rows;
}

function rowNamed(file, rows, key, predicate, what) {
  const row = rows.find((r) => predicate(r[key]));
  if (!row) fail(file, what);
  return row;
}

// ------------------------------------------------------------------ anomaly
function anomaly() {
  const file = "anomaly_benchmark.md";
  const t = read(file);
  const models = {};
  for (const r of table(file, t, "Model")) {
    models[r.Model] = {
      precision: num(r.Precision), recall: num(r.Recall), f1: num(r.F1), auroc: num(r.AUROC),
    };
  }
  for (const need of ["Autoencoder", "Ensemble", "Local Outlier Factor", "Isolation Forest"]) {
    if (!models[need]) fail(file, `a "${need}" results row`);
  }
  const acc = t.match(/\*\*(PASS|FAIL)\*\* - ensemble F1 ([\d.]+) beats (.+?) \(([\d.]+)\)/);
  if (!acc) fail(file, "the acceptance line (ensemble F1 beats ...)");
  return {
    source: `${SOURCE_PREFIX}/${file}`,
    dataset: grab(file, t, /\*\*Dataset:\*\* (.+?),/, "the dataset line"),
    rows: num(grab(file, t, /\*\*Rows scored:\*\* ([\d,]+)/, "rows scored")),
    labelled_anomalies: num(grab(file, t, /\*\*Rows scored:\*\* [\d,]+ \(([\d,]+) labelled/, "labelled anomalies")),
    seed: num(grab(file, t, /Seed: (\d+)\./, "the seed")),
    models: {
      autoencoder: models.Autoencoder,
      ensemble: models.Ensemble,
      local_outlier_factor: models["Local Outlier Factor"],
      isolation_forest: models["Isolation Forest"],
    },
    acceptance: {
      passed: acc[1] === "PASS", ensemble_f1: num(acc[2]), beats: acc[3], beats_f1: num(acc[4]),
    },
  };
}

// -------------------------------------------------------------------- drift
function drift() {
  const file = "drift_benchmark.md";
  const t = read(file);
  const detectors = table(file, t, "Detector").map((r) => ({
    name: r.Detector,
    first_alarm_index: num(r["First alarm index"]),
    delay_samples: num(r["Detection delay (samples)"]),
    control_alarms: r["Alarms on control stream"],
    quiet_on_control: r["Quiet on control"] === "yes",
  }));
  if (detectors.length === 0) fail(file, "detector rows");
  return {
    source: `${SOURCE_PREFIX}/${file}`,
    seed: num(grab(file, t, /\*\*Seed:\*\* (\d+)/, "the seed")),
    stream_length: num(grab(file, t, /\*\*Stream length:\*\* ([\d,]+) samples/, "the stream length")),
    planted_index: num(grab(file, t, /\*\*(\d+)\*\* \(0-based/, "the planted drift index")),
    shift_sigma: num(grab(file, t, /jumps by \+([\d.]+) standard deviations/, "the planted shift size")),
    detectors,
    false_alarms_on_control: /False alarms: none\./.test(t) ? 0 : fail(file, "the 'False alarms: none.' line"),
    passed: /\*\*PASSED\*\*/.test(t),
  };
}

// ------------------------------------------------------------------- dedupe
function dedupe() {
  const file = "dedupe_benchmark.md";
  const t = read(file);
  const row = rowNamed(file, table(file, t, "Dataset"), "Dataset", (d) => d.startsWith("Generated"),
    'the "Generated (...)" pair-quality row');
  const label = row.Dataset;
  return {
    source: `${SOURCE_PREFIX}/${file}`,
    dataset: "Generated name/address set",
    seed: num(grab(file, label, /seed (\d+)/, "the generated-set seed")),
    entities: num(grab(file, label, /(\d+) entities/, "the entity count")),
    rows: num(row.Rows),
    true_pairs: num(row["True pairs"]),
    precision: num(row["Pair precision"]),
    recall: num(row["Pair recall"]),
    f1: num(row["Pair F1"]),
  };
}

// -------------------------------------------------------------------- scale
function scale() {
  const file = "scale_benchmark.md";
  const t = read(file);
  const head = t.split("## Headline")[1]?.split("\n## ")[0];
  if (!head) fail(file, "the Headline section");
  return {
    source: `${SOURCE_PREFIX}/${file}`,
    rows: num(grab(file, head, /\*\*Rows:\*\* ([\d,]+) x/, "headline rows")),
    columns: num(grab(file, head, /\*\*Rows:\*\* [\d,]+ x (\d+) columns/, "headline columns")),
    csv_mb: num(grab(file, head, /\(([\d.]+) MB of CSV\)/, "headline CSV size")),
    wall_seconds: num(grab(file, head, /\*\*Wall time:\*\* ([\d.]+) s/, "headline wall time")),
    rows_per_second: num(grab(file, head, /\(([\d,]+) rows\/s\)/, "headline rows/s")),
    peak_memory_mb: num(grab(file, head, /\*\*Peak memory:\*\* ([\d,]+) MB/, "headline peak memory")),
    measured_on: grab(file, t, /\*\*Date \(UTC\):\*\* (\d{4}-\d{2}-\d{2})/, "the measurement date"),
  };
}

const data = {
  _generated_by: "apps/web/scripts/extract-benchmarks.mjs - do not edit by hand; re-run the script.",
  anomaly: anomaly(),
  drift: drift(),
  dedupe: dedupe(),
  scale: scale(),
};
const json = JSON.stringify(data, null, 2) + "\n";

if (process.argv.includes("--check")) {
  let current = "";
  try { current = readFileSync(OUT_FILE, "utf8"); } catch { /* missing counts as stale */ }
  if (current !== json) {
    console.error("lib/benchmarks.json is out of date; run: npm run extract-benchmarks");
    process.exit(1);
  }
  console.log("lib/benchmarks.json matches the benchmark reports.");
} else {
  writeFileSync(OUT_FILE, json);
  console.log(`wrote ${OUT_FILE}`);
}
