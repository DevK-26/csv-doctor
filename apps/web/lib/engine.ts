// Typed client for the CSV Doctor engine (services/engine/app/main.py).
// Shapes mirror what the engine actually returns; nothing here fabricates data.

export const ENGINE_URL = (process.env.NEXT_PUBLIC_ENGINE_URL ?? "http://localhost:8000").replace(/\/+$/, "");

/** The engine's in-memory path refuses files over 10 MB (app/engine/parse.py MAX_BYTES). */
export const IN_MEMORY_LIMIT_BYTES = 10 * 1024 * 1024;

// ------------------------------------------------------------------- shapes

export interface ColumnProfile {
  name: string;
  type: string;
  missing_pct: number;
  unique: number;
  top_values?: Record<string, number>;
  min?: number;
  max?: number;
  mean?: number;
  median?: number;
  negatives?: number;
}

export interface Profile {
  rows: number;
  cols: number;
  missing_pct: number;
  duplicate_rows: number;
  duplicate_pct: number;
  columns: ColumnProfile[];
}

export interface FeatureContribution {
  feature: string;
  value: number | string | null;
  value_was_missing: boolean;
  typical: number | string | null;
  usual_low: number | null;
  usual_high: number | null;
  direction: string;
  contribution: number;
}

export interface AnomalyRow {
  row_index: number;
  line_number: number;
  score: number;
  agreement: string;
  votes: Record<string, boolean>;
  model_scores: Record<string, number>;
  top_features: FeatureContribution[];
  reasons: string[];
}

export interface Anomalies {
  skipped: boolean;
  reason: string | null;
  method?: string;
  features_used?: string[];
  rows_scored?: number;
  threshold?: number;
  flagged_count?: number;
  flagged_pct?: number;
  rows_returned?: number;
  truncated?: boolean;
  rows?: AnomalyRow[];
}

export interface ChangePoint {
  column: string;
  detector: string;
  change_at: string | number;
  change_line: number;
  detected_at: string | number;
  detection_delay_rows: number;
  before: { rows: number; mean: number; std: number };
  after: { rows: number; mean: number; std: number };
  direction: string;
  pct_change: number | null;
  reason: string;
}

export interface DriftColumn {
  column: string;
  kind: "numeric" | "categorical" | string;
  skipped: boolean;
  reason: string | null;
  split_at?: string;
  n_reference?: number;
  n_current?: number;
  psi?: number;
  ks_statistic?: number;
  ks_p_value?: number;
  js_divergence?: number;
  p_value?: number;
  level?: string;
  drifted?: boolean;
  change_points?: number;
}

export interface Drift {
  skipped: boolean;
  reason: string | null;
  time_column?: string;
  period?: { start: string; end: string };
  rows_used?: number;
  columns?: DriftColumn[];
  change_points: ChangePoint[];
  change_points_total?: number;
  truncated?: boolean;
}

export interface MatchedField {
  column: string;
  similarity: number;
  identical: boolean;
  identical_after_cleanup: boolean;
  values: string[];
}

export interface DuplicateGroup {
  group_id: number;
  kind: "exact" | "near" | string;
  rows: number[];
  size: number;
  confidence: number;
  weakest_pair: number;
  matched_fields: MatchedField[];
  differs_in: string[];
  reasons: string[];
}

export interface Relations {
  skipped: boolean;
  reason: string | null;
  duplicate_groups: number;
  duplicate_rows: number;
  groups: DuplicateGroup[];
  groups_truncated: boolean;
}

export interface FactsPack {
  rows: number;
  cols: number;
  health_score: number;
  score_reasons: string[];
  missing_pct: number;
  duplicate_rows: number;
  columns: { name: string; type: string; missing_pct: number; unique: number }[];
  anomalies: Anomalies;
  drift: Drift;
  relations: {
    skipped: boolean;
    reason: string | null;
    duplicate_groups: number;
    duplicate_rows: number;
    exact_groups: number;
    near_groups: number;
    low_confidence_groups: number;
  };
}

/** POST /analyse (in-memory path). */
export interface Analysis {
  kind: "analysis";
  file_hash: string;
  engine_version: string;
  health_score: number;
  score_reasons: string[];
  profile: Profile;
  anomalies: Anomalies;
  drift: Drift;
  relations: Relations;
  facts_pack: FactsPack;
}

/** Result of a finished /jobs run (scale path): profile + score only. */
export interface ScaleReport {
  kind: "scale";
  file_hash: string;
  engine_version: string;
  health_score: number;
  score_reasons: string[];
  profile: Profile;
  scale: { chunk_rows: number; chunks: number; encoding: string; delimiter: string };
}

export type Report = Analysis | ScaleReport;

export interface AskCheck {
  name: string;
  passed: boolean;
  detail: string;
}

export interface AskAnswer {
  kind: string;
  text: string;
  value?: number | null;
}

export interface AskResult {
  question: string;
  status: "verified" | "unverified" | "failed";
  reason: string | null;
  message: string;
  answer: AskAnswer | null;
  program: string | null;
  assumptions: string[];
  verification: { passed: boolean; method: string; checks: AskCheck[] } | null;
  generator: string;
}

export interface JobProgress {
  status: string;
  percent: number;
  stage: string;
}

// ------------------------------------------------------------------- errors

export type EngineErrorKind = "unreachable" | "rejected" | "too_big_for_memory" | "failed";

export class EngineError extends Error {
  constructor(public kind: EngineErrorKind, message: string, public status?: number) {
    super(message);
    this.name = "EngineError";
  }
}

const unreachable = () =>
  new EngineError(
    "unreachable",
    `Cannot reach the CSV Doctor engine at ${ENGINE_URL}. Start it (uvicorn app.main:app in services/engine) ` +
      `or set NEXT_PUBLIC_ENGINE_URL, then try again. No results are shown without a real engine response.`,
  );

async function send(path: string, body: FormData): Promise<Response> {
  try {
    return await fetch(`${ENGINE_URL}${path}`, { method: "POST", body });
  } catch {
    throw unreachable();
  }
}

async function getJson<T>(path: string): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${ENGINE_URL}${path}`);
  } catch {
    throw unreachable();
  }
  if (!res.ok) throw await httpError(res);
  return (await res.json()) as T;
}

/** FastAPI sends {"detail": "..."} for 4xx/5xx; anything else gets a generic message. */
async function httpError(res: Response): Promise<EngineError> {
  let detail = "";
  try {
    const body = await res.json();
    detail = typeof body?.detail === "string" ? body.detail : JSON.stringify(body?.detail ?? "");
  } catch {
    /* non-JSON error body */
  }
  const message = detail || `The engine answered HTTP ${res.status}.`;
  const tooBig = /^(file_too_large|too_many_rows)\b/.test(detail);
  return new EngineError(tooBig && res.status === 422 ? "too_big_for_memory" : "rejected", message, res.status);
}

function form(file: File, question?: string): FormData {
  const body = new FormData();
  body.append("file", file, file.name);
  if (question !== undefined) body.append("question", question);
  return body;
}

// ---------------------------------------------------------------- endpoints

export async function analyse(file: File): Promise<Analysis> {
  const res = await send("/analyse", form(file));
  if (!res.ok) throw await httpError(res);
  return { kind: "analysis", ...(await res.json()) };
}

export async function ask(file: File, question: string): Promise<AskResult> {
  const res = await send("/ask", form(file, question));
  if (!res.ok) throw await httpError(res);
  return (await res.json()) as AskResult;
}

interface JobRecord {
  job_id: string;
  status: string;
  percent: number;
  stage: string;
  result: Omit<ScaleReport, "kind"> | null;
  error: string | null;
}

/**
 * Scale path: POST /jobs, follow GET /jobs/{id}/events (SSE) and report each real
 * progress event, then fetch the finished record. Resolves with the job result.
 */
export async function runJob(
  file: File,
  onProgress: (p: JobProgress) => void,
  signal?: AbortSignal,
): Promise<ScaleReport> {
  const res = await send("/jobs", form(file));
  if (!res.ok) throw await httpError(res);
  const { job_id } = (await res.json()) as { job_id: string };

  await followEvents(job_id, onProgress, signal);
  const record = await getJson<JobRecord>(`/jobs/${job_id}`);
  if (record.status === "failed" || !record.result) {
    throw new EngineError("failed", record.error ?? "The job failed without an error message.");
  }
  return { kind: "scale", ...record.result };
}

const MAX_RECONNECTS = 5;

/** Resolves once the stream reports a terminal event; reconnects (checking job state) on drops. */
function followEvents(jobId: string, onProgress: (p: JobProgress) => void, signal?: AbortSignal): Promise<void> {
  return new Promise<void>((resolve, reject) => {
    let source: EventSource | null = null;
    let reconnects = 0;
    let settled = false;

    const finish = (err?: Error) => {
      if (settled) return;
      settled = true;
      source?.close();
      signal?.removeEventListener("abort", onAbort);
      err ? reject(err) : resolve();
    };
    const onAbort = () => finish(new EngineError("failed", "Cancelled."));
    signal?.addEventListener("abort", onAbort);

    const handle = (ev: MessageEvent) => {
      const data = JSON.parse(ev.data) as JobProgress;
      reconnects = 0;
      onProgress({ status: data.status, percent: data.percent, stage: data.stage });
      if (ev.type === "complete" || ev.type === "failed") finish();
    };

    const open = () => {
      source = new EventSource(`${ENGINE_URL}/jobs/${jobId}/events`);
      source.addEventListener("progress", handle as EventListener);
      source.addEventListener("complete", handle as EventListener);
      source.addEventListener("failed", handle as EventListener);
      source.onerror = async () => {
        source?.close();
        if (settled) return;
        // The stream dropped: the job may have finished meanwhile, or the engine may be gone.
        try {
          const record = await getJson<JobRecord>(`/jobs/${jobId}`);
          if (record.status === "done" || record.status === "failed") {
            onProgress({ status: record.status, percent: record.percent, stage: record.stage });
            return finish();
          }
        } catch (err) {
          return finish(err as Error);
        }
        if (++reconnects > MAX_RECONNECTS) {
          return finish(new EngineError("unreachable", "Lost the progress stream from the engine and could not resume it."));
        }
        setTimeout(() => !settled && open(), 1000);
      };
    };
    open();
  });
}
