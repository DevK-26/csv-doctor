"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import UploadPanel from "../../components/UploadPanel";
import HealthSection from "../../components/report/HealthSection";
import DriftChart from "../../components/report/DriftChart";
import AnomalyTable from "../../components/report/AnomalyTable";
import DedupeClusters from "../../components/report/DedupeClusters";
import Summary from "../../components/report/Summary";
import AskBox from "../../components/report/AskBox";
import {
  analyse, runJob, EngineError, IN_MEMORY_LIMIT_BYTES,
  type JobProgress, type Report,
} from "../../lib/engine";
import { takePendingFile } from "../../lib/pendingFile";
import { int } from "../../lib/format";

type Phase =
  | { name: "idle" }
  | { name: "analysing"; fileName: string }
  | { name: "job"; fileName: string; note: string; progress: JobProgress | null }
  | { name: "done"; report: Report }
  | { name: "error"; message: string };

const mb = (bytes: number) => (bytes / (1024 * 1024)).toFixed(1);

export default function ReportPage() {
  const [phase, setPhase] = useState<Phase>({ name: "idle" });
  const [file, setFile] = useState<File | null>(null);   // kept in memory for Ask-Your-Data
  const abort = useRef<AbortController | null>(null);
  const started = useRef(false);

  const startJob = useCallback(async (f: File, note: string, signal: AbortSignal) => {
    setPhase({ name: "job", fileName: f.name, note, progress: null });
    const report = await runJob(f, (progress) =>
      setPhase((p) => (p.name === "job" ? { ...p, progress } : p)), signal);
    setPhase({ name: "done", report });
  }, []);

  const run = useCallback(async (f: File) => {
    abort.current?.abort();
    const ctl = new AbortController();
    abort.current = ctl;
    setFile(f);
    try {
      if (f.size > IN_MEMORY_LIMIT_BYTES) {
        await startJob(f, `${f.name} is ${mb(f.size)} MB, over the ${mb(IN_MEMORY_LIMIT_BYTES)} MB limit of the in-memory path.`, ctl.signal);
        return;
      }
      setPhase({ name: "analysing", fileName: f.name });
      try {
        setPhase({ name: "done", report: await analyse(f) });
      } catch (e) {
        if (!(e instanceof EngineError && e.kind === "too_big_for_memory")) throw e;
        await startJob(f, `The engine refused the in-memory path: ${e.message}`, ctl.signal);
      }
    } catch (e) {
      if (ctl.signal.aborted) return;
      setPhase({ name: "error", message: e instanceof Error ? e.message : "Unexpected error." });
    }
  }, [startJob]);

  useEffect(() => {
    if (started.current) return;      // React strict mode runs effects twice in dev
    started.current = true;
    const pending = takePendingFile();
    if (pending) void run(pending);
  }, [run]);

  const busy = phase.name === "analysing" || phase.name === "job";

  return (
    <main className="wrap">
      <p><Link href="/">← CSV Doctor</Link></p>
      <h1 style={{ fontSize: 34 }}>Report</h1>

      {(phase.name === "idle" || phase.name === "error") && (
        <div className="narrow">
          {phase.name === "error" && <div className="alert err" role="alert"><b>Could not analyse the file.</b> {phase.message}</div>}
          {phase.name === "idle" && <p className="lede">Choose a CSV to analyse.</p>}
          <UploadPanel onFile={run} />
        </div>
      )}

      {phase.name === "analysing" && (
        <div className="alert info" role="status">Analysing {phase.fileName}… the engine is profiling, scoring and checking for anomalies, drift and duplicates.</div>
      )}

      {phase.name === "job" && (
        <div className="card" role="status">
          <p style={{ marginTop: 0 }}><b>{phase.fileName}</b> is being processed on the async scale path.</p>
          <p className="muted">{phase.note}</p>
          {phase.progress ? (
            <>
              <div className="progress" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={phase.progress.percent}>
                <div style={{ width: `${phase.progress.percent}%` }} />
              </div>
              <p style={{ marginBottom: 0 }}>{phase.progress.percent}% · {phase.progress.stage}</p>
            </>
          ) : (
            <p style={{ marginBottom: 0 }}>Sending the file and waiting for the engine to accept the job… (no percentage is shown until the engine reports real progress)</p>
          )}
        </div>
      )}

      {phase.name === "done" && (
        <>
          <div className="row" style={{ justifyContent: "space-between" }}>
            <p className="muted" style={{ margin: 0 }}>
              {file?.name} · engine {phase.report.engine_version} · file hash {phase.report.file_hash.slice(0, 12)}…
            </p>
            <button className="btn secondary" onClick={() => { abort.current?.abort(); setPhase({ name: "idle" }); }}>
              Analyse another file
            </button>
          </div>

          {phase.report.kind === "analysis" ? (
            <>
              <Summary analysis={phase.report} />
              <HealthSection report={phase.report} />
              <DriftChart drift={phase.report.drift} />
              <AnomalyTable anomalies={phase.report.anomalies} />
              <DedupeClusters relations={phase.report.relations} />
              <AskBox file={file} columns={phase.report.profile.columns} />
            </>
          ) : (
            <>
              <div className="alert info">
                This file went through the scale path ({int(phase.report.scale.chunks)} chunk(s) of up to{" "}
                {int(phase.report.scale.chunk_rows)} rows). That path computes the profile and Health Score only;
                anomaly detection, drift, duplicate clusters and the summary run on files up to 10 MB / 50,000 rows.
              </div>
              <HealthSection report={phase.report} />
              <AskBox file={file} columns={phase.report.profile.columns}
                unavailable="Ask-Your-Data loads the whole file into memory, which the engine limits to 10 MB / 50,000 rows, so it is not available for this file." />
            </>
          )}
        </>
      )}

      {busy && <p className="muted" style={{ marginTop: 20 }}>Your file is sent only to the engine at the configured address; nothing is mocked.</p>}
    </main>
  );
}
