// Day 1 landing skeleton. Drop zone is wired on Day 7-9 (see BLUEPRINT §9).
// For now this page's job is to state the promise clearly and link samples.
export default function Home() {
  return (
    <main style={{ maxWidth: 720, margin: "0 auto", padding: "64px 20px" }}>
      <h1 style={{ fontSize: 44, lineHeight: 1.1 }}>CSV Doctor</h1>
      <p style={{ fontSize: 20, color: "#444" }}>
        Drop any CSV. In 30 seconds, know what&apos;s <b>wrong</b>, what&apos;s{" "}
        <b>weird</b>, and what&apos;s <b>interesting</b> in your data — in plain English.
      </p>
      <div
        style={{
          border: "2px dashed #999", borderRadius: 12, padding: 40,
          textAlign: "center", color: "#666", margin: "32px 0",
        }}
      >
        Drop your .csv here (Day 7-9: upload wiring)
        <br />
        <small>Max 10 MB / 50,000 rows · No signup · File auto-deleted within 1 hour</small>
      </div>
      <p>
        No file handy? Try a sample: <b>Sales</b> · <b>Survey</b> · <b>Website analytics</b>{" "}
        (in <code>/samples</code>)
      </p>
      <p style={{ color: "#666", fontSize: 14 }}>
        Real ML runs on your full file — profiling, anomaly detection, drift
        detection. A free AI only explains the computed facts. It never sees
        your raw data, and it never invents a number.
      </p>
    </main>
  );
}
