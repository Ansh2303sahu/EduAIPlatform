"use client";

export default function ReportPanel({ title, report }: { title: string; report: any }) {
  if (!report) return null;

  return (
    <div style={{ marginTop: 16, border: "1px solid #ddd", borderRadius: 10, padding: 16 }}>
      <h3 style={{ marginTop: 0 }}>{title}</h3>

      {"summary" in report && (
        <>
          <h4>Summary</h4>
          <p>{report.summary || "—"}</p>

          <h4>Issues</h4>
          <ul>
            {(report.issues || []).map((x: string, i: number) => <li key={i}>{x}</li>)}
          </ul>

          <h4>Checklist</h4>
          <ul>
            {(report.checklist || []).map((x: string, i: number) => <li key={i}>{x}</li>)}
          </ul>
        </>
      )}

      {"rubric_breakdown" in report && (
        <>
          <h4>Rubric breakdown</h4>
          <pre style={{ whiteSpace: "pre-wrap" }}>{JSON.stringify(report.rubric_breakdown, null, 2)}</pre>

          <h4>Moderation notes</h4>
          <ul>
            {(report.moderation_notes || []).map((x: string, i: number) => <li key={i}>{x}</li>)}
          </ul>
        </>
      )}
    </div>
  );
}