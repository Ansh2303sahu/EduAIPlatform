"use client";

type StepStatus = "pending" | "running" | "done" | "failed";

export type Step = {
  key: "scan" | "extract" | "ml" | "llm" | "report";
  label: string;
  status: StepStatus;
  message?: string;
  at?: string | null;
};

function iconFor(s: StepStatus) {
  if (s === "done") return "✅";
  if (s === "failed") return "❌";
  if (s === "running") return "⏳";
  return "•";
}

export default function ProgressStepper({ steps }: { steps: Step[] }) {
  return (
    <div style={{ display: "grid", gap: 10 }}>
      {steps.map((st) => (
        <div
          key={st.key}
          style={{
            display: "grid",
            gridTemplateColumns: "28px 1fr",
            gap: 10,
            padding: 12,
            border: "1px solid #e5e5e5",
            borderRadius: 10,
            background: "#fff",
          }}
        >
          <div style={{ fontSize: 18, lineHeight: "18px" }}>{iconFor(st.status)}</div>
          <div style={{ display: "grid", gap: 4 }}>
            <div style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
              <b>{st.label}</b>
              <span style={{ fontSize: 12, opacity: 0.8 }}>{st.status}</span>
            </div>
            {st.message ? <div style={{ fontSize: 13 }}>{st.message}</div> : null}
            {st.at ? <div style={{ fontSize: 12, opacity: 0.7 }}>{st.at}</div> : null}
          </div>
        </div>
      ))}
    </div>
  );
}