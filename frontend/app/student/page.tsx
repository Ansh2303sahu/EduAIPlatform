"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  ArrowRight,
  BadgeCheck,
  Bell,
  FileText,
  Gauge,
  Sparkles,
  Zap,
} from "lucide-react";

import AssignmentUpload from "@/components/AssignmentUpload";
import { backendUrl } from "@/lib/backendUrl";
import { fetchWithAuth } from "@/lib/fetchWithAuth";

type HistoryItem = {
  id: string;
  file_id: string;
  created_at?: string;
  report_json?: {
    summary?: string;
    model_agreement?: { final_confidence?: number };
  };
  model_versions?: {
    agreement?: {
      final_confidence?: number;
      ml_bucket_0_to_4?: number;
      injected?: boolean;
    };
  };
  needs_review?: boolean;
};

type LatestResp = {
  found: boolean;
  item?: {
    file_id: string;
    created_at?: string;
    report_json?: {
      summary?: string;
      model_agreement?: { final_confidence?: number };
    };
    model_versions?: {
      agreement?: {
        final_confidence?: number;
        ml_bucket_0_to_4?: number;
        injected?: boolean;
      };
    };
  };
};

type ResultsResp = {
  jobs?: Array<{ status: string }>;
};

function confidenceBandFrom(finalConfidence?: number) {
  if (typeof finalConfidence !== "number") return "—";
  if (finalConfidence >= 0.85) return "High";
  if (finalConfidence >= 0.65) return "Medium";
  return "Low";
}

function pct(n?: number) {
  if (typeof n !== "number") return "—";
  return `${Math.round(n * 100)}%`;
}

function fmtDate(iso?: string) {
  if (!iso) return "";
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

function shortFileId(id: string) {
  if (!id) return "—";
  if (id.length <= 14) return id;
  return `${id.slice(0, 8)}...${id.slice(-6)}`;
}

export default function StudentDashboard() {
  const [latestFileId, setLatestFileId] = useState<string | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [latest, setLatest] = useState<LatestResp | null>(null);
  const [inFlight, setInFlight] = useState<number | null>(null);

  async function loadHistory() {
    try {
      const res = await fetchWithAuth(backendUrl(`/phase7/history/student?limit=6`));
      if (!res.ok) return;

      const json = await res.json();
      const items = Array.isArray(json?.items)
        ? json.items
        : Array.isArray(json)
        ? json
        : [];

      setHistory(items);
    } catch {
      // demo safe
    }
  }

  async function loadLatestForFile(fileId: string) {
    try {
      const res = await fetchWithAuth(backendUrl(`/phase7/latest/student/${fileId}`));
      if (!res.ok) {
        setLatest({ found: false });
        return;
      }
      const json = (await res.json()) as LatestResp;
      setLatest(json);
    } catch {
      setLatest({ found: false });
    }
  }

  async function loadInFlightJobs(fileId: string) {
    try {
      const res = await fetchWithAuth(backendUrl(`/results/${fileId}`));
      if (!res.ok) {
        setInFlight(null);
        return;
      }
      const json = (await res.json()) as ResultsResp;
      const jobs = Array.isArray(json?.jobs) ? json.jobs : [];
      const n = jobs.filter((j) =>
        ["queued", "running"].includes(String(j.status).toLowerCase())
      ).length;
      setInFlight(n);
    } catch {
      setInFlight(null);
    }
  }

  useEffect(() => {
    void loadHistory();
  }, []);

  useEffect(() => {
    if (!latestFileId) return;

    void loadLatestForFile(latestFileId);
    void loadInFlightJobs(latestFileId);

    const interval = setInterval(() => {
      void Promise.all([loadLatestForFile(latestFileId), loadInFlightJobs(latestFileId)]);
    }, 2500);

    return () => clearInterval(interval);
  }, [latestFileId]);

  useEffect(() => {
    if (latest?.found) {
      void loadHistory();
    }
  }, [latest?.found]);

  const finalConfidence =
    latest?.item?.model_versions?.agreement?.final_confidence ??
    latest?.item?.report_json?.model_agreement?.final_confidence ??
    history?.[0]?.model_versions?.agreement?.final_confidence ??
    history?.[0]?.report_json?.model_agreement?.final_confidence;

  const confidenceBand = confidenceBandFrom(finalConfidence);

  const confidenceSub =
    confidenceBand === "—"
      ? "Upload an assignment to generate confidence"
      : `Final confidence ${pct(finalConfidence)}`;

  const reportReady = latest?.found === true && !!latest?.item?.file_id;

  const inFlightLabel = useMemo(() => {
    if (!latestFileId) return "Upload to begin job tracking";
    if (inFlight === null) return "Loading job state";
    if (inFlight === 0) return "All jobs finished";
    return "Jobs currently processing";
  }, [latestFileId, inFlight]);

  const totalReports = history.length;

  const avgConfidenceValue = useMemo(() => {
    const arr = history
      .map(
        (h) =>
          h?.model_versions?.agreement?.final_confidence ??
          h?.report_json?.model_agreement?.final_confidence
      )
      .filter((v): v is number => typeof v === "number");

    if (!arr.length) return undefined;
    return arr.reduce((a, b) => a + b, 0) / arr.length;
  }, [history]);

  const reviewCount = useMemo(() => {
    return history.filter((h) => h.needs_review).length;
  }, [history]);

  const confidenceTrendData = useMemo(() => {
    const source = [...history].reverse();
    return source.map((h, i) => {
      const fc =
        h?.model_versions?.agreement?.final_confidence ??
        h?.report_json?.model_agreement?.final_confidence ??
        0;

      return {
        name: `R${i + 1}`,
        confidence: Math.round(fc * 100),
      };
    });
  }, [history]);

  const jobsBarData = useMemo(() => {
    const base = latestFileId ? 1 : 0;
    return [
      { name: "Scan", value: inFlight && inFlight > 0 ? 1 : base ? 1 : 0 },
      { name: "Extract", value: latestFileId ? 1 : 0 },
      { name: "ML", value: latestFileId ? 1 : 0 },
      { name: "LLM", value: reportReady ? 1 : latestFileId ? 1 : 0 },
      { name: "Report", value: reportReady ? 1 : 0 },
    ];
  }, [latestFileId, inFlight, reportReady]);

  const donutData = useMemo(() => {
    const safe = typeof finalConfidence === "number" ? Math.round(finalConfidence * 100) : 0;
    return [
      { name: "Confidence", value: safe },
      { name: "Remaining", value: 100 - safe },
    ];
  }, [finalConfidence]);

  return (
    <div className="grid gap-6">
      <section className="rounded-[30px] border border-white/10 bg-[linear-gradient(135deg,rgba(19,33,84,0.96),rgba(8,18,50,0.96))] p-6 shadow-[0_20px_80px_rgba(0,0,0,0.28)]">
        <div className="mb-5 flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <div className="inline-flex rounded-full border border-blue-300/20 bg-white/10 px-3 py-1 text-xs font-semibold text-blue-100">
              New Assessment
            </div>
            <h2 className="mt-4 text-3xl font-black leading-tight">
              Upload and track your assignment in real time
            </h2>
            <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-300">
              Start a new analysis run, monitor each processing stage, and open feedback as soon as the final report is ready.
            </p>
          </div>

          {latestFileId && (
            <div className="flex flex-wrap items-center gap-3">
              <div className="rounded-full border border-blue-300/15 bg-white/8 px-3 py-2 text-xs text-slate-200">
                Tracking file: <span className="font-bold">{shortFileId(latestFileId)}</span>
              </div>

              {reportReady && (
                <Link
                  href={`/student/results/${latestFileId}`}
                  className="inline-flex items-center gap-2 rounded-full bg-gradient-to-r from-blue-500 to-indigo-500 px-4 py-2 text-sm font-bold text-white shadow-lg transition hover:scale-[1.02]"
                >
                  View feedback
                  <ArrowRight size={15} />
                </Link>
              )}
            </div>
          )}
        </div>

        <div className="rounded-[24px] border border-white/10 bg-white/[0.06] p-4">
          <AssignmentUpload onFileId={(id) => setLatestFileId(id)} />
        </div>
      </section>

      <section className="grid grid-cols-1 gap-6 md:grid-cols-2 xl:grid-cols-4">
        <StatCard
          title="Confidence"
          value={confidenceBand}
          sub={confidenceSub}
          accent="from-blue-500/25 to-indigo-500/10"
          icon={<BadgeCheck size={18} />}
        />
        <StatCard
          title="Jobs"
          value={inFlight === null ? "—" : String(inFlight)}
          sub={inFlightLabel}
          accent="from-cyan-500/25 to-blue-500/10"
          icon={<Zap size={18} />}
        />
        <StatCard
          title="Recent reports"
          value={String(totalReports)}
          sub="Latest student analyses"
          accent="from-indigo-500/25 to-blue-500/10"
          icon={<FileText size={18} />}
        />
        <StatCard
          title="Average confidence"
          value={pct(avgConfidenceValue)}
          sub="Based on recent reports"
          accent="from-violet-500/25 to-indigo-500/10"
          icon={<Gauge size={18} />}
        />
      </section>

      <section className="grid grid-cols-1 gap-6 xl:grid-cols-[1.2fr_0.8fr]">
        <div className="rounded-[30px] border border-white/10 bg-white/[0.05] p-5 shadow-[0_20px_80px_rgba(0,0,0,0.24)] backdrop-blur-xl">
          <div className="mb-5 flex items-center justify-between">
            <div>
              <div className="text-sm font-bold text-slate-200">Confidence trend</div>
              <div className="mt-1 text-xs text-slate-400">
                Performance trend from recent feedback reports
              </div>
            </div>
            <div className="rounded-full border border-blue-300/15 bg-blue-500/10 px-3 py-1 text-xs font-semibold text-blue-100">
              Analytics
            </div>
          </div>

          <div className="h-[280px]">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={confidenceTrendData}>
                <defs>
                  <linearGradient id="studentArea" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#4f7cff" stopOpacity={0.65} />
                    <stop offset="100%" stopColor="#4f7cff" stopOpacity={0.04} />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke="rgba(255,255,255,0.08)" vertical={false} />
                <XAxis dataKey="name" stroke="#94a3b8" tickLine={false} axisLine={false} />
                <YAxis stroke="#94a3b8" tickLine={false} axisLine={false} />
                <Tooltip
                  contentStyle={{
                    background: "#0d1736",
                    border: "1px solid rgba(255,255,255,0.08)",
                    borderRadius: 16,
                    color: "#fff",
                  }}
                />
                <Area
                  type="monotone"
                  dataKey="confidence"
                  stroke="#5c86ff"
                  strokeWidth={3}
                  fill="url(#studentArea)"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="grid gap-6">
          <div className="rounded-[30px] border border-white/10 bg-white/[0.05] p-5 shadow-[0_20px_80px_rgba(0,0,0,0.24)] backdrop-blur-xl">
            <div className="mb-4">
              <div className="text-sm font-bold text-slate-200">Confidence score</div>
              <div className="mt-1 text-xs text-slate-400">Latest agreement confidence</div>
            </div>

            <div className="h-[220px]">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={donutData}
                    innerRadius={62}
                    outerRadius={84}
                    dataKey="value"
                    stroke="none"
                  >
                    <Cell fill="#5c86ff" />
                    <Cell fill="rgba(255,255,255,0.08)" />
                  </Pie>
                  <Tooltip
                    contentStyle={{
                      background: "#0d1736",
                      border: "1px solid rgba(255,255,255,0.08)",
                      borderRadius: 16,
                      color: "#fff",
                    }}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>

            <div className="-mt-28 text-center">
              <div className="text-4xl font-black">{pct(finalConfidence)}</div>
              <div className="mt-1 text-sm text-slate-400">{confidenceBand} confidence</div>
            </div>
          </div>

          <div className="rounded-[30px] border border-white/10 bg-white/[0.05] p-5 shadow-[0_20px_80px_rgba(0,0,0,0.24)] backdrop-blur-xl">
            <div className="mb-5">
              <div className="text-sm font-bold text-slate-200">Pipeline overview</div>
              <div className="mt-1 text-xs text-slate-400">Assignment processing stages</div>
            </div>

            <div className="h-[190px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={jobsBarData}>
                  <CartesianGrid stroke="rgba(255,255,255,0.08)" vertical={false} />
                  <XAxis dataKey="name" stroke="#94a3b8" tickLine={false} axisLine={false} />
                  <YAxis hide />
                  <Tooltip
                    contentStyle={{
                      background: "#0d1736",
                      border: "1px solid rgba(255,255,255,0.08)",
                      borderRadius: 16,
                      color: "#fff",
                    }}
                  />
                  <Bar dataKey="value" radius={[10, 10, 0, 0]} fill="#87a2ff" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <StatCard
            title="Needs review"
            value={String(reviewCount)}
            sub="Reports flagged for review"
            accent="from-rose-500/15 to-violet-500/10"
            icon={<Bell size={18} />}
          />
        </div>
      </section>

      <section className="rounded-[30px] border border-white/10 bg-white/[0.05] p-5 shadow-[0_20px_80px_rgba(0,0,0,0.24)] backdrop-blur-xl">
        <div className="mb-5 flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <div className="text-xl font-black">Recent feedback</div>
            <div className="mt-1 text-sm text-slate-400">
              Open your recent AI-generated reports and summaries
            </div>
          </div>

          <div className="rounded-full border border-blue-300/15 bg-blue-500/10 px-3 py-1 text-xs font-semibold text-blue-100">
            Last {history.length} items
          </div>
        </div>

        {history.length === 0 ? (
          <div className="rounded-[24px] border border-dashed border-white/10 bg-white/[0.03] p-8 text-center text-slate-400">
            No reports yet. Upload your first assessment to populate analytics and feedback cards.
          </div>
        ) : (
          <div className="grid gap-4">
            {history.map((h) => {
              const fc =
                h?.model_versions?.agreement?.final_confidence ??
                h?.report_json?.model_agreement?.final_confidence;

              const band = confidenceBandFrom(fc);
              const summary = h?.report_json?.summary;

              return (
                <Link
                  key={h.id}
                  href={`/student/results/${h.file_id}`}
                  className="group rounded-[24px] border border-white/10 bg-[linear-gradient(180deg,rgba(255,255,255,0.045),rgba(255,255,255,0.03))] p-4 transition hover:border-blue-300/25 hover:bg-white/[0.065]"
                >
                  <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
                    <div className="min-w-0">
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="inline-flex rounded-full border border-blue-300/15 bg-blue-500/10 px-3 py-1 text-xs font-semibold text-blue-100">
                          {band}
                        </span>
                        <span className="inline-flex rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-slate-300">
                          {pct(fc)}
                        </span>
                        {h.needs_review && (
                          <span className="inline-flex rounded-full border border-amber-300/20 bg-amber-400/10 px-3 py-1 text-xs font-semibold text-amber-200">
                            Needs review
                          </span>
                        )}
                      </div>

                      <div className="mt-3 text-sm font-semibold text-white">
                        File ID: <span className="font-mono text-slate-300">{h.file_id}</span>
                      </div>

                      <div className="mt-2 max-w-4xl text-sm leading-6 text-slate-400">
                        {summary || "No summary available for this report yet."}
                      </div>
                    </div>

                    <div className="flex flex-col items-start gap-3 lg:items-end">
                      <div className="text-xs text-slate-400">{fmtDate(h.created_at)}</div>
                      <div className="inline-flex items-center gap-2 text-sm font-bold text-blue-200 transition group-hover:text-white">
                        Open results
                        <ArrowRight size={15} />
                      </div>
                    </div>
                  </div>
                </Link>
              );
            })}
          </div>
        )}
      </section>
    </div>
  );
}

function StatCard({
  title,
  value,
  sub,
  accent,
  icon,
}: {
  title: string;
  value: string;
  sub: string;
  accent: string;
  icon: React.ReactNode;
}) {
  return (
    <div className="relative overflow-hidden rounded-[28px] border border-white/10 bg-white/[0.05] p-5 shadow-[0_20px_80px_rgba(0,0,0,0.22)]">
      <div className={`absolute inset-x-0 top-0 h-24 bg-gradient-to-r ${accent} blur-2xl`} />
      <div className="relative">
        <div className="flex items-center justify-between">
          <div className="text-sm font-bold text-slate-200">{title}</div>
          <div className="grid h-10 w-10 place-items-center rounded-2xl bg-white/10 text-blue-200">
            {icon}
          </div>
        </div>
        <div className="mt-5 text-4xl font-black">{value}</div>
        <div className="mt-2 text-sm text-slate-400">{sub}</div>
      </div>
    </div>
  );
}