"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import {
  ArrowLeft,
  Clock3,
  RefreshCcw,
  ServerCog,
  TriangleAlert,
  Workflow,
} from "lucide-react";

import { backendUrl } from "@/lib/backendUrl";
import { fetchJsonWithAuth } from "@/lib/fetchWithAuth";

export default function AdminWorkersPage() {
  const [data, setData] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  async function loadData() {
    try {
      setLoading(true);
      setErr(null);
      const j = await fetchJsonWithAuth(backendUrl("/admin/workers"), { method: "GET" });
      setData(j);
    } catch (e: any) {
      setErr(e?.message || "Failed to load workers/jobs");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadData();
  }, []);

  const counts = data?.counts || {};
  const countEntries = useMemo(() => Object.entries(counts), [counts]);

  return (
    <div className="min-h-screen bg-[linear-gradient(180deg,#07111f_0%,#081426_45%,#050b16_100%)] text-white">
      <div className="mx-auto max-w-[1450px] px-4 py-6 sm:px-6 lg:px-8">
        <div className="grid gap-6">
          <section className="rounded-[30px] border border-white/10 bg-white/[0.04] p-5 shadow-[0_20px_80px_rgba(0,0,0,0.35)] backdrop-blur-xl">
            <div className="flex flex-col gap-4 xl:flex-row xl:items-center xl:justify-between">
              <div>
                <div className="mb-2 inline-flex items-center gap-2 rounded-full border border-cyan-400/20 bg-cyan-500/10 px-3 py-1 text-xs font-medium text-cyan-200">
                  <Workflow className="h-3.5 w-3.5" />
                  Job Monitoring
                </div>
                <h1 className="text-3xl font-semibold tracking-tight">Workers / Jobs</h1>
                <p className="mt-2 text-sm text-slate-300">
                  Monitor worker activity, counts, and recent job records from the backend pipeline.
                </p>
              </div>

              <div className="flex gap-3">
                <Link
                  href="/admin"
                  className="inline-flex items-center gap-2 rounded-2xl border border-white/10 bg-white/5 px-4 py-2 text-sm text-slate-200 transition hover:bg-white/10"
                >
                  <ArrowLeft className="h-4 w-4" />
                  Back
                </Link>
                <button
                  onClick={loadData}
                  className="inline-flex items-center gap-2 rounded-2xl border border-white/10 bg-white/5 px-4 py-2 text-sm text-slate-200 transition hover:bg-white/10"
                >
                  <RefreshCcw className="h-4 w-4" />
                  {loading ? "Refreshing..." : "Refresh"}
                </button>
              </div>
            </div>
          </section>

          {err ? (
            <div className="rounded-2xl border border-rose-400/20 bg-rose-500/10 p-4 text-rose-200">
              {err}
            </div>
          ) : null}

          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <MetricCard
              title="Job Types"
              value={String(countEntries.length)}
              icon={<ServerCog className="h-5 w-5" />}
            />
            <MetricCard
              title="Recent Jobs"
              value={String(data?.recent_jobs?.length ?? 0)}
              icon={<Clock3 className="h-5 w-5" />}
            />
            <MetricCard
              title="Pipeline Health"
              value={(data?.recent_jobs?.length ?? 0) > 0 ? "Active" : "Idle"}
              icon={<Workflow className="h-5 w-5" />}
            />
            <MetricCard
              title="Attention"
              value={err ? "Issue" : "OK"}
              icon={<TriangleAlert className="h-5 w-5" />}
            />
          </div>

          <div className="grid gap-4 xl:grid-cols-[0.95fr_1.05fr]">
            <section className="rounded-[28px] border border-white/10 bg-white/[0.04] p-5 shadow-[0_18px_60px_rgba(0,0,0,0.22)] backdrop-blur-xl">
              <h3 className="text-lg font-semibold text-white">Counts (24h)</h3>
              <p className="mt-1 text-sm text-slate-400">
                Window: {data?.window?.since || "—"} → {data?.window?.now || "—"}
              </p>

              <div className="mt-4 grid gap-3">
                {countEntries.length ? (
                  countEntries.map(([key, value]) => (
                    <div
                      key={String(key)}
                      className="flex items-center justify-between rounded-2xl border border-white/10 bg-white/[0.03] px-4 py-3"
                    >
                      <span className="text-sm text-slate-300 capitalize">{String(key)}</span>
                      <span className="text-lg font-semibold text-white">{String(value)}</span>
                    </div>
                  ))
                ) : (
                  <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-4 text-sm text-slate-400">
                    {loading ? "Loading counts..." : "No counts available."}
                  </div>
                )}
              </div>
            </section>

            <section className="rounded-[28px] border border-white/10 bg-white/[0.04] p-5 shadow-[0_18px_60px_rgba(0,0,0,0.22)] backdrop-blur-xl">
              <h3 className="text-lg font-semibold text-white">Recent Jobs</h3>
              <p className="mt-1 text-sm text-slate-400">Latest worker and job records</p>

              <div className="mt-4 grid gap-4">
                {(data?.recent_jobs || []).length ? (
                  (data.recent_jobs || []).map((job: any, idx: number) => (
                    <div
                      key={job?.id || idx}
                      className="rounded-2xl border border-white/10 bg-white/[0.03] p-4"
                    >
                      <pre className="overflow-x-auto rounded-xl border border-white/10 bg-[#050b16] p-3 text-xs text-slate-300">
                        {JSON.stringify(job, null, 2)}
                      </pre>
                    </div>
                  ))
                ) : (
                  <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-4 text-sm text-slate-400">
                    {loading ? "Loading jobs..." : "No recent jobs."}
                  </div>
                )}
              </div>
            </section>
          </div>
        </div>
      </div>
    </div>
  );
}

function MetricCard({
  title,
  value,
  icon,
}: {
  title: string;
  value: string;
  icon: React.ReactNode;
}) {
  return (
    <div className="rounded-[24px] border border-white/10 bg-white/[0.04] p-5 shadow-[0_18px_60px_rgba(0,0,0,0.22)] backdrop-blur-xl">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-sm text-slate-400">{title}</div>
          <div className="mt-3 text-3xl font-bold text-white">{value}</div>
        </div>
        <div className="rounded-2xl border border-white/10 bg-white/10 p-3 text-slate-100">
          {icon}
        </div>
      </div>
    </div>
  );
}