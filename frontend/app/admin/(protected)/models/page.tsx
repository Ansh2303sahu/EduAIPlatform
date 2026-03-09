"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { ArrowLeft, Bot, Cpu, RefreshCcw, Workflow } from "lucide-react";

import { backendUrl } from "@/lib/backendUrl";
import { fetchJsonWithAuth } from "@/lib/fetchWithAuth";

export default function AdminModelsPage() {
  const [data, setData] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  async function loadData() {
    try {
      setLoading(true);
      setErr(null);
      const j = await fetchJsonWithAuth(backendUrl("/admin/models?limit=50"), { method: "GET" });
      setData(j);
    } catch (e: any) {
      setErr(e?.message || "Failed to load models");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadData();
  }, []);

  return (
    <div className="min-h-screen bg-[linear-gradient(180deg,#07111f_0%,#081426_45%,#050b16_100%)] text-white">
      <div className="mx-auto max-w-[1450px] px-4 py-6 sm:px-6 lg:px-8">
        <div className="grid gap-6">
          <Header
            title="Model Registry"
            subtitle="Inspect latest student and professor model versions, plus recent AI report runs."
            onRefresh={loadData}
            loading={loading}
          />

          {err ? (
            <div className="rounded-2xl border border-rose-400/20 bg-rose-500/10 p-4 text-rose-200">
              {err}
            </div>
          ) : null}

          <div className="grid gap-4 xl:grid-cols-2">
            <InfoCard title="Latest Student Model" icon={<Bot className="h-5 w-5" />}>
              <JsonBlock value={data?.latest_student || {}} />
            </InfoCard>

            <InfoCard title="Latest Professor Model" icon={<Cpu className="h-5 w-5" />}>
              <JsonBlock value={data?.latest_professor || {}} />
            </InfoCard>
          </div>

          <section className="rounded-[28px] border border-white/10 bg-white/[0.04] p-5 shadow-[0_18px_60px_rgba(0,0,0,0.22)] backdrop-blur-xl">
            <div className="mb-4 flex items-center gap-3">
              <div className="rounded-2xl border border-white/10 bg-white/10 p-3 text-slate-100">
                <Workflow className="h-5 w-5" />
              </div>
              <div>
                <h3 className="text-lg font-semibold text-white">Recent Runs</h3>
                <p className="text-sm text-slate-400">Latest model-linked report records</p>
              </div>
            </div>

            <div className="grid gap-4">
              {(data?.items || []).length ? (
                (data.items || []).map((x: any) => (
                  <div
                    key={x.id}
                    className="rounded-2xl border border-white/10 bg-white/[0.03] p-4"
                  >
                    <div className="mb-3 flex flex-wrap gap-3 text-xs text-slate-400">
                      <span>{x.created_at || "—"}</span>
                      <span>role={x.role || "—"}</span>
                      <span>file_id={x.file_id || "—"}</span>
                    </div>

                    <pre className="overflow-x-auto rounded-xl border border-white/10 bg-[#050b16] p-3 text-xs text-slate-300">
                      {JSON.stringify(x.model_versions || {}, null, 2)}
                    </pre>
                  </div>
                ))
              ) : (
                <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-4 text-sm text-slate-400">
                  {loading ? "Loading runs..." : "No ai_reports yet."}
                </div>
              )}
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}

function Header({
  title,
  subtitle,
  onRefresh,
  loading,
}: {
  title: string;
  subtitle: string;
  onRefresh: () => void;
  loading: boolean;
}) {
  return (
    <section className="rounded-[30px] border border-white/10 bg-white/[0.04] p-5 shadow-[0_20px_80px_rgba(0,0,0,0.35)] backdrop-blur-xl">
      <div className="flex flex-col gap-4 xl:flex-row xl:items-center xl:justify-between">
        <div>
          <h1 className="text-3xl font-semibold tracking-tight">{title}</h1>
          <p className="mt-2 text-sm text-slate-300">{subtitle}</p>
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
            onClick={onRefresh}
            className="inline-flex items-center gap-2 rounded-2xl border border-white/10 bg-white/5 px-4 py-2 text-sm text-slate-200 transition hover:bg-white/10"
          >
            <RefreshCcw className="h-4 w-4" />
            {loading ? "Refreshing..." : "Refresh"}
          </button>
        </div>
      </div>
    </section>
  );
}

function InfoCard({
  title,
  icon,
  children,
}: {
  title: string;
  icon: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <section className="rounded-[28px] border border-white/10 bg-white/[0.04] p-5 shadow-[0_18px_60px_rgba(0,0,0,0.22)] backdrop-blur-xl">
      <div className="mb-4 flex items-center gap-3">
        <div className="rounded-2xl border border-white/10 bg-white/10 p-3 text-slate-100">
          {icon}
        </div>
        <h3 className="text-lg font-semibold text-white">{title}</h3>
      </div>
      {children}
    </section>
  );
}

function JsonBlock({ value }: { value: any }) {
  return (
    <pre className="overflow-x-auto rounded-2xl border border-white/10 bg-[#050b16] p-4 text-xs text-slate-300">
      {JSON.stringify(value ?? {}, null, 2)}
    </pre>
  );
}