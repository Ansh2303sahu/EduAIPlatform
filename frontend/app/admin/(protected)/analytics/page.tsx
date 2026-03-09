"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  ArrowLeft,
  BarChart3,
  CheckCircle2,
  FileText,
  RefreshCcw,
  Sparkles,
} from "lucide-react";
import {
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

import { backendUrl } from "@/lib/backendUrl";
import { fetchJsonWithAuth } from "@/lib/fetchWithAuth";

export default function AdminAnalyticsPage() {
  const [data, setData] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  async function loadData() {
    try {
      setLoading(true);
      setErr(null);
      const j = await fetchJsonWithAuth(backendUrl("/admin/analytics"), { method: "GET" });
      setData(j);
    } catch (e: any) {
      setErr(e?.message || "Failed to load analytics");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadData();
  }, []);

  const confidenceData = useMemo(() => {
    const buckets = data?.confidence_buckets || {};
    return Object.entries(buckets).map(([key, value]) => ({
      bucket: `Band ${key}`,
      value: Number(value || 0),
    }));
  }, [data]);

  const roleData = useMemo(() => {
    const roles = data?.reports_by_role || {};
    return Object.entries(roles).map(([key, value]) => ({
      name: key,
      value: Number(value || 0),
    }));
  }, [data]);

  const needsReviewPercent = Math.round((data?.needs_review_rate || 0) * 100);

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top,_rgba(59,130,246,0.16),_transparent_24%),linear-gradient(180deg,#07111f_0%,#081426_45%,#050b16_100%)] text-white">
      <div className="mx-auto max-w-[1450px] px-4 py-6 sm:px-6 lg:px-8">
        <div className="grid gap-6">
          <TopBar
            title="Admin Analytics"
            subtitle="Confidence patterns, review rates, and report distribution across the platform."
            onRefresh={loadData}
            loading={loading}
          />

          {err ? (
            <ErrorCard message={err} />
          ) : null}

          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <MetricCard
              title="Needs Review Rate"
              value={`${needsReviewPercent}%`}
              icon={<AlertTriangle className="h-5 w-5" />}
            />
            <MetricCard
              title="Student Reports"
              value={String(data?.reports_by_role?.student ?? 0)}
              icon={<FileText className="h-5 w-5" />}
            />
            <MetricCard
              title="Professor Reports"
              value={String(data?.reports_by_role?.professor ?? 0)}
              icon={<BarChart3 className="h-5 w-5" />}
            />
            <MetricCard
              title="Analytics Status"
              value={loading ? "Loading" : "Ready"}
              icon={<CheckCircle2 className="h-5 w-5" />}
            />
          </div>

          <div className="grid gap-4 xl:grid-cols-[1.3fr_1fr]">
            <CardShell
              title="Confidence Buckets"
              subtitle="ML confidence grouped into buckets 0–4"
            >
              <div className="h-[320px]">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={confidenceData}>
                    <CartesianGrid stroke="rgba(255,255,255,0.06)" vertical={false} />
                    <XAxis dataKey="bucket" stroke="#94a3b8" tickLine={false} axisLine={false} />
                    <YAxis stroke="#94a3b8" tickLine={false} axisLine={false} />
                    <Tooltip
                      contentStyle={{
                        background: "rgba(7,17,31,0.96)",
                        border: "1px solid rgba(255,255,255,0.08)",
                        borderRadius: 16,
                        color: "#fff",
                      }}
                    />
                    <Bar dataKey="value" radius={[10, 10, 0, 0]} fill="#60a5fa" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </CardShell>

            <CardShell
              title="Reports by Role"
              subtitle="Current role split"
            >
              <div className="h-[320px]">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={roleData}
                      dataKey="value"
                      nameKey="name"
                      innerRadius={65}
                      outerRadius={105}
                      paddingAngle={4}
                    >
                      {roleData.map((entry, idx) => (
                        <Cell
                          key={`${entry.name}-${idx}`}
                          fill={idx === 0 ? "#3b82f6" : "#8b5cf6"}
                        />
                      ))}
                    </Pie>
                    <Tooltip
                      contentStyle={{
                        background: "rgba(7,17,31,0.96)",
                        border: "1px solid rgba(255,255,255,0.08)",
                        borderRadius: 16,
                        color: "#fff",
                      }}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </div>

              <div className="grid gap-3 sm:grid-cols-2">
                {roleData.map((item, idx) => (
                  <LegendCard
                    key={item.name}
                    label={item.name}
                    value={item.value}
                    color={idx === 0 ? "#3b82f6" : "#8b5cf6"}
                  />
                ))}
              </div>
            </CardShell>
          </div>

          <CardShell
            title="Raw Analytics Snapshot"
            subtitle="Useful while backend analytics APIs are still evolving"
          >
            <JsonBlock value={data} />
          </CardShell>
        </div>
      </div>
    </div>
  );
}

function TopBar({
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
          <div className="mb-2 inline-flex items-center gap-2 rounded-full border border-blue-400/20 bg-blue-500/10 px-3 py-1 text-xs font-medium text-blue-200">
            <Sparkles className="h-3.5 w-3.5" />
            Admin Panel
          </div>
          <h1 className="text-3xl font-semibold tracking-tight">{title}</h1>
          <p className="mt-2 text-sm text-slate-300">{subtitle}</p>
        </div>

        <div className="flex flex-wrap items-center gap-3">
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

function CardShell({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle: string;
  children: React.ReactNode;
}) {
  return (
    <section className="rounded-[28px] border border-white/10 bg-white/[0.04] p-5 shadow-[0_18px_60px_rgba(0,0,0,0.22)] backdrop-blur-xl">
      <h3 className="text-lg font-semibold text-white">{title}</h3>
      <p className="mt-1 text-sm text-slate-400">{subtitle}</p>
      <div className="mt-4">{children}</div>
    </section>
  );
}

function LegendCard({
  label,
  value,
  color,
}: {
  label: string;
  value: number;
  color: string;
}) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-3">
      <div className="flex items-center gap-2">
        <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: color }} />
        <span className="text-sm text-slate-300 capitalize">{label}</span>
      </div>
      <div className="mt-2 text-xl font-semibold text-white">{value}</div>
    </div>
  );
}

function JsonBlock({ value }: { value: any }) {
  return (
    <pre className="overflow-x-auto rounded-2xl border border-white/10 bg-[#050b16] p-4 text-xs text-slate-300">
      {JSON.stringify(value ?? {}, null, 2)}
    </pre>
  );
}

function ErrorCard({ message }: { message: string }) {
  return (
    <div className="rounded-2xl border border-rose-400/20 bg-rose-500/10 p-4 text-rose-200">
      {message}
    </div>
  );
}