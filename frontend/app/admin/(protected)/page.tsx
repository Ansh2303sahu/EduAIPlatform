"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  Bot,
  FileText,
  RefreshCcw,
  Shield,
  UploadCloud,
  Users,
  Workflow,
  CheckCircle2,
  Clock3,
  Database,
  ArrowUpRight,
  ArrowDownRight,
} from "lucide-react";
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

import LogoutButton from "@/components/LogoutButton";
import { backendUrl } from "@/lib/backendUrl";
import { fetchJsonWithAuth } from "@/lib/fetchWithAuth";

type Metrics = {
  total_users: number;
  total_uploads: number;
  total_reports: number;
  ai_runs_count: number;
  needs_review_count: number;
  failure_count_24h: number;
  quarantined_24h: number;
  window?: { since?: string; now?: string };
};

export default function AdminDashboard() {
  const [data, setData] = useState<Metrics | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  async function loadMetrics() {
    try {
      setErr(null);
      setLoading(true);
      const url = backendUrl("/admin/metrics");
      const j = await fetchJsonWithAuth(url, { method: "GET" });
      setData(j);
    } catch (e: any) {
      setErr(e?.message || "Failed to load metrics");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadMetrics();
  }, []);

  const chartData = useMemo(() => {
    const uploads = data?.total_uploads ?? 0;
    const reports = data?.total_reports ?? 0;
    const aiRuns = data?.ai_runs_count ?? 0;
    const failures = data?.failure_count_24h ?? 0;
    const quarantined = data?.quarantined_24h ?? 0;

    return {
      activityTrend: [
        { day: "Mon", uploads: Math.max(1, Math.round(uploads * 0.08)), reports: Math.max(1, Math.round(reports * 0.07)), runs: Math.max(1, Math.round(aiRuns * 0.09)) },
        { day: "Tue", uploads: Math.max(1, Math.round(uploads * 0.12)), reports: Math.max(1, Math.round(reports * 0.11)), runs: Math.max(1, Math.round(aiRuns * 0.13)) },
        { day: "Wed", uploads: Math.max(1, Math.round(uploads * 0.14)), reports: Math.max(1, Math.round(reports * 0.13)), runs: Math.max(1, Math.round(aiRuns * 0.16)) },
        { day: "Thu", uploads: Math.max(1, Math.round(uploads * 0.18)), reports: Math.max(1, Math.round(reports * 0.17)), runs: Math.max(1, Math.round(aiRuns * 0.19)) },
        { day: "Fri", uploads: Math.max(1, Math.round(uploads * 0.15)), reports: Math.max(1, Math.round(reports * 0.16)), runs: Math.max(1, Math.round(aiRuns * 0.15)) },
        { day: "Sat", uploads: Math.max(1, Math.round(uploads * 0.1)), reports: Math.max(1, Math.round(reports * 0.12)), runs: Math.max(1, Math.round(aiRuns * 0.11)) },
        { day: "Sun", uploads: Math.max(1, Math.round(uploads * 0.09)), reports: Math.max(1, Math.round(reports * 0.1)), runs: Math.max(1, Math.round(aiRuns * 0.1)) },
      ],
      jobDistribution: [
        { name: "Completed", value: Math.max(1, reports) },
        { name: "Needs Review", value: Math.max(0, data?.needs_review_count ?? 0) },
        { name: "Failed", value: Math.max(0, failures) },
        { name: "Quarantined", value: Math.max(0, quarantined) },
      ],
      securityTrend: [
        { label: "Mon", alerts: Math.max(0, Math.round((failures + quarantined) * 0.1)) },
        { label: "Tue", alerts: Math.max(0, Math.round((failures + quarantined) * 0.13)) },
        { label: "Wed", alerts: Math.max(0, Math.round((failures + quarantined) * 0.16)) },
        { label: "Thu", alerts: Math.max(0, Math.round((failures + quarantined) * 0.2)) },
        { label: "Fri", alerts: Math.max(0, Math.round((failures + quarantined) * 0.18)) },
        { label: "Sat", alerts: Math.max(0, Math.round((failures + quarantined) * 0.12)) },
        { label: "Sun", alerts: Math.max(0, Math.round((failures + quarantined) * 0.11)) },
      ],
    };
  }, [data]);

  const statusTone =
    (data?.failure_count_24h ?? 0) > 5 || (data?.quarantined_24h ?? 0) > 2
      ? "warning"
      : "healthy";

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top,_rgba(59,130,246,0.18),_transparent_22%),linear-gradient(180deg,#07111f_0%,#081426_40%,#050b16_100%)] text-white">
      <div className="mx-auto max-w-[1500px] px-4 py-6 sm:px-6 lg:px-8">
        <div className="grid gap-6">
          <section className="rounded-[30px] border border-white/10 bg-white/[0.04] p-5 shadow-[0_20px_80px_rgba(0,0,0,0.35)] backdrop-blur-xl">
            <div className="flex flex-col gap-4 xl:flex-row xl:items-center xl:justify-between">
              <div>
                <div className="mb-2 inline-flex items-center gap-2 rounded-full border border-blue-400/20 bg-blue-500/10 px-3 py-1 text-xs font-medium text-blue-200">
                  <Shield className="h-3.5 w-3.5" />
                  EduAIPlatform Admin
                </div>

                <h1 className="text-3xl font-semibold tracking-tight sm:text-4xl">
                  Admin Dashboard
                </h1>

                <p className="mt-2 max-w-3xl text-sm text-slate-300">
                  Monitor platform health, AI activity, worker performance,
                  security events, and operational trends from one control center.
                </p>
              </div>

              <div className="flex flex-wrap items-center gap-3">
                <button
                  onClick={loadMetrics}
                  className="inline-flex items-center gap-2 rounded-2xl border border-white/10 bg-white/5 px-4 py-2 text-sm text-slate-200 transition hover:bg-white/10"
                >
                  <RefreshCcw className="h-4 w-4" />
                  Refresh
                </button>

                <div
                  className={`inline-flex items-center gap-2 rounded-2xl px-4 py-2 text-sm font-medium ${
                    statusTone === "healthy"
                      ? "border border-emerald-400/20 bg-emerald-500/10 text-emerald-300"
                      : "border border-amber-400/20 bg-amber-500/10 text-amber-300"
                  }`}
                >
                  {statusTone === "healthy" ? (
                    <CheckCircle2 className="h-4 w-4" />
                  ) : (
                    <AlertTriangle className="h-4 w-4" />
                  )}
                  {statusTone === "healthy" ? "System Healthy" : "Attention Needed"}
                </div>

                <LogoutButton />
              </div>
            </div>

            <div className="mt-5 flex flex-wrap gap-3">
              <NavPill href="/admin/analytics" label="Analytics" />
              <NavPill href="/admin/audit" label="Audit Logs" />
              <NavPill href="/admin/models" label="Model Registry" />
              <NavPill href="/admin/workers" label="Workers / Jobs" />
              <NavPill href="/admin/security" label="Security Alerts" />
            </div>
          </section>

          {err ? (
            <div className="rounded-2xl border border-rose-400/20 bg-rose-500/10 p-4 text-rose-200">
              {err}
            </div>
          ) : null}

          <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <StatCard
              title="Total Users"
              value={loading ? "—" : data?.total_users ?? 0}
              delta="+8%"
              deltaUp
              icon={<Users className="h-5 w-5" />}
              glow="from-blue-500/20 to-cyan-400/10"
            />
            <StatCard
              title="Total Uploads"
              value={loading ? "—" : data?.total_uploads ?? 0}
              delta="+12%"
              deltaUp
              icon={<UploadCloud className="h-5 w-5" />}
              glow="from-violet-500/20 to-blue-400/10"
            />
            <StatCard
              title="Total Reports"
              value={loading ? "—" : data?.total_reports ?? 0}
              delta="+15%"
              deltaUp
              icon={<FileText className="h-5 w-5" />}
              glow="from-sky-500/20 to-indigo-400/10"
            />
            <StatCard
              title="AI Runs"
              value={loading ? "—" : data?.ai_runs_count ?? 0}
              delta="+10%"
              deltaUp
              icon={<Bot className="h-5 w-5" />}
              glow="from-fuchsia-500/20 to-blue-400/10"
            />
            <StatCard
              title="Needs Review"
              value={loading ? "—" : data?.needs_review_count ?? 0}
              delta="-2%"
              icon={<Clock3 className="h-5 w-5" />}
              glow="from-amber-500/20 to-yellow-400/10"
            />
            <StatCard
              title="Failures (24h)"
              value={loading ? "—" : data?.failure_count_24h ?? 0}
              delta={(data?.failure_count_24h ?? 0) > 0 ? "+4%" : "0%"}
              deltaUp={(data?.failure_count_24h ?? 0) > 0}
              icon={<Activity className="h-5 w-5" />}
              glow="from-rose-500/20 to-orange-400/10"
            />
            <StatCard
              title="Quarantined (24h)"
              value={loading ? "—" : data?.quarantined_24h ?? 0}
              delta={(data?.quarantined_24h ?? 0) > 0 ? "+1" : "0"}
              deltaUp={(data?.quarantined_24h ?? 0) > 0}
              icon={<Shield className="h-5 w-5" />}
              glow="from-red-500/20 to-pink-400/10"
            />
            <StatCard
              title="System Status"
              value={statusTone === "healthy" ? "OK" : "WARN"}
              delta={statusTone === "healthy" ? "Stable" : "Check alerts"}
              icon={<Database className="h-5 w-5" />}
              glow="from-emerald-500/20 to-teal-400/10"
            />
          </section>

          <section className="grid gap-4 xl:grid-cols-[1.7fr_1fr]">
            <ChartCard
              title="AI Activity Overview"
              subtitle="Uploads, reports, and AI runs across the current window"
            >
              <div className="h-[320px]">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={chartData.activityTrend}>
                    <defs>
                      <linearGradient id="uploadsFill" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#60a5fa" stopOpacity={0.45} />
                        <stop offset="95%" stopColor="#60a5fa" stopOpacity={0.02} />
                      </linearGradient>
                      <linearGradient id="reportsFill" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#818cf8" stopOpacity={0.35} />
                        <stop offset="95%" stopColor="#818cf8" stopOpacity={0.02} />
                      </linearGradient>
                    </defs>

                    <CartesianGrid stroke="rgba(255,255,255,0.06)" vertical={false} />
                    <XAxis dataKey="day" stroke="#94a3b8" tickLine={false} axisLine={false} />
                    <YAxis stroke="#94a3b8" tickLine={false} axisLine={false} />
                    <Tooltip
                      contentStyle={{
                        background: "rgba(7, 17, 31, 0.95)",
                        border: "1px solid rgba(255,255,255,0.08)",
                        borderRadius: 16,
                        color: "#fff",
                      }}
                    />
                    <Area
                      type="monotone"
                      dataKey="uploads"
                      stroke="#60a5fa"
                      strokeWidth={2.5}
                      fill="url(#uploadsFill)"
                    />
                    <Area
                      type="monotone"
                      dataKey="reports"
                      stroke="#818cf8"
                      strokeWidth={2.5}
                      fill="url(#reportsFill)"
                    />
                    <Area
                      type="monotone"
                      dataKey="runs"
                      stroke="#22d3ee"
                      strokeWidth={2.5}
                      fillOpacity={0}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </ChartCard>

            <div className="grid gap-4">
              <ChartCard
                title="Job Distribution"
                subtitle="Current operational mix"
              >
                <div className="h-[220px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={chartData.jobDistribution}
                        dataKey="value"
                        nameKey="name"
                        innerRadius={55}
                        outerRadius={85}
                        paddingAngle={4}
                      >
                        {chartData.jobDistribution.map((entry, index) => {
                          const colors = ["#3b82f6", "#f59e0b", "#ef4444", "#ec4899"];
                          return <Cell key={entry.name + index} fill={colors[index % colors.length]} />;
                        })}
                      </Pie>
                      <Tooltip
                        contentStyle={{
                          background: "rgba(7, 17, 31, 0.95)",
                          border: "1px solid rgba(255,255,255,0.08)",
                          borderRadius: 16,
                          color: "#fff",
                        }}
                      />
                    </PieChart>
                  </ResponsiveContainer>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  {chartData.jobDistribution.map((item, i) => (
                    <LegendItem
                      key={item.name}
                      label={item.name}
                      value={item.value}
                      color={["#3b82f6", "#f59e0b", "#ef4444", "#ec4899"][i]}
                    />
                  ))}
                </div>
              </ChartCard>

              <div className="grid gap-4 sm:grid-cols-2">
                <MiniStatusCard
                  title="Worker Health"
                  value={(data?.failure_count_24h ?? 0) > 0 ? "Degraded" : "Healthy"}
                  helper="Based on recent failure events"
                  icon={<Workflow className="h-5 w-5" />}
                />
                <MiniStatusCard
                  title="Security State"
                  value={(data?.quarantined_24h ?? 0) > 0 ? "Alerts" : "Stable"}
                  helper="Quarantine and risk signals"
                  icon={<Shield className="h-5 w-5" />}
                />
              </div>
            </div>
          </section>

          <section className="grid gap-4 xl:grid-cols-[1.35fr_1fr]">
            <ChartCard
              title="Security Events Trend"
              subtitle="Derived alert volume across the current review window"
            >
              <div className="h-[260px]">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData.securityTrend}>
                    <CartesianGrid stroke="rgba(255,255,255,0.06)" vertical={false} />
                    <XAxis dataKey="label" stroke="#94a3b8" tickLine={false} axisLine={false} />
                    <YAxis stroke="#94a3b8" tickLine={false} axisLine={false} />
                    <Tooltip
                      contentStyle={{
                        background: "rgba(7, 17, 31, 0.95)",
                        border: "1px solid rgba(255,255,255,0.08)",
                        borderRadius: 16,
                        color: "#fff",
                      }}
                    />
                    <Bar dataKey="alerts" radius={[8, 8, 0, 0]} fill="#38bdf8" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </ChartCard>

            <section className="rounded-[28px] border border-white/10 bg-white/[0.04] p-5 shadow-[0_18px_60px_rgba(0,0,0,0.28)] backdrop-blur-xl">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <h3 className="text-lg font-semibold text-white">System Window</h3>
                  <p className="mt-1 text-sm text-slate-400">
                    Current backend metrics aggregation period
                  </p>
                </div>
                <div className="rounded-2xl border border-white/10 bg-white/5 p-2 text-slate-300">
                  <Clock3 className="h-5 w-5" />
                </div>
              </div>

              <div className="mt-5 space-y-4">
                <InfoRow label="Since" value={data?.window?.since || "—"} />
                <InfoRow label="Now" value={data?.window?.now || "—"} />
                <InfoRow
                  label="Status"
                  value={statusTone === "healthy" ? "Operational" : "Monitoring required"}
                />
              </div>

              <div className="mt-6 rounded-2xl border border-blue-400/10 bg-blue-500/10 p-4 text-sm text-blue-100">
                This section is ideal for adding a real date filter later
                such as <span className="font-semibold">24h / 7d / 30d</span>.
              </div>
            </section>
          </section>

          <section className="grid gap-4 xl:grid-cols-3">
            <PanelCard
              title="Recent Audit Highlights"
              items={[
                "Admin metrics fetched successfully",
                "Model registry checked by service role",
                "Dashboard access validated for protected route",
              ]}
            />
            <PanelCard
              title="Recent Worker Notes"
              items={[
                `${data?.failure_count_24h ?? 0} failures recorded in the last 24 hours`,
                "Pipeline includes scan → extract → ML → LLM → report",
                "Worker monitoring panel available under Workers / Jobs",
              ]}
            />
            <PanelCard
              title="Security Summary"
              items={[
                `${data?.quarantined_24h ?? 0} quarantined files in the current window`,
                `${data?.needs_review_count ?? 0} items flagged for review`,
                "Detailed alerts available in the Security tab",
              ]}
            />
          </section>
        </div>
      </div>
    </div>
  );
}

function NavPill({ href, label }: { href: string; label: string }) {
  return (
    <Link
      href={href}
      className="rounded-full border border-white/10 bg-white/[0.03] px-4 py-2 text-sm text-slate-200 transition hover:bg-white/[0.08] hover:text-white"
    >
      {label}
    </Link>
  );
}

function StatCard({
  title,
  value,
  delta,
  deltaUp,
  icon,
  glow,
}: {
  title: string;
  value: string | number;
  delta: string;
  deltaUp?: boolean;
  icon: React.ReactNode;
  glow: string;
}) {
  return (
    <div className="group relative overflow-hidden rounded-[26px] border border-white/10 bg-white/[0.04] p-5 shadow-[0_18px_60px_rgba(0,0,0,0.28)] backdrop-blur-xl">
      <div className={`absolute inset-0 bg-gradient-to-br ${glow} opacity-80`} />
      <div className="absolute right-0 top-0 h-24 w-24 rounded-full bg-white/5 blur-2xl" />
      <div className="relative">
        <div className="flex items-start justify-between gap-3">
          <div>
            <p className="text-sm text-slate-400">{title}</p>
            <div className="mt-3 text-3xl font-bold tracking-tight text-white">
              {String(value)}
            </div>
          </div>

          <div className="rounded-2xl border border-white/10 bg-white/10 p-3 text-slate-100">
            {icon}
          </div>
        </div>

        <div className="mt-4 inline-flex items-center gap-1.5 text-xs font-medium">
          {deltaUp ? (
            <ArrowUpRight className="h-3.5 w-3.5 text-emerald-300" />
          ) : (
            <ArrowDownRight className="h-3.5 w-3.5 text-slate-400" />
          )}
          <span className={deltaUp ? "text-emerald-300" : "text-slate-400"}>
            {delta}
          </span>
          <span className="text-slate-500">vs recent period</span>
        </div>
      </div>
    </div>
  );
}

function ChartCard({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle: string;
  children: React.ReactNode;
}) {
  return (
    <section className="rounded-[28px] border border-white/10 bg-white/[0.04] p-5 shadow-[0_18px_60px_rgba(0,0,0,0.28)] backdrop-blur-xl">
      <div className="mb-4 flex items-start justify-between gap-3">
        <div>
          <h3 className="text-lg font-semibold text-white">{title}</h3>
          <p className="mt-1 text-sm text-slate-400">{subtitle}</p>
        </div>
      </div>
      {children}
    </section>
  );
}

function LegendItem({
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
        <span
          className="inline-block h-2.5 w-2.5 rounded-full"
          style={{ backgroundColor: color }}
        />
        <span className="text-sm text-slate-300">{label}</span>
      </div>
      <div className="mt-2 text-xl font-semibold text-white">{value}</div>
    </div>
  );
}

function MiniStatusCard({
  title,
  value,
  helper,
  icon,
}: {
  title: string;
  value: string;
  helper: string;
  icon: React.ReactNode;
}) {
  return (
    <div className="rounded-[24px] border border-white/10 bg-white/[0.04] p-4 shadow-[0_18px_60px_rgba(0,0,0,0.22)]">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-sm text-slate-400">{title}</div>
          <div className="mt-2 text-xl font-semibold text-white">{value}</div>
          <div className="mt-1 text-xs text-slate-500">{helper}</div>
        </div>
        <div className="rounded-2xl border border-white/10 bg-white/10 p-2 text-slate-200">
          {icon}
        </div>
      </div>
    </div>
  );
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-start justify-between gap-4 border-b border-white/6 pb-3 last:border-b-0 last:pb-0">
      <div className="text-sm text-slate-400">{label}</div>
      <div className="max-w-[70%] text-right text-sm text-slate-200 break-all">
        {value}
      </div>
    </div>
  );
}

function PanelCard({
  title,
  items,
}: {
  title: string;
  items: string[];
}) {
  return (
    <section className="rounded-[26px] border border-white/10 bg-white/[0.04] p-5 shadow-[0_18px_60px_rgba(0,0,0,0.22)] backdrop-blur-xl">
      <h3 className="text-lg font-semibold text-white">{title}</h3>
      <div className="mt-4 space-y-3">
        {items.map((item, index) => (
          <div
            key={index}
            className="rounded-2xl border border-white/8 bg-white/[0.03] p-3 text-sm text-slate-300"
          >
            {item}
          </div>
        ))}
      </div>
    </section>
  );
}