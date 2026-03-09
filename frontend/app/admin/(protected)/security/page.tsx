"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import {
  AlertTriangle,
  ArrowLeft,
  Lock,
  RefreshCcw,
  Shield,
  ShieldAlert,
} from "lucide-react";

import { backendUrl } from "@/lib/backendUrl";
import { fetchJsonWithAuth } from "@/lib/fetchWithAuth";

export default function AdminSecurityPage() {
  const [data, setData] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  async function loadData() {
    try {
      setLoading(true);
      setErr(null);
      const j = await fetchJsonWithAuth(backendUrl("/admin/security-alerts?limit=50"), {
        method: "GET",
      });
      setData(j);
    } catch (e: any) {
      setErr(e?.message || "Failed to load security alerts");
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
          <section className="rounded-[30px] border border-white/10 bg-white/[0.04] p-5 shadow-[0_20px_80px_rgba(0,0,0,0.35)] backdrop-blur-xl">
            <div className="flex flex-col gap-4 xl:flex-row xl:items-center xl:justify-between">
              <div>
                <div className="mb-2 inline-flex items-center gap-2 rounded-full border border-rose-400/20 bg-rose-500/10 px-3 py-1 text-xs font-medium text-rose-200">
                  <ShieldAlert className="h-3.5 w-3.5" />
                  Security Center
                </div>
                <h1 className="text-3xl font-semibold tracking-tight">Security Alerts</h1>
                <p className="mt-2 text-sm text-slate-300">
                  Review quarantined files and security-related audit events across the platform.
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

          <div className="grid gap-4 sm:grid-cols-3">
            <MetricBox
              title="Quarantined Files"
              value={String(data?.quarantined_files?.length ?? 0)}
              icon={<Shield className="h-5 w-5" />}
            />
            <MetricBox
              title="Security Audit Events"
              value={String(data?.security_audit?.length ?? 0)}
              icon={<Lock className="h-5 w-5" />}
            />
            <MetricBox
              title="Risk Posture"
              value={(data?.quarantined_files?.length ?? 0) > 0 ? "Watch" : "Stable"}
              icon={<AlertTriangle className="h-5 w-5" />}
            />
          </div>

          <div className="grid gap-4 xl:grid-cols-2">
            <DataPanel title="Quarantined Files" subtitle="Files isolated for security reasons">
              <JsonBlock value={data?.quarantined_files || []} />
            </DataPanel>

            <DataPanel title="Security Audit Events" subtitle="Security-related event trail">
              <JsonBlock value={data?.security_audit || []} />
            </DataPanel>
          </div>
        </div>
      </div>
    </div>
  );
}

function MetricBox({
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

function DataPanel({
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

function JsonBlock({ value }: { value: any }) {
  return (
    <pre className="overflow-x-auto rounded-2xl border border-white/10 bg-[#050b16] p-4 text-xs text-slate-300">
      {JSON.stringify(value ?? {}, null, 2)}
    </pre>
  );
}