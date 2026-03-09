"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { ArrowLeft, RefreshCcw, ScrollText, ShieldCheck } from "lucide-react";

import { backendUrl } from "@/lib/backendUrl";
import { fetchJsonWithAuth } from "@/lib/fetchWithAuth";

export default function AdminAuditPage() {
  const [rows, setRows] = useState<any[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  async function loadData() {
    try {
      setLoading(true);
      setErr(null);
      const j = await fetchJsonWithAuth(backendUrl("/admin/audit?limit=100"), { method: "GET" });
      setRows(Array.isArray(j) ? j : []);
    } catch (e: any) {
      setErr(e?.message || "Failed to load audit logs");
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
                <div className="mb-2 inline-flex items-center gap-2 rounded-full border border-emerald-400/20 bg-emerald-500/10 px-3 py-1 text-xs font-medium text-emerald-200">
                  <ShieldCheck className="h-3.5 w-3.5" />
                  Compliance & Traceability
                </div>
                <h1 className="text-3xl font-semibold tracking-tight">Audit Logs</h1>
                <p className="mt-2 text-sm text-slate-300">
                  Review tracked admin and system actions with metadata and event timing.
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

          <section className="rounded-[28px] border border-white/10 bg-white/[0.04] p-5 shadow-[0_18px_60px_rgba(0,0,0,0.22)] backdrop-blur-xl">
            <div className="mb-4 flex items-center gap-3">
              <div className="rounded-2xl border border-white/10 bg-white/10 p-3 text-slate-100">
                <ScrollText className="h-5 w-5" />
              </div>
              <div>
                <h3 className="text-lg font-semibold text-white">Recent Audit Events</h3>
                <p className="text-sm text-slate-400">Showing latest 100 records</p>
              </div>
            </div>

            <div className="overflow-hidden rounded-2xl border border-white/10">
              <div className="grid grid-cols-[180px_180px_1fr] border-b border-white/10 bg-white/[0.05] px-4 py-3 text-xs font-semibold uppercase tracking-wide text-slate-300">
                <div>Time</div>
                <div>Action</div>
                <div>Metadata</div>
              </div>

              {rows.length ? (
                rows.map((r, idx) => (
                  <div
                    key={idx}
                    className="grid grid-cols-[180px_180px_1fr] gap-4 border-b border-white/6 px-4 py-4 last:border-b-0"
                  >
                    <div className="text-xs text-slate-300 break-all">{r.created_at || "—"}</div>
                    <div className="text-xs font-medium text-blue-200">{r.action || "—"}</div>
                    <pre className="overflow-x-auto rounded-xl border border-white/10 bg-[#050b16] p-3 text-xs text-slate-300">
                      {JSON.stringify(
                        {
                          actor_user_id: r.actor_user_id,
                          entity_type: r.entity_type,
                          entity_id: r.entity_id,
                          metadata: r.metadata,
                        },
                        null,
                        2
                      )}
                    </pre>
                  </div>
                ))
              ) : (
                <div className="px-4 py-8 text-sm text-slate-400">
                  {loading ? "Loading audit logs..." : "No audit logs yet."}
                </div>
              )}
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}