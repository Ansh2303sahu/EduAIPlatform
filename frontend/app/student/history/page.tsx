"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import {
  ArrowLeft,
  FileText,
  GitCompareArrows,
  RefreshCw,
  Search,
  ShieldAlert,
  ShieldCheck,
} from "lucide-react";
import { backendUrl } from "@/lib/backendUrl";
import { fetchJsonWithAuth } from "@/lib/fetchWithAuth";

type HistoryItem = {
  id: string;
  file_id: string;
  created_at?: string | null;
  needs_review?: boolean;
  model_versions?: Record<string, any> | null;
};

function formatDate(value?: string | null) {
  if (!value) return "—";
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return value;
  return d.toLocaleString();
}

export default function StudentHistoryPage() {
  const [items, setItems] = useState<HistoryItem[]>([]);
  const [q, setQ] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  async function loadHistory(showRefreshing = false) {
    try {
      setErr(null);
      if (showRefreshing) setRefreshing(true);
      else setLoading(true);

      const j = await fetchJsonWithAuth(backendUrl("/phase7/history/student?limit=60"), {
        method: "GET",
      });

      setItems(Array.isArray(j?.items) ? j.items : []);
    } catch (e: any) {
      setErr(e?.message || "Failed to load history");
      setItems([]);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }

  useEffect(() => {
    let alive = true;

    (async () => {
      try {
        setErr(null);
        setLoading(true);

        const j = await fetchJsonWithAuth(backendUrl("/phase7/history/student?limit=60"), {
          method: "GET",
        });

        if (!alive) return;
        setItems(Array.isArray(j?.items) ? j.items : []);
      } catch (e: any) {
        if (!alive) return;
        setErr(e?.message || "Failed to load history");
        setItems([]);
      } finally {
        if (!alive) return;
        setLoading(false);
      }
    })();

    return () => {
      alive = false;
    };
  }, []);

  const filtered = useMemo(() => {
    const s = q.trim().toLowerCase();
    if (!s) return items;

    return items.filter((x) => {
      const mv =
        x?.model_versions && typeof x.model_versions === "object"
          ? JSON.stringify(x.model_versions).toLowerCase()
          : "";

      return (
        String(x?.created_at || "").toLowerCase().includes(s) ||
        String(x?.file_id || "").toLowerCase().includes(s) ||
        String(x?.id || "").toLowerCase().includes(s) ||
        mv.includes(s)
      );
    });
  }, [items, q]);

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top,_#213b9f_0%,_#0b1537_38%,_#071126_100%)] text-white">
      <div className="mx-auto max-w-[1450px] px-4 py-6 sm:px-6 lg:px-8">
        <div className="grid gap-6">
          <div className="rounded-[30px] border border-white/10 bg-white/[0.05] p-5 shadow-[0_20px_70px_rgba(0,0,0,0.28)] backdrop-blur-xl">
            <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
              <div>
                <div className="text-xs font-semibold uppercase tracking-[0.24em] text-blue-200/70">
                  Student / History
                </div>
                <h1 className="mt-2 text-3xl font-black tracking-tight">History</h1>
                <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-300">
                  Search and revisit past student reports, check review flags, and open comparison view.
                </p>
              </div>

              <div className="flex flex-wrap gap-3">
                <button
                  onClick={() => loadHistory(true)}
                  disabled={loading || refreshing}
                  className="inline-flex items-center gap-2 rounded-2xl border border-white/10 bg-white/10 px-4 py-3 text-sm font-bold text-white transition hover:bg-white/15 disabled:cursor-not-allowed disabled:opacity-60"
                >
                  <RefreshCw size={16} className={refreshing ? "animate-spin" : ""} />
                  {refreshing ? "Refreshing..." : "Refresh"}
                </button>

                <Link
                  href="/student"
                  className="inline-flex items-center gap-2 rounded-2xl border border-white/10 bg-white/10 px-4 py-3 text-sm font-bold text-white transition hover:bg-white/15"
                >
                  <ArrowLeft size={16} />
                  Back
                </Link>
              </div>
            </div>
          </div>

          <div className="rounded-[30px] border border-white/10 bg-white/[0.05] p-5 shadow-[0_20px_70px_rgba(0,0,0,0.25)] backdrop-blur-xl">
            <div className="relative">
              <Search size={17} className="pointer-events-none absolute left-4 top-1/2 -translate-y-1/2 text-slate-400" />
              <input
                placeholder="Search by file ID, report ID, date, model..."
                value={q}
                onChange={(e) => setQ(e.target.value)}
                className="w-full rounded-2xl border border-white/10 bg-[#0c1737] py-3 pl-11 pr-4 text-sm text-white outline-none transition placeholder:text-slate-500 focus:border-blue-400/40"
              />
            </div>

            {err ? (
              <div className="mt-4 rounded-2xl border border-rose-300/20 bg-rose-400/10 px-4 py-3 text-sm text-rose-200">
                {err}
              </div>
            ) : null}
          </div>

          <div className="rounded-[30px] border border-white/10 bg-white/[0.05] shadow-[0_20px_70px_rgba(0,0,0,0.24)] backdrop-blur-xl overflow-hidden">
            <div className="grid grid-cols-[220px_minmax(0,1fr)_160px_200px] gap-4 border-b border-white/10 bg-white/[0.04] px-5 py-4 text-xs font-extrabold uppercase tracking-[0.18em] text-slate-400 max-lg:hidden">
              <div>Date</div>
              <div>File</div>
              <div>Needs review</div>
              <div>Actions</div>
            </div>

            {loading ? (
              <div className="px-5 py-10 text-sm text-slate-400">Loading history...</div>
            ) : !filtered.length ? (
              <div className="px-5 py-10 text-sm text-slate-400">
                {items.length ? "No matching history found." : "No history yet."}
              </div>
            ) : (
              <div className="grid">
                {filtered.map((x) => (
                  <div
                    key={x.id}
                    className="grid gap-4 border-t border-white/10 px-5 py-4 max-lg:grid-cols-1 lg:grid-cols-[220px_minmax(0,1fr)_160px_200px] lg:items-center"
                  >
                    <div className="text-sm text-slate-300">{formatDate(x.created_at)}</div>

                    <div className="min-w-0">
                      <div className="flex items-start gap-3">
                        <div className="grid h-10 w-10 shrink-0 place-items-center rounded-xl bg-blue-500/15 text-blue-200">
                          <FileText size={16} />
                        </div>
                        <div className="min-w-0">
                          <div className="break-all text-sm font-bold text-white">{x.file_id || "—"}</div>
                          <div className="mt-1 break-all text-xs text-slate-400">report_id: {x.id}</div>
                        </div>
                      </div>
                    </div>

                    <div>
                      {x.needs_review ? (
                        <span className="inline-flex items-center gap-2 rounded-full border border-amber-300/20 bg-amber-400/10 px-3 py-1.5 text-xs font-bold text-amber-200">
                          <ShieldAlert size={14} />
                          Yes
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-2 rounded-full border border-emerald-300/20 bg-emerald-400/10 px-3 py-1.5 text-xs font-bold text-emerald-200">
                          <ShieldCheck size={14} />
                          No
                        </span>
                      )}
                    </div>

                    <div className="flex flex-wrap gap-3">
                      <Link
                        href={`/student/results/${x.file_id}`}
                        className="inline-flex items-center gap-2 rounded-2xl border border-blue-300/20 bg-blue-500/10 px-4 py-2.5 text-sm font-bold text-blue-100 transition hover:bg-blue-500/15"
                      >
                        <FileText size={15} />
                        Open
                      </Link>

                      <Link
                        href={`/student/compare?a=${x.id}`}
                        className="inline-flex items-center gap-2 rounded-2xl border border-white/10 bg-white/8 px-4 py-2.5 text-sm font-bold text-slate-100 transition hover:bg-white/12"
                      >
                        <GitCompareArrows size={15} />
                        Compare
                      </Link>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}