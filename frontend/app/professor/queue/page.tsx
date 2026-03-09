"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import {
  AlertTriangle,
  ArrowLeft,
  ClipboardCheck,
  RefreshCw,
  Search,
  ShieldCheck,
  ShieldAlert,
} from "lucide-react";
import { backendUrl } from "@/lib/backendUrl";
import { fetchJsonWithAuth } from "@/lib/fetchWithAuth";

function fmtDate(iso?: string) {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

export default function ProfessorQueuePage() {
  const [items, setItems] = useState<any[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [q, setQ] = useState("");

  async function loadQueue() {
    try {
      setLoading(true);
      setErr(null);
      const j = await fetchJsonWithAuth(backendUrl("/phase7/professor/queue?limit=60"), {
        method: "GET",
      });
      setItems(j.items || []);
    } catch (e: any) {
      setErr(e?.message || "Failed to load queue");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void loadQueue();
  }, []);

  const filtered = useMemo(() => {
    const s = q.trim().toLowerCase();
    if (!s) return items;
    return items.filter((x) =>
      [x.file_id, x.id, x.created_at]
        .filter(Boolean)
        .join(" ")
        .toLowerCase()
        .includes(s)
    );
  }, [items, q]);

  const needsReviewCount = useMemo(
    () => items.filter((x) => x.needs_review).length,
    [items]
  );

  return (
    <div className="grid gap-6">
      <section className="rounded-[30px] border border-white/10 bg-white/[0.04] p-5 shadow-[0_20px_70px_rgba(0,0,0,0.24)] backdrop-blur-xl">
        <div className="flex flex-col gap-4 xl:flex-row xl:items-center xl:justify-between">
          <div>
            <div className="text-xs font-semibold uppercase tracking-[0.25em] text-blue-200/65">
              Professor / Queue
            </div>
            <h1 className="mt-2 text-3xl font-black tracking-tight">Professor queue</h1>
            <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-300">
              Review pending and flagged professor reports, prioritising moderation needs first.
            </p>
          </div>

          <div className="flex flex-wrap gap-3">
            <button
              onClick={loadQueue}
              className="inline-flex items-center gap-2 rounded-2xl border border-white/10 bg-white/10 px-4 py-3 text-sm font-bold text-white transition hover:bg-white/15"
            >
              <RefreshCw size={16} className={loading ? "animate-spin" : ""} />
              {loading ? "Loading..." : "Refresh"}
            </button>

            <Link
              href="/professor"
              className="inline-flex items-center gap-2 rounded-2xl border border-white/10 bg-white/10 px-4 py-3 text-sm font-bold text-white transition hover:bg-white/15"
            >
              <ArrowLeft size={16} />
              Back
            </Link>
          </div>
        </div>
      </section>

      <section className="grid grid-cols-1 gap-6 md:grid-cols-3">
        <StatCard
          title="Queue items"
          value={String(items.length)}
          sub="Current professor queue"
          icon={<ClipboardCheck size={18} />}
          accent="from-blue-500/20 to-indigo-500/10"
        />
        <StatCard
          title="Needs review"
          value={String(needsReviewCount)}
          sub="Priority moderation items"
          icon={<ShieldAlert size={18} />}
          accent="from-amber-500/20 to-rose-500/10"
        />
        <StatCard
          title="Safe items"
          value={String(Math.max(0, items.length - needsReviewCount))}
          sub="Lower-risk reports"
          icon={<ShieldCheck size={18} />}
          accent="from-emerald-500/20 to-cyan-500/10"
        />
      </section>

      <section className="rounded-[30px] border border-white/10 bg-white/[0.05] p-5 shadow-[0_20px_70px_rgba(0,0,0,0.22)] backdrop-blur-xl">
        <div className="relative">
          <Search size={17} className="pointer-events-none absolute left-4 top-1/2 -translate-y-1/2 text-slate-400" />
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Search queue by file id, report id, or date..."
            className="w-full rounded-2xl border border-white/10 bg-[#0c1737] py-3 pl-11 pr-4 text-sm text-white outline-none transition placeholder:text-slate-500 focus:border-blue-400/40"
          />
        </div>
      </section>

      {err ? (
        <div className="rounded-2xl border border-rose-300/20 bg-rose-400/10 px-4 py-3 text-sm text-rose-200">
          {err}
        </div>
      ) : null}

      <section className="rounded-[30px] border border-white/10 bg-white/[0.05] p-5 shadow-[0_20px_70px_rgba(0,0,0,0.22)] backdrop-blur-xl">
        <div className="mb-5 text-lg font-black text-white">Queue items</div>

        {loading ? (
          <div className="text-sm text-slate-400">Loading queue...</div>
        ) : filtered.length === 0 ? (
          <div className="text-sm text-slate-400">No items yet.</div>
        ) : (
          <div className="grid gap-4">
            {filtered.map((x) => (
              <div
                key={x.id}
                className="rounded-[24px] border border-white/10 bg-[linear-gradient(180deg,rgba(255,255,255,0.045),rgba(255,255,255,0.03))] p-5"
              >
                <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
                  <div className="grid gap-2">
                    <div className="text-sm text-slate-400">{fmtDate(x.created_at)}</div>
                    <div className="text-base font-black text-white">{x.file_id}</div>
                    <div className="text-xs text-slate-400">report_id: {x.id}</div>
                  </div>

                  <div className="flex flex-wrap items-center gap-3">
                    {x.needs_review ? (
                      <span className="inline-flex items-center gap-2 rounded-full border border-amber-300/20 bg-amber-400/10 px-3 py-1 text-xs font-bold text-amber-200">
                        <ShieldAlert size={14} />
                        Needs review
                      </span>
                    ) : (
                      <span className="inline-flex items-center gap-2 rounded-full border border-emerald-300/20 bg-emerald-400/10 px-3 py-1 text-xs font-bold text-emerald-200">
                        <ShieldCheck size={14} />
                        Safe
                      </span>
                    )}

                    <Link
                      href={`/professor/results/${x.file_id}`}
                      className="rounded-2xl bg-gradient-to-r from-blue-500 to-indigo-500 px-4 py-2.5 text-sm font-bold text-white shadow-[0_12px_28px_rgba(59,130,246,0.28)] transition hover:scale-[1.01]"
                    >
                      Open
                    </Link>
                  </div>
                </div>
              </div>
            ))}
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
  icon,
  accent,
}: {
  title: string;
  value: string;
  sub: string;
  icon: React.ReactNode;
  accent: string;
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
        <div className="mt-5 text-4xl font-black text-white">{value}</div>
        <div className="mt-2 text-sm text-slate-400">{sub}</div>
      </div>
    </div>
  );
}