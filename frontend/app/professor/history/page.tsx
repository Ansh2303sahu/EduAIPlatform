"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import {
  AlertTriangle,
  ArrowLeft,
  Clock3,
  FileSearch,
  RefreshCw,
  Search,
  ShieldCheck,
  ShieldAlert,
  Sparkles,
} from "lucide-react";
import { backendUrl } from "@/lib/backendUrl";
import { fetchJsonWithAuth } from "@/lib/fetchWithAuth";

type HistoryItem = {
  id: string;
  file_id: string;
  role: "student" | "professor";
  created_at?: string;
  needs_review?: boolean;
  report_json?: {
    rubric_breakdown?: Array<{
      criterion?: string;
      band?: string;
      justification?: string;
    }>;
    feedback_explanation?: string;
    moderation_notes?: Array<{
      risk?: string;
      note?: string;
    }>;
    safety?: {
      needs_review?: boolean;
      reason?: string;
    };
  };
  model_versions?: {
    llm_model_used?: string;
    llm_primary?: string;
    llm_fallback?: string;
    timings_ms?: {
      total?: number;
      ingestion?: number;
      ai_service?: number;
      llm_service?: number;
    };
    agreement?: {
      final_confidence?: number;
      injected?: boolean;
      ml_bucket_0_to_4?: number;
    };
    request_id?: string;
  };
};

type HistoryResp = {
  items?: HistoryItem[];
};

function fmtDate(iso?: string) {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

function pct(n?: number) {
  if (typeof n !== "number") return "—";
  return `${Math.round(n * 100)}%`;
}

function msToSec(ms?: number) {
  if (typeof ms !== "number") return "—";
  return `${(ms / 1000).toFixed(1)}s`;
}

function confidenceBand(n?: number) {
  if (typeof n !== "number") return "—";
  if (n >= 0.85) return "High";
  if (n >= 0.65) return "Medium";
  return "Low";
}

export default function ProfessorHistoryPage() {
  const [items, setItems] = useState<HistoryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);

  const [search, setSearch] = useState("");
  const [reviewFilter, setReviewFilter] = useState<"all" | "needs_review" | "safe">("all");

  async function loadHistory() {
    setErr(null);
    try {
      setLoading(true);

      const j = (await fetchJsonWithAuth(
        backendUrl("/phase7/history/professor?limit=60"),
        { method: "GET" }
      )) as HistoryResp;

      setItems(j.items || []);
    } catch (e: any) {
      setErr(e?.message || "Failed to load professor history");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void loadHistory();
  }, []);

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();

    return items.filter((x) => {
      const textBlob = [
        x.file_id,
        x.id,
        x.model_versions?.llm_model_used,
        x.model_versions?.llm_primary,
        x.report_json?.feedback_explanation,
      ]
        .filter(Boolean)
        .join(" ")
        .toLowerCase();

      const matchesSearch = !q || textBlob.includes(q);

      const needsReview = !!(x.needs_review || x.report_json?.safety?.needs_review);

      const matchesReview =
        reviewFilter === "all" ||
        (reviewFilter === "needs_review" && needsReview) ||
        (reviewFilter === "safe" && !needsReview);

      return matchesSearch && matchesReview;
    });
  }, [items, search, reviewFilter]);

  const stats = useMemo(() => {
    const total = items.length;
    const needsReview = items.filter(
      (x) => x.needs_review || x.report_json?.safety?.needs_review
    ).length;

    const avgConfidenceRaw =
      items.reduce(
        (sum, x) => sum + (x.model_versions?.agreement?.final_confidence || 0),
        0
      ) / (items.length || 1);

    return {
      total,
      needsReview,
      avgConfidence: total ? avgConfidenceRaw : undefined,
    };
  }, [items]);

  return (
    <div className="grid gap-6">
      <section className="rounded-[30px] border border-white/10 bg-white/[0.04] p-5 shadow-[0_20px_70px_rgba(0,0,0,0.24)] backdrop-blur-xl">
        <div className="flex flex-col gap-4 xl:flex-row xl:items-center xl:justify-between">
          <div>
            <div className="text-xs font-semibold uppercase tracking-[0.25em] text-blue-200/65">
              Professor / History
            </div>
            <h1 className="mt-2 text-3xl font-black tracking-tight">Professor report history</h1>
            <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-300">
              Review previous professor reports, moderation flags, confidence metadata, and model timings.
            </p>
          </div>

          <div className="flex flex-wrap gap-3">
            <button
              onClick={loadHistory}
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
          title="Total reports"
          value={String(stats.total)}
          sub="Stored professor reports"
          icon={<FileSearch size={18} />}
          accent="from-blue-500/20 to-indigo-500/10"
        />
        <StatCard
          title="Needs review"
          value={String(stats.needsReview)}
          sub="Safety or moderation attention required"
          icon={<ShieldAlert size={18} />}
          accent="from-amber-500/20 to-rose-500/10"
        />
        <StatCard
          title="Avg confidence"
          value={pct(stats.avgConfidence)}
          sub={`Band: ${confidenceBand(stats.avgConfidence)}`}
          icon={<Sparkles size={18} />}
          accent="from-cyan-500/20 to-blue-500/10"
        />
      </section>

      <section className="rounded-[30px] border border-white/10 bg-white/[0.05] p-5 shadow-[0_20px_70px_rgba(0,0,0,0.22)] backdrop-blur-xl">
        <div className="mb-4 text-lg font-black text-white">Filters</div>

        <div className="grid gap-4 lg:grid-cols-[1.5fr_240px]">
          <div className="relative">
            <Search size={17} className="pointer-events-none absolute left-4 top-1/2 -translate-y-1/2 text-slate-400" />
            <input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search by file id, report id, model, or text..."
              className="w-full rounded-2xl border border-white/10 bg-[#0c1737] py-3 pl-11 pr-4 text-sm text-white outline-none transition placeholder:text-slate-500 focus:border-blue-400/40"
            />
          </div>

          <select
            value={reviewFilter}
            onChange={(e) => setReviewFilter(e.target.value as any)}
            className="rounded-2xl border border-white/10 bg-[#0c1737] px-4 py-3 text-sm text-white outline-none transition focus:border-blue-400/40"
          >
            <option value="all">All reports</option>
            <option value="needs_review">Needs review only</option>
            <option value="safe">Safe only</option>
          </select>
        </div>
      </section>

      {err ? (
        <div className="rounded-2xl border border-rose-300/20 bg-rose-400/10 px-4 py-3 text-sm text-rose-200">
          {err}
        </div>
      ) : null}

      <section className="rounded-[30px] border border-white/10 bg-white/[0.05] p-5 shadow-[0_20px_70px_rgba(0,0,0,0.22)] backdrop-blur-xl">
        <div className="mb-5 text-lg font-black text-white">Professor reports</div>

        {loading ? (
          <div className="text-sm text-slate-400">Loading history...</div>
        ) : filtered.length === 0 ? (
          <div className="text-sm text-slate-400">No history items found.</div>
        ) : (
          <div className="grid gap-4">
            {filtered.map((x) => {
              const needsReview = !!(x.needs_review || x.report_json?.safety?.needs_review);
              const finalConfidence = x.model_versions?.agreement?.final_confidence;
              const rubricCount = x.report_json?.rubric_breakdown?.length || 0;
              const moderationCount = x.report_json?.moderation_notes?.length || 0;

              return (
                <div
                  key={x.id}
                  className="rounded-[24px] border border-white/10 bg-[linear-gradient(180deg,rgba(255,255,255,0.045),rgba(255,255,255,0.03))] p-5"
                >
                  <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
                    <div className="grid gap-2">
                      <div className="text-base font-black text-white">
                        File ID: <span className="font-semibold">{x.file_id}</span>
                      </div>
                      <div className="text-xs text-slate-400">Report ID: {x.id}</div>
                      <div className="text-xs text-slate-400">Created: {fmtDate(x.created_at)}</div>
                    </div>

                    <div className="flex flex-wrap gap-2">
                      {needsReview ? (
                        <Badge tone="bad" text="Needs review" />
                      ) : (
                        <Badge tone="good" text="Safe" />
                      )}
                      <Badge tone="primary" text={`Confidence: ${pct(finalConfidence)}`} />
                    </div>
                  </div>

                  <div className="mt-4 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
                    <MiniInfo label="Band" value={confidenceBand(finalConfidence)} />
                    <MiniInfo label="Rubric rows" value={String(rubricCount)} />
                    <MiniInfo label="Moderation notes" value={String(moderationCount)} />
                    <MiniInfo
                      label="LLM used"
                      value={x.model_versions?.llm_model_used || x.model_versions?.llm_primary || "—"}
                    />
                  </div>

                  <div className="mt-5 grid gap-4 xl:grid-cols-2">
                    <div>
                      <div className="mb-2 text-xs font-extrabold uppercase tracking-[0.16em] text-slate-400">
                        Feedback explanation
                      </div>
                      <div className="text-sm leading-6 text-slate-300">
                        {x.report_json?.feedback_explanation
                          ? x.report_json.feedback_explanation.slice(0, 220) +
                            (x.report_json.feedback_explanation.length > 220 ? "..." : "")
                          : "—"}
                      </div>
                    </div>

                    <div className="grid gap-4 md:grid-cols-2">
                      <div className="rounded-[20px] border border-white/10 bg-white/[0.04] p-4">
                        <div className="mb-2 text-xs font-extrabold uppercase tracking-[0.16em] text-slate-400">
                          Timing
                        </div>
                        <div className="grid gap-1 text-sm text-slate-300">
                          <div>Total: <b className="text-white">{msToSec(x.model_versions?.timings_ms?.total)}</b></div>
                          <div>Ingestion: <b className="text-white">{msToSec(x.model_versions?.timings_ms?.ingestion)}</b></div>
                          <div>ML: <b className="text-white">{msToSec(x.model_versions?.timings_ms?.ai_service)}</b></div>
                          <div>LLM: <b className="text-white">{msToSec(x.model_versions?.timings_ms?.llm_service)}</b></div>
                        </div>
                      </div>

                      <div className="rounded-[20px] border border-white/10 bg-white/[0.04] p-4">
                        <div className="mb-2 text-xs font-extrabold uppercase tracking-[0.16em] text-slate-400">
                          Safety
                        </div>
                        <div className="grid gap-1 text-sm text-slate-300">
                          <div>Needs review: <b className="text-white">{needsReview ? "Yes" : "No"}</b></div>
                          <div>Reason: <b className="text-white">{x.report_json?.safety?.reason || "—"}</b></div>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="mt-5 flex flex-wrap items-center gap-3">
                    <Link
                      href={`/professor/results/${x.file_id}`}
                      className="inline-flex items-center rounded-2xl bg-gradient-to-r from-blue-500 to-indigo-500 px-4 py-2.5 text-sm font-bold text-white shadow-[0_12px_28px_rgba(59,130,246,0.28)] transition hover:scale-[1.01]"
                    >
                      Open report
                    </Link>

                    <button
                      type="button"
                      onClick={() => navigator.clipboard.writeText(x.file_id)}
                      className="rounded-2xl border border-white/10 bg-white/10 px-4 py-2.5 text-sm font-bold text-white transition hover:bg-white/15"
                    >
                      Copy File ID
                    </button>

                    {x.model_versions?.request_id ? (
                      <div className="text-xs text-slate-400">
                        Request ID: <code>{x.model_versions.request_id}</code>
                      </div>
                    ) : null}
                  </div>
                </div>
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

function MiniInfo({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-[20px] border border-white/10 bg-white/[0.04] p-4">
      <div className="text-xs font-extrabold uppercase tracking-[0.16em] text-slate-400">{label}</div>
      <div className="mt-2 text-base font-bold text-white break-words">{value}</div>
    </div>
  );
}

function Badge({ text, tone }: { text: string; tone: "good" | "bad" | "primary" }) {
  const style =
    tone === "good"
      ? "border-emerald-300/20 bg-emerald-400/10 text-emerald-200"
      : tone === "bad"
      ? "border-rose-300/20 bg-rose-400/10 text-rose-200"
      : "border-blue-300/20 bg-blue-500/10 text-blue-100";

  return (
    <span className={`inline-flex rounded-full border px-3 py-1 text-xs font-bold ${style}`}>
      {text}
    </span>
  );
}