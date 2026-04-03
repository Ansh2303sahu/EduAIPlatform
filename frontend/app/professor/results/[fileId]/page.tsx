"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import {
  AlertTriangle,
  ArrowLeft,
  ClipboardList,
  RefreshCw,
  Sparkles,
  Timer,
} from "lucide-react";
import { backendUrl } from "@/lib/backendUrl";
import { fetchJsonWithAuth, fetchWithAuth } from "@/lib/fetchWithAuth";
import { buildRenderableRagMeta } from "@/lib/ragMeta";
import RagEvidencePanel from "@/components/rag/RagEvidencePanel";

type ProfessorLatestResp = {
  found?: boolean;
  item?: {
    id: string;
    file_id: string;
    role: "professor";
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
      rag_meta?: {
        enabled?: boolean;
        confidence_score?: number | null;
        confidence_label?: string | null;
        safe_review?: boolean | null;
        citations?: Array<{
          title: string;
          section: string;
          document_id: string;
          chunk_id: string;
          category?: string | null;
          score?: number | null;
        }> | null;
        retrieved_chunks?: Array<{
          chunk_id: string;
          document_id: string;
          document_title: string;
          section: string;
          category: string;
          audience: "student" | "professor";
          content: string;
          score: number;
          source_priority?: number;
          is_official?: boolean;
          metadata?: Record<string, any>;
        }> | null;
        trace?: {
          audience?: string;
          query?: string;
          collection_name?: string;
          rewritten_queries?: string[];
          retrieved_chunk_ids?: string[];
          final_chunk_ids?: string[];
          scores?: number[];
        } | null;
      } | null;
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
    rag_trace?: {
      audience?: string;
      query?: string;
      collection_name?: string;
      rewritten_queries?: string[];
      retrieved_chunk_ids?: string[];
      final_chunk_ids?: string[];
      scores?: number[];
    } | null;
    citations?: Array<{
      title: string;
      section: string;
      document_id: string;
      chunk_id: string;
      category?: string | null;
      score?: number | null;
    }> | null;
    retrieval_confidence?: number | null;
    retrieval_confidence_label?: string | null;
    safe_review?: boolean | null;
    retrieved_chunks?: Array<{
      chunk_id: string;
      document_id: string;
      document_title: string;
      section: string;
      category: string;
      audience: "student" | "professor";
      content: string;
      score: number;
      source_priority?: number;
      is_official?: boolean;
      metadata?: Record<string, any>;
    }> | null;
  };
};

type LangChainMeta = {
  role: string;
  needs_review: boolean;
  model_used: string;
  fallback_used: boolean;
  decision_source: string;
  execution_mode: string;
  discrepancy_flag?: boolean | null;
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

export default function ProfessorReport({ params }: { params: { fileId: string } }) {
  const fileId = params.fileId;

  const [item, setItem] = useState<ProfessorLatestResp["item"] | null>(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  const [langchainGenerating, setLangchainGenerating] = useState(false);
  const [langchainMeta, setLangchainMeta] = useState<LangChainMeta | null>(null);

  async function loadReport() {
    setErr(null);
    try {
      setLoading(true);
      const j = (await fetchJsonWithAuth(
        backendUrl(`/phase7/latest/professor/${fileId}`),
        { method: "GET" }
      )) as ProfessorLatestResp;

      setItem(j.item || null);
    } catch (e: any) {
      setErr(e?.message || "Failed to load report");
    } finally {
      setLoading(false);
    }
  }

  async function generateLangChain() {
    if (!fileId || langchainGenerating) return;
    setLangchainGenerating(true);
    setErr(null);
    try {
      const res = await fetchWithAuth(
        backendUrl(`/langchain/professor/generate`),
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ file_id: fileId }),
        },
        { timeoutMs: 600000 }
      );
      if (!res.ok) {
        const txt = await res.text().catch(() => "");
        throw new Error(txt || `LangChain generate failed (${res.status})`);
      }
      const json = await res.json();
      setLangchainMeta(json.meta || null);
      setItem({
        id: json.stored?.id || "",
        file_id: fileId,
        role: "professor",
        created_at: json.stored?.created_at || undefined,
        needs_review: json.meta?.needs_review || false,
        report_json: json.report,
      });
    } catch (e: any) {
      setErr(e?.message || "LangChain generate failed");
    } finally {
      setLangchainGenerating(false);
    }
  }

  useEffect(() => {
    void loadReport();
  }, [fileId]);

  const report = item?.report_json || null;
  const mv = item?.model_versions || {};
  const needsReview = !!(item?.needs_review || report?.safety?.needs_review);
  const finalConfidence = mv?.agreement?.final_confidence;

  const rubricRows = useMemo(() => {
    const rows = report?.rubric_breakdown || [];
    return Array.isArray(rows) ? rows : [];
  }, [report]);

  const moderationNotes = useMemo(() => {
    const rows = report?.moderation_notes || [];
    return Array.isArray(rows) ? rows : [];
  }, [report]);

  const ragMeta = useMemo(() => buildRenderableRagMeta(item), [item]);

  return (
    <div className="grid gap-6">
      <section className="rounded-[30px] border border-white/10 bg-white/[0.04] p-5 shadow-[0_20px_70px_rgba(0,0,0,0.24)] backdrop-blur-xl">
        <div className="flex flex-col gap-4 xl:flex-row xl:items-center xl:justify-between">
          <div>
            <div className="text-xs font-semibold uppercase tracking-[0.25em] text-blue-200/65">
              Professor / Results
            </div>
            <h1 className="mt-2 text-3xl font-black tracking-tight">Professor report</h1>
            <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-300">
              Structured rubric feedback, moderation notes, confidence metadata, and review signals.
            </p>
          </div>

          <div className="flex flex-wrap gap-3">
            <button
              onClick={loadReport}
              className="inline-flex items-center gap-2 rounded-2xl border border-white/10 bg-white/10 px-4 py-3 text-sm font-bold text-white transition hover:bg-white/15"
            >
              <RefreshCw size={16} className={loading ? "animate-spin" : ""} />
              {loading ? "Loading..." : "Refresh"}
            </button>

            <button
              onClick={() => void generateLangChain()}
              disabled={langchainGenerating}
              className="inline-flex items-center gap-2 rounded-2xl border border-violet-400/30 bg-violet-500/20 px-4 py-3 text-sm font-bold text-violet-100 transition hover:bg-violet-500/30 disabled:opacity-50"
            >
              <RefreshCw size={16} className={langchainGenerating ? "animate-spin" : ""} />
              {langchainGenerating ? "Generating..." : "LangChain"}
            </button>
            <Link
              href="/professor/queue"
              className="inline-flex items-center gap-2 rounded-2xl border border-white/10 bg-white/10 px-4 py-3 text-sm font-bold text-white transition hover:bg-white/15"
            >
              <ArrowLeft size={16} />
              Back to queue
            </Link>
          </div>
        </div>
      </section>

      {err ? (
        <div className="rounded-2xl border border-rose-300/20 bg-rose-400/10 px-4 py-3 text-sm text-rose-200">
          {err}
        </div>
      ) : null}

      {loading ? (
        <section className="rounded-[24px] border border-white/10 bg-white/[0.05] p-5 text-sm text-slate-400">
          Loading professor report...
        </section>
      ) : !report ? (
        <section className="rounded-[24px] border border-white/10 bg-white/[0.05] p-5 text-sm text-slate-400">
          No professor report yet for this file.
        </section>
      ) : (
        <>
          <section className="rounded-[30px] border border-white/10 bg-white/[0.05] p-5 shadow-[0_20px_70px_rgba(0,0,0,0.22)] backdrop-blur-xl">
            <div className="flex flex-col gap-4 xl:flex-row xl:items-start xl:justify-between">
              <div className="grid gap-2">
                <div className="text-xs font-extrabold uppercase tracking-[0.16em] text-slate-400">File ID</div>
                <div className="text-xl font-black text-white">{fileId}</div>
                <div className="text-sm text-slate-400">Created: {fmtDate(item?.created_at)}</div>
              </div>

              <div className="flex flex-wrap gap-2">
                {needsReview ? (
                  <Badge tone="bad" text="Needs review" />
                ) : (
                  <Badge tone="good" text="Safe" />
                )}
                <Badge tone="primary" text={`Confidence: ${pct(finalConfidence)}`} />
                <Badge tone="neutral" text={`Band: ${confidenceBand(finalConfidence)}`} />
                {langchainMeta?.discrepancy_flag ? (
                  <Badge tone="bad" text="⚠ ML–LLM Disagreement" />
                ) : null}
                {langchainMeta?.execution_mode ? (
                  <Badge tone="neutral" text={`Mode: ${langchainMeta.execution_mode}`} />
                ) : null}
              </div>
            </div>

            <div className="mt-5 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
              <InfoTile label="LLM Used" value={mv?.llm_model_used || mv?.llm_primary || "—"} />
              <InfoTile label="Fallback" value={mv?.llm_fallback || "—"} />
              <InfoTile label="Request ID" value={mv?.request_id || "—"} />
              <InfoTile label="Injected" value={String(mv?.agreement?.injected ?? "—")} />
            </div>
          </section>

          <section className="rounded-[30px] border border-white/10 bg-white/[0.05] p-5 shadow-[0_20px_70px_rgba(0,0,0,0.22)] backdrop-blur-xl">
            <div className="mb-4 flex items-center gap-3">
              <div className="grid h-10 w-10 place-items-center rounded-2xl bg-blue-500/15 text-blue-200">
                <Sparkles size={18} />
              </div>
              <h3 className="text-lg font-black text-white">Feedback explanation</h3>
            </div>
            <div className="whitespace-pre-wrap text-sm leading-7 text-slate-300">
              {report.feedback_explanation || "—"}
            </div>
          </section>

          <section className="rounded-[30px] border border-white/10 bg-white/[0.05] p-5 shadow-[0_20px_70px_rgba(0,0,0,0.22)] backdrop-blur-xl">
            <div className="mb-4 flex items-center gap-3">
              <div className="grid h-10 w-10 place-items-center rounded-2xl bg-blue-500/15 text-blue-200">
                <ClipboardList size={18} />
              </div>
              <h3 className="text-lg font-black text-white">Rubric breakdown</h3>
            </div>

            <RubricTable rows={rubricRows} />
          </section>

          <section className="grid gap-6 xl:grid-cols-[1.1fr_0.9fr]">
            <section className="rounded-[30px] border border-white/10 bg-white/[0.05] p-5 shadow-[0_20px_70px_rgba(0,0,0,0.22)] backdrop-blur-xl">
              <div className="mb-4 flex items-center gap-3">
                <div className="grid h-10 w-10 place-items-center rounded-2xl bg-blue-500/15 text-blue-200">
                  <AlertTriangle size={18} />
                </div>
                <h3 className="text-lg font-black text-white">Moderation notes</h3>
              </div>

              {moderationNotes.length === 0 ? (
                <div className="text-sm text-slate-400">No moderation notes.</div>
              ) : (
                <div className="grid gap-3">
                  {moderationNotes.map((n, i) => (
                    <div
                      key={i}
                      className="rounded-[22px] border border-white/10 bg-white/[0.04] p-4"
                    >
                      <div className="text-xs font-extrabold uppercase tracking-[0.16em] text-slate-400">
                        Risk
                      </div>
                      <div className="mt-1 text-base font-bold text-white">{n.risk || "—"}</div>
                      <div className="mt-4 text-xs font-extrabold uppercase tracking-[0.16em] text-slate-400">
                        Note
                      </div>
                      <div className="mt-1 text-sm leading-6 text-slate-300">{n.note || "—"}</div>
                    </div>
                  ))}
                </div>
              )}
            </section>

            <section className="rounded-[30px] border border-white/10 bg-white/[0.05] p-5 shadow-[0_20px_70px_rgba(0,0,0,0.22)] backdrop-blur-xl">
              <div className="mb-4 flex items-center gap-3">
                <div className="grid h-10 w-10 place-items-center rounded-2xl bg-blue-500/15 text-blue-200">
                  <Timer size={18} />
                </div>
                <h3 className="text-lg font-black text-white">Safety + timing</h3>
              </div>

              <div className="grid gap-3">
                <MiniRow label="Needs review" value={needsReview ? "Yes" : "No"} />
                <MiniRow label="Safety reason" value={report?.safety?.reason || "—"} />
                <MiniRow label="Total time" value={msToSec(mv?.timings_ms?.total)} />
                <MiniRow label="Ingestion" value={msToSec(mv?.timings_ms?.ingestion)} />
                <MiniRow label="ML" value={msToSec(mv?.timings_ms?.ai_service)} />
                <MiniRow label="LLM" value={msToSec(mv?.timings_ms?.llm_service)} />
              </div>
            </section>
          </section>

          <RagEvidencePanel
            ragMeta={ragMeta}
            title="Professor RAG Evidence"
            emptyMessage="No RAG evidence was attached to this professor report. If you expected citations or snippets, the stored report currently has empty RAG fields."
          />

          <section className="rounded-[30px] border border-white/10 bg-white/[0.05] p-5 shadow-[0_20px_70px_rgba(0,0,0,0.22)] backdrop-blur-xl">
            <h3 className="mb-4 text-lg font-black text-white">Raw JSON (Demo)</h3>
            <pre className="overflow-x-auto whitespace-pre-wrap rounded-[20px] border border-white/10 bg-[#091327]/90 p-4 text-xs leading-6 text-slate-300">
              {JSON.stringify(report, null, 2)}
            </pre>
          </section>
        </>
      )}
    </div>
  );
}

function RubricTable({
  rows,
}: {
  rows: Array<{
    criterion?: string;
    band?: string;
    justification?: string;
  }>;
}) {
  if (!rows.length) {
    return (
      <div className="text-sm text-slate-400">
        No rubric breakdown found in report JSON yet.
      </div>
    );
  }

  return (
    <div className="overflow-x-auto rounded-[20px] border border-white/10">
      <table className="min-w-full border-collapse">
        <thead className="bg-white/[0.05]">
          <tr>
            <th className="border-b border-white/10 px-4 py-3 text-left text-xs font-extrabold uppercase tracking-[0.16em] text-slate-300">
              Criterion
            </th>
            <th className="border-b border-white/10 px-4 py-3 text-left text-xs font-extrabold uppercase tracking-[0.16em] text-slate-300">
              Band
            </th>
            <th className="border-b border-white/10 px-4 py-3 text-left text-xs font-extrabold uppercase tracking-[0.16em] text-slate-300">
              Justification
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="border-b border-white/5">
              <td className="px-4 py-3 text-sm text-slate-300">{r.criterion || "—"}</td>
              <td className="px-4 py-3 text-sm font-bold text-white">{r.band || "—"}</td>
              <td className="px-4 py-3 text-sm text-slate-300">{r.justification || "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function InfoTile({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-[20px] border border-white/10 bg-white/[0.04] p-4">
      <div className="text-xs font-extrabold uppercase tracking-[0.16em] text-slate-400">{label}</div>
      <div className="mt-2 text-sm font-bold text-white break-words">{value}</div>
    </div>
  );
}

function MiniRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-[18px] border border-white/10 bg-white/[0.04] p-4">
      <div className="text-xs font-extrabold uppercase tracking-[0.16em] text-slate-400">{label}</div>
      <div className="mt-2 text-sm font-bold text-white">{value}</div>
    </div>
  );
}

function Badge({
  text,
  tone,
}: {
  text: string;
  tone: "good" | "bad" | "primary" | "neutral";
}) {
  const style =
    tone === "good"
      ? "border-emerald-300/20 bg-emerald-400/10 text-emerald-200"
      : tone === "bad"
      ? "border-rose-300/20 bg-rose-400/10 text-rose-200"
      : tone === "primary"
      ? "border-blue-300/20 bg-blue-500/10 text-blue-100"
      : "border-white/10 bg-white/8 text-slate-200";

  return <span className={`inline-flex rounded-full border px-3 py-1 text-xs font-bold ${style}`}>{text}</span>;
}
