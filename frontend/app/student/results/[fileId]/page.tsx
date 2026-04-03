"use client";

import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";
import { useParams } from "next/navigation";
import {
  AlertTriangle,
  ArrowLeft,
  CheckCircle2,
  Clock3,
  FileText,
  ImageIcon,
  Mic,
  RefreshCw,
  ShieldAlert,
  Table2,
} from "lucide-react";
import { backendUrl } from "@/lib/backendUrl";
import { fetchWithAuth } from "@/lib/fetchWithAuth";
import { buildRenderableRagMeta } from "@/lib/ragMeta";
import RagEvidencePanel from "@/components/rag/RagEvidencePanel";

type ResultsResp = {
  file: {
    id: string;
    status: string;
    mime_type?: string;
    submission_id?: string | null;
    created_at?: string;
    processed_at?: string | null;
  };
  jobs: Array<{
    id: string;
    status: string;
    job_type: string;
    created_at?: string;
    error_code?: string | null;
    error_message?: string | null;
  }>;
  text?: { redacted_text: string } | null;
  tables?: Array<{
    table_index: number;
    sheet_name?: string | null;
    columns: string[];
    rows: any[][];
  }>;
  media?: Array<{
    media_index: number;
    media_type: string;
    width?: number | null;
    height?: number | null;
    caption?: string | null;
    metadata?: any;
  }>;
  transcript?: { redacted_transcript: string } | null;
  events?: Array<{ event_type: string; created_at?: string }>;
};

type ReportIssue =
  | string
  | {
      title?: string;
      issue?: string;
      label?: string;
      text?: string;
      evidence?: string;
      details?: string;
      description?: string;
      severity?: string;
      level?: string;
    };

type ReportPlanItem =
  | string
  | {
      action?: string;
      item?: string;
      step?: string;
      title?: string;
      text?: string;
      why?: string;
      reason?: string;
      how?: string;
      details?: string;
      priority?: number;
    };

type ReportChecklistItem =
  | string
  | {
      item?: string;
      label?: string;
      title?: string;
      text?: string;
      done?: boolean;
      checked?: boolean;
      complete?: boolean;
    };

type Phase7LatestResp = {
  found: boolean;
  item?: {
    id: string;
    file_id: string;
    role: "student" | "professor";
    created_at?: string;
    needs_review?: boolean;
    report_json?: {
      summary?: string;
      issues?: ReportIssue[];
      improvement_plan?: ReportPlanItem[];
      checklist?: ReportChecklistItem[];
      model_agreement?: { ml_confidence?: number; llm_confidence?: number; final_confidence?: number };
      safety?: { needs_review?: boolean; reason?: string };
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
      timings_ms?: { total?: number; ingestion?: number; ai_service?: number; llm_service?: number };
      agreement?: { final_confidence?: number; ml_bucket_0_to_4?: number; injected?: boolean };
      ml_models?: { feedback?: string; confidence?: string };
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

type NormalizedIssue = {
  title: string;
  evidence?: string;
  severity?: string;
};

type NormalizedPlanItem = {
  action: string;
  why?: string;
  how?: string;
  priority?: number;
};

type NormalizedChecklistItem = {
  item: string;
  done?: boolean;
};

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function asArray(value: unknown): unknown[] {
  if (Array.isArray(value)) return value;
  if (value == null) return [];

  const record = asRecord(value);
  if (record) {
    for (const key of ["items", "data", "values", "results"]) {
      if (Array.isArray(record[key])) return record[key] as unknown[];
    }
  }

  return [value];
}

function extractText(value: unknown): string | undefined {
  if (typeof value === "string") {
    const trimmed = value.trim();
    return trimmed || undefined;
  }

  if (typeof value === "number" && Number.isFinite(value)) {
    return String(value);
  }

  if (Array.isArray(value)) {
    const parts = value.map((item) => extractText(item)).filter(Boolean) as string[];
    return parts.length ? parts.join(" ") : undefined;
  }

  const record = asRecord(value);
  if (!record) return undefined;

  return firstNonEmptyText(
    record.text,
    record.title,
    record.label,
    record.item,
    record.action,
    record.issue,
    record.evidence,
    record.details,
    record.description,
    record.reason,
    record.how,
    record.content,
    record.message,
    record.value,
  );
}

function firstNonEmptyText(...values: unknown[]) {
  for (const value of values) {
    const text = extractText(value);
    if (text) return text;
  }
  return undefined;
}

function normalizeSeverity(value: unknown): string | undefined {
  const raw = extractText(value)?.toLowerCase();
  if (!raw) return undefined;
  if (raw.includes("high")) return "high";
  if (raw.includes("med")) return "med";
  if (raw.includes("low")) return "low";
  return raw;
}

function normalizePriority(value: unknown): number | undefined {
  if (typeof value === "number" && Number.isFinite(value)) {
    return Math.max(1, Math.round(value));
  }

  if (typeof value === "string") {
    const parsed = Number.parseInt(value.trim(), 10);
    if (Number.isFinite(parsed)) return Math.max(1, parsed);
  }

  return undefined;
}

function normalizeDone(value: unknown): boolean | undefined {
  if (typeof value === "boolean") return value;
  if (typeof value === "number") return value !== 0;
  if (typeof value !== "string") return undefined;

  const raw = value.trim().toLowerCase();
  if (["true", "yes", "done", "checked", "complete", "completed"].includes(raw)) return true;
  if (["false", "no", "pending", "unchecked", "incomplete"].includes(raw)) return false;
  return undefined;
}

function normalizeIssues(items?: ReportIssue[] | null): NormalizedIssue[] {
  return (items || [])
    .map((item) => {
      if (typeof item === "string") {
        const title = item.trim();
        return title ? { title } : null;
      }
      if (!item || typeof item !== "object") return null;

      const title = firstNonEmptyText(item.title, item.issue, item.label, item.text);
      const evidence = firstNonEmptyText(item.evidence, item.details, item.description);
      const severity = firstNonEmptyText(item.severity, item.level);

      if (!title && !evidence && !severity) return null;

      return {
        title: title || "Unnamed issue",
        evidence,
        severity,
      };
    })
    .filter(Boolean) as NormalizedIssue[];
}

function normalizeImprovementPlan(items?: ReportPlanItem[] | null): NormalizedPlanItem[] {
  return asArray(items)
    .map((item, idx) => {
      if (typeof item === "string") {
        const action = item.trim();
        return action ? { action } : null;
      }
      const record = asRecord(item);
      if (!record) return null;

      const action = firstNonEmptyText(
        record.action,
        record.item,
        record.step,
        record.title,
        record.text,
        record.message,
      );
      const why = firstNonEmptyText(record.why, record.reason, record.because);
      const how = firstNonEmptyText(record.how, record.details, record.description);
      const priority = normalizePriority(record.priority) ?? idx + 1;

      return action || why || how
        ? { action: action || "Suggested improvement", why, how, priority }
        : null;
    })
    .filter(Boolean) as NormalizedPlanItem[];
}

function normalizeChecklist(items?: ReportChecklistItem[] | null): NormalizedChecklistItem[] {
  return asArray(items)
    .map((item) => {
      if (typeof item === "string") {
        const text = item.trim();
        return text ? { item: text } : null;
      }
      const record = asRecord(item);
      if (!record) return null;

      const text = firstNonEmptyText(
        record.item,
        record.label,
        record.title,
        record.text,
        record.message,
      );
      const done =
        normalizeDone(record.done) ??
        normalizeDone(record.checked) ??
        normalizeDone(record.complete) ??
        normalizeDone(record.status);

      return text ? { item: text, done } : null;
    })
    .filter(Boolean) as NormalizedChecklistItem[];
}

function confidenceBandFrom(finalConfidence?: number) {
  if (typeof finalConfidence !== "number") return "—";
  if (finalConfidence >= 0.85) return "High";
  if (finalConfidence >= 0.65) return "Medium";
  return "Low";
}

function pct(n?: number) {
  if (typeof n !== "number") return "—";
  return `${Math.round(n * 100)}%`;
}

function msToSec(ms?: number) {
  if (typeof ms !== "number") return "—";
  return `${(ms / 1000).toFixed(1)}s`;
}

function fmtDate(iso?: string) {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

function clamp01(n?: number) {
  if (typeof n !== "number") return undefined;
  return Math.max(0, Math.min(1, n));
}

function statusBadgeClass(status: string) {
  const key = String(status || "unknown").toLowerCase();

  if (["done", "clean", "completed", "success", "stored"].includes(key)) {
    return "bg-emerald-400/12 text-emerald-200 border-emerald-400/20";
  }
  if (["running", "queued", "uploaded", "scanning"].includes(key)) {
    return "bg-blue-400/12 text-blue-100 border-blue-400/20";
  }
  if (["failed", "quarantined", "error"].includes(key)) {
    return "bg-rose-400/12 text-rose-200 border-rose-400/20";
  }

  return "bg-white/8 text-slate-300 border-white/10";
}

function ConfidenceMeter({ value }: { value?: number }) {
  const v = clamp01(value);
  const pctValue = typeof v === "number" ? Math.round(v * 100) : null;

  return (
    <div className="grid gap-3">
      <div className="flex items-center justify-between text-xs text-slate-400">
        <span>Confidence</span>
        <span className="font-black text-white">{pctValue === null ? "—" : `${pctValue}%`}</span>
      </div>

      <div className="h-3 overflow-hidden rounded-full bg-white/10">
        <div
          className="h-full rounded-full bg-gradient-to-r from-blue-500 to-indigo-400"
          style={{ width: pctValue === null ? "0%" : `${pctValue}%` }}
        />
      </div>

      <div className="text-xs text-slate-400">
        Band: <span className="font-bold text-white">{confidenceBandFrom(v)}</span>
      </div>
    </div>
  );
}

function MediaItem({
  fileId,
  media_index,
  media_type,
  metadata,
}: {
  fileId: string;
  media_index: number;
  media_type: string;
  metadata?: any;
}) {
  const [url, setUrl] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const derived_path = metadata?.derived_path;
  const derived_bucket = metadata?.derived_bucket;

  useEffect(() => {
    if (!derived_path) {
      setUrl(null);
      setErr(null);
      return;
    }

    let cancelled = false;

    (async () => {
      try {
        setErr(null);

        const qs = new URLSearchParams({
          file_id: fileId,
          derived_path: String(derived_path),
        });
        if (derived_bucket) qs.set("derived_bucket", String(derived_bucket));

        const res = await fetchWithAuth(backendUrl(`/media/signed-url?${qs.toString()}`));
        if (!res.ok) {
          const txt = await res.text().catch(() => "");
          throw new Error(txt || `Failed to get signed URL (${res.status})`);
        }

        const json = await res.json();
        const nextUrl = json?.url || null;
        if (!cancelled) setUrl(nextUrl);
      } catch (e: any) {
        if (!cancelled) setErr(e?.message || "Failed to load media URL");
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [fileId, derived_path, derived_bucket]);

  const kind = (media_type || "").toLowerCase();

  return (
    <div className="rounded-[24px] border border-white/10 bg-white/[0.04] p-4">
      <div className="mb-3 flex flex-wrap items-center gap-3">
        <div className="rounded-full border border-blue-300/15 bg-blue-500/10 px-3 py-1 text-xs font-bold text-blue-100">
          {kind} #{media_index}
        </div>
        <span className="break-all text-xs text-slate-400">{derived_path || ""}</span>
      </div>

      {err && <div className="text-sm text-rose-200">{err}</div>}

      {!derived_path ? (
        <div className="text-sm text-slate-400">No derived_path in metadata.</div>
      ) : !url ? (
        <div className="text-sm text-slate-400">Loading media...</div>
      ) : kind === "image" ? (
        <img
          src={url}
          alt={`media-${media_index}`}
          className="mt-2 w-full max-w-[760px] rounded-2xl border border-white/10"
        />
      ) : kind === "video" ? (
        <video
          src={url}
          controls
          className="mt-2 w-full max-w-[760px] rounded-2xl border border-white/10"
        />
      ) : kind === "audio" ? (
        <audio src={url} controls className="mt-2 w-full max-w-[760px]" />
      ) : (
        <a href={url} target="_blank" rel="noreferrer" className="inline-flex rounded-xl border border-white/10 bg-white/8 px-3 py-2 text-sm font-semibold text-blue-100">
          Open media
        </a>
      )}
    </div>
  );
}

export default function ResultsPage() {
  const params = useParams();
  const fileId = typeof params?.fileId === "string" ? params.fileId : "";

  const [data, setData] = useState<ResultsResp | null>(null);
  const [latest, setLatest] = useState<Phase7LatestResp | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [reanalyzing, setReanalyzing] = useState(false);
  const [langchainGenerating, setLangchainGenerating] = useState(false);
  const [langchainMeta, setLangchainMeta] = useState<LangChainMeta | null>(null);

  const inflightRef = useRef(false);

  async function loadAll() {
    if (!fileId || inflightRef.current) return;

    inflightRef.current = true;
    setError(null);

    try {
      const [resA, resB] = await Promise.all([
        fetchWithAuth(backendUrl(`/results/${fileId}`)),
        fetchWithAuth(backendUrl(`/phase7/latest/student/${fileId}`)),
      ]);

      if (!resA.ok) {
        const txt = await resA.text().catch(() => "");
        throw new Error(txt || `Failed to load results (${resA.status})`);
      }

      const jsonA = (await resA.json()) as ResultsResp;
      setData(jsonA);

      if (resB.ok) {
        const jsonB = (await resB.json()) as Phase7LatestResp;
        setLatest(jsonB);
      } else {
        setLatest({ found: false });
      }
    } catch (e: any) {
      setError(e?.message || "Failed to load");
    } finally {
      setLoading(false);
      inflightRef.current = false;
    }
  }

  async function reanalyze() {
    if (!fileId || reanalyzing) return;
    setReanalyzing(true);
    setError(null);
    try {
      // 10-minute timeout — LLM generation can take several minutes
      const res = await fetchWithAuth(
        backendUrl(`/phase7/student/generate`),
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ file_id: fileId, force: true }),
        },
        { timeoutMs: 600000 }
      );
      if (!res.ok) {
        const txt = await res.text().catch(() => "");
        throw new Error(txt || `Re-analyze failed (${res.status})`);
      }
      const json = await res.json();
      if (json?.stored?.report_json) {
        setLatest({ found: true, item: json.stored });
      } else {
        await loadAll();
      }
    } catch (e: any) {
      setError(e?.message || "Re-analyze failed");
    } finally {
      setReanalyzing(false);
    }
  }

  async function generateLangChain() {
    if (!fileId || langchainGenerating) return;
    setLangchainGenerating(true);
    setError(null);
    try {
      const res = await fetchWithAuth(
        backendUrl(`/langchain/student/generate`),
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
      setLatest({
        found: true,
        item: {
          id: json.stored?.id || "",
          file_id: fileId,
          role: "student",
          created_at: json.stored?.created_at || undefined,
          needs_review: json.meta?.needs_review || false,
          report_json: json.report,
        },
      });
    } catch (e: any) {
      setError(e?.message || "LangChain generate failed");
    } finally {
      setLangchainGenerating(false);
    }
  }

  useEffect(() => {
    if (!fileId) return;
    setLoading(true);
    void loadAll();
  }, [fileId]);

  useEffect(() => {
    const jobs = data?.jobs ?? [];
    const runningJobs = jobs.some((j) => {
      const s = String(j.status || "").toLowerCase();
      return s === "queued" || s === "running";
    });
    const reportNotReady = latest ? !latest.found : true;

    if (!runningJobs && !reportNotReady) return;

    const interval = setInterval(() => {
      void loadAll();
    }, 2500);

    return () => clearInterval(interval);
  }, [data?.jobs, latest?.found]);

  const jobs = useMemo(() => data?.jobs ?? [], [data?.jobs]);
  const tables = useMemo(() => data?.tables ?? [], [data?.tables]);
  const media = useMemo(() => data?.media ?? [], [data?.media]);
  const events = useMemo(() => data?.events ?? [], [data?.events]);

  const report = latest?.item?.report_json;
  const agreement = latest?.item?.model_versions?.agreement;

  const finalConf =
    agreement?.final_confidence ??
    report?.model_agreement?.final_confidence;

  const mlBucket = agreement?.ml_bucket_0_to_4;
  const mlConf =
    typeof mlBucket === "number"
      ? Math.max(0, Math.min(1, mlBucket / 4))
      : report?.model_agreement?.ml_confidence;

  const llmConf = report?.model_agreement?.llm_confidence;
  const llmMode = latest?.item?.model_versions?.llm_model_used;
  const isSafeMode = llmMode === "safe_mode";

  const band = confidenceBandFrom(finalConf);
  const needsReview = !!(latest?.item?.needs_review || report?.safety?.needs_review);
  const normalizedReport = useMemo(() => {
    const safeReport = report || {};

    const issues = normalizeIssues(safeReport.issues);
    const improvementPlan = normalizeImprovementPlan(safeReport.improvement_plan);
    const checklist = normalizeChecklist(safeReport.checklist);

    return {
      summary:
        typeof safeReport.summary === "string" && safeReport.summary.trim()
          ? safeReport.summary.trim()
          : "No summary available.",
      issues,
      improvementPlan,
      checklist,
    };
  }, [report]);
  const issueItems = normalizedReport.issues;
  const improvementItems = normalizedReport.improvementPlan;
  const checklistItems = normalizedReport.checklist;
  const hasRenderableReport =
    !!report &&
    Object.keys(report).length > 0 &&
    (
      !!normalizedReport.summary ||
      normalizedReport.issues.length > 0 ||
      normalizedReport.improvementPlan.length > 0 ||
      normalizedReport.checklist.length > 0
    );

  const ragMeta = useMemo(() => buildRenderableRagMeta(latest?.item), [latest?.item]);

  if (!fileId) {
    return <div className="p-6 text-white">Missing file ID</div>;
  }

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top,_#213b9f_0%,_#0b1537_38%,_#071126_100%)] text-white">
      <div className="mx-auto max-w-[1450px] px-4 py-6 sm:px-6 lg:px-8">
        <div className="grid gap-6">
          <div className="rounded-[30px] border border-white/10 bg-white/[0.05] p-5 shadow-[0_20px_70px_rgba(0,0,0,0.28)] backdrop-blur-xl">
            <div className="flex flex-col gap-4 xl:flex-row xl:items-center xl:justify-between">
              <div>
                <div className="text-xs font-semibold uppercase tracking-[0.24em] text-blue-200/70">
                  Student / Results
                </div>
                <h1 className="mt-2 text-3xl font-black tracking-tight">Results</h1>
                <div className="mt-2 text-sm text-slate-300">
                  File ID: <span className="font-mono">{fileId}</span>
                </div>

                <div className="mt-3 flex flex-wrap items-center gap-3">
                  <span className={`inline-flex rounded-full border px-3 py-1 text-xs font-bold ${statusBadgeClass(data?.file?.status || "unknown")}`}>
                    {String(data?.file?.status || "unknown").toLowerCase()}
                  </span>
                  {data?.file?.mime_type ? (
                    <span className="text-xs text-slate-400">{data.file.mime_type}</span>
                  ) : null}
                  {needsReview ? (
                    <span className="inline-flex items-center gap-2 rounded-full border border-amber-300/20 bg-amber-400/10 px-3 py-1 text-xs font-bold text-amber-200">
                      <ShieldAlert size={14} />
                      Needs review
                    </span>
                  ) : null}
                  {langchainMeta?.discrepancy_flag ? (
                    <span className="inline-flex items-center gap-2 rounded-full border border-orange-300/20 bg-orange-400/10 px-3 py-1 text-xs font-bold text-orange-200">
                      <AlertTriangle size={14} />
                      ML–LLM Disagreement
                    </span>
                  ) : null}
                  {langchainMeta?.execution_mode ? (
                    <span className="inline-flex rounded-full border border-violet-300/20 bg-violet-500/10 px-3 py-1 text-xs font-bold text-violet-200">
                      Mode: {langchainMeta.execution_mode}
                    </span>
                  ) : null}
                </div>
              </div>

              <div className="flex flex-wrap gap-3">
                <Link
                  href="/student"
                  className="inline-flex items-center gap-2 rounded-2xl border border-white/10 bg-white/10 px-4 py-3 text-sm font-bold text-white transition hover:bg-white/15"
                >
                  <ArrowLeft size={16} />
                  Back
                </Link>
                <button
                  onClick={() => void loadAll()}
                  className="inline-flex items-center gap-2 rounded-2xl border border-white/10 bg-white/10 px-4 py-3 text-sm font-bold text-white transition hover:bg-white/15"
                >
                  <RefreshCw size={16} className={loading ? "animate-spin" : ""} />
                  {loading ? "Loading..." : "Refresh"}
                </button>
                <button
                  onClick={() => void reanalyze()}
                  disabled={reanalyzing}
                  className="inline-flex items-center gap-2 rounded-2xl border border-blue-400/30 bg-blue-500/20 px-4 py-3 text-sm font-bold text-blue-100 transition hover:bg-blue-500/30 disabled:opacity-50"
                >
                  <RefreshCw size={16} className={reanalyzing ? "animate-spin" : ""} />
                  {reanalyzing ? "Re-analyzing..." : "Re-analyze"}
                </button>
                <button
                  onClick={() => void generateLangChain()}
                  disabled={langchainGenerating}
                  className="inline-flex items-center gap-2 rounded-2xl border border-violet-400/30 bg-violet-500/20 px-4 py-3 text-sm font-bold text-violet-100 transition hover:bg-violet-500/30 disabled:opacity-50"
                >
                  <RefreshCw size={16} className={langchainGenerating ? "animate-spin" : ""} />
                  {langchainGenerating ? "Generating..." : "LangChain"}
                </button>
              </div>
            </div>
          </div>

          {error ? (
            <div className="rounded-[24px] border border-rose-300/20 bg-rose-400/10 px-4 py-3 text-sm text-rose-200">
              {error}
            </div>
          ) : null}

          <div className="grid gap-6 xl:grid-cols-[1.2fr_0.8fr]">
            <section className="rounded-[30px] border border-white/10 bg-white/[0.05] p-5 shadow-[0_20px_70px_rgba(0,0,0,0.25)] backdrop-blur-xl">
              <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
                <div>
                  <h2 className="text-xl font-black">AI Feedback</h2>
                  <div className="mt-1 text-sm text-slate-400">
                    {!latest ? "Loading..." : latest.found ? "Stored report available" : "Not stored yet"}
                    {latest?.item?.created_at ? ` • ${fmtDate(latest.item.created_at)}` : ""}
                  </div>
                </div>

                <div className="flex flex-wrap items-center gap-3 text-sm">
                  <span className="rounded-full border border-blue-300/15 bg-blue-500/10 px-3 py-1 text-blue-100">
                    Band: <b>{band}</b>
                  </span>
                  <span className="rounded-full border border-white/10 bg-white/8 px-3 py-1 text-slate-200">
                    Final: <b>{pct(finalConf)}</b>
                  </span>
                </div>
              </div>

              {!latest ? (
                <MutedBox text="Loading AI feedback..." />
              ) : !latest.found ? (
                <MutedBox text="No AI report stored yet. If ingestion is complete, the report should appear shortly." />
              ) : !hasRenderableReport ? (
                <MutedBox text="AI report is being prepared..." />
              ) : (
                <div className="grid gap-5">
                  <div className="grid gap-4 md:grid-cols-3">
                    <MetricCard title="Final">
                      <ConfidenceMeter value={finalConf} />
                    </MetricCard>
                    <MetricCard title="ML">
                      <ConfidenceMeter value={mlConf} />
                    </MetricCard>
                    <MetricCard title="LLM">
                      {typeof llmConf === "number" ? (
                        <ConfidenceMeter value={llmConf} />
                      ) : (
                        <div className="text-sm leading-6 text-slate-400">
                          {isSafeMode ? "Unavailable in safe mode." : "LLM confidence unavailable."}
                        </div>
                      )}
                    </MetricCard>
                  </div>

                  <ContentCard title="Summary">
                    <div className="whitespace-pre-wrap text-sm leading-7 text-slate-300">
                      {normalizedReport.summary}
                    </div>
                  </ContentCard>

                  <ContentCard title="Issues">
                    {issueItems.length === 0 ? (
                      <MutedText text="No issues listed." />
                    ) : (
                      <div className="grid gap-3">
                        {issueItems.map((it, idx) => (
                          <div key={idx} className="rounded-[22px] border border-white/10 bg-white/[0.04] p-4">
                            <div className="flex flex-wrap items-center justify-between gap-3">
                              <strong className="text-white">{it.title}</strong>
                              <span className="rounded-full border border-white/10 bg-white/8 px-3 py-1 text-xs text-slate-300">
                                Severity: {it.severity || "—"}
                              </span>
                            </div>
                            {it.evidence ? (
                              <div className="mt-2 text-sm leading-6 text-slate-300">{it.evidence}</div>
                            ) : null}
                          </div>
                        ))}
                      </div>
                    )}
                  </ContentCard>

                  <ContentCard title="Improvement plan">
                    {improvementItems.length === 0 ? (
                      <MutedText text="No improvement steps." />
                    ) : (
                      <ol className="grid gap-4 pl-5">
                        {improvementItems.slice(0, 10).map((p, idx) => (
                          <li key={idx} className="text-sm text-slate-300">
                            <div className="font-bold text-white">
                              {p.action}
                              {typeof p.priority === "number" ? (
                                <span className="ml-2 text-xs text-slate-400">(Priority {p.priority})</span>
                              ) : null}
                            </div>
                            {p.why ? <div className="mt-1 text-slate-400">Why: {p.why}</div> : null}
                            {p.how ? <div className="mt-1 text-slate-400">How: {p.how}</div> : null}
                          </li>
                        ))}
                      </ol>
                    )}
                  </ContentCard>

                  <ContentCard title="Checklist">
                    {checklistItems.length === 0 ? (
                      <MutedText text="No checklist items." />
                    ) : (
                      <div className="grid gap-3">
                        {checklistItems.map((c, idx) => (
                          <label
                            key={idx}
                            className="flex items-center gap-3 rounded-2xl border border-white/10 bg-white/[0.03] px-4 py-3 text-sm text-slate-300"
                          >
                            <input type="checkbox" checked={!!c.done} readOnly className="h-4 w-4 accent-blue-500" />
                            <span>{c.item}</span>
                          </label>
                        ))}
                      </div>
                    )}
                  </ContentCard>

                  <RagEvidencePanel
                    ragMeta={ragMeta}
                    title="Student RAG Evidence"
                    emptyMessage="No RAG evidence was attached to this student report. If you expected citations or snippets, the stored report currently has empty RAG fields."
                  />
                </div>
              )}
            </section>

            <section className="grid gap-6">
              <ContentCard title="Model + timing">
                <div className="grid gap-3 text-sm text-slate-300">
                  {isSafeMode ? (
                    <>
                      <div>
                        LLM mode: <b className="text-white">Safe mode</b>
                      </div>
                      {latest?.item?.model_versions?.llm_fallback ? (
                        <div>
                          Fallback model: <b className="text-white">{latest.item.model_versions.llm_fallback}</b>
                        </div>
                      ) : null}
                    </>
                  ) : (
                    <div>
                      LLM used:{" "}
                      <b className="text-white">
                        {latest?.item?.model_versions?.llm_model_used ||
                          latest?.item?.model_versions?.llm_primary ||
                          "—"}
                      </b>
                      {latest?.item?.model_versions?.llm_fallback ? (
                        <span className="text-slate-400">
                          {" "}
                          (fallback: {latest.item.model_versions.llm_fallback})
                        </span>
                      ) : null}
                    </div>
                  )}

                  <div>
                    Timings: total <b className="text-white">{msToSec(latest?.item?.model_versions?.timings_ms?.total)}</b>, ingestion{" "}
                    <b className="text-white">{msToSec(latest?.item?.model_versions?.timings_ms?.ingestion)}</b>, ML{" "}
                    <b className="text-white">{msToSec(latest?.item?.model_versions?.timings_ms?.ai_service)}</b>, LLM{" "}
                    <b className="text-white">{msToSec(latest?.item?.model_versions?.timings_ms?.llm_service)}</b>
                  </div>

                  {latest?.item?.model_versions?.request_id ? (
                    <div>
                      Request ID: <code>{latest.item.model_versions.request_id}</code>
                    </div>
                  ) : null}

                  {report?.safety?.reason ? (
                    <div>
                      Safety: <b className="text-white">{report.safety.needs_review ? "Needs review" : "OK"}</b>{" "}
                      <span className="text-slate-400">— {report.safety.reason}</span>
                    </div>
                  ) : null}
                </div>
              </ContentCard>

              <ContentCard title={`Jobs (${jobs.length})`}>
                {jobs.length === 0 ? (
                  <MutedText text="No jobs yet." />
                ) : (
                  <div className="grid gap-3">
                    {jobs.map((j) => (
                      <div key={j.id} className="rounded-[22px] border border-white/10 bg-white/[0.04] p-4">
                        <div className="flex flex-wrap items-center justify-between gap-3">
                          <div className="font-bold text-white">{j.job_type}</div>
                          <span className={`inline-flex rounded-full border px-3 py-1 text-xs font-bold ${statusBadgeClass(j.status)}`}>
                            {String(j.status || "").toLowerCase()}
                          </span>
                        </div>
                        {j.error_message ? (
                          <div className="mt-2 text-sm text-rose-200">{j.error_message}</div>
                        ) : null}
                        {j.error_code && !j.error_message ? (
                          <div className="mt-2 text-sm text-rose-200">{j.error_code}</div>
                        ) : null}
                      </div>
                    ))}
                  </div>
                )}
              </ContentCard>

              <ContentCard title={`Processing events (${events.length})`}>
                {events.length === 0 ? (
                  <MutedText text="No events." />
                ) : (
                  <div className="grid gap-3">
                    {events.map((e, i) => (
                      <div key={i} className="flex items-center justify-between gap-3 rounded-2xl border border-white/10 bg-white/[0.03] px-4 py-3 text-sm">
                        <div className="font-semibold text-white">{e.event_type}</div>
                        <div className="text-slate-400">{fmtDate(e.created_at)}</div>
                      </div>
                    ))}
                  </div>
                )}
              </ContentCard>
            </section>
          </div>

          <div className="grid gap-6 xl:grid-cols-2">
            <ContentCard title="Extracted text" icon={<FileText size={18} />}>
              {!data?.text?.redacted_text ? (
                <MutedText text="No extracted text." />
              ) : (
                <pre className="whitespace-pre-wrap break-words rounded-2xl border border-white/10 bg-[#091327]/90 p-4 text-sm leading-6 text-slate-300">
                  {data.text.redacted_text}
                </pre>
              )}
            </ContentCard>

            <ContentCard title="Transcript" icon={<Mic size={18} />}>
              {!data?.transcript?.redacted_transcript ? (
                <MutedText text="No transcript." />
              ) : (
                <pre className="whitespace-pre-wrap break-words rounded-2xl border border-white/10 bg-[#091327]/90 p-4 text-sm leading-6 text-slate-300">
                  {data.transcript.redacted_transcript}
                </pre>
              )}
            </ContentCard>
          </div>

          <ContentCard title={`Tables (${tables.length})`} icon={<Table2 size={18} />}>
            {tables.length === 0 ? (
              <MutedText text="No tables." />
            ) : (
              <div className="grid gap-6">
                {tables.map((t) => (
                  <div key={t.table_index} className="rounded-[24px] border border-white/10 bg-white/[0.03] p-4">
                    <div className="mb-4 font-bold text-white">
                      Table #{t.table_index} {t.sheet_name ? `(${t.sheet_name})` : ""}
                    </div>

                    <div className="overflow-x-auto rounded-2xl border border-white/10">
                      <table className="min-w-full border-collapse">
                        <thead className="bg-white/[0.05]">
                          <tr>
                            {t.columns.map((c, i) => (
                              <th key={i} className="border-b border-white/10 px-4 py-3 text-left text-xs font-extrabold uppercase tracking-[0.16em] text-slate-300">
                                {c}
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {t.rows.slice(0, 50).map((r, i) => (
                            <tr key={i} className="border-b border-white/5">
                              {r.map((cell, j) => (
                                <td key={j} className="px-4 py-3 text-sm text-slate-300">
                                  {String(cell ?? "")}
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>

                    {t.rows.length > 50 ? (
                      <div className="mt-3 text-xs text-slate-400">
                        Showing first 50 rows of {t.rows.length}.
                      </div>
                    ) : null}
                  </div>
                ))}
              </div>
            )}
          </ContentCard>

          <ContentCard title={`Media (${media.length})`} icon={<ImageIcon size={18} />}>
            {media.length === 0 ? (
              <MutedText text="No media records." />
            ) : (
              <div className="grid gap-4">
                {media.map((m) => (
                  <MediaItem
                    key={`${m.media_type}-${m.media_index}`}
                    fileId={fileId}
                    media_index={m.media_index}
                    media_type={m.media_type}
                    metadata={m.metadata}
                  />
                ))}
              </div>
            )}
          </ContentCard>
        </div>
      </div>
    </div>
  );
}

function MetricCard({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-[24px] border border-white/10 bg-white/[0.04] p-4">
      <div className="mb-3 text-xs font-extrabold uppercase tracking-[0.18em] text-slate-400">{title}</div>
      {children}
    </div>
  );
}

function ContentCard({
  title,
  icon,
  children,
}: {
  title: string;
  icon?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <section className="rounded-[30px] border border-white/10 bg-white/[0.05] p-5 shadow-[0_20px_70px_rgba(0,0,0,0.22)] backdrop-blur-xl">
      <div className="mb-4 flex items-center gap-3">
        {icon ? (
          <div className="grid h-10 w-10 place-items-center rounded-2xl bg-blue-500/15 text-blue-200">
            {icon}
          </div>
        ) : null}
        <h3 className="text-lg font-black">{title}</h3>
      </div>
      {children}
    </section>
  );
}

function MutedBox({ text }: { text: string }) {
  return (
    <div className="rounded-[22px] border border-dashed border-white/10 bg-white/[0.03] px-4 py-5 text-sm text-slate-400">
      {text}
    </div>
  );
}

function MutedText({ text }: { text: string }) {
  return <div className="text-sm text-slate-400">{text}</div>;
}
