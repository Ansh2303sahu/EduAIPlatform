"use client";

import { useEffect, useState } from "react";
import {
  Bot,
  BrainCircuit,
  GitBranch,
  Network,
  RefreshCw,
  ShieldCheck,
  Sparkles,
  Workflow,
} from "lucide-react";

import { backendUrl } from "@/lib/backendUrl";
import { fetchJsonWithAuth } from "@/lib/fetchWithAuth";
import type { ReactNode } from "react";

type Role = "student" | "professor";

type AICheckResponse = {
  file_id: string;
  role: Role;
  summary: {
    selected_pipeline: string;
    status: string;
    created_at?: string | null;
    needs_review: boolean;
    confidence_score?: number | null;
    confidence_band?: string | null;
    report_summary: string;
  };
  langchain: {
    available: boolean;
    pipeline: string;
    chain_name: string;
    chain_version: string;
    prompt_version: string;
    schema_version: string;
    provider: string;
    model_used: string;
    primary_model: string;
    fallback_model: string;
    fallback_used: boolean;
    execution_mode: string;
    decision_source: string;
    discrepancy_flag?: boolean | null;
    retrieval_mode: string;
    retrieved_chunk_count: number;
    confidence_score?: number | null;
    summary: string;
  };
  langgraph: {
    available: boolean;
    pipeline: string;
    graph_name: string;
    graph_version: string;
    prompt_version: string;
    output_version: string;
    final_status: string;
    safe_mode: boolean;
    total_steps: number;
    total_latency_ms: number;
    node_count: number;
    decision_count: number;
    failure_count: number;
    trace_summary: string;
    warnings: string[];
  };
  genai: {
    available: boolean;
    pipeline: string;
    model_version: string;
    validator_model_version: string;
    final_status: string;
    confidence_score?: number | null;
    confidence_band?: string | null;
    report_summary: string;
    warning_count: number;
  };
  rag: {
    enabled: boolean;
    confidence_score?: number | null;
    confidence_label: string;
    citations_count: number;
    retrieved_chunk_count: number;
    query: string;
    collection_name: string;
    safe_review: boolean;
    summary: string;
  };
  ml: {
    available: boolean;
    confidence_score?: number | null;
    model_names: string[];
    source: string;
    summary: string;
  };
  llm: {
    available: boolean;
    model_used: string;
    primary_model: string;
    fallback_model: string;
    route: string;
    source: string;
  };
  mcp: {
    enabled: boolean;
    orchestration_enabled: boolean;
    llm_enabled: boolean;
    graph_used: boolean;
    tool_call_count: number;
    visible_tools: string[];
    summary: string;
  };
  n8n: {
    configured: boolean;
    generation_bridge_active: boolean;
    integrations: {
      assessment: boolean;
      file_upload: boolean;
      low_confidence: boolean;
      pipeline_failure: boolean;
    };
    summary: string;
  };
};

function fmtDate(iso?: string | null) {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

function pct(value?: number | null) {
  if (typeof value !== "number") return "—";
  return `${Math.round(value * 100)}%`;
}

function msToSec(ms?: number | null) {
  if (typeof ms !== "number") return "—";
  return `${(ms / 1000).toFixed(1)}s`;
}

function humanize(value?: string | null) {
  if (!value) return "—";
  return value.replace(/_/g, " ");
}

function boolTone(value: boolean) {
  return value
    ? "border-emerald-300/20 bg-emerald-400/10 text-emerald-200"
    : "border-white/10 bg-white/8 text-slate-300";
}

export default function AICheckPanel({
  fileId,
  role,
}: {
  fileId: string;
  role: Role;
}) {
  const [data, setData] = useState<AICheckResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function load({ keepData = true }: { keepData?: boolean } = {}) {
    if (!fileId) return;
    if (!keepData) setData(null);
    setError(null);

    try {
      const json = (await fetchJsonWithAuth(
        backendUrl(`/ai/check/${fileId}?role=${role}`),
        { method: "GET" },
        { timeoutMs: 30000 }
      )) as AICheckResponse;
      setData(json);
    } catch (e: any) {
      const message = String(e?.message || "Failed to load AI check");
      if (e?.status === 404 && (message === "Not Found" || message === "Request failed (404)")) {
        setError(
          "AI check route is not available from the running backend yet. If you are using Docker Compose, restart or recreate the backend service once."
        );
      } else {
        setError(message);
      }
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }

  useEffect(() => {
    setLoading(true);
    void load({ keepData: false });
  }, [fileId, role]);

  return (
    <section className="rounded-[30px] border border-white/10 bg-white/[0.05] p-5 shadow-[0_20px_70px_rgba(0,0,0,0.22)] backdrop-blur-xl">
      <div className="flex flex-col gap-4 xl:flex-row xl:items-start xl:justify-between">
        <div>
          <div className="text-xs font-extrabold uppercase tracking-[0.18em] text-blue-200/70">
            AI Check
          </div>
          <h3 className="mt-2 text-2xl font-black text-white">Pipeline map and result quality</h3>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-300">
            One view for LangChain, LangGraph, GenAI, RAG, LLM, ML, MCP, and n8n wiring for this {role} file.
          </p>
          <p className="mt-2 max-w-3xl text-xs leading-5 text-slate-400">
            Refresh reloads the latest stored diagnostics only. To apply backend changes or generate a new report for this file,
            run the main AI generation action again.
          </p>
          <div className="mt-3 text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
            File {fileId}
          </div>
        </div>

        <button
          onClick={() => {
            setRefreshing(true);
            void load();
          }}
          disabled={refreshing}
          className="inline-flex items-center gap-2 rounded-2xl border border-white/10 bg-white/10 px-4 py-3 text-sm font-bold text-white transition hover:bg-white/15 disabled:opacity-60"
        >
          <RefreshCw size={16} className={refreshing ? "animate-spin" : ""} />
          {refreshing ? "Refreshing..." : "Refresh AI Check"}
        </button>
      </div>

      {error ? (
        <div className="mt-5 rounded-[22px] border border-rose-300/20 bg-rose-400/10 px-4 py-3 text-sm text-rose-200">
          {error}
        </div>
      ) : null}

      {loading && !data ? (
        <div className="mt-5 rounded-[22px] border border-dashed border-white/10 bg-white/[0.03] px-4 py-5 text-sm text-slate-400">
          Loading AI check...
        </div>
      ) : data ? (
        <div className="mt-6 grid gap-6">
          <div className="rounded-[24px] border border-white/10 bg-[#0a1730]/80 p-5">
            <div className="flex flex-col gap-4 xl:flex-row xl:items-start xl:justify-between">
              <div className="grid gap-3">
                <div className="text-sm text-slate-300">
                  Pipeline:{" "}
                  <b className="text-white">{data.summary.selected_pipeline || "not available"}</b>
                </div>
                <div className="text-sm text-slate-300">
                  Created: <b className="text-white">{fmtDate(data.summary.created_at)}</b>
                </div>
                <div className="max-w-3xl text-sm leading-7 text-slate-300">
                  {data.summary.report_summary || "No stored AI report summary yet."}
                </div>
              </div>

              <div className="flex flex-wrap gap-2">
                <Pill label={`Status: ${humanize(data.summary.status)}`} tone="primary" />
                <Pill label={`Confidence: ${pct(data.summary.confidence_score)}`} tone="neutral" />
                <Pill
                  label={`Band: ${humanize(data.summary.confidence_band)}`}
                  tone={data.summary.confidence_band === "high" ? "good" : "neutral"}
                />
                <Pill
                  label={data.summary.needs_review ? "Needs review" : "Review clear"}
                  tone={data.summary.needs_review ? "warn" : "good"}
                />
              </div>
            </div>
          </div>

          <div className="grid gap-6 xl:grid-cols-3">
            <InfoCard
              icon={<Bot size={18} />}
              title="LangChain"
              subtitle={
                data.langchain.available
                  ? data.langchain.summary || "Stored LangChain execution metadata is available."
                  : "No stored LangChain result was found for this file yet."
              }
            >
              <MetricGrid
                items={[
                  { label: "Chain", value: data.langchain.chain_name || "â€”" },
                  { label: "Version", value: data.langchain.chain_version || "â€”" },
                  { label: "Prompt", value: data.langchain.prompt_version || "â€”" },
                  {
                    label: "Model",
                    value: data.langchain.model_used || data.langchain.primary_model || "â€”",
                  },
                  { label: "Mode", value: humanize(data.langchain.execution_mode) },
                  { label: "Confidence", value: pct(data.langchain.confidence_score) },
                ]}
              />

              <div className="mt-4 flex flex-wrap gap-2">
                <span
                  className={`rounded-full border px-3 py-1 text-xs font-bold ${boolTone(
                    data.langchain.available
                  )}`}
                >
                  {data.langchain.available ? "Stored result" : "Not stored"}
                </span>
                <span
                  className={`rounded-full border px-3 py-1 text-xs font-bold ${boolTone(
                    data.langchain.fallback_used
                  )}`}
                >
                  {data.langchain.fallback_used ? "Fallback used" : "Primary path"}
                </span>
                {data.langchain.discrepancy_flag !== null &&
                data.langchain.discrepancy_flag !== undefined ? (
                  <span
                    className={`rounded-full border px-3 py-1 text-xs font-bold ${boolTone(
                      !data.langchain.discrepancy_flag
                    )}`}
                  >
                    {data.langchain.discrepancy_flag ? "Discrepancy flagged" : "No discrepancy"}
                  </span>
                ) : null}
              </div>

              <div className="mt-4 grid gap-2 rounded-[22px] border border-white/10 bg-white/[0.03] p-4 text-sm text-slate-300">
                <div>
                  Retrieval:{" "}
                  <span className="text-white">
                    {humanize(data.langchain.retrieval_mode)} / {data.langchain.retrieved_chunk_count || 0} chunks
                  </span>
                </div>
                <div>
                  Provider: <span className="text-white">{data.langchain.provider || "â€”"}</span>
                </div>
                <div>
                  Decision source: <span className="text-white">{humanize(data.langchain.decision_source)}</span>
                </div>
                <div>
                  Primary / fallback:{" "}
                  <span className="text-white">
                    {data.langchain.primary_model || "â€”"} / {data.langchain.fallback_model || "â€”"}
                  </span>
                </div>
              </div>
            </InfoCard>

            <InfoCard
              icon={<Workflow size={18} />}
              title="LangGraph"
              subtitle={
                data.langgraph.available
                  ? data.langgraph.trace_summary || "Stored LangGraph execution metadata is available."
                  : "No Phase 15/16 LangGraph run stored yet."
              }
            >
              <MetricGrid
                items={[
                  { label: "Graph", value: data.langgraph.graph_name || "—" },
                  { label: "Version", value: data.langgraph.graph_version || "—" },
                  { label: "Final", value: humanize(data.langgraph.final_status) },
                  { label: "Latency", value: msToSec(data.langgraph.total_latency_ms) },
                  { label: "Steps", value: String(data.langgraph.total_steps || 0) },
                  { label: "Safe mode", value: data.langgraph.safe_mode ? "Yes" : "No" },
                ]}
              />

              {data.langgraph.warnings.length > 0 ? (
                <div className="mt-4 grid gap-2">
                  {data.langgraph.warnings.slice(0, 3).map((warning, index) => (
                    <div
                      key={`${warning}-${index}`}
                      className="rounded-2xl border border-amber-300/20 bg-amber-400/10 px-3 py-2 text-sm text-amber-100"
                    >
                      {warning}
                    </div>
                  ))}
                </div>
              ) : null}
            </InfoCard>

            <InfoCard
              icon={<Sparkles size={18} />}
              title="GenAI"
              subtitle={
                data.genai.available
                  ? data.genai.report_summary || "Structured GenAI output is stored."
                  : "No stored Phase 15/16 GenAI prediction yet."
              }
            >
              <MetricGrid
                items={[
                  { label: "Model", value: data.genai.model_version || "—" },
                  { label: "Validator", value: data.genai.validator_model_version || "—" },
                  { label: "Status", value: humanize(data.genai.final_status) },
                  { label: "Confidence", value: pct(data.genai.confidence_score) },
                  { label: "Band", value: humanize(data.genai.confidence_band) },
                  { label: "Warnings", value: String(data.genai.warning_count || 0) },
                ]}
              />
            </InfoCard>
          </div>

          <div className="grid gap-6 xl:grid-cols-[1.1fr_0.9fr]">
            <InfoCard
              icon={<ShieldCheck size={18} />}
              title="RAG + Grounding"
              subtitle={data.rag.summary || "No retrieval summary yet."}
            >
              <MetricGrid
                items={[
                  { label: "Enabled", value: data.rag.enabled ? "Yes" : "No" },
                  { label: "Confidence", value: pct(data.rag.confidence_score) },
                  { label: "Label", value: humanize(data.rag.confidence_label) },
                  { label: "Citations", value: String(data.rag.citations_count || 0) },
                  { label: "Chunks", value: String(data.rag.retrieved_chunk_count || 0) },
                  { label: "Safe review", value: data.rag.safe_review ? "Yes" : "No" },
                ]}
              />

              {(data.rag.query || data.rag.collection_name) ? (
                <div className="mt-4 grid gap-2 rounded-[22px] border border-white/10 bg-white/[0.03] p-4 text-sm text-slate-300">
                  <div>
                    Query: <span className="text-white">{data.rag.query || "—"}</span>
                  </div>
                  <div>
                    Collection: <span className="text-white">{data.rag.collection_name || "—"}</span>
                  </div>
                </div>
              ) : null}
            </InfoCard>

            <InfoCard
              icon={<BrainCircuit size={18} />}
              title="LLM + ML"
              subtitle={data.ml.summary || "Model routing details unavailable."}
            >
              <MetricGrid
                items={[
                  { label: "LLM route", value: data.llm.route || "—" },
                  { label: "LLM source", value: data.llm.source || "—" },
                  { label: "Model used", value: data.llm.model_used || "—" },
                  { label: "ML score", value: pct(data.ml.confidence_score) },
                  { label: "ML source", value: data.ml.source || "—" },
                  { label: "ML active", value: data.ml.available ? "Yes" : "No" },
                ]}
              />

              {data.ml.model_names.length > 0 ? (
                <div className="mt-4 flex flex-wrap gap-2">
                  {data.ml.model_names.map((name) => (
                    <span
                      key={name}
                      className="rounded-full border border-blue-300/15 bg-blue-500/10 px-3 py-1 text-xs font-semibold text-blue-100"
                    >
                      {name}
                    </span>
                  ))}
                </div>
              ) : null}
            </InfoCard>
          </div>

          <div className="grid gap-6 xl:grid-cols-2">
            <InfoCard
              icon={<GitBranch size={18} />}
              title="MCP"
              subtitle={data.mcp.summary || "No MCP summary available."}
            >
              <div className="flex flex-wrap gap-2">
                <span className={`rounded-full border px-3 py-1 text-xs font-bold ${boolTone(data.mcp.enabled)}`}>
                  {data.mcp.enabled ? "Enabled" : "Disabled"}
                </span>
                <span
                  className={`rounded-full border px-3 py-1 text-xs font-bold ${boolTone(
                    data.mcp.orchestration_enabled
                  )}`}
                >
                  {data.mcp.orchestration_enabled ? "Workflow ready" : "Workflow off"}
                </span>
                <span className={`rounded-full border px-3 py-1 text-xs font-bold ${boolTone(data.mcp.llm_enabled)}`}>
                  {data.mcp.llm_enabled ? "LLM tools on" : "LLM tools off"}
                </span>
                <span className={`rounded-full border px-3 py-1 text-xs font-bold ${boolTone(data.mcp.graph_used)}`}>
                  {data.mcp.graph_used ? "Used in graph" : "Not used in graph"}
                </span>
              </div>

              <div className="mt-4 text-sm text-slate-300">
                Stored MCP tool calls for this file:{" "}
                <b className="text-white">{data.mcp.tool_call_count}</b>
              </div>

              {data.mcp.visible_tools.length > 0 ? (
                <div className="mt-4 flex flex-wrap gap-2">
                  {data.mcp.visible_tools.map((tool) => (
                    <span
                      key={tool}
                      className="rounded-full border border-white/10 bg-white/[0.05] px-3 py-1 text-xs font-semibold text-slate-200"
                    >
                      {tool}
                    </span>
                  ))}
                </div>
              ) : null}
            </InfoCard>

            <InfoCard
              icon={<Network size={18} />}
              title="n8n"
              subtitle={data.n8n.summary || "No n8n summary available."}
            >
              <div className="flex flex-wrap gap-2">
                <span className={`rounded-full border px-3 py-1 text-xs font-bold ${boolTone(data.n8n.configured)}`}>
                  {data.n8n.configured ? "Configured" : "Not configured"}
                </span>
                <span
                  className={`rounded-full border px-3 py-1 text-xs font-bold ${boolTone(
                    data.n8n.generation_bridge_active
                  )}`}
                >
                  {data.n8n.generation_bridge_active ? "Generation bridge active" : "No stored generation trace"}
                </span>
              </div>

              <div className="mt-4 grid gap-3 sm:grid-cols-2">
                <MiniStatus label="Assessment" active={data.n8n.integrations.assessment} />
                <MiniStatus label="File Upload" active={data.n8n.integrations.file_upload} />
                <MiniStatus label="Low Confidence" active={data.n8n.integrations.low_confidence} />
                <MiniStatus label="Pipeline Failure" active={data.n8n.integrations.pipeline_failure} />
              </div>
            </InfoCard>
          </div>
        </div>
      ) : null}
    </section>
  );
}

function InfoCard({
  icon,
  title,
  subtitle,
  children,
}: {
  icon: ReactNode;
  title: string;
  subtitle: string;
  children: ReactNode;
}) {
  return (
    <div className="rounded-[24px] border border-white/10 bg-white/[0.04] p-5">
      <div className="flex items-start gap-3">
        <div className="grid h-11 w-11 place-items-center rounded-2xl bg-blue-500/15 text-blue-100">
          {icon}
        </div>
        <div>
          <h4 className="text-lg font-black text-white">{title}</h4>
          <div className="mt-1 text-sm leading-6 text-slate-400">{subtitle}</div>
        </div>
      </div>
      <div className="mt-5">{children}</div>
    </div>
  );
}

function MetricGrid({
  items,
}: {
  items: Array<{ label: string; value: string }>;
}) {
  return (
    <div className="grid gap-3 sm:grid-cols-2">
      {items.map((item) => (
        <div key={`${item.label}-${item.value}`} className="rounded-[18px] border border-white/10 bg-white/[0.03] p-4">
          <div className="text-xs font-extrabold uppercase tracking-[0.16em] text-slate-400">{item.label}</div>
          <div className="mt-2 break-words text-sm font-bold text-white">{item.value || "—"}</div>
        </div>
      ))}
    </div>
  );
}

function Pill({
  label,
  tone,
}: {
  label: string;
  tone: "good" | "warn" | "primary" | "neutral";
}) {
  const style =
    tone === "good"
      ? "border-emerald-300/20 bg-emerald-400/10 text-emerald-200"
      : tone === "warn"
      ? "border-amber-300/20 bg-amber-400/10 text-amber-100"
      : tone === "primary"
      ? "border-blue-300/20 bg-blue-500/10 text-blue-100"
      : "border-white/10 bg-white/8 text-slate-200";

  return <span className={`rounded-full border px-3 py-1 text-xs font-bold ${style}`}>{label}</span>;
}

function MiniStatus({
  label,
  active,
}: {
  label: string;
  active: boolean;
}) {
  return (
    <div className="rounded-[18px] border border-white/10 bg-white/[0.03] p-4">
      <div className="text-xs font-extrabold uppercase tracking-[0.16em] text-slate-400">{label}</div>
      <div className="mt-2 flex items-center gap-2 text-sm font-bold text-white">
        <span className={`h-2.5 w-2.5 rounded-full ${active ? "bg-emerald-300" : "bg-slate-500"}`} />
        {active ? "Connected" : "Not wired"}
      </div>
    </div>
  );
}
