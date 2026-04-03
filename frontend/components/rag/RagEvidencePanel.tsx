"use client";

type Citation = {
  title: string;
  section: string;
  document_id: string;
  chunk_id: string;
  category?: string | null;
  score?: number | null;
};

type RetrievedChunk = {
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
};

type RagMeta = {
  enabled?: boolean;
  confidence_score?: number;
  confidence_label?: string;
  safe_review?: boolean;
  citations?: Citation[];
  retrieved_chunks?: RetrievedChunk[];
  trace?: {
    audience?: string;
    query?: string;
    collection_name?: string;
    rewritten_queries?: string[];
    retrieved_chunk_ids?: string[];
    final_chunk_ids?: string[];
    scores?: number[];
  };
};

type Props = {
  ragMeta?: RagMeta | null;
  title?: string;
  emptyMessage?: string;
};

function ScoreBadge({ label }: { label?: string }) {
  const normalized = (label || "unknown").toLowerCase();

  const className =
    normalized === "high"
      ? "border-emerald-400/30 bg-emerald-400/10 text-emerald-200"
      : normalized === "medium"
      ? "border-amber-400/30 bg-amber-400/10 text-amber-200"
      : "border-rose-400/30 bg-rose-400/10 text-rose-200";

  return (
    <span className={`rounded-full border px-3 py-1 text-xs font-semibold uppercase tracking-wide ${className}`}>
      {label || "unknown"}
    </span>
  );
}

export default function RagEvidencePanel({
  ragMeta,
  title = "RAG Evidence",
  emptyMessage = "No RAG evidence was attached to this report.",
}: Props) {
  if (!ragMeta?.enabled) {
    return (
      <section className="rounded-[28px] border border-white/10 bg-white/[0.04] p-5 shadow-[0_20px_80px_rgba(0,0,0,0.22)] backdrop-blur-xl">
        <div className="mb-3">
          <h3 className="text-lg font-semibold text-white">{title}</h3>
          <p className="mt-1 text-sm text-white/60">
            Grounding sources, confidence, and retrieved evidence used for this AI output.
          </p>
        </div>
        <div className="rounded-[22px] border border-dashed border-white/10 bg-black/20 p-4 text-sm text-white/65">
          {emptyMessage}
        </div>
      </section>
    );
  }

  const citations = ragMeta.citations || [];
  const retrievedChunks = ragMeta.retrieved_chunks || [];
  const rewrittenQueries = ragMeta.trace?.rewritten_queries || [];

  return (
    <section className="rounded-[28px] border border-white/10 bg-white/[0.04] p-5 shadow-[0_20px_80px_rgba(0,0,0,0.22)] backdrop-blur-xl">
      <div className="mb-5 flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
        <div>
          <h3 className="text-lg font-semibold text-white">{title}</h3>
          <p className="text-sm text-white/60">
            Grounding sources, confidence, and retrieved evidence used for this AI output.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <span className="rounded-full border border-cyan-400/30 bg-cyan-400/10 px-3 py-1 text-xs font-medium text-cyan-200">
            Confidence: {typeof ragMeta.confidence_score === "number" ? ragMeta.confidence_score.toFixed(2) : "—"}
          </span>
          <ScoreBadge label={ragMeta.confidence_label} />
          <span
            className={`rounded-full border px-3 py-1 text-xs font-medium ${
              ragMeta.safe_review
                ? "border-rose-400/30 bg-rose-400/10 text-rose-200"
                : "border-white/15 bg-white/5 text-white/75"
            }`}
          >
            Safe Review: {ragMeta.safe_review ? "On" : "Off"}
          </span>
        </div>
      </div>

      {rewrittenQueries.length > 0 && (
        <div className="mb-5 rounded-[22px] border border-white/10 bg-black/20 p-4">
          <p className="mb-2 text-sm font-semibold text-white/85">Expanded Queries</p>
          <div className="flex flex-wrap gap-2">
            {rewrittenQueries.map((q, i) => (
              <span
                key={`${q}-${i}`}
                className="rounded-full border border-white/10 bg-white/[0.03] px-3 py-1 text-xs text-white/70"
              >
                {q}
              </span>
            ))}
          </div>
        </div>
      )}

      <div className="grid gap-5 xl:grid-cols-2">
        <div className="rounded-[22px] border border-white/10 bg-black/20 p-4">
          <h4 className="mb-3 text-sm font-semibold text-white/90">Citations</h4>
          <div className="space-y-3">
            {citations.length === 0 ? (
              <p className="text-sm text-white/55">No citations available.</p>
            ) : (
              citations.map((citation, idx) => (
                <div
                  key={`${citation.chunk_id}-${idx}`}
                  className="rounded-2xl border border-white/10 bg-white/[0.03] p-3"
                >
                  <div className="flex flex-wrap items-start justify-between gap-2">
                    <p className="text-sm font-medium text-white">{citation.title}</p>
                    {typeof citation.score === "number" && (
                      <span className="text-xs text-white/50">score {citation.score.toFixed(2)}</span>
                    )}
                  </div>
                  <p className="mt-1 text-xs text-white/55">
                    {citation.category || "general"} • section {citation.section || "unknown"}
                  </p>
                  <p className="mt-1 break-all text-[11px] text-white/35">{citation.chunk_id}</p>
                </div>
              ))
            )}
          </div>
        </div>

        <div className="rounded-[22px] border border-white/10 bg-black/20 p-4">
          <h4 className="mb-3 text-sm font-semibold text-white/90">Retrieved Snippets</h4>
          <div className="space-y-3">
            {retrievedChunks.length === 0 ? (
              <p className="text-sm text-white/55">No retrieved snippets available.</p>
            ) : (
              retrievedChunks.map((chunk) => (
                <div
                  key={chunk.chunk_id}
                  className="rounded-2xl border border-white/10 bg-white/[0.03] p-3"
                >
                  <div className="flex flex-wrap items-start justify-between gap-2">
                    <p className="text-sm font-medium text-white">{chunk.document_title}</p>
                    <span className="text-xs text-white/50">score {chunk.score.toFixed(2)}</span>
                  </div>
                  <p className="mt-1 text-xs text-white/55">
                    {chunk.category} • section {chunk.section}
                  </p>
                  <p className="mt-2 text-sm leading-6 text-white/75">{chunk.content}</p>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
