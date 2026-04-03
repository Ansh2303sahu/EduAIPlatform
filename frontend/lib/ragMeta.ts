export type RagCitation = {
  title: string;
  section: string;
  document_id: string;
  chunk_id: string;
  category?: string | null;
  score?: number | null;
};

export type RagRetrievedChunk = {
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

export type RagTrace = {
  audience?: string;
  query?: string;
  collection_name?: string;
  rewritten_queries?: string[];
  retrieved_chunk_ids?: string[];
  final_chunk_ids?: string[];
  scores?: number[];
};

export type RagMeta = {
  enabled?: boolean;
  confidence_score?: number;
  confidence_label?: string;
  safe_review?: boolean;
  citations?: RagCitation[];
  retrieved_chunks?: RagRetrievedChunk[];
  trace?: RagTrace;
};

type RagCarrier = {
  citations?: RagCitation[] | null;
  retrieved_chunks?: RagRetrievedChunk[] | null;
  rag_trace?: RagTrace | null;
  retrieval_confidence?: number | null;
  retrieval_confidence_label?: string | null;
  safe_review?: boolean | null;
  report_json?: {
    rag_meta?: {
      enabled?: boolean | null;
      confidence_score?: number | null;
      confidence_label?: string | null;
      safe_review?: boolean | null;
      citations?: RagCitation[] | null;
      retrieved_chunks?: RagRetrievedChunk[] | null;
      trace?: RagTrace | null;
    } | null;
  } | null;
};

function nonEmptyArray<T>(value?: T[] | null): T[] {
  return Array.isArray(value) ? value.filter(Boolean) : [];
}

function hasMeaningfulTrace(trace?: RagTrace | null) {
  return !!(
    trace &&
    (
      (trace.rewritten_queries?.length ?? 0) > 0 ||
      (trace.retrieved_chunk_ids?.length ?? 0) > 0 ||
      (trace.final_chunk_ids?.length ?? 0) > 0
    )
  );
}

export function buildRenderableRagMeta(item?: RagCarrier | null): RagMeta | null {
  if (!item) return null;

  const nested = item.report_json?.rag_meta || null;
  const citations = nonEmptyArray(item.citations).length > 0
    ? nonEmptyArray(item.citations)
    : nonEmptyArray(nested?.citations);
  const retrievedChunks = nonEmptyArray(item.retrieved_chunks).length > 0
    ? nonEmptyArray(item.retrieved_chunks)
    : nonEmptyArray(nested?.retrieved_chunks);
  const trace = item.rag_trace || nested?.trace || undefined;
  const confidenceScore =
    typeof item.retrieval_confidence === "number"
      ? item.retrieval_confidence
      : typeof nested?.confidence_score === "number"
        ? nested.confidence_score
        : undefined;
  const confidenceLabel =
    typeof item.retrieval_confidence_label === "string" && item.retrieval_confidence_label.trim()
      ? item.retrieval_confidence_label
      : typeof nested?.confidence_label === "string" && nested.confidence_label.trim()
        ? nested.confidence_label
        : undefined;
  const safeReview =
    typeof item.safe_review === "boolean"
      ? item.safe_review
      : typeof nested?.safe_review === "boolean"
        ? nested.safe_review
        : undefined;

  const hasRagData =
    citations.length > 0 ||
    retrievedChunks.length > 0 ||
    hasMeaningfulTrace(trace) ||
    (typeof confidenceScore === "number" && confidenceScore > 0) ||
    !!confidenceLabel ||
    !!nested?.enabled;

  if (!hasRagData) return null;

  return {
    enabled: true,
    confidence_score: confidenceScore,
    confidence_label: confidenceLabel,
    safe_review: safeReview,
    citations,
    retrieved_chunks: retrievedChunks,
    trace,
  };
}
