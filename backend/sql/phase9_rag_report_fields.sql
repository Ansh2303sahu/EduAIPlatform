alter table if exists ai_reports
    add column if not exists rag_trace jsonb default '[]'::jsonb,
    add column if not exists citations jsonb default '[]'::jsonb,
    add column if not exists retrieval_confidence double precision,
    add column if not exists retrieval_confidence_label text,
    add column if not exists retrieved_chunks jsonb default '[]'::jsonb;

create index if not exists idx_ai_reports_retrieval_confidence
    on ai_reports(retrieval_confidence);

create index if not exists idx_ai_reports_rag_trace_gin
    on ai_reports using gin(rag_trace);

create index if not exists idx_ai_reports_citations_gin
    on ai_reports using gin(citations);

create index if not exists idx_ai_reports_retrieved_chunks_gin
    on ai_reports using gin(retrieved_chunks);