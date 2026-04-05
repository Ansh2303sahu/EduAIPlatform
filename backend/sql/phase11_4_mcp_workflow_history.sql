create table if not exists public.mcp_workflow_runs (
    workflow_run_id uuid primary key,
    workflow_name text not null,
    user_id text not null,
    role text not null,
    correlation_id text not null,
    request_id text not null,
    final_status text not null,
    blocked_reason text null,
    partial_reason text null,
    executed_steps jsonb not null default '[]'::jsonb,
    skipped_steps jsonb not null default '[]'::jsonb,
    step_count integer not null default 0,
    started_at timestamptz not null,
    finished_at timestamptz not null,
    duration_ms double precision not null default 0,
    tool_order jsonb not null default '[]'::jsonb,
    cache_hits_count integer not null default 0,
    llm_steps_count integer not null default 0,
    fallback_steps_count integer not null default 0,
    ownership_context_present boolean not null default false,
    request_fingerprint text not null default '',
    request_meta jsonb not null default '{}'::jsonb,
    warnings jsonb not null default '[]'::jsonb,
    error_code text null,
    message text null
);

create index if not exists idx_mcp_workflow_runs_started_at
    on public.mcp_workflow_runs (started_at desc);

create index if not exists idx_mcp_workflow_runs_workflow_name
    on public.mcp_workflow_runs (workflow_name);

create index if not exists idx_mcp_workflow_runs_role
    on public.mcp_workflow_runs (role);

create index if not exists idx_mcp_workflow_runs_final_status
    on public.mcp_workflow_runs (final_status);

create index if not exists idx_mcp_workflow_runs_user_id
    on public.mcp_workflow_runs (user_id);


create table if not exists public.mcp_workflow_steps (
    id bigserial primary key,
    workflow_run_id uuid not null references public.mcp_workflow_runs(workflow_run_id) on delete cascade,
    step_index integer not null,
    step_name text not null,
    tool_name text not null,
    tool_version text null,
    step_status text not null,
    execution_ms double precision not null default 0,
    cache_hit boolean not null default false,
    llm_used boolean not null default false,
    deterministic_fallback boolean not null default false,
    error_code text null,
    warning_count integer not null default 0,
    created_at timestamptz not null default now()
);

create index if not exists idx_mcp_workflow_steps_run_id
    on public.mcp_workflow_steps (workflow_run_id, step_index);

create index if not exists idx_mcp_workflow_steps_tool_name
    on public.mcp_workflow_steps (tool_name);

create index if not exists idx_mcp_workflow_steps_status
    on public.mcp_workflow_steps (step_status);
