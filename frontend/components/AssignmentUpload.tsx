"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import {
  AlertTriangle,
  CheckCircle2,
  FileText,
  Loader2,
  RefreshCw,
  Sparkles,
  UploadCloud,
  Wand2,
} from "lucide-react";
import { backendUrl } from "@/lib/backendUrl";

type StepKey = "scan" | "extract" | "ml" | "llm" | "report";
type StepStatus = "pending" | "running" | "done" | "error";
type Step = { key: StepKey; label: string; status: StepStatus };

type ExtractedBundle = {
  exText: any[];
  exTables: any[];
  exTranscript: any[];
};

type ResultsOut = {
  file?: any;
  jobs?: any[];
  text?: any | null;
  tables?: any[];
  media?: any[];
  transcript?: any | null;
  events?: any[];
  extracted?: {
    exText?: any[];
    exTables?: any[];
    exTranscript?: any[];
  };
};

const EMPTY_EXTRACTED: ExtractedBundle = { exText: [], exTables: [], exTranscript: [] };
const EMPTY_RESULTS: ResultsOut = { jobs: [], tables: [], media: [], events: [] };

function sleep(ms: number) {
  return new Promise((r) => setTimeout(r, ms));
}

function lower(v: any) {
  return String(v ?? "").trim().toLowerCase();
}

function scanOkFrom(s: any): boolean {
  if (!s) return false;

  const vals = [lower(s?.status), lower(s?.scan_status), lower(s?.scan_result)];
  if (vals.some((v) => ["clean", "ok", "passed", "done", "completed"].includes(v))) {
    return true;
  }

  return !!s?.scanned_at;
}

function reportFoundFrom(l: any): boolean {
  return !!(l?.found === true && l?.item);
}

function extractOkFromResults(r: ResultsOut | null): boolean {
  if (!r) return false;

  const jobs = Array.isArray(r.jobs) ? r.jobs : [];
  const events = Array.isArray(r.events) ? r.events : [];

  const anyJobDone = jobs.some((j) => {
    const st = lower(j?.status);
    return ["done", "completed", "success", "succeeded"].includes(st);
  });

  const anyClassicOutputs =
    !!r.text ||
    !!r.transcript ||
    (Array.isArray(r.tables) && r.tables.length > 0) ||
    (Array.isArray(r.media) && r.media.length > 0);

  const ex = r.extracted ?? {};
  const anyNestedExtracted =
    (Array.isArray(ex.exText) && ex.exText.length > 0) ||
    (Array.isArray(ex.exTables) && ex.exTables.length > 0) ||
    (Array.isArray(ex.exTranscript) && ex.exTranscript.length > 0);

  const anyDoneEvent = events.some((e) => {
    const t = lower(e?.event_type);
    return t === "file_processing_done" || t === "job_done" || t === "text_extracted";
  });

  return anyJobDone || anyClassicOutputs || anyNestedExtracted || anyDoneEvent;
}

function mlDoneFrom(gen: any, latest: any): boolean {
  return !!(
    gen?.ml ||
    gen?.stored?.model_versions?.ml_models ||
    latest?.item?.model_versions?.ml_models
  );
}

function llmDoneFrom(gen: any, latest: any): boolean {
  return !!(
    gen?.report ||
    gen?.stored?.report_json ||
    latest?.item?.report_json ||
    latest?.item
  );
}

function shortFileName(name?: string) {
  if (!name) return "No file selected";
  if (name.length <= 38) return name;
  return `${name.slice(0, 22)}...${name.slice(-12)}`;
}

function stepMeta(status: StepStatus) {
  switch (status) {
    case "done":
      return {
        dot: "bg-emerald-400 shadow-[0_0_18px_rgba(52,211,153,0.55)]",
        ring: "border-emerald-400/40",
        text: "text-emerald-200",
        badge: "bg-emerald-400/12 text-emerald-200 border-emerald-400/20",
        label: "Done",
      };
    case "running":
      return {
        dot: "bg-blue-400 shadow-[0_0_18px_rgba(96,165,250,0.55)]",
        ring: "border-blue-400/40",
        text: "text-blue-100",
        badge: "bg-blue-400/12 text-blue-100 border-blue-400/20",
        label: "Running",
      };
    case "error":
      return {
        dot: "bg-rose-400 shadow-[0_0_18px_rgba(251,113,133,0.55)]",
        ring: "border-rose-400/40",
        text: "text-rose-200",
        badge: "bg-rose-400/12 text-rose-200 border-rose-400/20",
        label: "Error",
      };
    default:
      return {
        dot: "bg-white/20",
        ring: "border-white/10",
        text: "text-slate-400",
        badge: "bg-white/5 text-slate-300 border-white/10",
        label: "Pending",
      };
  }
}

async function authFetch(input: RequestInfo, init: RequestInit = {}) {
  const { supabase } = await import("@/lib/supabaseClient");
  const { data, error } = await supabase.auth.getSession();
  if (error) throw new Error(error.message);

  const token = data.session?.access_token;
  if (!token) throw new Error("Not logged in (no Supabase session token)");

  const headers = new Headers(init.headers || {});
  headers.set("Authorization", `Bearer ${token}`);

  const isFormData = typeof FormData !== "undefined" && init.body instanceof FormData;
  if (!isFormData && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  return fetch(input, { ...init, headers });
}

async function authFetchJson<T = any>(url: string, init: RequestInit = {}): Promise<T> {
  const res = await authFetch(url, init);
  const text = await res.text();

  let json: any = {};
  try {
    json = text ? JSON.parse(text) : {};
  } catch {
    json = {};
  }

  if (!res.ok) {
    throw new Error(json?.detail || `Request failed ${res.status}: ${text || "no body"}`);
  }

  return json as T;
}

async function authFetchJsonSafe<T = any>(url: string, init: RequestInit = {}) {
  const res = await authFetch(url, init);
  const text = await res.text();

  let json: any = {};
  try {
    json = text ? JSON.parse(text) : {};
  } catch {
    json = {};
  }

  return {
    ok: res.ok,
    status: res.status,
    headers: res.headers,
    text,
    json: json as T,
    detail: (json as any)?.detail,
  };
}

export default function AssignmentUpload({ onFileId }: { onFileId?: (id: string) => void }) {
  const router = useRouter();

  const [file, setFile] = useState<File | null>(null);
  const [fileId, setFileId] = useState<string | null>(null);

  const [status, setStatus] = useState<any>(null);
  const [latest, setLatest] = useState<any>(null);
  const [gen, setGen] = useState<any>(null);
  const [extracted, setExtracted] = useState<ExtractedBundle>(EMPTY_EXTRACTED);
  const [results, setResults] = useState<ResultsOut>(EMPTY_RESULTS);

  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState("");
  const [err, setErr] = useState("");

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const navigatedRef = useRef(false);
  const extractedRef = useRef<ExtractedBundle>(EMPTY_EXTRACTED);

  const genCtrlRef = useRef({
    started: false,
    accepted: false,
    inFlight: false,
    attempts: 0,
    nextAllowedAt: 0,
    maxAttempts: 999,
    lastError: "",
  });

  async function getFileStatus(id: string, signal?: AbortSignal) {
    return authFetchJson(backendUrl(`/files/${id}/status`), { method: "GET", signal });
  }

  async function getLatestStudentReport(id: string, signal?: AbortSignal) {
    return authFetchJson(backendUrl(`/phase7/latest/student/${id}`), { method: "GET", signal });
  }

  async function getExtracted(id: string, signal?: AbortSignal): Promise<ExtractedBundle> {
    const out = await authFetchJson<any>(backendUrl(`/files/${id}/extracted`), {
      method: "GET",
      signal,
    });

    return {
      exText: Array.isArray(out?.exText) ? out.exText : [],
      exTables: Array.isArray(out?.exTables) ? out.exTables : [],
      exTranscript: Array.isArray(out?.exTranscript) ? out.exTranscript : [],
    };
  }

  async function getResults(id: string, signal?: AbortSignal): Promise<ResultsOut> {
    return authFetchJson<ResultsOut>(backendUrl(`/results/${id}`), { method: "GET", signal });
  }

  async function generateStudentReportSafe(id: string, force = false, signal?: AbortSignal) {
    return authFetchJsonSafe(backendUrl(`/phase7/student/generate`), {
      method: "POST",
      body: JSON.stringify({ file_id: id, force }),
      signal,
    });
  }

  async function uploadFile() {
    if (!file) return;

    setBusy(true);
    setErr("");
    setMsg("");

    try {
      const fd = new FormData();
      fd.append("file", file);

      const res = await authFetch(backendUrl("/files/upload"), {
        method: "POST",
        body: fd,
      });

      const text = await res.text();
      let j: any = {};
      try {
        j = text ? JSON.parse(text) : {};
      } catch {
        j = {};
      }

      if (!res.ok) {
        throw new Error(j?.detail || `Upload failed ${res.status}: ${text || "no body"}`);
      }

      const id =
        j?.file_id ??
        j?.id ??
        j?.data?.file_id ??
        j?.data?.id ??
        j?.item?.file_id ??
        j?.item?.id ??
        null;

      if (!id) {
        console.error("Upload response missing id:", j);
        throw new Error("Upload succeeded but file_id missing");
      }

      navigatedRef.current = false;
      extractedRef.current = EMPTY_EXTRACTED;

      genCtrlRef.current = {
        started: false,
        accepted: false,
        inFlight: false,
        attempts: 0,
        nextAllowedAt: 0,
        maxAttempts: 999,
        lastError: "",
      };

      setStatus(null);
      setLatest(null);
      setGen(null);
      setExtracted(EMPTY_EXTRACTED);
      setResults(EMPTY_RESULTS);

      const idStr = String(id);
      setFileId(idStr);
      onFileId?.(idStr);

      setMsg("Uploaded successfully. Scan and ingestion have started.");
    } catch (e: any) {
      setErr(e?.message || "Upload failed");
    } finally {
      setBusy(false);
    }
  }

  const scanOk = useMemo(() => scanOkFrom(status), [status]);

  const extractOk = useMemo(() => {
    if (extractOkFromResults(results)) return true;

    return (
      extracted.exText.length > 0 ||
      extracted.exTables.length > 0 ||
      extracted.exTranscript.length > 0
    );
  }, [results, extracted]);

  const reportFound = useMemo(() => reportFoundFrom(latest), [latest]);
  const mlDone = useMemo(() => mlDoneFrom(gen, latest), [gen, latest]);
  const llmDone = useMemo(() => llmDoneFrom(gen, latest), [gen, latest]);

  const steps: Step[] = useMemo(() => {
    const base: Step[] = [
      { key: "scan", label: "Scan", status: "pending" },
      { key: "extract", label: "Extract", status: "pending" },
      { key: "ml", label: "ML Inference", status: "pending" },
      { key: "llm", label: "LLM Draft", status: "pending" },
      { key: "report", label: "Report Stored", status: "pending" },
    ];

    if (!fileId) return base;

    base[0].status = scanOk ? "done" : "running";
    base[1].status = extractOk ? "done" : scanOk ? "running" : "pending";

    const started = genCtrlRef.current.started;
    const accepted = genCtrlRef.current.accepted;

    if (!started) {
      base[2].status = "pending";
      base[3].status = "pending";
    } else if (!accepted) {
      base[2].status = "running";
      base[3].status = "pending";
    } else {
      base[2].status = mlDone ? "done" : "running";
      base[3].status = llmDone ? "done" : "running";
    }

    base[4].status = reportFound ? "done" : accepted ? "running" : "pending";

    return base;
  }, [fileId, scanOk, extractOk, mlDone, llmDone, reportFound]);

  useEffect(() => {
    extractedRef.current = extracted;
  }, [extracted]);

  useEffect(() => {
    if (!fileId) return;

    const ctrl = new AbortController();
    let alive = true;

    async function maybeGenerate(fileIdInner: string, currentLatest: any) {
      const gc = genCtrlRef.current;
      const now = Date.now();

      if (reportFoundFrom(currentLatest)) return;
      if (gc.accepted) return;
      if (gc.inFlight) return;
      if (now < gc.nextAllowedAt) return;

      gc.started = true;
      gc.inFlight = true;
      gc.attempts += 1;

      setMsg(`Generating AI report… attempt ${gc.attempts}`);

      try {
        const res = await generateStudentReportSafe(fileIdInner, false, ctrl.signal);

        if (res.ok) {
          gc.accepted = true;
          gc.lastError = "";
          setGen(res.json);
          setErr("");

          const llmMs = (res.json as any)?.stored?.model_versions?.timings_ms?.llm_service;
          if (typeof llmMs === "number" && llmMs > 0) {
            setMsg(`Generation accepted. LLM last run ~${Math.round(llmMs / 1000)}s.`);
          } else {
            setMsg("Generation accepted. Waiting for report to be stored.");
          }
          return;
        }

        if (res.status === 429) {
          const ra = res.headers.get("Retry-After");
          const retryAfterMs = ra ? Math.max(0, Number(ra) * 1000) : 0;
          const expMs = Math.min(60000, 3000 * Math.pow(2, Math.max(0, gc.attempts - 1)));
          const waitMs = Math.max(expMs, retryAfterMs);

          gc.nextAllowedAt = Date.now() + waitMs;
          gc.lastError = res.detail || `Rate limited (${res.status})`;

          setErr(gc.lastError);
          setMsg(`AI busy. Retrying in ~${Math.ceil(waitMs / 1000)}s.`);
          return;
        }

        const waitMs = Math.min(60000, 5000 * Math.pow(2, Math.max(0, gc.attempts - 1)));
        gc.nextAllowedAt = Date.now() + waitMs;
        gc.lastError = res.detail || `Generate failed (${res.status})`;

        setErr(gc.lastError);
        setMsg(`Generate failed (${res.status}). Retrying in ~${Math.ceil(waitMs / 1000)}s.`);
      } catch (e: any) {
        const waitMs = Math.min(60000, 5000 * Math.pow(2, Math.max(0, gc.attempts - 1)));
        gc.nextAllowedAt = Date.now() + waitMs;
        gc.lastError = e?.message || "Generate request failed";

        setErr(gc.lastError);
        setMsg(`Generate request failed. Retrying in ~${Math.ceil(waitMs / 1000)}s.`);
      } finally {
        gc.inFlight = false;
      }
    }

    async function loop() {
      setMsg("Tracking progress.");

      let extractedEveryN = 0;

      while (alive) {
        try {
          const [s, r, l] = await Promise.all([
            getFileStatus(fileId, ctrl.signal),
            getResults(fileId, ctrl.signal),
            getLatestStudentReport(fileId, ctrl.signal),
          ]);

          setStatus(s);
          setResults(r);
          setLatest(l);

          extractedEveryN = (extractedEveryN + 1) % 4;
          let ex = extractedRef.current;

          if (extractedEveryN === 0) {
            try {
              ex = await getExtracted(fileId, ctrl.signal);
              extractedRef.current = ex;
              setExtracted(ex);
            } catch {
              // keep last known extracted values
            }
          }

          const _scanOk = scanOkFrom(s);
          const _extractOk =
            extractOkFromResults(r) ||
            ex.exText.length > 0 ||
            ex.exTables.length > 0 ||
            ex.exTranscript.length > 0;

          const _reportFound = reportFoundFrom(l);

          if (_reportFound) {
            setErr("");
            setMsg("Report stored. Opening results.");

            if (!navigatedRef.current) {
              navigatedRef.current = true;
              router.push(`/student/results/${fileId}`);
            }
            break;
          }

          if (_scanOk && !_extractOk) {
            setMsg("Scan complete. Waiting for extraction outputs.");
          }

          if (_scanOk && _extractOk) {
            await maybeGenerate(fileId, l);
          }
        } catch (e: any) {
          if (e?.name === "AbortError") break;
          setErr(e?.message || "Polling failed");
        }

        await sleep(2500);
      }
    }

    loop();

    return () => {
      alive = false;
      ctrl.abort();
    };
  }, [fileId, router]);

  async function forceRegenerate() {
    if (!fileId) return;

    setBusy(true);
    setErr("");
    setMsg("Force regenerating.");

    try {
      genCtrlRef.current = {
        started: true,
        accepted: false,
        inFlight: false,
        attempts: 0,
        nextAllowedAt: 0,
        maxAttempts: 999,
        lastError: "",
      };

      const res = await generateStudentReportSafe(fileId, true);
      if (!res.ok) {
        throw new Error(res.detail || `Force regenerate failed (${res.status})`);
      }

      genCtrlRef.current.accepted = true;
      setGen(res.json);
      setMsg("Force regenerate accepted. Waiting for report to be stored.");
    } catch (e: any) {
      setErr(e?.message || "Force regenerate failed");
      genCtrlRef.current.nextAllowedAt = Date.now() + 15000;
    } finally {
      setBusy(false);
    }
  }

  function resetAll() {
    setFile(null);
    setFileId(null);
    setStatus(null);
    setLatest(null);
    setGen(null);
    setExtracted(EMPTY_EXTRACTED);
    setResults(EMPTY_RESULTS);
    setMsg("");
    setErr("");
    setBusy(false);

    extractedRef.current = EMPTY_EXTRACTED;

    genCtrlRef.current = {
      started: false,
      accepted: false,
      inFlight: false,
      attempts: 0,
      nextAllowedAt: 0,
      maxAttempts: 999,
      lastError: "",
    };

    navigatedRef.current = false;
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  const selectedFileLabel = useMemo(() => shortFileName(file?.name), [file?.name]);

  return (
    <div className="grid gap-4">
      <div className="rounded-[26px] border border-white/10 bg-white/[0.06] p-4 shadow-[0_18px_60px_rgba(0,0,0,0.18)] backdrop-blur-xl sm:p-5">
        <div className="grid gap-4 xl:grid-cols-[1.2fr_0.8fr]">
          <div className="rounded-[24px] border border-dashed border-blue-300/20 bg-[linear-gradient(180deg,rgba(255,255,255,0.06),rgba(255,255,255,0.03))] p-5">
            <input
              ref={fileInputRef}
              type="file"
              className="hidden"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
            />

            <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
              <div className="flex items-start gap-4">
                <div className="grid h-14 w-14 shrink-0 place-items-center rounded-2xl bg-gradient-to-br from-blue-500/30 to-indigo-500/20 text-blue-100 shadow-[0_10px_30px_rgba(59,130,246,0.18)]">
                  <UploadCloud size={24} />
                </div>

                <div>
                  <div className="text-sm font-extrabold text-white sm:text-base">
                    Upload assignment file
                  </div>
                  <p className="mt-1 max-w-xl text-sm leading-6 text-slate-300">
                    Submit PDF, DOCX, TXT, images, tables, or transcript-supported files and track the full AI pipeline live.
                  </p>
                </div>
              </div>

              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="inline-flex items-center justify-center gap-2 rounded-2xl border border-blue-300/20 bg-white/10 px-4 py-3 text-sm font-bold text-blue-100 transition hover:bg-white/15"
              >
                <FileText size={16} />
                Choose file
              </button>
            </div>

            <div className="mt-4 rounded-2xl border border-white/10 bg-[#0b1737]/70 px-4 py-3">
              <div className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">
                Selected file
              </div>
              <div className="mt-2 flex items-center gap-3">
                <div className="grid h-10 w-10 place-items-center rounded-xl bg-blue-500/15 text-blue-200">
                  <FileText size={18} />
                </div>
                <div className="min-w-0">
                  <div className="truncate text-sm font-semibold text-white">{selectedFileLabel}</div>
                  <div className="text-xs text-slate-400">
                    {file ? `${Math.max(1, Math.round(file.size / 1024))} KB` : "No file selected yet"}
                  </div>
                </div>
              </div>
            </div>

            {fileId && (
              <div className="mt-4 rounded-2xl border border-blue-300/15 bg-blue-500/10 px-4 py-3 text-xs text-blue-100">
                File ID: <span className="font-mono font-bold">{fileId}</span>
              </div>
            )}
          </div>

          <div className="grid gap-3">
            <ActionButton
              onClick={uploadFile}
              disabled={!file || busy}
              icon={busy ? <Loader2 size={16} className="animate-spin" /> : <UploadCloud size={16} />}
              label={busy ? "Working..." : "Upload"}
              variant="primary"
            />

            <ActionButton
              onClick={forceRegenerate}
              disabled={!fileId || busy}
              icon={<Wand2 size={16} />}
              label="Force regenerate"
              variant="secondary"
            />

            <ActionButton
              onClick={resetAll}
              disabled={busy}
              icon={<RefreshCw size={16} />}
              label="Reset"
              variant="ghost"
            />
          </div>
        </div>
      </div>

      {msg && (
        <div className="flex items-start gap-3 rounded-[22px] border border-blue-300/15 bg-blue-500/10 px-4 py-3 text-sm text-blue-100">
          <Sparkles size={18} className="mt-0.5 shrink-0" />
          <div>{msg}</div>
        </div>
      )}

      {err && (
        <div className="flex items-start gap-3 rounded-[22px] border border-rose-300/20 bg-rose-400/10 px-4 py-3 text-sm text-rose-200">
          <AlertTriangle size={18} className="mt-0.5 shrink-0" />
          <div>{err}</div>
        </div>
      )}

      <div className="rounded-[26px] border border-white/10 bg-white/[0.05] p-4 shadow-[0_18px_60px_rgba(0,0,0,0.16)] backdrop-blur-xl sm:p-5">
        <div className="mb-4 flex items-center justify-between gap-3">
          <div>
            <div className="text-sm font-extrabold text-white">Pipeline progress</div>
            <div className="mt-1 text-xs text-slate-400">
              Live status for scan, extraction, ML inference, LLM drafting, and report storage
            </div>
          </div>

          <div className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs font-semibold text-slate-300">
            {fileId ? "Tracking active" : "Waiting for upload"}
          </div>
        </div>

        <div className="grid gap-3 md:grid-cols-5">
          {steps.map((s, idx) => {
            const meta = stepMeta(s.status);

            return (
              <div
                key={s.key}
                className={`relative rounded-[22px] border ${meta.ring} bg-[linear-gradient(180deg,rgba(255,255,255,0.06),rgba(255,255,255,0.03))] p-4`}
              >
                <div className="mb-4 flex items-center justify-between gap-3">
                  <div className={`h-3 w-3 rounded-full ${meta.dot}`} />
                  <div className={`rounded-full border px-2.5 py-1 text-[11px] font-bold ${meta.badge}`}>
                    {meta.label}
                  </div>
                </div>

                <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
                  Step {idx + 1}
                </div>
                <div className={`mt-2 text-sm font-extrabold ${meta.text}`}>{s.label}</div>

                {s.status === "running" && (
                  <div className="mt-4 h-1.5 overflow-hidden rounded-full bg-white/10">
                    <div className="h-full w-1/2 animate-pulse rounded-full bg-blue-400" />
                  </div>
                )}

                {s.status === "done" && (
                  <div className="mt-4 inline-flex items-center gap-2 text-xs font-semibold text-emerald-200">
                    <CheckCircle2 size={14} />
                    Completed
                  </div>
                )}

                {s.status === "pending" && (
                  <div className="mt-4 text-xs text-slate-400">Waiting for previous stage</div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      <details className="rounded-[26px] border border-white/10 bg-white/[0.04] p-4 backdrop-blur-xl">
        <summary className="cursor-pointer list-none">
          <div className="flex items-center justify-between gap-3">
            <div>
              <div className="text-sm font-extrabold text-white">Debug payload</div>
              <div className="mt-1 text-xs text-slate-400">
                Inspect backend state, extraction bundle, latest report payload, and generation control state
              </div>
            </div>
            <div className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs font-semibold text-slate-300">
              Expand
            </div>
          </div>
        </summary>

        <div className="mt-4 rounded-[20px] border border-white/10 bg-[#091327]/90 p-4">
          <pre className="overflow-x-auto whitespace-pre-wrap break-words text-xs leading-6 text-slate-300">
            {JSON.stringify(
              {
                status,
                results,
                extracted,
                latest,
                gen,
                scanOk,
                extractOk,
                reportFound,
                genCtrl: genCtrlRef.current,
              },
              null,
              2
            )}
          </pre>
        </div>
      </details>
    </div>
  );
}

function ActionButton({
  onClick,
  disabled,
  icon,
  label,
  variant,
}: {
  onClick: () => void;
  disabled?: boolean;
  icon: React.ReactNode;
  label: string;
  variant: "primary" | "secondary" | "ghost";
}) {
  const styles =
    variant === "primary"
      ? "bg-gradient-to-r from-blue-500 to-indigo-500 text-white shadow-[0_12px_28px_rgba(59,130,246,0.28)] hover:scale-[1.01]"
      : variant === "secondary"
      ? "bg-white/10 text-blue-100 border border-blue-300/15 hover:bg-white/15"
      : "bg-transparent text-slate-200 border border-white/10 hover:bg-white/5";

  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={`inline-flex items-center justify-center gap-2 rounded-2xl px-4 py-3 text-sm font-extrabold transition disabled:cursor-not-allowed disabled:opacity-45 ${styles}`}
    >
      {icon}
      <span>{label}</span>
    </button>
  );
}