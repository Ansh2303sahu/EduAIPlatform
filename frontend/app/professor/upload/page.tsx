"use client";

import { useMemo, useRef, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  ArrowLeft,
  CheckCircle2,
  FileText,
  Loader2,
  RefreshCw,
  Sparkles,
  UploadCloud,
  Wand2,
} from "lucide-react";
import { backendUrl } from "@/lib/backendUrl";
import { fetchWithAuth, fetchJsonWithAuth } from "@/lib/fetchWithAuth";

type StepKey = "upload" | "scan" | "extract" | "ml" | "llm" | "report";
type StepStatus = "pending" | "running" | "done" | "error";

type Step = {
  key: StepKey;
  label: string;
  status: StepStatus;
};

type FileStatusResponse = {
  id?: string;
  status?: string;
  scan_engine?: string | null;
  scan_result?: string | null;
  scanned_at?: string | null;
  processed_at?: string | null;
  submission_id?: string | null;
};

type ProfessorLatestResponse = {
  found?: boolean;
  item?: {
    id?: string;
    file_id?: string;
    role?: "professor";
    created_at?: string;
    needs_review?: boolean;
    report_json?: unknown;
    model_versions?: unknown;
  };
};

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function inferSteps(args: {
  uploaded: boolean;
  fileStatus?: string;
  reportReady: boolean;
  failed?: boolean;
}): Step[] {
  const { uploaded, fileStatus, reportReady, failed } = args;
  const s = String(fileStatus || "").toLowerCase();

  const base: Step[] = [
    { key: "upload", label: "Upload", status: uploaded ? "done" : "pending" },
    { key: "scan", label: "Scan", status: "pending" },
    { key: "extract", label: "Extract", status: "pending" },
    { key: "ml", label: "ML", status: "pending" },
    { key: "llm", label: "LLM", status: "pending" },
    { key: "report", label: "Report", status: "pending" },
  ];

  if (!uploaded) return base;

  if (failed) {
    return base.map((x, idx) => {
      if (idx === 0) return x;
      return { ...x, status: "error" };
    });
  }

  if (reportReady) {
    base[1].status = "done";
    base[2].status = "done";
    base[3].status = "done";
    base[4].status = "done";
    base[5].status = "done";
    return base;
  }

  if (s === "uploaded" || s === "scanning") {
    base[1].status = "running";
    return base;
  }

  if (s === "clean") {
    base[1].status = "done";
    base[2].status = "running";
    return base;
  }

  if (s === "extracted") {
    base[1].status = "done";
    base[2].status = "done";
    base[3].status = "running";
    return base;
  }

  if (s === "ml_done") {
    base[1].status = "done";
    base[2].status = "done";
    base[3].status = "done";
    base[4].status = "running";
    return base;
  }

  if (s === "quarantined" || s === "failed") {
    return base.map((x, idx) => {
      if (idx === 0) return x;
      return { ...x, status: "error" };
    });
  }

  return base;
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

export default function ProfessorUploadPage() {
  const router = useRouter();
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const [file, setFile] = useState<File | null>(null);
  const [fileId, setFileId] = useState<string | null>(null);

  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [fileStatus, setFileStatus] = useState<string>("");
  const [reportReady, setReportReady] = useState(false);
  const [reportId, setReportId] = useState<string | null>(null);

  const [uploadDone, setUploadDone] = useState(false);
  const [progressNote, setProgressNote] = useState<string>("");

  const steps = useMemo(
    () =>
      inferSteps({
        uploaded: uploadDone,
        fileStatus,
        reportReady,
        failed: !!error && !!fileId,
      }),
    [uploadDone, fileStatus, reportReady, error, fileId]
  );

  async function onUploadAndGenerate() {
    if (!file) {
      setError("Please choose a file first.");
      return;
    }

    setBusy(true);
    setError(null);
    setProgressNote("Uploading file...");
    setReportReady(false);
    setReportId(null);
    setFileId(null);
    setFileStatus("");
    setUploadDone(false);

    try {
      const form = new FormData();
      form.append("file", file);

      const uploadRes = await fetchWithAuth(
        backendUrl("/files/upload"),
        { method: "POST", body: form },
        { timeoutMs: 60000 }
      );

      if (!uploadRes.ok) {
        const txt = await uploadRes.text().catch(() => "");
        throw new Error(txt || `Upload failed (${uploadRes.status})`);
      }

      const uploadJson = await uploadRes.json();
      const nextFileId = uploadJson?.file_id;

      if (!nextFileId) {
        throw new Error("Upload succeeded but file_id is missing.");
      }

      setFileId(nextFileId);
      setUploadDone(true);
      setFileStatus("uploaded");
      setProgressNote("File uploaded. Monitoring ingestion...");

      await monitorProfessorFlow(nextFileId);
    } catch (e: any) {
      setError(e?.message || "Upload failed");
    } finally {
      setBusy(false);
    }
  }

  async function monitorProfessorFlow(nextFileId: string) {
    let professorTriggered = false;

    for (let i = 0; i < 120; i++) {
      const statusRes = (await fetchJsonWithAuth(
        backendUrl(`/files/${nextFileId}/status`),
        { method: "GET" },
        { timeoutMs: 20000 }
      ).catch(() => null)) as FileStatusResponse | null;

      const latestRes = (await fetchJsonWithAuth(
        backendUrl(`/phase7/latest/professor/${nextFileId}`),
        { method: "GET" },
        { timeoutMs: 20000 }
      ).catch(() => null)) as ProfessorLatestResponse | null;

      const fileStatusNow = String(statusRes?.status || "");
      setFileStatus(fileStatusNow);

      if (fileStatusNow === "quarantined" || fileStatusNow === "failed") {
        throw new Error("Processing failed or file was quarantined.");
      }

      if (latestRes?.found && latestRes?.item?.file_id === nextFileId) {
        setReportReady(true);
        setReportId(String(latestRes?.item?.id || ""));
        setProgressNote("Professor report is ready.");
        return;
      }

      if (
        !professorTriggered &&
        (fileStatusNow === "clean" || fileStatusNow === "extracted" || fileStatusNow === "ml_done")
      ) {
        setProgressNote("Generating professor AI report...");

        const genRes = await fetchWithAuth(
          backendUrl("/phase7/professor/generate"),
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ file_id: nextFileId, force: false }),
          },
          { timeoutMs: 240000 }
        );

        if (!genRes.ok && genRes.status !== 409 && genRes.status !== 429) {
          const txt = await genRes.text().catch(() => "");
          throw new Error(txt || `Professor generate failed (${genRes.status})`);
        }

        professorTriggered = true;
        setFileStatus("ml_done");
      }

      if (!professorTriggered) {
        if (fileStatusNow === "uploaded" || fileStatusNow === "scanning") {
          setProgressNote("Scanning uploaded file...");
        } else if (fileStatusNow === "clean") {
          setProgressNote("Extracting content...");
        } else if (fileStatusNow === "extracted") {
          setProgressNote("Preparing professor pipeline...");
        } else if (!fileStatusNow) {
          setProgressNote("Waiting for backend status...");
        }
      } else if (!reportReady) {
        setProgressNote("Waiting for professor report to be stored...");
      }

      await sleep(2500);
    }

    throw new Error("Timed out waiting for professor report.");
  }

  function openResults() {
    if (!fileId || !reportReady) return;
    router.push(`/professor/results/${fileId}`);
  }

  function resetForm() {
    setFile(null);
    setFileId(null);
    setUploadDone(false);
    setFileStatus("");
    setProgressNote("");
    setReportReady(false);
    setReportId(null);
    setError(null);
    setBusy(false);

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }

  const selectedFileLabel = useMemo(() => shortFileName(file?.name), [file?.name]);

  return (
    <div className="grid gap-6">
      <section className="rounded-[30px] border border-white/10 bg-white/[0.04] p-5 shadow-[0_20px_70px_rgba(0,0,0,0.24)] backdrop-blur-xl">
        <div className="flex flex-col gap-4 xl:flex-row xl:items-center xl:justify-between">
          <div>
            <div className="text-xs font-semibold uppercase tracking-[0.25em] text-blue-200/65">
              Professor / Upload
            </div>
            <h1 className="mt-2 text-3xl font-black tracking-tight">Upload submission</h1>
            <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-300">
              Upload a submission, wait for ingestion, and trigger the professor-side rubric report.
            </p>
          </div>

          <Link
            href="/professor"
            className="inline-flex items-center gap-2 rounded-2xl border border-white/10 bg-white/10 px-4 py-3 text-sm font-bold text-white transition hover:bg-white/15"
          >
            <ArrowLeft size={16} />
            Back to dashboard
          </Link>
        </div>
      </section>

      <section className="rounded-[30px] border border-blue-300/10 bg-[linear-gradient(135deg,rgba(19,33,84,0.96),rgba(8,18,50,0.96))] p-6 shadow-[0_20px_80px_rgba(0,0,0,0.28)]">
        <div className="mb-5">
          <div className="inline-flex rounded-full border border-blue-300/20 bg-white/10 px-3 py-1 text-xs font-semibold text-blue-100">
            New Professor Review Run
          </div>
          <h2 className="mt-4 text-3xl font-black leading-tight">
            Upload and generate a professor report
          </h2>
          <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-300">
            This uploads the file, waits for ingestion, then triggers the professor AI report automatically.
          </p>
        </div>

        <div className="grid gap-4 xl:grid-cols-[1.2fr_0.8fr]">
          <div className="rounded-[24px] border border-dashed border-blue-300/20 bg-[linear-gradient(180deg,rgba(255,255,255,0.06),rgba(255,255,255,0.03))] p-5">
            <input
              ref={fileInputRef}
              type="file"
              className="hidden"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
              disabled={busy}
            />

            <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
              <div className="flex items-start gap-4">
                <div className="grid h-14 w-14 shrink-0 place-items-center rounded-2xl bg-gradient-to-br from-blue-500/30 to-indigo-500/20 text-blue-100 shadow-[0_10px_30px_rgba(59,130,246,0.18)]">
                  <UploadCloud size={24} />
                </div>

                <div>
                  <div className="text-sm font-extrabold text-white sm:text-base">
                    Upload submission file
                  </div>
                  <p className="mt-1 max-w-xl text-sm leading-6 text-slate-300">
                    Select a file and generate professor-side rubric review, safety metadata, and moderation output.
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
              onClick={onUploadAndGenerate}
              disabled={busy || !file}
              icon={busy ? <Loader2 size={16} className="animate-spin" /> : <UploadCloud size={16} />}
              label={busy ? "Processing..." : "Upload and Generate"}
              variant="primary"
            />

            <ActionButton
              onClick={openResults}
              disabled={!fileId || !reportReady}
              icon={<Wand2 size={16} />}
              label="Open Professor Results"
              variant="secondary"
            />

            <ActionButton
              onClick={resetForm}
              disabled={busy}
              icon={<RefreshCw size={16} />}
              label="Reset"
              variant="ghost"
            />
          </div>
        </div>

        {error ? (
          <div className="mt-4 rounded-2xl border border-rose-300/20 bg-rose-400/10 px-4 py-3 text-sm text-rose-200 whitespace-pre-wrap">
            {error}
          </div>
        ) : null}

        {progressNote ? (
          <div className="mt-4 flex items-start gap-3 rounded-2xl border border-blue-300/15 bg-blue-500/10 px-4 py-3 text-sm text-blue-100">
            <Sparkles size={18} className="mt-0.5 shrink-0" />
            <div>{progressNote}</div>
          </div>
        ) : null}
      </section>

      <section className="rounded-[30px] border border-white/10 bg-white/[0.05] p-5 shadow-[0_20px_70px_rgba(0,0,0,0.22)] backdrop-blur-xl">
        <div className="mb-5">
          <div className="text-lg font-black text-white">Pipeline progress</div>
          <div className="mt-1 text-sm text-slate-400">
            Upload, scan, extraction, professor ML and LLM generation, then report storage
          </div>
        </div>

        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
          {steps.map((step, idx) => {
            const meta = stepMeta(step.status);

            return (
              <div
                key={step.key}
                className={`rounded-[22px] border ${meta.ring} bg-[linear-gradient(180deg,rgba(255,255,255,0.06),rgba(255,255,255,0.03))] p-4`}
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
                <div className={`mt-2 text-sm font-extrabold ${meta.text}`}>{step.label}</div>

                <div className="mt-3 text-xs leading-5 text-slate-400">
                  {step.key === "upload" && "File is uploaded to the platform"}
                  {step.key === "scan" && "Antivirus and safety checks"}
                  {step.key === "extract" && "Text, tables, OCR, and media extraction"}
                  {step.key === "ml" && "Professor ML signals and preparation"}
                  {step.key === "llm" && "Professor rubric reasoning and structured generation"}
                  {step.key === "report" && "Stored professor AI output ready to view"}
                </div>

                {step.status === "running" ? (
                  <div className="mt-4 h-1.5 overflow-hidden rounded-full bg-white/10">
                    <div className="h-full w-1/2 animate-pulse rounded-full bg-blue-400" />
                  </div>
                ) : null}

                {step.status === "done" ? (
                  <div className="mt-4 inline-flex items-center gap-2 text-xs font-semibold text-emerald-200">
                    <CheckCircle2 size={14} />
                    Completed
                  </div>
                ) : null}
              </div>
            );
          })}
        </div>
      </section>

      <section className="rounded-[30px] border border-white/10 bg-white/[0.05] p-5 shadow-[0_20px_70px_rgba(0,0,0,0.22)] backdrop-blur-xl">
        <div className="mb-4 text-lg font-black text-white">Run details</div>

        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          <InfoTile label="File ID" value={fileId || "—"} />
          <InfoTile label="File status" value={fileStatus || "—"} />
          <InfoTile label="Report stored" value={reportReady ? "Yes" : "No"} />
          <InfoTile label="Report ID" value={reportId || "—"} />
        </div>
      </section>
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

function InfoTile({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-[20px] border border-white/10 bg-white/[0.04] p-4">
      <div className="text-xs font-extrabold uppercase tracking-[0.16em] text-slate-400">{label}</div>
      <div className="mt-2 text-sm font-bold text-white break-words">{value}</div>
    </div>
  );
}