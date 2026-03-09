"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import {
  ArrowLeft,
  ArrowRightLeft,
  CheckCircle2,
  FileDiff,
  AlertTriangle,
  MinusCircle,
  PlusCircle,
} from "lucide-react";
import { backendUrl } from "@/lib/backendUrl";
import { fetchJsonWithAuth } from "@/lib/fetchWithAuth";

function fmtDate(value?: string) {
  if (!value) return "—";
  try {
    return new Date(value).toLocaleString();
  } catch {
    return value;
  }
}

export default function StudentComparePage() {
  const sp = useSearchParams();
  const preA = sp.get("a") || "";

  const [items, setItems] = useState<any[]>([]);
  const [a, setA] = useState(preA);
  const [b, setB] = useState("");
  const [diff, setDiff] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loadingItems, setLoadingItems] = useState(true);
  const [running, setRunning] = useState(false);

  useEffect(() => {
    (async () => {
      try {
        setErr(null);
        setLoadingItems(true);
        const j = await fetchJsonWithAuth(backendUrl("/phase7/history/student?limit=60"), {
          method: "GET",
        });
        setItems(Array.isArray(j?.items) ? j.items : []);
      } catch (e: any) {
        setErr(e?.message || "Failed to load history");
      } finally {
        setLoadingItems(false);
      }
    })();
  }, []);

  async function run() {
    setErr(null);
    setDiff(null);

    if (!a || !b) {
      setErr("Pick two reports");
      return;
    }

    try {
      setRunning(true);
      const j = await fetchJsonWithAuth(
        backendUrl(`/phase7/compare/student?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}`),
        { method: "GET" }
      );
      setDiff(j);
    } catch (e: any) {
      setErr(e?.message || "Compare failed");
    } finally {
      setRunning(false);
    }
  }

  const selectedA = useMemo(() => items.find((x) => x.id === a), [items, a]);
  const selectedB = useMemo(() => items.find((x) => x.id === b), [items, b]);

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top,_#213b9f_0%,_#0b1537_38%,_#071126_100%)] text-white">
      <div className="mx-auto max-w-[1400px] px-4 py-6 sm:px-6 lg:px-8">
        <div className="grid gap-6">
          <div className="rounded-[30px] border border-white/10 bg-white/[0.05] p-5 shadow-[0_20px_70px_rgba(0,0,0,0.28)] backdrop-blur-xl">
            <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
              <div>
                <div className="text-xs font-semibold uppercase tracking-[0.24em] text-blue-200/70">
                  Student / Compare
                </div>
                <h1 className="mt-2 text-3xl font-black tracking-tight">Compare Reports</h1>
                <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-300">
                  Compare two student reports to see improvement, newly introduced issues, and checklist differences.
                </p>
              </div>

              <Link
                href="/student/history"
                className="inline-flex items-center gap-2 rounded-2xl border border-white/10 bg-white/10 px-4 py-3 text-sm font-bold text-white transition hover:bg-white/15"
              >
                <ArrowLeft size={16} />
                Back to history
              </Link>
            </div>
          </div>

          <div className="grid gap-6 xl:grid-cols-[1.2fr_0.8fr]">
            <div className="rounded-[30px] border border-white/10 bg-white/[0.05] p-5 shadow-[0_20px_70px_rgba(0,0,0,0.25)] backdrop-blur-xl">
              <div className="mb-4">
                <div className="text-lg font-black">Select reports</div>
                <div className="mt-1 text-sm text-slate-400">
                  Choose any two reports from your history and compare their outputs.
                </div>
              </div>

              <div className="grid gap-4 lg:grid-cols-[1fr_1fr_180px]">
                <select
                  value={a}
                  onChange={(e) => setA(e.target.value)}
                  className="rounded-2xl border border-white/10 bg-[#0c1737] px-4 py-3 text-sm text-white outline-none transition focus:border-blue-400/40"
                >
                  <option value="">Select report A</option>
                  {items.map((x) => (
                    <option key={x.id} value={x.id}>
                      {fmtDate(x.created_at)} — {x.file_id} — {x.id}
                    </option>
                  ))}
                </select>

                <select
                  value={b}
                  onChange={(e) => setB(e.target.value)}
                  className="rounded-2xl border border-white/10 bg-[#0c1737] px-4 py-3 text-sm text-white outline-none transition focus:border-blue-400/40"
                >
                  <option value="">Select report B</option>
                  {items.map((x) => (
                    <option key={x.id} value={x.id}>
                      {fmtDate(x.created_at)} — {x.file_id} — {x.id}
                    </option>
                  ))}
                </select>

                <button
                  onClick={run}
                  disabled={running || loadingItems}
                  className="inline-flex items-center justify-center gap-2 rounded-2xl bg-gradient-to-r from-blue-500 to-indigo-500 px-4 py-3 text-sm font-extrabold text-white shadow-[0_12px_28px_rgba(59,130,246,0.28)] transition hover:scale-[1.01] disabled:cursor-not-allowed disabled:opacity-50"
                >
                  <ArrowRightLeft size={16} />
                  {running ? "Comparing..." : "Compare"}
                </button>
              </div>

              {err && (
                <div className="mt-4 flex items-start gap-3 rounded-2xl border border-rose-300/20 bg-rose-400/10 px-4 py-3 text-sm text-rose-200">
                  <AlertTriangle size={18} className="mt-0.5 shrink-0" />
                  <div>{err}</div>
                </div>
              )}
            </div>

            <div className="rounded-[30px] border border-white/10 bg-white/[0.05] p-5 shadow-[0_20px_70px_rgba(0,0,0,0.25)] backdrop-blur-xl">
              <div className="mb-4 text-lg font-black">Selected metadata</div>

              <div className="grid gap-4">
                <MiniReportCard title="Report A" item={selectedA} />
                <MiniReportCard title="Report B" item={selectedB} />
              </div>
            </div>
          </div>

          {diff ? (
            <div className="grid gap-6">
              <div className="rounded-[30px] border border-white/10 bg-white/[0.05] p-5 shadow-[0_20px_70px_rgba(0,0,0,0.25)] backdrop-blur-xl">
                <div className="mb-4 flex items-center gap-3">
                  <div className="grid h-11 w-11 place-items-center rounded-2xl bg-blue-500/15 text-blue-200">
                    <FileDiff size={20} />
                  </div>
                  <div>
                    <div className="text-lg font-black">Summary comparison</div>
                    <div className="text-sm text-slate-400">Compare overall narrative between both reports</div>
                  </div>
                </div>

                <div className="grid gap-4 xl:grid-cols-2">
                  <SummaryCard title="Report A" value={diff?.diff?.summary_a || "—"} />
                  <SummaryCard title="Report B" value={diff?.diff?.summary_b || "—"} />
                </div>
              </div>

              <div className="grid gap-6 xl:grid-cols-2">
                <DiffBox
                  title="Issues Removed (Improved)"
                  items={diff?.diff?.issues_removed}
                  icon={<CheckCircle2 size={17} />}
                  tone="good"
                />
                <DiffBox
                  title="Issues Added (Declined)"
                  items={diff?.diff?.issues_added}
                  icon={<PlusCircle size={17} />}
                  tone="bad"
                />
              </div>

              <div className="grid gap-6 xl:grid-cols-2">
                <DiffBox
                  title="Checklist Removed"
                  items={diff?.diff?.checklist_removed}
                  icon={<MinusCircle size={17} />}
                  tone="neutral"
                />
                <DiffBox
                  title="Checklist Added"
                  items={diff?.diff?.checklist_added}
                  icon={<PlusCircle size={17} />}
                  tone="neutral"
                />
              </div>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}

function MiniReportCard({ title, item }: { title: string; item?: any }) {
  return (
    <div className="rounded-[24px] border border-white/10 bg-white/[0.04] p-4">
      <div className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">{title}</div>
      {!item ? (
        <div className="mt-3 text-sm text-slate-400">No report selected</div>
      ) : (
        <div className="mt-3 grid gap-2 text-sm text-slate-300">
          <div>
            <span className="font-semibold text-white">Date:</span> {fmtDate(item.created_at)}
          </div>
          <div className="break-all">
            <span className="font-semibold text-white">File:</span> {item.file_id}
          </div>
          <div className="break-all">
            <span className="font-semibold text-white">Report ID:</span> {item.id}
          </div>
        </div>
      )}
    </div>
  );
}

function SummaryCard({ title, value }: { title: string; value: string }) {
  return (
    <div className="rounded-[24px] border border-white/10 bg-[linear-gradient(180deg,rgba(255,255,255,0.05),rgba(255,255,255,0.03))] p-5">
      <div className="mb-3 text-sm font-bold text-blue-100">{title}</div>
      <div className="text-sm leading-7 text-slate-300">{value}</div>
    </div>
  );
}

function DiffBox({
  title,
  items,
  icon,
  tone,
}: {
  title: string;
  items: string[];
  icon: React.ReactNode;
  tone: "good" | "bad" | "neutral";
}) {
  const toneClass =
    tone === "good"
      ? "bg-emerald-400/10 text-emerald-200 border-emerald-400/20"
      : tone === "bad"
      ? "bg-rose-400/10 text-rose-200 border-rose-400/20"
      : "bg-blue-400/10 text-blue-100 border-blue-300/20";

  return (
    <div className="rounded-[30px] border border-white/10 bg-white/[0.05] p-5 shadow-[0_20px_70px_rgba(0,0,0,0.22)] backdrop-blur-xl">
      <div className="mb-4 flex items-center gap-3">
        <div className={`grid h-10 w-10 place-items-center rounded-2xl border ${toneClass}`}>{icon}</div>
        <div className="text-lg font-black">{title}</div>
      </div>

      {(items || []).length ? (
        <ul className="grid gap-3">
          {items.map((x, i) => (
            <li key={i} className="rounded-2xl border border-white/10 bg-white/[0.03] px-4 py-3 text-sm text-slate-300">
              {x}
            </li>
          ))}
        </ul>
      ) : (
        <div className="rounded-2xl border border-dashed border-white/10 bg-white/[0.03] px-4 py-5 text-sm text-slate-400">
          No differences found.
        </div>
      )}
    </div>
  );
}