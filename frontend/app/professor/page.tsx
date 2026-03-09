"use client";

import Link from "next/link";
import {
  AlertTriangle,
  ArrowRight,
  ClipboardCheck,
  Clock3,
  Download,
  FileSearch,
  History,
  ShieldAlert,
  Sparkles,
  UploadCloud,
  Users,
} from "lucide-react";

export default function ProfessorDashboard() {
  return (
    <div className="grid gap-6">
      <section className="grid grid-cols-1 gap-6 xl:grid-cols-[1.25fr_0.75fr]">
        <div className="rounded-[30px] border border-blue-300/10 bg-[linear-gradient(135deg,rgba(19,33,84,0.96),rgba(8,18,50,0.96))] p-6 shadow-[0_20px_80px_rgba(0,0,0,0.28)]">
          <div className="mb-5">
            <div className="inline-flex rounded-full border border-blue-300/20 bg-white/10 px-3 py-1 text-xs font-semibold text-blue-100">
              Quick Actions
            </div>
            <h2 className="mt-4 text-3xl font-black leading-tight text-white">
              Open your main professor workflows
            </h2>
            <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-300">
              Jump directly into review queue, upload flow, history, and exports without leaving the moderation dashboard.
            </p>
          </div>

          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
            <QuickActionCard
              href="/professor/queue"
              icon={<ShieldAlert size={18} />}
              title="Open Queue"
              sub="Check flagged and pending reviews"
            />
            <QuickActionCard
              href="/professor/upload"
              icon={<UploadCloud size={18} />}
              title="Upload Submission"
              sub="Start a new professor-side review flow"
            />
            <QuickActionCard
              href="/professor/history"
              icon={<History size={18} />}
              title="View History"
              sub="See past reports and outputs"
            />
          </div>
        </div>

        <div className="grid gap-6">
          <StatCard
            title="Needs review"
            value="—"
            sub="From ai_reports.needs_review"
            accent="from-amber-500/15 to-rose-500/10"
            icon={<AlertTriangle size={18} />}
          />
          <StatCard
            title="Recent runs"
            value="—"
            sub="From prof_events / inference_events"
            accent="from-cyan-500/20 to-blue-500/10"
            icon={<Clock3 size={18} />}
          />
          <StatCard
            title="Exports"
            value="—"
            sub="PDF / JSON export panel"
            accent="from-violet-500/20 to-indigo-500/10"
            icon={<Download size={18} />}
          />
        </div>
      </section>

      <section className="grid grid-cols-1 gap-6 xl:grid-cols-3">
        <PanelCard
          title="Moderation Queue Overview"
          subtitle="Latest ingestions first, needs-review items prioritised"
          icon={<ClipboardCheck size={18} />}
        >
          <div className="rounded-[24px] border border-dashed border-blue-300/20 bg-blue-500/10 p-4 text-sm text-slate-300">
            <div className="font-bold text-white">Queue preview</div>
            <ul className="mt-3 grid list-disc gap-2 pl-5 text-slate-300">
              <li>latest file_id</li>
              <li>needs_review badge</li>
              <li>confidence band</li>
              <li>open results button</li>
            </ul>
          </div>
        </PanelCard>

        <PanelCard
          title="Review Safety"
          subtitle="Escalation and moderation checks"
          icon={<ShieldAlert size={18} />}
        >
          <div className="grid gap-3">
            <InfoStrip
              tone="warning"
              title="Flagged submissions"
              sub="High-priority review items will surface here."
            />
            <InfoStrip
              tone="neutral"
              title="Prompt-injection review"
              sub="Suspicious AI safety markers can be monitored here."
            />
            <InfoStrip
              tone="good"
              title="Policy-aligned outputs"
              sub="Approved reviews and safe reports can be summarised here."
            />
          </div>
        </PanelCard>

        <PanelCard
          title="Professor Insights"
          subtitle="Operational view of teaching-side AI review"
          icon={<Users size={18} />}
        >
          <div className="grid gap-4">
            <MiniMetric label="Rubric reports reviewed" value="—" />
            <MiniMetric label="Submissions awaiting moderation" value="—" />
            <MiniMetric label="Exports generated" value="—" />
          </div>
        </PanelCard>
      </section>

      <section className="grid grid-cols-1 gap-6 xl:grid-cols-[1.2fr_0.8fr]">
        <PanelCard
          title="Review pipeline"
          subtitle="Typical professor-side workflow"
          icon={<FileSearch size={18} />}
        >
          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
            <StepCard
              step="01"
              title="Upload"
              sub="Professor uploads a submission or review target"
            />
            <StepCard
              step="02"
              title="Analyse"
              sub="AI generates rubric-aware feedback and confidence"
            />
            <StepCard
              step="03"
              title="Moderate"
              sub="Professor inspects flagged evidence and review notes"
            />
            <StepCard
              step="04"
              title="Export"
              sub="Final result prepared for PDF / JSON delivery"
            />
          </div>
        </PanelCard>

        <PanelCard
          title="Next implementation"
          subtitle="Suggested next professor pages"
          icon={<Sparkles size={18} />}
        >
          <div className="grid gap-3">
            <NextItem text="Professor queue with real flagged items" />
            <NextItem text="Professor upload with processing progress" />
            <NextItem text="Professor results page with rubric breakdown" />
            <NextItem text="Export center for PDF / JSON output" />
          </div>
        </PanelCard>
      </section>
    </div>
  );
}

function StatCard({
  title,
  value,
  sub,
  accent,
  icon,
}: {
  title: string;
  value: string;
  sub: string;
  accent: string;
  icon: React.ReactNode;
}) {
  return (
    <div className="relative overflow-hidden rounded-[28px] border border-white/10 bg-white/[0.05] p-5 shadow-[0_20px_80px_rgba(0,0,0,0.22)]">
      <div className={`absolute inset-x-0 top-0 h-24 bg-gradient-to-r ${accent} blur-2xl`} />
      <div className="relative">
        <div className="flex items-center justify-between">
          <div className="text-sm font-bold text-slate-200">{title}</div>
          <div className="grid h-10 w-10 place-items-center rounded-2xl bg-white/10 text-blue-200">
            {icon}
          </div>
        </div>
        <div className="mt-5 text-4xl font-black text-white">{value}</div>
        <div className="mt-2 text-sm text-slate-400">{sub}</div>
      </div>
    </div>
  );
}

function QuickActionCard({
  href,
  icon,
  title,
  sub,
}: {
  href: string;
  icon: React.ReactNode;
  title: string;
  sub: string;
}) {
  return (
    <Link
      href={href}
      className="group rounded-[24px] border border-white/10 bg-white/[0.05] p-5 transition hover:border-blue-300/25 hover:bg-white/[0.08]"
    >
      <div className="flex items-center justify-between gap-3">
        <div className="grid h-11 w-11 place-items-center rounded-2xl bg-blue-500/15 text-blue-200">
          {icon}
        </div>
        <ArrowRight
          className="text-slate-400 transition group-hover:translate-x-1 group-hover:text-white"
          size={18}
        />
      </div>

      <div className="mt-5 text-lg font-black text-white">{title}</div>
      <div className="mt-2 text-sm leading-6 text-slate-400">{sub}</div>
    </Link>
  );
}

function PanelCard({
  title,
  subtitle,
  icon,
  children,
}: {
  title: string;
  subtitle: string;
  icon: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <section className="rounded-[30px] border border-white/10 bg-white/[0.05] p-5 shadow-[0_20px_80px_rgba(0,0,0,0.22)] backdrop-blur-xl">
      <div className="mb-5 flex items-center gap-3">
        <div className="grid h-10 w-10 place-items-center rounded-2xl bg-blue-500/15 text-blue-200">
          {icon}
        </div>
        <div>
          <div className="text-lg font-black text-white">{title}</div>
          <div className="text-sm text-slate-400">{subtitle}</div>
        </div>
      </div>
      {children}
    </section>
  );
}

function InfoStrip({
  tone,
  title,
  sub,
}: {
  tone: "warning" | "neutral" | "good";
  title: string;
  sub: string;
}) {
  const toneClass =
    tone === "warning"
      ? "border-amber-300/20 bg-amber-400/10 text-amber-100"
      : tone === "good"
      ? "border-emerald-300/20 bg-emerald-400/10 text-emerald-100"
      : "border-white/10 bg-white/[0.04] text-slate-200";

  return (
    <div className={`rounded-[22px] border p-4 ${toneClass}`}>
      <div className="font-bold">{title}</div>
      <div className="mt-1 text-sm opacity-85">{sub}</div>
    </div>
  );
}

function MiniMetric({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="rounded-[22px] border border-white/10 bg-white/[0.04] p-4">
      <div className="text-sm text-slate-400">{label}</div>
      <div className="mt-2 text-3xl font-black text-white">{value}</div>
    </div>
  );
}

function StepCard({
  step,
  title,
  sub,
}: {
  step: string;
  title: string;
  sub: string;
}) {
  return (
    <div className="rounded-[24px] border border-white/10 bg-white/[0.04] p-4">
      <div className="text-xs font-extrabold uppercase tracking-[0.18em] text-blue-200/70">
        Step {step}
      </div>
      <div className="mt-2 text-lg font-black text-white">{title}</div>
      <div className="mt-2 text-sm leading-6 text-slate-400">{sub}</div>
    </div>
  );
}

function NextItem({ text }: { text: string }) {
  return (
    <div className="rounded-[20px] border border-white/10 bg-white/[0.04] px-4 py-3 text-sm text-slate-300">
      {text}
    </div>
  );
}
