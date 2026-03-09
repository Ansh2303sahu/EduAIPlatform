"use client";

import { Bell, Search, Sparkles } from "lucide-react";
import { usePathname } from "next/navigation";

function pageMeta(pathname: string) {
  if (pathname === "/student") {
    return {
      eyebrow: "Student / Dashboard",
      title: "Welcome back",
      sub: "Monitor uploads, confidence analytics, and recent AI-generated feedback.",
    };
  }

  if (pathname.startsWith("/student/history")) {
    return {
      eyebrow: "Student / History",
      title: "Report history",
      sub: "Search past reports, inspect review flags, and reopen feedback anytime.",
    };
  }

  if (pathname.startsWith("/student/compare")) {
    return {
      eyebrow: "Student / Compare",
      title: "Compare reports",
      sub: "Review changes between submissions and track improvement over time.",
    };
  }

  if (pathname.startsWith("/student/results")) {
    return {
      eyebrow: "Student / Results",
      title: "Feedback results",
      sub: "View AI summary, issues, checklist, extracted content, and processing data.",
    };
  }

  return {
    eyebrow: "Student",
    title: "Workspace",
    sub: "Your student AI workspace.",
  };
}

export default function StudentTopbar() {
  const pathname = usePathname();
  const meta = pageMeta(pathname);

  return (
    <section className="rounded-[30px] border border-white/10 bg-white/[0.04] p-5 shadow-[0_20px_80px_rgba(0,0,0,0.25)] backdrop-blur-xl">
      <div className="flex flex-col gap-4 xl:flex-row xl:items-center xl:justify-between">
        <div>
          <div className="text-xs font-semibold uppercase tracking-[0.25em] text-blue-200/65">
            {meta.eyebrow}
          </div>
          <h1 className="mt-2 text-3xl font-black tracking-tight sm:text-4xl">
            {meta.title}
          </h1>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-300">
            {meta.sub}
          </p>
        </div>

        <div className="flex flex-wrap gap-3">
          <TopPill icon={<Search size={15} />} text="Smart search" />
          <TopPill icon={<Bell size={15} />} text="Notifications" />
          <TopPill icon={<Sparkles size={15} />} text="AI ready" />
        </div>
      </div>
    </section>
  );
}

function TopPill({
  icon,
  text,
}: {
  icon: React.ReactNode;
  text: string;
}) {
  return (
    <div className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-2 text-xs font-semibold text-slate-200">
      <span className="text-blue-200">{icon}</span>
      <span>{text}</span>
    </div>
  );
}