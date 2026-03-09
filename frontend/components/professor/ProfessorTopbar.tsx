"use client";

import { Bell, Search, ShieldAlert } from "lucide-react";
import { usePathname } from "next/navigation";

function pageMeta(pathname: string) {
  if (pathname === "/professor") {
    return {
      eyebrow: "Professor / Dashboard",
      title: "Professor workspace",
      sub: "Review AI-generated rubric reports, flagged submissions, and moderation activity.",
    };
  }

  if (pathname.startsWith("/professor/queue")) {
    return {
      eyebrow: "Professor / Queue",
      title: "Moderation queue",
      sub: "Inspect flagged and pending professor-side reports in one review queue.",
    };
  }

  if (pathname.startsWith("/professor/upload")) {
    return {
      eyebrow: "Professor / Upload",
      title: "Upload submission",
      sub: "Upload a submission and generate professor-side structured rubric feedback.",
    };
  }

  if (pathname.startsWith("/professor/history")) {
    return {
      eyebrow: "Professor / History",
      title: "Professor report history",
      sub: "Search, filter, and reopen stored professor reports and moderation data.",
    };
  }

  if (pathname.startsWith("/professor/results")) {
    return {
      eyebrow: "Professor / Results",
      title: "Professor report",
      sub: "Review rubric breakdown, confidence metadata, moderation notes, and safety signals.",
    };
  }

  return {
    eyebrow: "Professor",
    title: "Workspace",
    sub: "Your professor AI workspace.",
  };
}

export default function ProfessorTopbar() {
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
          <TopPill icon={<Search size={15} />} text="Search" />
          <TopPill icon={<Bell size={15} />} text="Alerts" />
          <TopPill icon={<ShieldAlert size={15} />} text="Moderation ready" />
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