"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Brain,
  ClipboardCheck,
  FileSearch,
  History,
  Home,
  ShieldAlert,
  Sparkles,
  UploadCloud,
} from "lucide-react";
import LogoutButton from "@/components/LogoutButton";

const navItems = [
  {
    href: "/professor",
    label: "Dashboard",
    icon: Home,
    exact: true,
  },
  {
    href: "/professor/queue",
    label: "Queue",
    icon: ClipboardCheck,
  },
  {
    href: "/professor/upload",
    label: "Upload",
    icon: UploadCloud,
  },
  {
    href: "/professor/history",
    label: "History",
    icon: History,
  },
];

export default function ProfessorSidebar() {
  const pathname = usePathname();

  function isActive(href: string, exact?: boolean) {
    if (exact) return pathname === href;
    return pathname === href || pathname.startsWith(`${href}/`);
  }

  return (
    <aside className="rounded-[30px] border border-white/10 bg-[linear-gradient(180deg,rgba(11,23,58,0.95),rgba(6,12,30,0.95))] p-5 shadow-[0_20px_80px_rgba(0,0,0,0.35)] backdrop-blur-xl">
      <div className="mb-8 flex items-center gap-3">
        <div className="grid h-11 w-11 place-items-center rounded-2xl bg-gradient-to-br from-blue-500 to-indigo-700 shadow-lg">
          <Brain size={20} />
        </div>

        <div>
          <div className="text-xs font-semibold uppercase tracking-[0.28em] text-blue-200/70">
            EduAI
          </div>
          <div className="text-lg font-extrabold">Professor Portal</div>
        </div>
      </div>

      <nav className="space-y-2">
        {navItems.map((item) => {
          const Icon = item.icon;
          const active = isActive(item.href, item.exact);

          return (
            <Link
              key={item.href}
              href={item.href}
              className={[
                "flex w-full items-center gap-3 rounded-2xl px-4 py-3 text-sm font-semibold transition",
                active
                  ? "bg-gradient-to-r from-blue-500/25 to-indigo-500/20 text-white shadow-[0_10px_30px_rgba(61,89,255,0.18)]"
                  : "text-slate-300 hover:bg-white/5 hover:text-white",
              ].join(" ")}
            >
              <span className="text-blue-200">
                <Icon size={17} />
              </span>
              <span>{item.label}</span>
            </Link>
          );
        })}
      </nav>

      <div className="mt-8 grid gap-4">
        <div className="rounded-[24px] border border-blue-400/20 bg-[linear-gradient(135deg,rgba(62,86,230,0.35),rgba(109,58,209,0.28))] p-4 shadow-[0_10px_40px_rgba(61,89,255,0.18)]">
          <div className="mb-3 inline-flex rounded-full border border-white/10 bg-white/10 px-3 py-1 text-xs font-semibold text-blue-100">
            Professor workspace
          </div>

          <h3 className="text-lg font-bold">Moderation and rubric review</h3>
          <p className="mt-2 text-sm leading-6 text-blue-100/75">
            Review flagged reports, inspect rubric breakdowns, and manage professor-side AI outputs.
          </p>

          <div className="mt-4 grid gap-2">
            <InfoPill icon={<ShieldAlert size={14} />} text="Moderation queue" />
            <InfoPill icon={<FileSearch size={14} />} text="Rubric reports" />
            <InfoPill icon={<Sparkles size={14} />} text="AI-assisted review" />
          </div>
        </div>

        <LogoutButton />
      </div>
    </aside>
  );
}

function InfoPill({
  icon,
  text,
}: {
  icon: React.ReactNode;
  text: string;
}) {
  return (
    <div className="inline-flex items-center gap-2 rounded-2xl border border-white/10 bg-white/8 px-3 py-2 text-xs font-semibold text-blue-100">
      {icon}
      <span>{text}</span>
    </div>
  );
}