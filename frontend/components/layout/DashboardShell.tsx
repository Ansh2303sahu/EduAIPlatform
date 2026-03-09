"use client";

import Link from "next/link";

export default function DashboardShell({
  title,
  role,
  children,
}: {
  title: string;
  role: "student" | "professor" | "admin";
  children: React.ReactNode;
}) {
  const links =
    role === "student"
      ? [
          { href: "/student", label: "Dashboard" },
          { href: "/student/history", label: "History" },
        ]
      : role === "professor"
      ? [
          { href: "/professor", label: "Dashboard" },
          { href: "/professor/results", label: "Results" },
        ]
      : [
          { href: "/admin", label: "Overview" },
          { href: "/admin/audit", label: "Audit Logs" },
          { href: "/admin/models", label: "Models" },
          { href: "/admin/workers", label: "Workers" },
          { href: "/admin/security", label: "Security" },
        ];

  return (
    <div style={{ minHeight: "100vh", background: "#f6f8fb", color: "#0f172a" }}>
      <div style={{ display: "grid", gridTemplateColumns: "260px 1fr" }}>
        {/* Sidebar */}
        <aside
          style={{
            minHeight: "100vh",
            borderRight: "1px solid #e5e7eb",
            background: "#fff",
            padding: 18,
            position: "sticky",
            top: 0,
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 18 }}>
            <div
              style={{
                width: 34,
                height: 34,
                borderRadius: 10,
                background: "#0b1b3a",
                display: "grid",
                placeItems: "center",
                color: "#fff",
                fontWeight: 900,
              }}
            >
              E
            </div>
            <div>
              <div style={{ fontWeight: 900 }}>EduAIPlatform</div>
              <div style={{ fontSize: 12, color: "#64748b" }}>{role.toUpperCase()}</div>
            </div>
          </div>

          <nav style={{ display: "grid", gap: 8 }}>
            {links.map((x) => (
              <Link
                key={x.href}
                href={x.href}
                style={{
                  textDecoration: "none",
                  color: "#0f172a",
                  padding: "10px 12px",
                  borderRadius: 12,
                  border: "1px solid #eef2f7",
                  background: "#fff",
                  fontWeight: 700,
                  fontSize: 13,
                }}
              >
                {x.label}
              </Link>
            ))}
          </nav>

          <div style={{ marginTop: 18, fontSize: 12, color: "#64748b" }}>
            Secure uploads • Role-safe reports • Local LLM
          </div>
        </aside>

        {/* Main */}
        <main style={{ padding: 22 }}>
          <div
            style={{
              background: "#fff",
              border: "1px solid #eef2f7",
              borderRadius: 16,
              padding: 16,
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              gap: 12,
            }}
          >
            <div>
              <div style={{ fontSize: 12, color: "#64748b" }}>Dashboard</div>
              <div style={{ fontSize: 22, fontWeight: 900 }}>{title}</div>
            </div>

            <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
              <Link
                href="/"
                style={{
                  textDecoration: "none",
                  padding: "10px 12px",
                  borderRadius: 12,
                  border: "1px solid #e5e7eb",
                  fontWeight: 800,
                  fontSize: 13,
                  color: "#0b1b3a",
                  background: "#fff",
                }}
              >
                Landing
              </Link>
            </div>
          </div>

          <div style={{ marginTop: 16 }}>{children}</div>
        </main>
      </div>
    </div>
  );
}