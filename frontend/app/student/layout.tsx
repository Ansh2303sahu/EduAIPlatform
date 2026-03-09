import RoleGuard from "@/components/RoleGuard";
import type { ReactNode } from "react";
import StudentSidebar from "@/components/student/StudentSidebar";
import StudentTopbar from "@/components/student/StudentTopbar";

export default function StudentLayout({ children }: { children: ReactNode }) {
  return (
    <RoleGuard allowedRoles={["student"]}>
      <div className="min-h-screen bg-[radial-gradient(circle_at_top,_#213b9f_0%,_#0b1537_38%,_#071126_100%)] text-white">
        <div className="mx-auto grid min-h-screen max-w-[1600px] grid-cols-1 gap-6 p-4 lg:grid-cols-[280px_minmax(0,1fr)] lg:p-7">
          <StudentSidebar />

          <main className="min-w-0 space-y-6">
            <StudentTopbar />
            {children}
          </main>
        </div>
      </div>
    </RoleGuard>
  );
}
