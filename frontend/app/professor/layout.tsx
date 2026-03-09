import RoleGuard from "../../components/RoleGuard";
import ProfessorSidebar from "../../components/professor/ProfessorSidebar";
import ProfessorTopbar from "../../components/professor/ProfessorTopbar";

export default function ProfessorLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <RoleGuard allowedRoles={["professor", "admin"]}>
      <div className="min-h-screen bg-[radial-gradient(circle_at_top,_#213b9f_0%,_#0b1537_38%,_#071126_100%)] text-white">
        <div className="mx-auto grid min-h-screen max-w-[1600px] grid-cols-1 gap-6 p-4 lg:grid-cols-[280px_minmax(0,1fr)] lg:p-7">
          <ProfessorSidebar />

          <main className="min-w-0 space-y-6">
            <ProfessorTopbar />
            {children}
          </main>
        </div>
      </div>
    </RoleGuard>
  );
}
