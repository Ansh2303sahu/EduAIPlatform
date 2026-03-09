import RoleGuard from "@/components/RoleGuard";

export default function AdminProtectedLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <RoleGuard allowedRoles={["admin"]} requireAal2={true}>
      {children}
    </RoleGuard>
  );
}