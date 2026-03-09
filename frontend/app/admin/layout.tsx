import RoleGuard from "@/components/RoleGuard";

export default function AdminLayout({ children }: { children: React.ReactNode }) {
  // ✅ only checks role=admin, does NOT require aal2
  // This allows /admin/mfa to load and upgrade the session.
  return <RoleGuard allowedRoles={["admin"]}>{children}</RoleGuard>;
}