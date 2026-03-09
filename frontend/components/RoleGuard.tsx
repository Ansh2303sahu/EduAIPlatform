"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { supabase } from "@/lib/supabaseClient";
import { fetchMe, routeForRole } from "@/lib/auth";

/**
 * Props:
 * - allowedRoles: string[]
 * - requireAal2: boolean (admin)
 */
export default function RoleGuard({
  allowedRoles,
  requireAal2 = false,
  children,
}: {
  allowedRoles: string[];
  requireAal2?: boolean;
  children: React.ReactNode;
}) {
  const router = useRouter();
  const [status, setStatus] = useState<"checking" | "ok">("checking");

  useEffect(() => {
    let mounted = true;

    async function run() {
      try {
        const { data } = await supabase.auth.getSession();
        const session = data.session;

        if (!session?.access_token) {
          router.replace("/login");
          return;
        }

        const me = await fetchMe();

        if (!allowedRoles.includes(me.role)) {
          router.replace(routeForRole(me.role));
          return;
        }

        if (requireAal2) {
          const { data: aalData, error } =
            await supabase.auth.mfa.getAuthenticatorAssuranceLevel();
          if (error) throw error;

          if (aalData.currentLevel !== "aal2") {
            router.replace("/admin/mfa");
            return;
          }
        }

        if (mounted) setStatus("ok");
      } catch {
        router.replace("/login");
      }
    }

    run();
    return () => {
      mounted = false;
    };
  }, [allowedRoles, requireAal2, router]);

  if (status !== "ok") return <p style={{ padding: 24 }}>Checking access…</p>;
  return <>{children}</>;
}