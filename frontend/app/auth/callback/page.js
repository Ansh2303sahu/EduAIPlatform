"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { supabase } from "@/lib/supabaseClient";
import { fetchMe, routeForRole } from "@/lib/auth";

export default function AuthCallbackPage() {
  const router = useRouter();
  const [msg, setMsg] = useState("Finishing sign-in...");

  useEffect(() => {
    async function run() {
      try {
        const { data, error } = await supabase.auth.getSession();
        if (error) throw error;

        if (!data.session) {
          const { data: refreshed } = await supabase.auth.refreshSession();
          if (!refreshed.session) {
            setMsg("Could not establish session. Please log in.");
            router.replace("/login");
            return;
          }
        }

        const { data: s2 } = await supabase.auth.getSession();
        const accessToken = s2.session?.access_token;

        if (!accessToken) {
          setMsg("Session missing. Please log in.");
          router.replace("/login");
          return;
        }

        const me = await fetchMe(accessToken);
        router.replace(routeForRole(me.role));
      } catch (e) {
        setMsg("Auth callback failed. Please log in.");
        router.replace("/login");
      }
    }

    run();
  }, [router]);

  return (
    <div style={{ padding: 24 }}>
      <h1>Auth</h1>
      <p>{msg}</p>
    </div>
  );
}
