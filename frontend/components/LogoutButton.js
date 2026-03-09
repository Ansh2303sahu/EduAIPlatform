"use client";

import { supabase } from "@/lib/supabaseClient";
import { useRouter } from "next/navigation";
import { LogOut } from "lucide-react";
import { useState } from "react";

export default function LogoutButton() {
  const router = useRouter();
  const [loading, setLoading] = useState(false);

  async function logout() {
    try {
      setLoading(true);
      await supabase.auth.signOut();
      router.replace("/login");
    } finally {
      setLoading(false);
    }
  }

  return (
    <button
      onClick={logout}
      disabled={loading}
      className="inline-flex items-center justify-center gap-2 rounded-2xl border border-white/10 bg-white/10 px-4 py-3 text-sm font-bold text-white transition hover:bg-white/15 disabled:cursor-not-allowed disabled:opacity-60"
    >
      <LogOut size={16} />
      <span>{loading ? "Signing out..." : "Logout"}</span>
    </button>
  );
}