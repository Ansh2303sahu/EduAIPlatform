"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import {
  ArrowRight,
  CheckCircle2,
  KeyRound,
  QrCode,
  RefreshCcw,
  Shield,
  Trash2,
} from "lucide-react";

import { supabase } from "@/lib/supabaseClient";
import { fetchMe } from "@/lib/auth";

type AAL = "aal1" | "aal2" | null;

type TotpFactor = {
  id: string;
  status?: string;
  friendly_name?: string | null;
  created_at?: string;
  factor_type?: string;
};

export default function AdminMfaPage() {
  const router = useRouter();

  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);

  const [msg, setMsg] = useState("");
  const [err, setErr] = useState("");

  const [currentAal, setCurrentAal] = useState<AAL>(null);
  const [nextAal, setNextAal] = useState<AAL>(null);

  const [factors, setFactors] = useState<TotpFactor[]>([]);
  const [selectedFactorId, setSelectedFactorId] = useState("");

  const [qr, setQr] = useState<string | null>(null);
  const [enrolledFactorId, setEnrolledFactorId] = useState<string | null>(null);

  const [code, setCode] = useState("");

  const selectedFactor = useMemo(
    () => factors.find((f) => f.id === selectedFactorId) || null,
    [factors, selectedFactorId]
  );

  const selectedStatus = (selectedFactor?.status || "").toLowerCase();
  const selectedIsVerified = selectedStatus === "verified";
  const selectedIsUnverified = selectedStatus === "unverified";
  const hasFactors = factors.length > 0;

  function ok(text: string) {
    setErr("");
    setMsg(text);
  }

  function bad(text: string) {
    setMsg("");
    setErr(text);
  }

  async function ensureAdminLoggedIn() {
    const { data, error } = await supabase.auth.getSession();
    if (error) throw error;

    const session = data.session;
    if (!session?.access_token) {
      router.replace("/login");
      return null;
    }

    const me = await fetchMe();
    if (me?.role !== "admin") {
      router.replace("/login");
      return null;
    }

    return session;
  }

  async function refreshAal() {
    const { data, error } = await supabase.auth.mfa.getAuthenticatorAssuranceLevel();
    if (error) throw error;

    const c = (data.currentLevel as AAL) || null;
    const n = (data.nextLevel as AAL) || null;

    setCurrentAal(c);
    setNextAal(n);

    return { currentLevel: c, nextLevel: n };
  }

  function mergeTotpFactors(listFactorsData: any): TotpFactor[] {
    const all = (listFactorsData?.all || []) as any[];
    const totpFromTotp = (listFactorsData?.totp || []) as any[];
    const totpDerived = all.filter((f) => (f.factor_type || "").toLowerCase() === "totp");

    const map = new Map<string, TotpFactor>();
    [...totpFromTotp, ...totpDerived].forEach((f: any) => {
      if (f?.id) map.set(String(f.id), f as TotpFactor);
    });

    return Array.from(map.values());
  }

  async function loadFactors(preferSelectId?: string) {
    const { data, error } = await supabase.auth.mfa.listFactors();
    if (error) throw error;

    const totp = mergeTotpFactors(data);

    const sorted = [...totp].sort((a, b) => {
      const av = (a.status || "").toLowerCase() === "verified" ? 1 : 0;
      const bv = (b.status || "").toLowerCase() === "verified" ? 1 : 0;
      if (av !== bv) return av - bv;
      const at = a.created_at ? new Date(a.created_at).getTime() : 0;
      const bt = b.created_at ? new Date(b.created_at).getTime() : 0;
      return bt - at;
    });

    setFactors(sorted);

    const ids = new Set(sorted.map((f) => f.id));
    if (preferSelectId && ids.has(preferSelectId)) {
      setSelectedFactorId(preferSelectId);
      return;
    }
    if (selectedFactorId && ids.has(selectedFactorId)) return;
    setSelectedFactorId(sorted[0]?.id || "");
  }

  async function boot() {
    try {
      setLoading(true);
      setMsg("");
      setErr("");
      setQr(null);
      setEnrolledFactorId(null);

      const s = await ensureAdminLoggedIn();
      if (!s) return;

      await refreshAal();
      await loadFactors();

      setLoading(false);
    } catch (e: any) {
      bad(e?.message || "Failed to load MFA state");
      setLoading(false);
    }
  }

  useEffect(() => {
    boot();
  }, []);

  async function refreshAll() {
    try {
      setBusy(true);
      setMsg("");
      setErr("");

      await supabase.auth.refreshSession();
      await refreshAal();
      await loadFactors(enrolledFactorId || undefined);

      ok("Refreshed.");
    } catch (e: any) {
      bad(e?.message || "Failed to refresh");
    } finally {
      setBusy(false);
    }
  }

  async function requireFreshOtpChallenge(factorId: string) {
    const trimmed = code.trim();
    if (!/^\d{6}$/.test(trimmed)) {
      throw new Error("Enter the 6-digit OTP code first.");
    }

    const { error } = await supabase.auth.mfa.challengeAndVerify({
      factorId,
      code: trimmed,
    });
    if (error) throw error;

    await supabase.auth.refreshSession();
    await refreshAal();
  }

  async function generateQrNewFactor() {
    try {
      setBusy(true);
      setMsg("");
      setErr("");
      setQr(null);
      setEnrolledFactorId(null);

      await ensureAdminLoggedIn();

      const friendlyName = `EduAI Admin TOTP ${new Date().toISOString()}`;

      const { data, error } = await supabase.auth.mfa.enroll({
        factorType: "totp",
        friendlyName,
      } as any);

      if (error) throw error;

      setEnrolledFactorId(data.id);
      setQr(data.totp.qr_code);

      await loadFactors(data.id);

      ok("QR generated. Scan it, then enter OTP and verify.");
    } catch (e: any) {
      bad(e?.message || "Failed to generate QR");
    } finally {
      setBusy(false);
    }
  }

  async function verifySelected() {
    try {
      setBusy(true);
      setMsg("");
      setErr("");

      await ensureAdminLoggedIn();

      if (!selectedFactorId) return bad("Select a factor first.");

      await requireFreshOtpChallenge(selectedFactorId);

      ok("Verified. Session upgraded.");
    } catch (e: any) {
      bad(e?.message || "Failed to verify");
    } finally {
      setBusy(false);
    }
  }

  async function removeSelectedFactor() {
    try {
      setBusy(true);
      setMsg("");
      setErr("");

      await ensureAdminLoggedIn();

      if (!selectedFactorId) return bad("Select a factor first.");

      if (selectedIsVerified) {
        await requireFreshOtpChallenge(selectedFactorId);
      }

      const { error } = await supabase.auth.mfa.unenroll({ factorId: selectedFactorId });
      if (error) throw error;

      setCode("");
      setQr(null);
      setEnrolledFactorId(null);

      await loadFactors();
      await refreshAal();

      ok("Factor removed. Now generate a new QR if needed.");
    } catch (e: any) {
      bad(e?.message || "Failed to remove factor");
    } finally {
      setBusy(false);
    }
  }

  async function resetAndScanAgain() {
    try {
      setBusy(true);
      setMsg("");
      setErr("");

      await ensureAdminLoggedIn();

      if (!selectedFactorId) return bad("Select a factor first.");

      if (selectedIsVerified) {
        await requireFreshOtpChallenge(selectedFactorId);
      }

      const { error: rmErr } = await supabase.auth.mfa.unenroll({ factorId: selectedFactorId });
      if (rmErr) throw rmErr;

      const friendlyName = `EduAI Admin TOTP ${new Date().toISOString()}`;
      const { data, error } = await supabase.auth.mfa.enroll({
        factorType: "totp",
        friendlyName,
      } as any);
      if (error) throw error;

      setEnrolledFactorId(data.id);
      setQr(data.totp.qr_code);
      setCode("");

      await loadFactors(data.id);
      await refreshAal();

      ok("Reset done. Scan the new QR and verify with the new code.");
    } catch (e: any) {
      bad(e?.message || "Reset failed");
    } finally {
      setBusy(false);
    }
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-[linear-gradient(180deg,#07111f_0%,#081426_45%,#050b16_100%)] px-6 py-10 text-white">
        Loading MFA...
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[linear-gradient(180deg,#07111f_0%,#081426_45%,#050b16_100%)] text-white">
      <div className="mx-auto max-w-[1350px] px-4 py-6 sm:px-6 lg:px-8">
        <div className="grid gap-6">
          <section className="rounded-[30px] border border-white/10 bg-white/[0.04] p-5 shadow-[0_20px_80px_rgba(0,0,0,0.35)] backdrop-blur-xl">
            <div className="flex flex-col gap-4 xl:flex-row xl:items-center xl:justify-between">
              <div>
                <div className="mb-2 inline-flex items-center gap-2 rounded-full border border-blue-400/20 bg-blue-500/10 px-3 py-1 text-xs font-medium text-blue-200">
                  <Shield className="h-3.5 w-3.5" />
                  Admin Security
                </div>
                <h1 className="text-3xl font-semibold tracking-tight">Admin MFA</h1>
                <p className="mt-2 text-sm text-slate-300">
                  Manage TOTP factors, verify sessions, and reset QR enrollment securely.
                </p>
              </div>

              <div className="flex flex-wrap gap-3">
                <button
                  onClick={refreshAll}
                  disabled={busy}
                  className="inline-flex items-center gap-2 rounded-2xl border border-white/10 bg-white/5 px-4 py-2 text-sm text-slate-200 transition hover:bg-white/10 disabled:opacity-60"
                >
                  <RefreshCcw className="h-4 w-4" />
                  {busy ? "Working..." : "Refresh"}
                </button>

                <button
                  onClick={() => router.replace("/admin")}
                  disabled={currentAal !== "aal2"}
                  className="inline-flex items-center gap-2 rounded-2xl bg-blue-500 px-4 py-2 text-sm font-medium text-white transition hover:bg-blue-400 disabled:opacity-60"
                >
                  Go to Admin
                  <ArrowRight className="h-4 w-4" />
                </button>
              </div>
            </div>
          </section>

          {msg ? (
            <div className="rounded-2xl border border-emerald-400/20 bg-emerald-500/10 p-4 text-emerald-200">
              {msg}
            </div>
          ) : null}

          {err ? (
            <div className="rounded-2xl border border-rose-400/20 bg-rose-500/10 p-4 text-rose-200">
              {err}
            </div>
          ) : null}

          <div className="grid gap-4 sm:grid-cols-2">
            <StatusPill label="Session" value={currentAal || "?"} good={currentAal === "aal2"} />
            <StatusPill label="Next" value={nextAal || "?"} good={nextAal === "aal2"} />
          </div>

          <div className="grid gap-4 xl:grid-cols-[1.1fr_0.9fr]">
            <section className="rounded-[28px] border border-white/10 bg-white/[0.04] p-5 shadow-[0_18px_60px_rgba(0,0,0,0.22)] backdrop-blur-xl">
              <div className="mb-4 flex items-center gap-3">
                <div className="rounded-2xl border border-white/10 bg-white/10 p-3">
                  <KeyRound className="h-5 w-5 text-white" />
                </div>
                <div>
                  <h2 className="text-lg font-semibold text-white">Factors</h2>
                  <p className="text-sm text-slate-400">
                    Verified factors need OTP before remove or reset.
                  </p>
                </div>
              </div>

              {!hasFactors ? (
                <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-4 text-sm text-slate-400">
                  No factors yet. Generate a QR on the right.
                </div>
              ) : (
                <div className="grid gap-4">
                  <div>
                    <label className="mb-2 block text-sm font-medium text-slate-300">
                      Select factor
                    </label>
                    <select
                      value={selectedFactorId}
                      onChange={(e) => setSelectedFactorId(e.target.value)}
                      className="w-full rounded-2xl border border-white/10 bg-[#0b1730] px-4 py-3 text-sm text-white outline-none"
                    >
                      {factors.map((f) => (
                        <option key={f.id} value={f.id}>
                          {(f.friendly_name || "TOTP")} — {(f.status || "unknown")} — {f.id.slice(0, 8)}…
                        </option>
                      ))}
                    </select>
                  </div>

                  <div className="flex flex-wrap items-center gap-3">
                    <Badge good={selectedIsVerified}>
                      {selectedFactor?.status || "unknown"}
                    </Badge>
                    <span className="text-xs text-slate-400">
                      Factor: {selectedFactorId ? `${selectedFactorId.slice(0, 12)}…` : "—"}
                    </span>
                  </div>

                  <div>
                    <label className="mb-2 block text-sm font-medium text-slate-300">
                      OTP code
                    </label>
                    <input
                      className="w-full rounded-2xl border border-white/10 bg-[#0b1730] px-4 py-3 text-sm text-white outline-none placeholder:text-slate-500"
                      placeholder="123456"
                      value={code}
                      onChange={(e) => setCode(e.target.value)}
                      inputMode="numeric"
                      autoComplete="one-time-code"
                    />
                  </div>

                  <div className="flex flex-wrap gap-3">
                    <button
                      onClick={verifySelected}
                      disabled={busy || !selectedFactorId}
                      className="inline-flex items-center gap-2 rounded-2xl bg-blue-500 px-4 py-3 text-sm font-medium text-white transition hover:bg-blue-400 disabled:opacity-60"
                    >
                      <CheckCircle2 className="h-4 w-4" />
                      Verify / Upgrade
                    </button>

                    <button
                      onClick={removeSelectedFactor}
                      disabled={busy || !selectedFactorId}
                      className="inline-flex items-center gap-2 rounded-2xl border border-rose-400/20 bg-rose-500/10 px-4 py-3 text-sm font-medium text-rose-200 transition hover:bg-rose-500/20 disabled:opacity-60"
                    >
                      <Trash2 className="h-4 w-4" />
                      Remove Factor
                    </button>

                    <button
                      onClick={resetAndScanAgain}
                      disabled={busy || !selectedFactorId}
                      className="inline-flex items-center gap-2 rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm font-medium text-slate-200 transition hover:bg-white/10 disabled:opacity-60"
                    >
                      <RefreshCcw className="h-4 w-4" />
                      Reset & Scan Again
                    </button>
                  </div>
                </div>
              )}
            </section>

            <section className="rounded-[28px] border border-white/10 bg-white/[0.04] p-5 shadow-[0_18px_60px_rgba(0,0,0,0.22)] backdrop-blur-xl">
              <div className="mb-4 flex items-center gap-3">
                <div className="rounded-2xl border border-white/10 bg-white/10 p-3">
                  <QrCode className="h-5 w-5 text-white" />
                </div>
                <div>
                  <h2 className="text-lg font-semibold text-white">Generate QR</h2>
                  <p className="text-sm text-slate-400">
                    This always creates a new factor so QR always appears.
                  </p>
                </div>
              </div>

              <button
                onClick={generateQrNewFactor}
                disabled={busy}
                className="inline-flex items-center gap-2 rounded-2xl bg-blue-500 px-4 py-3 text-sm font-medium text-white transition hover:bg-blue-400 disabled:opacity-60"
              >
                <QrCode className="h-4 w-4" />
                Generate QR
              </button>

              {qr ? (
                <div className="mt-5">
                  <div className="inline-block rounded-2xl border border-dashed border-white/20 bg-white/5 p-4">
                    <img src={qr} alt="TOTP QR" className="h-[220px] w-[220px] rounded-xl bg-white p-2" />
                  </div>
                  <p className="mt-3 text-sm text-slate-400">
                    Scan the QR, then enter the OTP on the left and verify.
                  </p>
                </div>
              ) : (
                <div className="mt-5 rounded-2xl border border-white/10 bg-white/[0.03] p-4 text-sm text-slate-400">
                  No QR yet.
                </div>
              )}
            </section>
          </div>
        </div>
      </div>
    </div>
  );
}

function StatusPill({
  label,
  value,
  good,
}: {
  label: string;
  value: string;
  good?: boolean;
}) {
  return (
    <div className="rounded-[24px] border border-white/10 bg-white/[0.04] p-5 shadow-[0_18px_60px_rgba(0,0,0,0.22)] backdrop-blur-xl">
      <div className="text-sm text-slate-400">{label}</div>
      <div className={`mt-3 text-2xl font-bold ${good ? "text-emerald-300" : "text-amber-300"}`}>
        {value}
      </div>
    </div>
  );
}

function Badge({
  children,
  good,
}: {
  children: React.ReactNode;
  good?: boolean;
}) {
  return (
    <span
      className={`rounded-full px-3 py-1 text-xs font-semibold ${
        good
          ? "border border-emerald-400/20 bg-emerald-500/10 text-emerald-200"
          : "border border-amber-400/20 bg-amber-500/10 text-amber-200"
      }`}
    >
      {children}
    </span>
  );
}