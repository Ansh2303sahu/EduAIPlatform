"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import { Eye, EyeOff, CheckSquare, PieChart, Search, Plus } from "lucide-react";
import { supabase } from "@/lib/supabaseClient";
import { fetchMe, routeForRole } from "@/lib/auth";

export default function LoginPage() {
  const router = useRouter();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [showPassword, setShowPassword] = useState(false);
  const [remember, setRemember] = useState(false);

  async function onLogin(e: React.FormEvent) {
    e.preventDefault();
    setErr(null);

    const cleanEmail = email.trim().toLowerCase();
    if (!cleanEmail || !password) {
      setErr("Enter email and password.");
      return;
    }

    try {
      setLoading(true);

      const { data, error } = await supabase.auth.signInWithPassword({
        email: cleanEmail,
        password,
      });
      if (error) throw error;

      const token = data.session?.access_token;
      if (!token) throw new Error("Login succeeded but session token is missing.");

      const me = await fetchMe();
      router.replace(routeForRole(me.role));
    } catch (e: any) {
      setErr(e?.message || "Login failed");
    } finally {
      setLoading(false);
    }
  }

  const rightItems = useMemo(
    () => [
      { icon: <CheckSquare size={16} />, text: "Assignments tracked" },
      { icon: <PieChart size={16} />, text: "Progress insights" },
      { icon: <Search size={16} />, text: "Smart analysis" },
    ],
    []
  );

  return (
    <div className="loginPage">
      <div className="loginBackdrop" />

      <motion.div
        className="loginCard"
        initial={{ opacity: 0, y: 24, scale: 0.98 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.4, ease: "easeOut" }}
      >
        <div className="loginLeft">
          <div className="loginLeftInner">
            <div className="brandBadge">
              <div className="brandDot">🧠</div>
            </div>

            <div className="loginHeadingBlock">
              <h1>Login</h1>
              <p>Access your EduAI workspace and continue your assessment workflow.</p>
            </div>

            <form onSubmit={onLogin} className="loginForm">
              <div className="fieldGroup">
                <label htmlFor="email">Username or email</label>
                <input
                  id="email"
                  type="email"
                  placeholder="you@example.com"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  autoComplete="email"
                />
              </div>

              <div className="fieldGroup">
                <div className="passwordTopRow">
                  <label htmlFor="password">Password</label>
                  <Link href="/forgot-password" className="forgotLink">
                    Forgot password?
                  </Link>
                </div>

                <div className="passwordWrap">
                  <input
                    id="password"
                    placeholder="••••••••"
                    type={showPassword ? "text" : "password"}
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    autoComplete="current-password"
                  />
                  <button
                    type="button"
                    className="eyeBtn"
                    onClick={() => setShowPassword((v) => !v)}
                    aria-label={showPassword ? "Hide password" : "Show password"}
                  >
                    {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
                  </button>
                </div>
              </div>

              <div className="optionsRow">
                <label className="rememberMe">
                  <input
                    type="checkbox"
                    checked={remember}
                    onChange={(e) => setRemember(e.target.checked)}
                  />
                  <span>Remember me</span>
                </label>
              </div>

              {err ? <div className="errorText">{err}</div> : null}

              <button type="submit" className="loginBtn" disabled={loading}>
                {loading ? "Logging in..." : "Login"}
              </button>
            </form>

            <div className="bottomText">
              Don&apos;t have an account?{" "}
              <Link href="/register">Sign up</Link>
            </div>
          </div>
        </div>

        <div className="loginRight">
          <div className="loginRightInner">
            <div className="illustrationArea">
              <div className="floatingCard listCard">
                {rightItems.map((item, i) => (
                  <div key={i} className="miniRow">
                    <span className="miniIcon">{item.icon}</span>
                    <span className="miniLine" />
                  </div>
                ))}
              </div>

              <div className="floatingCard pieCard">
                <PieChart size={20} />
              </div>

              <div className="floatingCard searchCard">
                <Search size={16} />
              </div>

              <div className="plusIcon">
                <Plus size={22} />
              </div>

              <div className="graphLine">
                <span />
                <span />
                <span />
              </div>

              <div className="mainDashboard">
                <div className="chartTop">
                  <div className="circleChart" />
                </div>
                <div className="chartBars">
                  <span />
                  <span />
                  <span />
                </div>
              </div>

              <div className="ideaBubble">💡</div>
            </div>

            <div className="rightContent">
              <h2>Check Your Project Progress</h2>
              <p>
                Track submissions, monitor AI feedback, and view insights from one secure dashboard.
              </p>

              <div className="sliderDots">
                <span className="active" />
                <span />
                <span />
              </div>
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  );
}