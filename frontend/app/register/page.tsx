"use client";

import Link from "next/link";

export default function RegisterPage() {
  return (
    <div className="register-page">
      <div className="register-card">
        <div className="register-left">
          <div className="form-wrap">
            <h1>Create Account</h1>

            <div className="input-group">
              <label htmlFor="fullName">Full name</label>
              <input id="fullName" type="text" placeholder="Your full name" />
            </div>

            <div className="input-group">
              <label htmlFor="email">Email</label>
              <input id="email" type="email" placeholder="you@example.com" />
            </div>

            <div className="input-group">
              <label htmlFor="password">Password</label>
              <input id="password" type="password" placeholder="••••••••" />
            </div>

            <button className="register-btn">Register</button>

            <p className="bottom-text">
              Already have an account? <Link href="/login">Login</Link>
            </p>
          </div>
        </div>

        <div className="register-right">
          <div className="illustration-box">
            <img
              src="/auth-illustration.png"
              alt="Register illustration"
              className="illustration-img"
            />
          </div>

          <h2>Check Your Project Progress</h2>
          <p>
            Create your account and start using the platform with a clean,
            modern experience.
          </p>

          <div className="slider-dots">
            <span className="dot active" />
            <span className="dot" />
            <span className="dot" />
          </div>
        </div>
      </div>
    </div>
  );
}