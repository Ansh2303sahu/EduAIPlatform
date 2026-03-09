"use client";

import { motion } from "framer-motion";

/**
 * Lightweight "man" animation:
 * - subtle float
 * - arm wave
 * - eye blink
 * Pure SVG + Framer Motion (no assets needed)
 */
export default function ManAnimation() {
  return (
    <motion.div
      aria-hidden
      style={{
        width: 240,
        height: 240,
        borderRadius: 24,
        background: "rgba(255,255,255,.14)",
        border: "1px solid rgba(255,255,255,.18)",
        display: "grid",
        placeItems: "center",
        overflow: "hidden",
      }}
      animate={{ y: [0, -8, 0] }}
      transition={{ duration: 2.2, repeat: Infinity, ease: "easeInOut" }}
    >
      <svg width="210" height="210" viewBox="0 0 210 210" fill="none">
        {/* soft glow */}
        <defs>
          <radialGradient id="g" cx="0" cy="0" r="1" gradientUnits="userSpaceOnUse" gradientTransform="translate(105 80) rotate(90) scale(90 90)">
            <stop stopColor="rgba(255,255,255,.35)" />
            <stop offset="1" stopColor="rgba(255,255,255,0)" />
          </radialGradient>
          <linearGradient id="shirt" x1="60" y1="110" x2="150" y2="170">
            <stop stopColor="#ffffff" stopOpacity="0.95" />
            <stop offset="1" stopColor="#e9eeff" stopOpacity="0.95" />
          </linearGradient>
        </defs>

        <circle cx="105" cy="105" r="95" fill="url(#g)" />

        {/* body */}
        <path
          d="M70 168c3-26 20-42 35-42s32 16 35 42"
          stroke="rgba(255,255,255,.6)"
          strokeWidth="10"
          strokeLinecap="round"
        />

        {/* shirt */}
        <path
          d="M70 168c4-24 18-36 35-36s31 12 35 36"
          fill="url(#shirt)"
          stroke="rgba(47,75,255,.25)"
          strokeWidth="2"
        />

        {/* neck */}
        <rect x="95" y="98" width="20" height="18" rx="8" fill="#f2c7a8" />

        {/* head */}
        <circle cx="105" cy="78" r="30" fill="#f2c7a8" />

        {/* hair */}
        <path
          d="M78 78c2-20 18-33 36-33 17 0 30 11 31 28-9-7-21-8-31-8-12 0-25 2-36 13z"
          fill="#1f2937"
          opacity="0.95"
        />

        {/* eyes (blink animation) */}
        <motion.g
          animate={{ scaleY: [1, 1, 0.1, 1, 1] }}
          transition={{ duration: 4, repeat: Infinity, times: [0, 0.45, 0.5, 0.55, 1] }}
          style={{ transformOrigin: "105px 82px" }}
        >
          <circle cx="95" cy="82" r="3" fill="#111827" />
          <circle cx="115" cy="82" r="3" fill="#111827" />
        </motion.g>

        {/* smile */}
        <path d="M96 92c5 6 13 6 18 0" stroke="#111827" strokeWidth="3" strokeLinecap="round" />

        {/* left arm */}
        <path
          d="M78 132c-12 10-18 20-16 30"
          stroke="rgba(255,255,255,.85)"
          strokeWidth="10"
          strokeLinecap="round"
        />

        {/* right arm (waving) */}
        <motion.g
          style={{ transformOrigin: "140px 130px" }}
          animate={{ rotate: [0, -10, 0, -8, 0] }}
          transition={{ duration: 2.4, repeat: Infinity, ease: "easeInOut" }}
        >
          <path
            d="M132 132c14 6 22 16 24 28"
            stroke="rgba(255,255,255,.85)"
            strokeWidth="10"
            strokeLinecap="round"
          />
          {/* hand */}
          <circle cx="158" cy="166" r="7" fill="#f2c7a8" />
        </motion.g>

        {/* laptop */}
        <path
          d="M58 176h94c6 0 10 4 10 10v6H48v-6c0-6 4-10 10-10z"
          fill="rgba(17,24,39,.55)"
        />
        <rect x="66" y="150" width="78" height="36" rx="10" fill="rgba(255,255,255,.12)" stroke="rgba(255,255,255,.20)" />
        <circle cx="105" cy="168" r="4" fill="rgba(255,255,255,.35)" />
      </svg>
    </motion.div>
  );
}