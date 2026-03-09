// lib/backendUrl.ts
const DEFAULT_BASE = "http://127.0.0.1:8000";

function stripTrailingSlash(s: string) {
  return s.endsWith("/") ? s.slice(0, -1) : s;
}

/**
 * backendUrl("me") -> `${BASE}/api/me`
 * backendUrl("/reports/123") -> `${BASE}/api/reports/123`
 *
 * If NEXT_PUBLIC_BACKEND_URL already ends with /api, it won't be duplicated.
 */
export function backendUrl(path: string) {
  const rawBase = process.env.NEXT_PUBLIC_BACKEND_URL || DEFAULT_BASE;
  const base = stripTrailingSlash(rawBase);

  const p = path.startsWith("/") ? path : `/${path}`;

  // If base already includes /api at the end, don't add again
  if (base.endsWith("/api")) return `${base}${p}`;

  return `${base}/api${p}`;
}