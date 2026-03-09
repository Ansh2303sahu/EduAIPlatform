// lib/fetchWithAuth.js
import { supabase } from "./supabaseClient";

function makeRequestId() {
  return (globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random()}`).toString();
}

async function safeReadJson(res) {
  const ct = res.headers.get("content-type") || "";
  if (!ct.includes("application/json")) return null;
  try {
    return await res.json();
  } catch {
    return null;
  }
}

/**
 * Raw fetch with auth + headers + timeout + refresh-on-401
 */
export async function fetchWithAuth(url, options = {}, config = {}) {
  const timeoutMs = config.timeoutMs ?? 15000;
  const rid = config.requestId ?? makeRequestId();
  const clientVersion =
    config.clientVersion ?? process.env.NEXT_PUBLIC_CLIENT_VERSION ?? "dev";

  const getToken = async () => {
    const { data, error } = await supabase.auth.getSession();
    if (error) throw error;
    return data.session?.access_token || null;
  };

  const doFetch = async (token) => {
    const controller = new AbortController();
    const t = setTimeout(() => controller.abort(), timeoutMs);

    const headers = new Headers(options.headers || {});
    headers.set("x-request-id", rid);
    headers.set("x-client-version", clientVersion);

    if (token && !headers.has("Authorization")) {
      headers.set("Authorization", `Bearer ${token}`);
    }

    try {
      return await fetch(url, {
        ...options,
        headers,
        signal: controller.signal,
        cache: "no-store",
      });
    } finally {
      clearTimeout(t);
    }
  };

  let token = await getToken();
  let res = await doFetch(token);

  if (res.status === 401) {
    const { data: refreshed, error } = await supabase.auth.refreshSession();
    if (error || !refreshed?.session?.access_token) {
      await supabase.auth.signOut();
      throw {
        code: "AUTH_EXPIRED",
        message: "Session expired. Please log in again.",
        status: 401,
        requestId: rid,
      };
    }
    token = refreshed.session.access_token;
    res = await doFetch(token);
  }

  if (res.status === 401) {
    await supabase.auth.signOut();
    throw {
      code: "UNAUTHORIZED",
      message: "Unauthorized. Please log in again.",
      status: 401,
      requestId: rid,
    };
  }

  return res;
}

/**
 * JSON helper: returns parsed JSON on 2xx; throws normalized error on non-2xx.
 */
export async function fetchJsonWithAuth(url, options = {}, config = {}) {
  let res;
  try {
    res = await fetchWithAuth(url, options, config);
  } catch (e) {
    if (typeof e === "object" && e?.code) throw e;
    throw { code: "FETCH_FAILED", message: "Network/auth request failed.", status: 0 };
  }

  const body = await safeReadJson(res);

  if (!res.ok) {
    throw {
      code: body?.code || `HTTP_${res.status}`,
      message: body?.message || `Request failed (${res.status})`,
      status: res.status,
      traceId: body?.traceId || body?.trace_id || res.headers.get("x-trace-id") || undefined,
      requestId: res.headers.get("x-request-id") || config.requestId,
    };
  }

  return body;
}