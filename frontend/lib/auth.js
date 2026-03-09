// lib/auth.js
import { supabase } from "./supabaseClient";
import { fetchJsonWithAuth } from "./fetchWithAuth";
import { backendUrl } from "./backendUrl";

export async function getSession() {
  const { data, error } = await supabase.auth.getSession();
  if (error) throw error;
  return data.session;
}

export async function fetchMe() {
  return fetchJsonWithAuth(backendUrl("/me"), { method: "GET" });
}

export function routeForRole(role) {
  if (role === "admin") return "/admin";
  if (role === "professor") return "/professor";
  return "/student";
}