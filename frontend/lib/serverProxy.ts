import { createDevJwt } from "@/lib/serverAuth";

const BACKEND_BASE_URL = process.env.BACKEND_BASE_URL ?? "http://127.0.0.1:8000";

export async function proxyToBackend(
  path: string,
  init: RequestInit = {},
): Promise<Response> {
  const token = createDevJwt();
  const headers = new Headers(init.headers);
  headers.set("Authorization", `Bearer ${token}`);

  return fetch(`${BACKEND_BASE_URL}${path}`, {
    ...init,
    headers,
    cache: "no-store",
  });
}
